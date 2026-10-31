"""스모크 테스트: 실제 라벨 없이 파이프라인 전체가 끝까지 도는지 확인한다.

  합성 얼굴 이미지 생성 → 가상 라벨러 5명 CSV → 다수결 병합 → 학습(soft label, mixup, freeze→finetune, AMP*)
  → temperature scaling → 테스트 평가 → 참가자 사진 예측(TTA) → Grad-CAM → Top-N 매칭

  python -m scripts.smoke_test                       # 기본: resnet18 (사전학습 가중치 다운로드 없이)
  python -m scripts.smoke_test --model simple_cnn
  python -m scripts.smoke_test --pretrained          # ImageNet 가중치 다운로드 포함 확인

합성 이미지는 클래스마다 귀 모양/색이 다른 도형일 뿐 실제 얼굴이 아니며, 성능 수치는 의미 없다.
결과물은 outputs/smoke/ (git 제외).
"""
import argparse
import csv
import json
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from faceapp import CLASSES  # noqa: E402
from faceapp.config import load_config  # noqa: E402
from faceapp.evaluate import evaluate, print_report  # noqa: E402
from faceapp.gradcam import save_gradcam_grid  # noqa: E402
from faceapp.matching import attach_face_probs, format_recommendations, recommend_top_n  # noqa: E402
from faceapp.predict import predict_images, read_predictions  # noqa: E402
from faceapp.train import train  # noqa: E402
from scripts import merge_labels  # noqa: E402

STYLE = {  # 클래스별 (얼굴색, 귀 모양)
    "dog": ((205, 160, 110), "floppy"),
    "cat": ((170, 170, 180), "pointy"),
    "rabbit": ((235, 225, 225), "long"),
    "fox": ((225, 120, 50), "wide"),
}


def synth_face(label: str, rng: random.Random, size: int = 96) -> Image.Image:
    bg = tuple(rng.randint(40, 120) for _ in range(3))
    im = Image.new("RGB", (size, size), bg)
    d = ImageDraw.Draw(im)
    color, ears = STYLE[label]
    color = tuple(int(np.clip(c + rng.randint(-25, 25), 0, 255)) for c in color)
    s, cx, cy = size / 96, size / 2 + rng.randint(-5, 5), size / 2 + 8 + rng.randint(-5, 5)
    if ears == "floppy":
        d.ellipse([cx - 40 * s, cy - 25 * s, cx - 22 * s, cy + 15 * s], fill=color)
        d.ellipse([cx + 22 * s, cy - 25 * s, cx + 40 * s, cy + 15 * s], fill=color)
    elif ears == "pointy":
        d.polygon([(cx - 28 * s, cy - 15 * s), (cx - 18 * s, cy - 42 * s), (cx - 6 * s, cy - 22 * s)], fill=color)
        d.polygon([(cx + 28 * s, cy - 15 * s), (cx + 18 * s, cy - 42 * s), (cx + 6 * s, cy - 22 * s)], fill=color)
    elif ears == "long":
        d.ellipse([cx - 18 * s, cy - 58 * s, cx - 6 * s, cy - 10 * s], fill=color)
        d.ellipse([cx + 6 * s, cy - 58 * s, cx + 18 * s, cy - 10 * s], fill=color)
    else:
        d.polygon([(cx - 36 * s, cy - 8 * s), (cx - 30 * s, cy - 40 * s), (cx - 8 * s, cy - 22 * s)], fill=color)
        d.polygon([(cx + 36 * s, cy - 8 * s), (cx + 30 * s, cy - 40 * s), (cx + 8 * s, cy - 22 * s)], fill=color)
    d.ellipse([cx - 28 * s, cy - 28 * s, cx + 28 * s, cy + 30 * s], fill=color)
    for ex in (-11, 11):
        d.ellipse([cx + (ex - 4) * s, cy - 8 * s, cx + (ex + 4) * s, cy], fill=(20, 20, 20))
    d.ellipse([cx - 4 * s, cy + 6 * s, cx + 4 * s, cy + 12 * s], fill=(60, 30, 30))
    im = im.filter(ImageFilter.GaussianBlur(0.6))
    arr = np.asarray(im).astype(np.int16) + np.random.RandomState(rng.randint(0, 1 << 30)).randint(-12, 13, (size, size, 3))
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def make_dataset(work: Path, ids_per_class: int, imgs_per_id: int, rng: random.Random):
    img_dir = work / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    truth = []
    ident = 0
    for label in CLASSES:
        for _ in range(ids_per_class):
            ident += 1
            for k in range(imgs_per_id):
                fname = f"{ident:05d}_{k}.jpg"
                synth_face(label, rng).save(img_dir / fname, quality=92)
                truth.append((fname, str(ident), label))
    return img_dir, truth


def simulate_annotators(truth, work: Path, rng: random.Random, n_annot=5, acc=0.8) -> Path:
    """가상 라벨러: 정답을 acc 확률로, 아니면 무작위 라벨. 일부는 기권(빈 칸)."""
    ann_dir = work / "annotators"
    ann_dir.mkdir(parents=True, exist_ok=True)
    for a in range(n_annot):
        with open(ann_dir / f"member{a + 1}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["filename", "identity", "label"])
            for fname, ident, label in truth:
                r = rng.random()
                lab = label if r < acc else ("" if r > 0.97 else rng.choice(CLASSES))
                w.writerow([fname, ident, lab])
    return ann_dir


def make_participants(work: Path, n: int, rng: random.Random) -> tuple[Path, Path]:
    photo_dir = work / "participants"
    photo_dir.mkdir(parents=True, exist_ok=True)
    hobbies = ["영화", "러닝", "독서", "게임", "요리", "여행", "사진", "음악"]
    people = []
    for i in range(n):
        pid = f"p{i + 1:02d}"
        synth_face(rng.choice(CLASSES), rng).save(photo_dir / f"{pid}.jpg")
        gender = "F" if i % 2 == 0 else "M"
        lo = rng.randint(20, 24)
        liked = rng.sample(CLASSES, 2)
        people.append({
            "id": pid, "gender": gender, "age": rng.randint(20, 27), "photo": f"{pid}.jpg",
            "hobbies": rng.sample(hobbies, 3),
            "preference": {"face_type": {liked[0]: 0.7, liked[1]: 0.3}, "age_min": lo, "age_max": lo + 4,
                           "gender": "M" if gender == "F" else "F"},
        })
    pj = work / "participants.json"
    pj.write_text(json.dumps(people, ensure_ascii=False, indent=2), encoding="utf-8")
    return photo_dir, pj


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workdir", default=str(ROOT / "outputs" / "smoke"))
    ap.add_argument("--model", default="resnet18")
    ap.add_argument("--pretrained", action="store_true", help="ImageNet 가중치 다운로드해서 사용")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--ids-per-class", type=int, default=10)
    ap.add_argument("--keep", action="store_true", help="기존 workdir 유지")
    args = ap.parse_args(argv)

    t0 = time.time()
    rng = random.Random(0)
    work = Path(args.workdir)
    if work.exists() and not args.keep:
        shutil.rmtree(work)
    work.mkdir(parents=True, exist_ok=True)

    print("== 1. 합성 데이터 + 가상 라벨러 5명")
    img_dir, truth = make_dataset(work, args.ids_per_class, imgs_per_id=2, rng=rng)
    ann_dir = simulate_annotators(truth, work, rng)
    stats = merge_labels.main(["--input-dir", str(ann_dir), "--out-dir", str(work / "labels"), "--min-votes", "3"])
    assert stats["n_final"] > 0

    print("\n== 2. 학습 (+ temperature scaling)")
    cfg = load_config(overrides=[
        f"output_dir={(work / 'run').as_posix()}",
        f"data.image_dir={img_dir.as_posix()}",
        f"data.labels_csv={(work / 'labels' / 'final_labels.csv').as_posix()}",
        "data.image_size=96", "data.num_workers=0",
        f"model.name={args.model}", f"model.pretrained={str(args.pretrained).lower()}",
        f"train.epochs={args.epochs}", "train.batch_size=16", "train.accum_steps=2",
        "train.freeze_backbone_epochs=1", "train.warmup_epochs=1", "train.soft_labels=true",
        "train.mixup_alpha=0.2", "train.early_stopping_patience=100",
    ])
    summary = train(cfg)
    assert (work / "run" / "best.pt").exists()

    print("\n== 3. 테스트 평가")
    m = evaluate(cfg, "test")
    print_report(m)
    assert 0.0 <= m["accuracy"] <= 1.0 and 0.0 <= m["macro_f1"] <= 1.0

    print("\n== 4. 참가자 사진 예측 (TTA, 보정 확률)")
    photo_dir, pj = make_participants(work, 12, rng)
    photos = sorted(photo_dir.glob("*.jpg"))
    pred_csv = work / "run" / "predictions.csv"
    rows = predict_images(cfg, photos, pred_csv)
    for r in rows:
        assert abs(sum(r[f"prob_{c}"] for c in CLASSES) - 1.0) < 1e-4
    emb = np.load(pred_csv.with_suffix(".embeddings.npy"))
    print(f"  {len(rows)}명 예측, 임베딩 shape={emb.shape}")

    print("\n== 5. Grad-CAM")
    cam_png = save_gradcam_grid(cfg, photos[:4], work / "run" / "gradcam.png")
    print(f"  {cam_png}")

    print("\n== 6. Top-N 매칭")
    people = attach_face_probs(json.loads(pj.read_text(encoding="utf-8")), read_predictions(pred_csv))
    recs = recommend_top_n(people, cfg["matching"], top_n=3)
    print(format_recommendations(dict(list(recs.items())[:4]), people))
    (work / "run" / "matches.json").write_text(json.dumps(recs, ensure_ascii=False, indent=2), encoding="utf-8")
    by_id = {p["id"]: p for p in people}
    for pid, cands in recs.items():
        scores = [c["score"] for c in cands]
        assert scores == sorted(scores, reverse=True)
        assert all(c["id"] != pid and by_id[c["id"]]["gender"] != by_id[pid]["gender"] for c in cands)

    print(f"\n[SMOKE TEST PASSED] {time.time() - t0:.1f}s  결과물: {work / 'run'}")
    print(f"  val macro-F1={summary['best_val_macro_f1']:.3f}, test acc={m['accuracy']:.3f} (합성 데이터라 수치는 의미 없음)")


if __name__ == "__main__":
    main()
