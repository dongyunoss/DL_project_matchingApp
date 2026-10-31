"""스모크 테스트: 실제 라벨 없이 파이프라인 전체가 끝까지 도는지 확인한다.

  합성 얼굴 이미지 생성 → 가상 라벨러 5명 CSV → 다수결 병합
  → faceapp.pipeline: 학습(soft label, mixup, freeze→finetune, AMP*) → temperature scaling → 테스트 평가
    → 참가자 사진 예측(TTA) → Grad-CAM → Top-N 매칭 → HTML 리포트
  → 수작업 vs AI 매칭 시간 비교 (가상 측정값)

  python -m scripts.smoke_test                       # 기본: resnet18 (사전학습 가중치 다운로드 없이), 96px
  python -m scripts.smoke_test --model simple_cnn
  python -m scripts.smoke_test --image-size 224      # 실제 해상도로 확인
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
from faceapp.pipeline import run_pipeline  # noqa: E402
from scripts import compare_matching_time, merge_labels  # noqa: E402

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


def build_synthetic_labels(work: Path, ids_per_class: int = 10, seed: int = 0):
    """합성 이미지 + 가상 라벨러 5명 + 다수결 병합. 반환: (img_dir, final_labels.csv, merge_stats)"""
    rng = random.Random(seed)
    img_dir, truth = make_dataset(work, ids_per_class, imgs_per_id=2, rng=rng)
    ann_dir = simulate_annotators(truth, work, rng)
    stats = merge_labels.main(["--input-dir", str(ann_dir), "--out-dir", str(work / "labels"), "--min-votes", "3"])
    return img_dir, work / "labels" / "final_labels.csv", stats


def smoke_overrides(run_dir: Path, img_dir: Path, labels_csv: Path, epochs: int = 3, image_size: int = 96,
                    pretrained: bool = False) -> list[str]:
    """CPU에서 수 초 안에 끝나도록 줄인 설정 (모델 종류는 호출하는 쪽에서 지정)."""
    return [
        f"output_dir={run_dir.as_posix()}",
        f"data.image_dir={img_dir.as_posix()}",
        f"data.labels_csv={labels_csv.as_posix()}",
        f"data.image_size={image_size}", "data.num_workers=0",
        f"model.pretrained={str(pretrained).lower()}",
        f"train.epochs={epochs}", "train.batch_size=16", "train.accum_steps=2",
        "train.freeze_backbone_epochs=1", "train.warmup_epochs=1", "train.soft_labels=true",
        "train.mixup_alpha=0.2", "train.early_stopping_patience=100",
    ]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workdir", default=str(ROOT / "outputs" / "smoke"))
    ap.add_argument("--model", default="resnet18")
    ap.add_argument("--pretrained", action="store_true", help="ImageNet 가중치 다운로드해서 사용")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--image-size", type=int, default=96)
    ap.add_argument("--ids-per-class", type=int, default=10)
    args = ap.parse_args(argv)

    t0 = time.time()
    work = Path(args.workdir)
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True, exist_ok=True)

    print("== 1. 합성 데이터 + 가상 라벨러 5명 → 다수결 병합")
    img_dir, labels_csv, stats = build_synthetic_labels(work, args.ids_per_class)
    assert stats["n_final"] > 0
    photo_dir, pj = make_participants(work, 12, random.Random(1))

    print("\n== 2. 파이프라인: 학습 → 보정 → 평가 → 예측 → Grad-CAM → 매칭 → 리포트")
    run_dir = work / "run"
    cfg = load_config(overrides=[*smoke_overrides(run_dir, img_dir, labels_csv, args.epochs, args.image_size,
                                                  args.pretrained), f"model.name={args.model}"])
    summary = run_pipeline(cfg, pj, photo_dir, top_n=3, gradcam_max=4)

    for name in ("best.pt", "splits.csv", "metrics_test.json", "confusion_matrix_test.png", "predictions.csv",
                 "predictions.embeddings.npy", "gradcam_participants.png", "matches.json", "report.html",
                 "pipeline_summary.json", "learning_curve.png"):
        assert (run_dir / name).exists(), f"결과물 누락: {name}"
    m = summary["test_metrics"]
    assert 0.0 <= m["accuracy"] <= 1.0 and 0.0 <= m["macro_f1"] <= 1.0
    with open(run_dir / "predictions.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            assert abs(sum(float(r[f"prob_{c}"]) for c in CLASSES) - 1.0) < 1e-4
    emb = np.load(run_dir / "predictions.embeddings.npy")
    assert emb.shape[0] == 12
    people = {p["id"]: p for p in json.loads(pj.read_text(encoding="utf-8"))}
    recs = json.loads((run_dir / "matches.json").read_text(encoding="utf-8"))
    for pid, cands in recs.items():
        scores = [c["score"] for c in cands]
        assert scores == sorted(scores, reverse=True)
        assert all(c["id"] != pid and people[c["id"]]["gender"] != people[pid]["gender"] for c in cands)
    assert "p01" in (run_dir / "report.html").read_text(encoding="utf-8")

    print("\n== 3. 수작업 vs AI 매칭 시간 비교 (가상 측정값)")
    timing = work / "timing.csv"
    timing.write_text("trial_id,method,n_participants,minutes,notes\n"
                      "1,manual,12,60,가상값\n1,ai_assisted,12,8,가상값\n", encoding="utf-8")
    res = compare_matching_time.main(["--timing", str(timing),
                                      "--pipeline-summary", str(run_dir / "pipeline_summary.json")])
    assert res["overall"]["reduction"] > 0

    print(f"\n[SMOKE TEST PASSED] {time.time() - t0:.1f}s  결과물: {run_dir}")
    print(f"  test acc={m['accuracy']:.3f}, macro-F1={m['macro_f1']:.3f} (합성 데이터라 수치는 의미 없음)")
    print(f"  리포트: {run_dir / 'report.html'}")


if __name__ == "__main__":
    main()
