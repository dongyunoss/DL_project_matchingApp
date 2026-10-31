"""End-to-end 실행: 학습 → 테스트 평가 → 참가자 사진 예측 → Grad-CAM → Top-N 매칭 → HTML 리포트.

각 단계 소요 시간을 기록하고, AI 매칭 시간(예측+매칭)을 pipeline_summary.json에 남긴다.
이 값은 수작업 매칭과 시간을 비교할 때 쓴다 (scripts/compare_matching_time.py).

    python -m faceapp.pipeline --participants data/participants.json --photos data/participants/photos
    python -m faceapp.pipeline --participants data/participants.json --photos data/participants/photos --skip-train
"""
import argparse
import time
from contextlib import contextmanager
from pathlib import Path

from .config import add_config_args, config_from_args
from .evaluate import evaluate, print_report
from .gradcam import save_gradcam_grid
from .matching import attach_face_probs, format_recommendations, load_participants, recommend_top_n
from .predict import predict_images, read_predictions
from .report import write_html_report
from .utils import ensure_dir, save_json


class StageTimer:
    def __init__(self):
        self.seconds: dict[str, float] = {}

    @contextmanager
    def __call__(self, name: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.seconds[name] = round(time.perf_counter() - t0, 3)
            print(f"[time] {name}: {self.seconds[name]:.2f}s")


def resolve_photos(participants: list[dict], photo_dir) -> list[Path]:
    photo_dir = Path(photo_dir)
    paths, missing = [], []
    for p in participants:
        if not p.get("photo"):
            missing.append(f"{p['id']}(photo 없음)")
            continue
        path = photo_dir / Path(p["photo"]).name
        if path.exists():
            paths.append(path)
        else:
            missing.append(str(path))
    if missing:
        raise FileNotFoundError(f"참가자 사진을 찾을 수 없습니다: {missing[:10]}")
    return paths


def run_pipeline(cfg: dict, participants_path, photo_dir, skip_train: bool = False, top_n: int | None = None,
                 gradcam_max: int = 8) -> dict:
    from .train import train  # 학습 의존성은 필요할 때만

    out = ensure_dir(cfg["output_dir"])
    timer = StageTimer()
    summary: dict = {"model": cfg["model"]["name"], "output_dir": str(out)}

    if not skip_train:
        with timer("train"):
            tr = train(cfg)
        summary["train"] = {k: v for k, v in tr.items() if k != "calibration"}
        summary["calibration"] = tr.get("calibration")
        with timer("evaluate"):
            m = evaluate(cfg, "test")
        print_report(m)
        summary["test_metrics"] = {k: m[k] for k in ("n", "accuracy", "macro_precision", "macro_recall", "macro_f1", "ece", "temperature")}
    elif not (out / "best.pt").exists():
        raise FileNotFoundError(f"{out / 'best.pt'} 없음 — --skip-train 없이 먼저 학습하세요.")

    people = load_participants(participants_path)
    photos = resolve_photos(people, photo_dir)

    # ---- AI 매칭 구간 (시간 비교 대상): 사진 → 확률 → Top-N
    with timer("predict"):
        predict_images(cfg, photos, out / "predictions.csv")
    with timer("matching"):
        people = attach_face_probs(people, read_predictions(out / "predictions.csv"))
        recs = recommend_top_n(people, cfg.get("matching", {}), top_n)
        save_json(recs, out / "matches.json")

    if gradcam_max > 0:
        with timer("gradcam"):
            save_gradcam_grid(cfg, photos[:gradcam_max], out / "gradcam_participants.png")
    with timer("report"):
        report = write_html_report(recs, people, photo_dir, out / "report.html", meta={
            "모델": cfg["model"]["name"], "참가자": len(people),
            "Top-N": top_n or cfg.get("matching", {}).get("top_n", 5),
            "가중치": ", ".join(f"{k} {v}" for k, v in (cfg.get("matching", {}).get("weights") or {}).items()),
        })

    ai_sec = timer.seconds["predict"] + timer.seconds["matching"]
    summary.update({
        "timing_sec": timer.seconds,
        "ai_matching": {"n_participants": len(people), "seconds": round(ai_sec, 3),
                        "seconds_per_participant": round(ai_sec / max(len(people), 1), 4)},
        "artifacts": {"matches": str(out / "matches.json"), "report": str(report),
                      "predictions": str(out / "predictions.csv")},
    })
    save_json(summary, out / "pipeline_summary.json")
    print(format_recommendations(recs, people))
    print(f"[ai matching] 참가자 {len(people)}명 예측+매칭 {ai_sec:.2f}s  → 리포트 {report}")
    return summary


def main():
    p = add_config_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    p.add_argument("--participants", required=True, help="participants.json (형식은 README 참고, 잠정)")
    p.add_argument("--photos", required=True, help="참가자 사진 폴더 (participants.json의 photo 파일명 기준)")
    p.add_argument("--skip-train", action="store_true", help="기존 <output_dir>/best.pt로 예측·매칭만 수행")
    p.add_argument("--top-n", type=int, default=None)
    p.add_argument("--gradcam-max", type=int, default=8, help="Grad-CAM 그릴 참가자 수 (0이면 생략)")
    args = p.parse_args()
    run_pipeline(config_from_args(args), args.participants, args.photos, args.skip_train, args.top_n, args.gradcam_max)


if __name__ == "__main__":
    main()
