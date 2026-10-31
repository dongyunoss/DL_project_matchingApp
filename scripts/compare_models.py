"""모델/학습 전략 비교 실험: 여러 설정을 같은 split으로 학습·평가하고 비교표를 만든다.

기본 비교: ResNet18 transfer learning (ImageNet 사전학습) vs SimpleCNN (처음부터 학습).

  python -m scripts.compare_models                                     # configs/default.yaml 데이터로
  python -m scripts.compare_models --runs "resnet18_transfer:model.name=resnet18" `
      "resnet18_scratch:model.name=resnet18 model.pretrained=false" "cnn_baseline:model.name=simple_cnn"
  python -m scripts.compare_models --runs "hard:train.soft_labels=false" "soft:train.soft_labels=true"
  python -m scripts.compare_models --smoke                             # 합성 데이터로 동작 확인

결과: <out>/comparison.csv, <out>/comparison.md (보고서에 붙여 넣을 수 있는 표), <out>/<run>/...
모든 run은 같은 seed·라벨 CSV를 쓰므로 split이 동일하며, 이를 assert로 확인한다.
"""
import argparse
import csv
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from faceapp.config import DEFAULT_CONFIG, load_config  # noqa: E402

DEFAULT_RUNS = ["resnet18_transfer:model.name=resnet18 model.pretrained=true",
                "cnn_baseline:model.name=simple_cnn"]
COLUMNS = ["run", "model", "pretrained", "params_M", "best_epoch", "train_min", "val_macro_f1",
           "test_accuracy", "test_macro_precision", "test_macro_recall", "test_macro_f1", "test_ece", "temperature"]


def parse_run_spec(spec: str) -> tuple[str, list[str]]:
    """'name:key=val key=val' → (name, ['key=val', ...])"""
    if ":" not in spec:
        raise ValueError(f"run 형식은 'name:key=val ...' 입니다: {spec}")
    name, rest = spec.split(":", 1)
    name = name.strip()
    if not name or any(c in name for c in r'\/:*?"<>| '):
        raise ValueError(f"run 이름이 올바르지 않습니다: '{name}'")
    overrides = rest.split()
    for o in overrides:
        if "=" not in o:
            raise ValueError(f"override 형식은 key=val 입니다: {o}")
    return name, overrides


def to_markdown(rows: list[dict]) -> str:
    def fmt(v):
        return f"{v:.4f}" if isinstance(v, float) else str(v)
    lines = ["| " + " | ".join(COLUMNS) + " |", "|" + "---|" * len(COLUMNS)]
    lines += ["| " + " | ".join(fmt(r.get(c, "")) for c in COLUMNS) + " |" for r in rows]
    return "\n".join(lines) + "\n"


def run_one(name: str, cfg: dict) -> dict:
    from faceapp.evaluate import evaluate
    from faceapp.models import build_model, count_parameters
    from faceapp.train import train

    t0 = time.time()
    summary = train(cfg)
    train_min = (time.time() - t0) / 60
    m = evaluate(cfg, "test")
    params = count_parameters(build_model(cfg, pretrained=False))[0] / 1e6
    return {"run": name, "model": cfg["model"]["name"], "pretrained": cfg["model"].get("pretrained"),
            "params_M": round(params, 2), "best_epoch": summary["best_epoch"], "train_min": round(train_min, 2),
            "val_macro_f1": summary["best_val_macro_f1"], "test_accuracy": m["accuracy"],
            "test_macro_precision": m["macro_precision"], "test_macro_recall": m["macro_recall"],
            "test_macro_f1": m["macro_f1"], "test_ece": m["ece"], "temperature": m["temperature"]}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="모든 run에 공통 적용할 override")
    ap.add_argument("--runs", nargs="*", default=DEFAULT_RUNS, help="'이름:key=val key=val' 목록")
    ap.add_argument("--out", default=None, help="결과 폴더 (기본: outputs/compare, --smoke면 outputs/smoke_compare)")
    ap.add_argument("--smoke", action="store_true", help="합성 데이터 + 짧은 학습으로 동작 확인 (사전학습 가중치 없이)")
    args = ap.parse_args(argv)

    out = Path(args.out or ROOT / "outputs" / ("smoke_compare" if args.smoke else "compare"))
    common = list(args.set)
    if args.smoke:
        from scripts.smoke_test import build_synthetic_labels, smoke_overrides
        if out.exists():
            shutil.rmtree(out)
        img_dir, labels_csv, _ = build_synthetic_labels(out / "data")
        common = smoke_overrides(out, img_dir, labels_csv, epochs=2) + common

    specs = [parse_run_spec(s) for s in args.runs]
    names = [n for n, _ in specs]
    if len(set(names)) != len(names):
        raise SystemExit(f"run 이름이 중복됩니다: {names}")

    rows, split_files = [], []
    for name, overrides in specs:
        extra = ["model.pretrained=false"] if args.smoke else []
        cfg = load_config(args.config, [*common, *overrides, *extra, f"output_dir={(out / name).as_posix()}"])
        print(f"\n===== [{name}] {' '.join(overrides)}")
        rows.append(run_one(name, cfg))
        split_files.append((out / name / "splits.csv").read_text(encoding="utf-8"))
    assert all(s == split_files[0] for s in split_files), "run마다 split이 다릅니다 — 공정한 비교가 아닙니다"

    out.mkdir(parents=True, exist_ok=True)
    with open(out / "comparison.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    md = to_markdown(rows)
    (out / "comparison.md").write_text(md, encoding="utf-8")
    print("\n" + md)
    print(f"[saved] {out / 'comparison.csv'}, {out / 'comparison.md'}")
    return rows


if __name__ == "__main__":
    main()
