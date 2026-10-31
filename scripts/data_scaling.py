"""데이터 규모별 성능 비교 (제안서 3-2).

학습 데이터를 단계적으로 늘려가며(클래스 균형 유지) 같은 val/test로 평가한다.
- val/test 분할은 한 번만 만들고 모든 단계에서 고정한다.
- 각 단계의 학습 데이터는 이전 단계를 포함한다(nested). 단계 사이의 차이가 '데이터 양'만 되도록.
- --seeds를 여러 개 주면 단계마다 반복해 평균±표준편차를 낸다 (소량 데이터에서 권장).

  python -m scripts.data_scaling                                   # 기본: 25/50/75/100%
  python -m scripts.data_scaling --fractions 0.1 0.25 0.5 1.0 --seeds 1 2 3
  python -m scripts.data_scaling --smoke                           # 합성 데이터로 동작 확인

결과(<out>/): scaling.csv(단계·seed별), scaling_summary.csv/.md(평균±표준편차),
              scaling.png(데이터 수 vs Accuracy/Macro P·R·F1), loss_curves.png(단계별 Train/Val Loss)
"""
import argparse
import csv
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from faceapp import CLASSES  # noqa: E402
from faceapp.config import DEFAULT_CONFIG, load_config  # noqa: E402

METRICS = ["accuracy", "macro_precision", "macro_recall", "macro_f1"]


def balanced_nested_subsets(train_records: list[dict], fractions, seed: int = 42, balanced: bool = True) -> dict:
    """{fraction: records}. 클래스별로 고정 순서를 정해 앞에서부터 자르므로 작은 단계 ⊂ 큰 단계.

    balanced=True: 클래스별 개수 = round(fraction × 가장 적은 클래스 수) → 모든 단계에서 클래스 수 동일.
    balanced=False: 클래스별로 자기 개수의 fraction (원래 비율 유지).
    """
    rng = random.Random(seed)
    by_class = defaultdict(list)
    for r in sorted(train_records, key=lambda r: r["filename"]):
        by_class[r["label"]].append(r)
    for recs in by_class.values():
        rng.shuffle(recs)
    present = [c for c in CLASSES if by_class.get(c)]
    min_n = min(len(by_class[c]) for c in present)
    out = {}
    for f in sorted(fractions):
        if not 0 < f <= 1:
            raise ValueError(f"fraction은 (0, 1] 범위여야 합니다: {f}")
        sub = []
        for c in present:
            k = max(1, round(f * (min_n if balanced else len(by_class[c]))))
            sub += by_class[c][:k]
        out[f] = sub
    return out


def summarize(rows: list[dict]) -> list[dict]:
    by_f = defaultdict(list)
    for r in rows:
        by_f[r["fraction"]].append(r)
    out = []
    for f in sorted(by_f):
        rs = by_f[f]
        s = {"fraction": f, "n_train": rs[0]["n_train"], "n_runs": len(rs)}
        for m in METRICS + ["best_val_loss"]:
            vals = np.array([r[m] for r in rs], dtype=float)
            s[f"{m}_mean"], s[f"{m}_std"] = float(vals.mean()), float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        out.append(s)
    return out


def to_markdown(summary: list[dict]) -> str:
    head = "| 학습 데이터 비율 | 학습 이미지 수 | 반복 | Accuracy | Macro Precision | Macro Recall | Macro F1 | Best Val Loss |"
    lines = [head, "|---|---|---|---|---|---|---|---|"]
    for s in summary:
        cells = [f"{s['fraction']:.0%}", str(s["n_train"]), str(s["n_runs"])]
        cells += [f"{s[f'{m}_mean']:.3f} ± {s[f'{m}_std']:.3f}" for m in METRICS + ["best_val_loss"]]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def plot(summary, histories, out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = [s["n_train"] for s in summary]
    fig, ax = plt.subplots(figsize=(6, 4))
    for m, label in zip(METRICS, ["Accuracy", "Macro Precision", "Macro Recall", "Macro F1"]):
        y = np.array([s[f"{m}_mean"] for s in summary])
        e = np.array([s[f"{m}_std"] for s in summary])
        ax.plot(x, y, marker="o", label=label)
        ax.fill_between(x, y - e, y + e, alpha=0.15)
    ax.set_xlabel("number of training images")
    ax.set_ylabel("test score")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend()
    ax.set_title("Data scale vs. performance")
    fig.tight_layout()
    fig.savefig(out_dir / "scaling.png", dpi=140)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for (f, hist), color in zip(sorted(histories.items()), plt.cm.viridis(np.linspace(0, 0.9, len(histories)))):
        ep = [h["epoch"] for h in hist]
        axes[0].plot(ep, [h["train_loss"] for h in hist], color=color, label=f"{f:.0%}")
        axes[1].plot(ep, [h["val_loss"] for h in hist], color=color, label=f"{f:.0%}")
    axes[0].set_title("Training loss")
    axes[1].set_title("Validation loss")
    for a in axes:
        a.set_xlabel("epoch")
        a.grid(alpha=0.3)
    axes[1].legend(title="train data")
    fig.tight_layout()
    fig.savefig(out_dir / "loss_curves.png", dpi=140)
    plt.close(fig)


def main(argv=None):
    from faceapp.evaluate import evaluate
    from faceapp.train import prepare_splits, train

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    ap.add_argument("--fractions", nargs="*", type=float, default=[0.25, 0.5, 0.75, 1.0])
    ap.add_argument("--seeds", nargs="*", type=int, default=[42], help="단계마다 반복할 학습 seed")
    ap.add_argument("--stratified", action="store_true", help="클래스 균형 대신 원래 클래스 비율 유지")
    ap.add_argument("--out", default=None, help="결과 폴더 (기본: outputs/data_scaling)")
    ap.add_argument("--smoke", action="store_true", help="합성 데이터 + 짧은 학습으로 동작 확인")
    args = ap.parse_args(argv)

    out = Path(args.out or ROOT / "outputs" / ("smoke_scaling" if args.smoke else "data_scaling"))
    overrides = list(args.set)
    if args.smoke:
        from scripts.smoke_test import build_synthetic_labels, smoke_overrides
        if out.exists():
            shutil.rmtree(out)
        img_dir, labels_csv, _ = build_synthetic_labels(out / "data", ids_per_class=14)
        overrides = smoke_overrides(out, img_dir, labels_csv, epochs=2) + ["model.name=resnet18"] + overrides
    base_cfg = load_config(args.config, [*overrides, f"output_dir={(out / 'base').as_posix()}"])
    splits = prepare_splits(base_cfg)  # val/test 고정
    print(f"[scaling] val={len(splits['val'])} test={len(splits['test'])} 고정, train 후보 {len(splits['train'])}장")

    rows, histories = [], {}
    for seed in args.seeds:
        subsets = balanced_nested_subsets(splits["train"], args.fractions, seed=seed, balanced=not args.stratified)
        for frac, sub in subsets.items():
            name = f"f{int(round(frac * 100)):03d}_s{seed}"
            cfg = load_config(args.config, [*overrides, f"output_dir={(out / name).as_posix()}", f"seed={seed}"])
            print(f"\n===== [{name}] train {len(sub)}장 (클래스별 {dict(zip(CLASSES, [sum(r['label'] == c for r in sub) for c in CLASSES]))})")
            summary = train(cfg, {"train": sub, "val": splits["val"], "test": splits["test"]})
            m = evaluate(cfg, "test")
            rows.append({"fraction": frac, "seed": seed, "n_train": len(sub), "best_epoch": summary["best_epoch"],
                         "best_val_loss": summary["best_val_loss"], **{k: m[k] for k in METRICS}, "ece": m["ece"]})
            if seed == args.seeds[0]:
                with open(out / name / "history.csv", encoding="utf-8") as f:
                    histories[frac] = [{k: float(v) if k != "phase" else v for k, v in h.items()}
                                       for h in csv.DictReader(f)]

    out.mkdir(parents=True, exist_ok=True)
    with open(out / "scaling.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summary = summarize(rows)
    with open(out / "scaling_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)
    md = to_markdown(summary)
    (out / "scaling_summary.md").write_text(md, encoding="utf-8")
    plot(summary, histories, out)
    print("\n" + md)
    print(f"[saved] {out / 'scaling_summary.md'}, {out / 'scaling.png'}, {out / 'loss_curves.png'}")
    return summary


if __name__ == "__main__":
    main()
