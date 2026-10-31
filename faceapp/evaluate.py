"""테스트셋 평가: Accuracy, Macro F1, 클래스별 P/R/F1, Confusion Matrix, ECE.

    python -m faceapp.evaluate --config configs/default.yaml [--split test]
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from . import CLASS_KO, CLASSES, NUM_CLASSES
from .calibrate import expected_calibration_error, nll
from .config import add_config_args, config_from_args
from .data import FaceDataset, build_eval_transforms
from .engine import run_inference
from .utils import ensure_dir, get_device, save_json


# --------------------------------------------------------------------------- metrics
def confusion_matrix(y_true, y_pred, num_classes: int = NUM_CLASSES) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)  # 행=정답, 열=예측
    for t, p in zip(np.asarray(y_true, dtype=int), np.asarray(y_pred, dtype=int)):
        cm[t, p] += 1
    return cm


def metrics_from_confusion(cm: np.ndarray) -> dict:
    tp = np.diag(cm).astype(float)
    pred_pos, true_pos = cm.sum(0).astype(float), cm.sum(1).astype(float)
    precision = np.divide(tp, pred_pos, out=np.zeros_like(tp), where=pred_pos > 0)
    recall = np.divide(tp, true_pos, out=np.zeros_like(tp), where=true_pos > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(tp), where=denom > 0)
    present = true_pos > 0  # 정답에 존재하는 클래스만 macro 평균 (sklearn labels=present와 동일)
    total = cm.sum()
    return {
        "accuracy": float(tp.sum() / total) if total else 0.0,
        "macro_f1": float(f1[present].mean()) if present.any() else 0.0,
        "per_class": {c: {"precision": float(precision[i]), "recall": float(recall[i]),
                          "f1": float(f1[i]), "support": int(true_pos[i])}
                      for i, c in enumerate(CLASSES)},
    }


def classification_metrics(y_true, y_pred) -> dict:
    return metrics_from_confusion(confusion_matrix(y_true, y_pred))


# --------------------------------------------------------------------------- helpers
def make_loader(ds, cfg, shuffle=False, drop_last=False):
    return DataLoader(ds, batch_size=cfg["train"]["batch_size"], shuffle=shuffle, drop_last=drop_last,
                      num_workers=cfg["data"].get("num_workers", 0), pin_memory=torch.cuda.is_available())


def load_split_records(cfg: dict, split: str) -> list[dict]:
    """학습 때 저장한 outputs/<run>/splits.csv에서 해당 split 레코드를 읽는다 (재현성 보장)."""
    path = Path(cfg["output_dir"]) / "splits.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} 없음 — 먼저 python -m faceapp.train 을 실행하세요.")
    with open(path, newline="", encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if r["split"] == split]


def save_confusion_plot(cm: np.ndarray, path, title="Confusion Matrix") -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(CLASSES)), CLASSES)
    ax.set_yticks(range(len(CLASSES)), CLASSES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    vmax = cm.max() if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > vmax / 2 else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------------------ main
def evaluate(cfg: dict, split: str = "test", ckpt_path=None) -> dict:
    from .models import load_checkpoint

    device = get_device()
    out_dir = ensure_dir(cfg["output_dir"])
    model, ckpt = load_checkpoint(ckpt_path or out_dir / "best.pt", device)
    t = float(ckpt.get("temperature", 1.0))

    recs = load_split_records(cfg, split)
    if not recs:
        raise ValueError(f"'{split}' split이 비어 있습니다.")
    ds = FaceDataset(recs, cfg["data"]["image_dir"], build_eval_transforms(cfg, ckpt))
    out = run_inference(model, make_loader(ds, cfg), device, amp=cfg["train"].get("amp", True),
                        tta=cfg["predict"].get("tta", False))
    logits, labels = out["logits"], out["labels"]
    probs = (logits / t).softmax(1)
    preds = probs.argmax(1)

    cm = confusion_matrix(labels.numpy(), preds.numpy())
    m = metrics_from_confusion(cm)
    m.update({
        "split": split, "n": len(recs), "model": ckpt["model_name"], "temperature": t,
        "nll": nll(logits, labels, t), "ece": expected_calibration_error(probs, labels),
        "ece_uncalibrated": expected_calibration_error(logits.softmax(1), labels),
        "confusion_matrix": cm.tolist(),
    })
    save_json(m, out_dir / f"metrics_{split}.json")
    np.savetxt(out_dir / f"confusion_matrix_{split}.csv", cm, fmt="%d", delimiter=",",
               header=",".join(CLASSES), comments="")
    save_confusion_plot(cm, out_dir / f"confusion_matrix_{split}.png", f"{ckpt['model_name']} ({split})")

    with open(out_dir / f"predictions_{split}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["filename", "identity", "label", "pred", *[f"prob_{c}" for c in CLASSES]])
        for r, p, pr in zip(recs, preds.tolist(), probs.tolist()):
            w.writerow([r["filename"], r["identity"], r["label"], CLASSES[p], *[f"{v:.4f}" for v in pr]])
    return m


def print_report(m: dict) -> None:
    print(f"[{m['split']}] n={m['n']}  Accuracy={m['accuracy']:.4f}  Macro-F1={m['macro_f1']:.4f}  "
          f"ECE={m['ece']:.4f} (보정 전 {m['ece_uncalibrated']:.4f}, T={m['temperature']:.3f})")
    for c, v in m["per_class"].items():
        print(f"  {c:>6} ({CLASS_KO[c]}): P={v['precision']:.3f} R={v['recall']:.3f} F1={v['f1']:.3f} n={v['support']}")
    print("  confusion matrix (행=정답, 열=예측, 순서", CLASSES, ")")
    for row in m["confusion_matrix"]:
        print("   ", row)


def main():
    p = add_config_args(argparse.ArgumentParser(description=__doc__))
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--checkpoint", default=None)
    args = p.parse_args()
    print_report(evaluate(config_from_args(args), args.split, args.checkpoint))


if __name__ == "__main__":
    main()
