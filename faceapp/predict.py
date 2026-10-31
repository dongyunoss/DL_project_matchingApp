"""참가자 사진 → 4클래스 확률(보정된 Softmax) CSV + 임베딩.

    python -m faceapp.predict --images data/participants/photos --out outputs/run1/predictions.csv
"""
import argparse
import csv
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader

from . import CLASSES
from .config import add_config_args, config_from_args
from .data import ImageListDataset, build_eval_transforms, list_images
from .engine import run_inference
from .models import load_checkpoint
from .utils import get_device


def predict_images(cfg: dict, image_paths, out_csv, ckpt_path=None) -> list[dict]:
    device = get_device()
    model, ckpt = load_checkpoint(ckpt_path or Path(cfg["output_dir"]) / "best.pt", device)
    t = float(ckpt.get("temperature", 1.0))
    paths = [Path(p) for p in image_paths]
    if not paths:
        raise ValueError("예측할 이미지가 없습니다.")
    ds = ImageListDataset(paths, build_eval_transforms(cfg, ckpt))
    loader = DataLoader(ds, batch_size=cfg["train"]["batch_size"], num_workers=cfg["data"].get("num_workers", 0))
    pc = cfg.get("predict", {})
    out = run_inference(model, loader, device, amp=cfg["train"].get("amp", True), tta=pc.get("tta", True),
                        return_embeddings=pc.get("save_embeddings", True))
    probs = (out["logits"] / t).softmax(1).numpy()

    rows = []
    for p, pr in zip(paths, probs):
        rows.append({"filename": p.name, "pred": CLASSES[int(pr.argmax())],
                     **{f"prob_{c}": float(v) for c, v in zip(CLASSES, pr)}})
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["filename", "pred", *[f"prob_{c}" for c in CLASSES]])
        w.writeheader()
        for r in rows:
            w.writerow({**r, **{f"prob_{c}": f"{r[f'prob_{c}']:.6f}" for c in CLASSES}})
    if out["embeddings"] is not None:
        np.save(out_csv.with_suffix(".embeddings.npy"), out["embeddings"].numpy())
    return rows


def read_predictions(path) -> dict[str, list[float]]:
    """predictions.csv → {filename: [p_dog, p_cat, p_rabbit, p_fox]}"""
    with open(path, newline="", encoding="utf-8") as f:
        return {r["filename"]: [float(r[f"prob_{c}"]) for c in CLASSES] for r in csv.DictReader(f)}


def main():
    p = add_config_args(argparse.ArgumentParser(description=__doc__))
    p.add_argument("--images", required=True, help="이미지 폴더 또는 단일 이미지 경로")
    p.add_argument("--out", default=None, help="출력 CSV (기본: <output_dir>/predictions.csv)")
    p.add_argument("--checkpoint", default=None)
    args = p.parse_args()
    cfg = config_from_args(args)
    src = Path(args.images)
    paths = list_images(src) if src.is_dir() else [src]
    out = args.out or Path(cfg["output_dir"]) / "predictions.csv"
    rows = predict_images(cfg, paths, out, args.checkpoint)
    for r in rows[:20]:
        probs = " ".join(f"{c}={r[f'prob_{c}']:.2f}" for c in CLASSES)
        print(f"{r['filename']:<24} → {r['pred']:<6} ({probs})")
    print(f"[saved] {out} ({len(rows)}장)")


if __name__ == "__main__":
    main()
