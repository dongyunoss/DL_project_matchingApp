"""학습: Transfer learning(ResNet18) → 단계적 fine-tuning.

    python -m faceapp.train --config configs/default.yaml
    python -m faceapp.train --set model.name=simple_cnn output_dir=outputs/cnn_baseline   # 베이스라인
    python -m faceapp.train --set train.batch_size=16 train.accum_steps=2                 # OOM 시

포함 기법: AMP, gradient accumulation, discriminative LR, warmup+cosine, backbone freeze→unfreeze,
label smoothing, soft label(투표 분포), class-weighted loss, Mixup/CutMix, early stopping,
학습 후 temperature scaling.
"""
import argparse
import csv
import math
import time
from pathlib import Path

import numpy as np
import torch

from . import CLASSES
from .calibrate import calibrate_checkpoint
from .config import add_config_args, config_from_args
from .data import (FaceDataset, build_transforms, class_counts, read_labels_csv, split_by_identity,
                   write_records_csv)
from .evaluate import classification_metrics, make_loader
from .engine import run_inference
from .losses import inverse_frequency_weights, mixup_cutmix, soft_cross_entropy
from .models import build_model, count_parameters, save_checkpoint
from .utils import ensure_dir, get_device, save_json, set_seed


def warmup_cosine(total_steps: int, warmup_steps: int):
    def fn(step):
        if warmup_steps > 0 and step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))
    return fn


def prepare_splits(cfg: dict) -> dict:
    records = read_labels_csv(cfg["data"]["labels_csv"])
    splits = split_by_identity(records, cfg["data"]["split_ratios"], cfg["seed"])
    out = ensure_dir(cfg["output_dir"])
    write_records_csv([r for s in ("train", "val", "test") for r in splits[s]], out / "splits.csv")
    return splits


def train_one_epoch(model, loader, optimizer, scheduler, scaler, device, cfg, class_weights, rng):
    tc = cfg["train"]
    model.train()
    if not any(p.requires_grad for p in model.backbone.parameters()):
        model.backbone.eval()  # freeze 단계: BN running stats도 사전학습 값 유지
    use_amp = tc.get("amp", True) and device.type == "cuda"
    accum = max(1, int(tc.get("accum_steps", 1)))
    max_batches = tc.get("max_batches_per_epoch")
    total_loss, correct, n = 0.0, 0, 0
    optimizer.zero_grad(set_to_none=True)
    n_batches = min(len(loader), max_batches or len(loader))

    for i, (x, y, target) in enumerate(loader):
        if i >= n_batches:
            break
        x, y, target = x.to(device, non_blocking=True), y.to(device), target.to(device)
        x, target_mixed = mixup_cutmix(x, target, tc.get("mixup_alpha", 0.0), tc.get("cutmix_prob", 0.5), rng)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(x)
        loss = soft_cross_entropy(logits, target_mixed, tc.get("label_smoothing", 0.0), class_weights)
        scaler.scale(loss / accum).backward()

        if (i + 1) % accum == 0 or (i + 1) == n_batches:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        n += x.size(0)
    return total_loss / max(n, 1), correct / max(n, 1)


def validate(model, loader, device, cfg, class_weights):
    out = run_inference(model, loader, device, amp=cfg["train"].get("amp", True))
    logits, labels = out["logits"], out["labels"]
    if len(labels) == 0:
        return {"loss": float("nan"), "accuracy": 0.0, "macro_f1": 0.0}
    onehot = torch.nn.functional.one_hot(labels.long(), len(CLASSES)).float()
    loss = soft_cross_entropy(logits, onehot, 0.0, class_weights).item()
    m = classification_metrics(labels.numpy(), logits.argmax(1).numpy())
    return {"loss": loss, "accuracy": m["accuracy"], "macro_f1": m["macro_f1"]}


def train(cfg: dict) -> dict:
    set_seed(cfg["seed"])
    rng = np.random.RandomState(cfg["seed"])
    device = get_device()
    tc = cfg["train"]
    out_dir = ensure_dir(cfg["output_dir"])

    splits = prepare_splits(cfg)
    counts = class_counts(splits["train"])
    print(f"[data] train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])} "
          f"| train class counts {dict(zip(CLASSES, counts))}")
    if not splits["train"] or not splits["val"]:
        raise ValueError("train/val split이 비어 있습니다. 라벨 수를 늘리세요.")

    img_dir, soft = cfg["data"]["image_dir"], tc.get("soft_labels", False)
    train_ds = FaceDataset(splits["train"], img_dir, build_transforms(cfg, train=True), soft_labels=soft)
    val_ds = FaceDataset(splits["val"], img_dir, build_transforms(cfg, train=False))
    train_loader = make_loader(train_ds, cfg, shuffle=True, drop_last=len(train_ds) > tc["batch_size"])
    val_loader = make_loader(val_ds, cfg)

    model = build_model(cfg).to(device)
    total, _ = count_parameters(model)
    print(f"[model] {model.name} pretrained={cfg['model'].get('pretrained')} params={total/1e6:.2f}M device={device}")

    class_weights = inverse_frequency_weights(counts).to(device) if tc.get("class_weighted_loss") else None
    optimizer = torch.optim.AdamW(model.param_groups(tc["lr"], tc.get("backbone_lr_factor", 0.1),
                                                     tc.get("weight_decay", 1e-4)))
    accum = max(1, int(tc.get("accum_steps", 1)))
    batches = min(len(train_loader), tc.get("max_batches_per_epoch") or len(train_loader))
    steps_per_epoch = math.ceil(batches / accum)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, warmup_cosine(steps_per_epoch * tc["epochs"], steps_per_epoch * tc.get("warmup_epochs", 0)))
    scaler = torch.amp.GradScaler("cuda", enabled=tc.get("amp", True) and device.type == "cuda")

    freeze_epochs = tc.get("freeze_backbone_epochs", 0) if model.name != "simple_cnn" else 0
    best, best_epoch, patience, history = (-1.0, float("inf")), -1, 0, []
    hist_path = out_dir / "history.csv"
    for epoch in range(tc["epochs"]):
        model.set_backbone_trainable(epoch >= freeze_epochs)
        phase = "head-only" if epoch < freeze_epochs else "finetune"
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, scheduler, scaler, device, cfg,
                                          class_weights, rng)
        val = validate(model, val_loader, device, cfg, class_weights)
        row = {"epoch": epoch + 1, "phase": phase, "train_loss": tr_loss, "train_acc": tr_acc,
               "val_loss": val["loss"], "val_acc": val["accuracy"], "val_macro_f1": val["macro_f1"],
               "lr_head": optimizer.param_groups[1]["lr"], "sec": time.time() - t0}
        history.append(row)
        print(f"[epoch {epoch+1:>3}/{tc['epochs']}] {phase:<9} train_loss={tr_loss:.4f} acc={tr_acc:.3f} | "
              f"val_loss={val['loss']:.4f} acc={val['accuracy']:.3f} macroF1={val['macro_f1']:.3f} "
              f"({row['sec']:.1f}s)")

        key = (val["macro_f1"], -val["loss"])
        if key > best:
            best, best_epoch, patience = key, epoch + 1, 0
            save_checkpoint(out_dir / "best.pt", model, cfg, epoch=epoch + 1, val=val)
        else:
            patience += 1
            if patience >= tc.get("early_stopping_patience", 1_000_000):
                print(f"[early stop] {patience} epoch 동안 개선 없음")
                break

    with open(hist_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0]))
        w.writeheader()
        w.writerows(history)
    save_learning_curve(history, out_dir / "learning_curve.png")

    summary = {"best_epoch": best_epoch, "best_val_macro_f1": best[0], "best_val_loss": -best[1]}
    if cfg.get("calibration", {}).get("enabled", True):
        summary["calibration"] = calibrate_checkpoint(cfg, out_dir / "best.pt")
        print(f"[calibration] T={summary['calibration']['temperature']:.3f} "
              f"val ECE {summary['calibration']['val_ece_before']:.4f} → {summary['calibration']['val_ece_after']:.4f}")
    save_json(summary, out_dir / "train_summary.json")
    print(f"[done] best epoch {best_epoch}, val macro-F1 {best[0]:.4f} → {Path(out_dir) / 'best.pt'}")
    return summary


def save_learning_curve(history, path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    ep = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    axes[0].plot(ep, [h["train_loss"] for h in history], label="train")
    axes[0].plot(ep, [h["val_loss"] for h in history], label="val")
    axes[0].set_title("loss")
    axes[1].plot(ep, [h["train_acc"] for h in history], label="train acc")
    axes[1].plot(ep, [h["val_acc"] for h in history], label="val acc")
    axes[1].plot(ep, [h["val_macro_f1"] for h in history], label="val macro-F1")
    axes[1].set_title("accuracy / F1")
    for a in axes:
        a.set_xlabel("epoch")
        a.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    args = add_config_args(argparse.ArgumentParser(description=__doc__)).parse_args()
    train(config_from_args(args))


if __name__ == "__main__":
    main()
