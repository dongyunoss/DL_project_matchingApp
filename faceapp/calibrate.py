"""Temperature scaling (Guo et al., ICML 2017) + ECE.

매칭 점수가 Softmax 확률을 직접 사용하므로, 과신(overconfident)된 확률을 보정한다.
정확도는 바뀌지 않고(argmax 불변) 확률의 신뢰도만 개선된다.

    python -m faceapp.calibrate --config configs/default.yaml
"""
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from .config import add_config_args, config_from_args


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor, t_min: float = 0.05, t_max: float = 20.0) -> float:
    """검증셋 NLL을 최소화하는 T > 0.

    1) log-스케일 grid search로 전역 후보를 찾고 2) LBFGS(strong Wolfe)로 log T를 미세조정한다.
    소량 데이터에서도 발산하지 않도록, NLL이 T=1보다 나빠지면 1.0을 반환한다.
    """
    if len(labels) == 0:
        return 1.0
    logits, labels = logits.detach().float(), labels.long()
    grid = torch.logspace(torch.log10(torch.tensor(t_min)), torch.log10(torch.tensor(t_max)), 200)
    losses = torch.stack([F.cross_entropy(logits / t, labels) for t in grid])
    best_t = float(grid[losses.argmin()])

    log_t = torch.tensor([float(torch.log(torch.tensor(best_t)))], requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=1.0, max_iter=50, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(logits / log_t.exp(), labels)
        loss.backward()
        return loss

    opt.step(closure)
    refined = float(log_t.exp().item())
    if t_min <= refined <= t_max and nll(logits, labels, refined) < nll(logits, labels, best_t):
        best_t = refined
    return best_t if nll(logits, labels, best_t) <= nll(logits, labels, 1.0) else 1.0


def expected_calibration_error(probs: torch.Tensor, labels: torch.Tensor, n_bins: int = 10) -> float:
    if len(labels) == 0:
        return float("nan")
    conf, pred = probs.max(dim=1)
    correct = (pred == labels).float()
    bins = torch.linspace(0, 1, n_bins + 1)
    ece = torch.zeros(1)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (conf > lo) & (conf <= hi)
        if mask.any():
            ece += mask.float().mean() * (conf[mask].mean() - correct[mask].mean()).abs()
    return float(ece.item())


def nll(logits, labels, temperature=1.0) -> float:
    if len(labels) == 0:
        return float("nan")
    return float(F.cross_entropy(logits.float() / temperature, labels.long()).item())


def calibrate_checkpoint(cfg: dict, ckpt_path=None) -> dict:
    """저장된 best.pt를 val split으로 보정하고 temperature를 체크포인트에 기록."""
    from .data import FaceDataset, build_eval_transforms
    from .engine import run_inference
    from .evaluate import load_split_records, make_loader
    from .models import load_checkpoint
    from .utils import get_device

    device = get_device()
    ckpt_path = Path(ckpt_path or Path(cfg["output_dir"]) / "best.pt")
    model, ckpt = load_checkpoint(ckpt_path, device)
    recs = load_split_records(cfg, "val")
    ds = FaceDataset(recs, cfg["data"]["image_dir"], build_eval_transforms(cfg, ckpt))
    out = run_inference(model, make_loader(ds, cfg), device, amp=cfg["train"].get("amp", True),
                        tta=cfg["predict"].get("tta", False))
    t = fit_temperature(out["logits"], out["labels"])
    lg, lb = out["logits"], out["labels"]
    result = {
        "temperature": t,
        "val_nll_before": nll(lg, lb), "val_nll_after": nll(lg, lb, t),
        "val_ece_before": expected_calibration_error(lg.softmax(1), lb),
        "val_ece_after": expected_calibration_error((lg / t).softmax(1), lb),
    }
    ckpt["temperature"] = t
    ckpt["calibration"] = result
    torch.save(ckpt, ckpt_path)
    return result


def main():
    p = add_config_args(argparse.ArgumentParser(description=__doc__))
    p.add_argument("--checkpoint", default=None)
    args = p.parse_args()
    res = calibrate_checkpoint(config_from_args(args), args.checkpoint)
    for k, v in res.items():
        print(f"{k:>16}: {v:.4f}")


if __name__ == "__main__":
    main()
