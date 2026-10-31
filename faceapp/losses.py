"""Soft-label cross entropy (+label smoothing, class weight)와 Mixup/CutMix."""
import numpy as np
import torch
import torch.nn.functional as F


def soft_cross_entropy(logits, target_probs, label_smoothing: float = 0.0, class_weights=None):
    """target_probs: (B, K) 확률 분포 (one-hot 또는 라벨러 투표 분포).

    class_weights가 있으면 각 샘플 가중치 = Σ_k t_k·w_k 로 두고 가중 평균 (nn.CrossEntropyLoss(weight=)와 동일 규칙).
    """
    k = logits.size(1)
    t = target_probs.float()
    if label_smoothing > 0:
        t = t * (1 - label_smoothing) + label_smoothing / k
    per_sample = -(t * F.log_softmax(logits.float(), dim=1)).sum(dim=1)
    if class_weights is None:
        return per_sample.mean()
    w = (target_probs.float() * class_weights.to(logits.device)).sum(dim=1)
    return (per_sample * w).sum() / w.sum().clamp_min(1e-8)


def inverse_frequency_weights(counts) -> torch.Tensor:
    """w_c = N / (K · n_c). 학습 데이터에 없는 클래스는 1."""
    counts = np.asarray(counts, dtype=np.float64)
    n, k = counts.sum(), len(counts)
    w = np.where(counts > 0, n / (k * np.maximum(counts, 1)), 1.0)
    return torch.tensor(w, dtype=torch.float32)


def mixup_cutmix(x, targets, alpha: float, cutmix_prob: float = 0.5, rng=None):
    """배치 내 섞기. targets는 (B, K) soft 타깃이며 같은 비율로 섞인다."""
    if alpha <= 0:
        return x, targets
    rng = rng or np.random
    lam = float(rng.beta(alpha, alpha))
    perm = torch.randperm(x.size(0), device=x.device)
    if rng.rand() < cutmix_prob:
        h, w = x.shape[2:]
        cut = np.sqrt(1 - lam)
        ch, cw = int(h * cut), int(w * cut)
        cy, cx = rng.randint(h), rng.randint(w)
        y1, y2 = np.clip(cy - ch // 2, 0, h), np.clip(cy + ch // 2, 0, h)
        x1, x2 = np.clip(cx - cw // 2, 0, w), np.clip(cx + cw // 2, 0, w)
        x = x.clone()
        x[:, :, y1:y2, x1:x2] = x[perm, :, y1:y2, x1:x2]
        lam = 1 - ((y2 - y1) * (x2 - x1) / (h * w))  # 실제 잘린 면적으로 보정
    else:
        x = lam * x + (1 - lam) * x[perm]
    return x, lam * targets + (1 - lam) * targets[perm]
