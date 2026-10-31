"""Grad-CAM (Selvaraju et al., ICCV 2017): 모델이 얼굴의 어느 부위를 근거로 분류했는지 시각화.

배경/머리카락/안경 등 얼굴 유형과 무관한 단서에 의존하는지 점검하는 용도.

    python -m faceapp.gradcam --images data/samples --out outputs/run1/gradcam --max 8
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from . import CLASSES
from .config import add_config_args, config_from_args
from .data import IMAGENET_MEAN, IMAGENET_STD, build_eval_transforms, list_images, load_image
from .models import load_checkpoint
from .utils import ensure_dir, get_device


class GradCAM:
    def __init__(self, model):
        self.model = model.eval()
        self.activations = None
        self.gradients = None
        self.handle = model.target_layer.register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        self.activations = output
        if output.requires_grad:
            output.register_hook(lambda g: setattr(self, "gradients", g))

    def remove(self):
        self.handle.remove()

    def __call__(self, x: torch.Tensor, class_idx: int | None = None):
        """x: (1,3,H,W). 반환: (cam (H,W) in [0,1], class_idx, logits (K,))"""
        x = x.requires_grad_(True)
        with torch.enable_grad():
            logits = self.model(x)
            idx = int(logits.argmax(1).item()) if class_idx is None else class_idx
            self.model.zero_grad(set_to_none=True)
            logits[0, idx].backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)          # GAP of gradients
        cam = F.relu((weights * self.activations).sum(1, keepdim=True))  # 가중합 + ReLU
        cam = F.interpolate(cam, size=x.shape[2:], mode="bilinear", align_corners=False)[0, 0]
        cam = cam - cam.min()
        cam = cam / cam.max().clamp_min(1e-8)
        return cam.detach().cpu().numpy(), idx, logits[0].detach().cpu()


def denormalize(x: torch.Tensor) -> np.ndarray:
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (x.detach().cpu() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()


def save_gradcam_grid(cfg: dict, image_paths, out_path, ckpt_path=None) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    device = get_device()
    model, ckpt = load_checkpoint(ckpt_path or Path(cfg["output_dir"]) / "best.pt", device)
    t = float(ckpt.get("temperature", 1.0))
    cam_fn = GradCAM(model)
    tf = build_eval_transforms(cfg, ckpt)
    paths = list(image_paths)
    fig, axes = plt.subplots(2, len(paths), figsize=(2.2 * len(paths), 4.6), squeeze=False)
    for j, p in enumerate(paths):
        x = tf(load_image(p)).unsqueeze(0).to(device)
        cam, idx, logits = cam_fn(x)
        probs = (logits / t).softmax(0).numpy()
        img = denormalize(x[0])
        axes[0, j].imshow(img)
        axes[0, j].set_title(Path(p).name[:14], fontsize=7)
        axes[1, j].imshow(img)
        axes[1, j].imshow(cam, cmap="jet", alpha=0.45)
        axes[1, j].set_title(f"{CLASSES[idx]} {probs[idx]:.2f}", fontsize=8)
        for a in axes[:, j]:
            a.axis("off")
    cam_fn.remove()
    fig.tight_layout()
    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def main():
    p = add_config_args(argparse.ArgumentParser(description=__doc__))
    p.add_argument("--images", required=True)
    p.add_argument("--out", default=None, help="출력 PNG (기본: <output_dir>/gradcam.png)")
    p.add_argument("--max", type=int, default=8)
    p.add_argument("--checkpoint", default=None)
    args = p.parse_args()
    cfg = config_from_args(args)
    src = Path(args.images)
    paths = (list_images(src) if src.is_dir() else [src])[: args.max]
    out = save_gradcam_grid(cfg, paths, args.out or Path(cfg["output_dir"]) / "gradcam.png", args.checkpoint)
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
