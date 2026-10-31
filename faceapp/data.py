"""라벨 CSV 로드, identity 기준 split, Dataset, 전처리/증강."""
import csv
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

from . import CLASS_TO_IDX, CLASSES, NUM_CLASSES

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
VOTE_COLS = [f"votes_{c}" for c in CLASSES]
SPLITS = ("train", "val", "test")


# ----------------------------------------------------------------------------- CSV
def read_labels_csv(path) -> list[dict]:
    """filename, identity, label [, votes_*] CSV를 읽어 레코드 리스트로 반환."""
    records = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        missing = {"filename", "identity", "label"} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: 필수 열 누락 {sorted(missing)} (필요: filename, identity, label)")
        has_votes = all(c in reader.fieldnames for c in VOTE_COLS)
        for i, row in enumerate(reader, start=2):
            label = row["label"].strip().lower()
            if label not in CLASS_TO_IDX:
                raise ValueError(f"{path}:{i} 알 수 없는 label '{label}' (허용: {CLASSES})")
            rec = {"filename": row["filename"].strip(), "identity": str(row["identity"]).strip(), "label": label}
            if has_votes:
                rec["votes"] = [float(row[c] or 0) for c in VOTE_COLS]
            records.append(rec)
    return records


def write_records_csv(records: list[dict], path, extra_cols=("split",)) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cols = ["filename", "identity", "label", *[c for c in extra_cols if any(c in r for r in records)]]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(records)


# --------------------------------------------------------------------------- split
def split_by_identity(records: list[dict], ratios=(0.7, 0.15, 0.15), seed: int = 42) -> dict[str, list[dict]]:
    """identity 단위로 train/val/test 분리. 각 identity의 최빈 라벨로 층화(stratify)한다.

    같은 인물의 모든 이미지는 반드시 하나의 split에만 들어간다.
    """
    assert abs(sum(ratios) - 1.0) < 1e-6, "split_ratios 합은 1이어야 합니다"
    by_id = defaultdict(list)
    for r in records:
        by_id[r["identity"]].append(r)

    by_label = defaultdict(list)
    for ident, recs in by_id.items():
        major = Counter(r["label"] for r in recs).most_common(1)[0][0]
        by_label[major].append(ident)

    rng = random.Random(seed)
    assign = {}
    for label in sorted(by_label):
        ids = sorted(by_label[label])
        rng.shuffle(ids)
        n = len(ids)
        n_val = int(round(n * ratios[1]))
        n_test = int(round(n * ratios[2]))
        if n >= 3:  # 소량 데이터에서도 val/test가 비지 않도록
            n_val, n_test = max(n_val, 1), max(n_test, 1)
        n_train = max(n - n_val - n_test, 0)
        for j, ident in enumerate(ids):
            assign[ident] = "train" if j < n_train else ("val" if j < n_train + n_val else "test")

    splits = {s: [] for s in SPLITS}
    for r in records:
        s = assign[r["identity"]]
        splits[s].append({**r, "split": s})
    assert_no_identity_leak(splits)
    return splits


def assert_no_identity_leak(splits: dict[str, list[dict]]) -> None:
    """같은 identity가 둘 이상의 split에 존재하면 AssertionError."""
    ids = {s: {r["identity"] for r in recs} for s, recs in splits.items()}
    names = list(ids)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            overlap = ids[names[i]] & ids[names[j]]
            assert not overlap, (
                f"identity 누수: {names[i]}와 {names[j]}에 같은 인물 {len(overlap)}명 "
                f"(예: {sorted(overlap)[:5]})"
            )


def class_counts(records: list[dict]) -> list[int]:
    c = Counter(r["label"] for r in records)
    return [c.get(k, 0) for k in CLASSES]


# ----------------------------------------------------------------------- transforms
def build_transforms(cfg: dict, train: bool):
    size = cfg["data"]["image_size"]
    norm = [T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    if not train:
        # TODO(미정): 얼굴 검출/정렬 기준. 현재는 CelebA aligned 이미지를 가정하고 resize+center crop.
        return T.Compose([T.Resize(int(size * 1.14)), T.CenterCrop(size), *norm])
    a = cfg.get("augment", {})
    ops = [T.RandomResizedCrop(size, scale=tuple(a.get("random_resized_crop_scale", [0.85, 1.0])), ratio=(0.9, 1.1))]
    if a.get("hflip", True):
        ops.append(T.RandomHorizontalFlip())
    if a.get("rotation_deg", 0):
        ops.append(T.RandomRotation(a["rotation_deg"]))
    cj = a.get("color_jitter", 0)
    if cj:
        ops.append(T.ColorJitter(brightness=cj, contrast=cj, saturation=cj))
    ops += norm
    if a.get("random_erasing", 0):
        ops.append(T.RandomErasing(p=a["random_erasing"], scale=(0.02, 0.1)))
    return T.Compose(ops)


def build_eval_transforms(cfg: dict, ckpt: dict | None = None):
    """평가/예측용 transform. 체크포인트가 있으면 학습 당시 image_size를 우선한다 (설정 불일치 방지)."""
    size = ((ckpt or {}).get("config") or {}).get("data", {}).get("image_size", cfg["data"]["image_size"])
    return build_transforms({**cfg, "data": {**cfg["data"], "image_size": size}}, train=False)


# ------------------------------------------------------------------------- dataset
def load_image(path) -> Image.Image:
    with Image.open(path) as im:
        return im.convert("RGB")


class FaceDataset(Dataset):
    """(image, hard_label, soft_target) 반환.

    soft_target은 soft_labels=True이고 votes가 있으면 투표 분포, 아니면 one-hot.
    """

    def __init__(self, records, image_dir, transform=None, soft_labels: bool = False):
        self.records = records
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.soft_labels = soft_labels

    def __len__(self):
        return len(self.records)

    def target_vector(self, rec) -> torch.Tensor:
        if self.soft_labels and rec.get("votes") is not None and sum(rec["votes"]) > 0:
            v = np.asarray(rec["votes"], dtype=np.float32)
            return torch.from_numpy(v / v.sum())
        t = torch.zeros(NUM_CLASSES)
        t[CLASS_TO_IDX[rec["label"]]] = 1.0
        return t

    def __getitem__(self, i):
        rec = self.records[i]
        img = load_image(self.image_dir / rec["filename"])
        if self.transform:
            img = self.transform(img)
        return img, CLASS_TO_IDX[rec["label"]], self.target_vector(rec)


class ImageListDataset(Dataset):
    """라벨 없는 이미지 목록 (예측용). (image, index) 반환."""

    def __init__(self, paths, transform):
        self.paths = [Path(p) for p in paths]
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.transform(load_image(self.paths[i])), i


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_images(folder) -> list[Path]:
    return sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in IMAGE_EXTS)
