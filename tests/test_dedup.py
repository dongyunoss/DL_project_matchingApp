import numpy as np
import pytest
from PIL import Image, ImageFilter, ImageOps

from faceapp.data import assert_no_identity_leak, split_by_identity
from faceapp.dedup import assign_groups, dhash, find_near_duplicates


def noise_img(seed, size=64):
    rng = np.random.RandomState(seed)
    return Image.fromarray(rng.randint(0, 255, (size, size, 3), dtype=np.uint8)).filter(ImageFilter.GaussianBlur(3))


def hashes(*imgs):
    return [(dhash(x), dhash(ImageOps.mirror(x))) for x in imgs]


def test_detects_resized_and_mirrored_copies_only():
    a = noise_img(0)
    pairs = find_near_duplicates(hashes(a, a.resize((50, 50)), ImageOps.mirror(a), noise_img(1)), threshold=5)
    assert {(i, j) for i, j, _ in pairs} == {(0, 1), (0, 2), (1, 2)}


def test_threshold_off():
    a = noise_img(0)
    assert find_near_duplicates(hashes(a, a), threshold=None) == []


def test_assign_groups_merges_identity_and_duplicates(tmp_path):
    imgs = {"a.jpg": noise_img(0), "a_copy.jpg": noise_img(0).resize((60, 60)), "b.jpg": noise_img(1),
            "b2.jpg": noise_img(2), "c.jpg": noise_img(3)}
    for name, im in imgs.items():
        im.save(tmp_path / name, quality=95)
    recs = [{"filename": "a.jpg", "identity": "1", "label": "dog"},
            {"filename": "a_copy.jpg", "identity": "2", "label": "dog"},   # 다른 identity지만 같은 이미지
            {"filename": "b.jpg", "identity": "3", "label": "cat"},
            {"filename": "b2.jpg", "identity": "3", "label": "cat"},        # 같은 identity
            {"filename": "c.jpg", "identity": "4", "label": "fox"}]
    out, dups = assign_groups(recs, tmp_path, threshold=5)
    g = {r["filename"]: r["group"] for r in out}
    assert g["a.jpg"] == g["a_copy.jpg"] and g["b.jpg"] == g["b2.jpg"]
    assert len({g["a.jpg"], g["b.jpg"], g["c.jpg"]}) == 3
    assert [(a, b) for a, b, _ in dups] == [("a.jpg", "a_copy.jpg")]


def test_split_keeps_duplicate_groups_together():
    recs = [{"filename": f"{i}.jpg", "identity": str(i), "label": ["dog", "cat", "rabbit", "fox"][i % 4],
             "group": f"g{i // 2}"} for i in range(80)]  # 2장씩 같은 그룹
    splits = split_by_identity(recs, (0.7, 0.15, 0.15), seed=0)
    where = {r["group"]: s for s, rs in splits.items() for r in rs}
    for s, rs in splits.items():
        for r in rs:
            assert where[r["group"]] == s


def test_group_leak_detected():
    splits = {"train": [{"identity": "1", "group": "g1"}], "val": [{"identity": "2", "group": "g1"}], "test": []}
    with pytest.raises(AssertionError, match="group 누수"):
        assert_no_identity_leak(splits)
