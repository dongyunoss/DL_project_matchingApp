import pytest

from faceapp import CLASSES
from faceapp.data import FaceDataset, assert_no_identity_leak, read_labels_csv, split_by_identity


def make_records(n_ids=40, imgs_per_id=3):
    return [{"filename": f"{i}_{k}.jpg", "identity": str(i), "label": CLASSES[i % 4]}
            for i in range(n_ids) for k in range(imgs_per_id)]


def test_split_has_no_identity_leak_and_covers_all():
    recs = make_records()
    splits = split_by_identity(recs, (0.7, 0.15, 0.15), seed=1)
    assert sum(len(v) for v in splits.values()) == len(recs)
    ids = [{r["identity"] for r in v} for v in splits.values()]
    assert not (ids[0] & ids[1]) and not (ids[0] & ids[2]) and not (ids[1] & ids[2])


def test_split_is_stratified_and_deterministic():
    recs = make_records(n_ids=80, imgs_per_id=1)
    s1 = split_by_identity(recs, (0.7, 0.15, 0.15), seed=7)
    s2 = split_by_identity(recs, (0.7, 0.15, 0.15), seed=7)
    assert [r["filename"] for r in s1["test"]] == [r["filename"] for r in s2["test"]]
    for split in ("train", "val", "test"):
        assert {r["label"] for r in s1[split]} == set(CLASSES)


def test_leak_detection_raises():
    splits = {"train": [{"identity": "1"}], "val": [{"identity": "2"}], "test": [{"identity": "1"}]}
    with pytest.raises(AssertionError, match="identity 누수"):
        assert_no_identity_leak(splits)


def test_read_labels_csv_validates(tmp_path):
    p = tmp_path / "labels.csv"
    p.write_text("filename,identity,label\na.jpg,1,dog\nb.jpg,2,hamster\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hamster"):
        read_labels_csv(p)
    p.write_text("filename,label\na.jpg,dog\n", encoding="utf-8")
    with pytest.raises(ValueError, match="identity"):
        read_labels_csv(p)


def test_soft_target_from_votes():
    ds = FaceDataset([], ".", soft_labels=True)
    t = ds.target_vector({"label": "dog", "votes": [3, 2, 0, 0]})
    assert t.tolist() == pytest.approx([0.6, 0.4, 0, 0])
    hard = FaceDataset([], ".", soft_labels=False).target_vector({"label": "fox", "votes": [3, 2, 0, 0]})
    assert hard.tolist() == [0, 0, 0, 1]
