import csv

import pytest

from scripts.merge_labels import fleiss_kappa, main, majority_vote, merge, normalize_label


def test_normalize_label_aliases():
    assert normalize_label("Dog") == "dog"
    assert normalize_label("고양이상") == "cat"
    assert normalize_label("토끼") == "rabbit"
    assert normalize_label("  ") is None
    with pytest.raises(ValueError):
        normalize_label("hamster")


def test_majority_vote_clear_winner():
    label, votes, reason = majority_vote(["dog", "dog", "dog", "cat", "fox"])
    assert label == "dog" and votes["dog"] == 3 and reason == "ok"


def test_majority_vote_plurality_wins():
    assert majority_vote(["dog", "dog", "cat", "fox", "rabbit"])[0] == "dog"


def test_majority_vote_tie_goes_to_review():
    label, _, reason = majority_vote(["dog", "dog", "cat", "cat", "fox"])
    assert label is None and reason.startswith("tie:") and "cat" in reason and "dog" in reason


def test_majority_vote_abstain_and_min_votes():
    assert majority_vote(["dog", None, None, None, None], min_votes=3)[0] is None
    assert majority_vote(["dog", "dog", None, "cat", None], min_votes=3)[0] == "dog"


def test_fleiss_kappa_perfect_and_nan():
    assert fleiss_kappa([[5, 0, 0, 0], [0, 5, 0, 0], [0, 0, 5, 0]]) == pytest.approx(1.0)
    assert fleiss_kappa([]) != fleiss_kappa([])  # NaN


def test_merge_identity_mismatch_raises():
    ann = {"a": {"x.jpg": ("1", "dog")}, "b": {"x.jpg": ("2", "dog")}}
    with pytest.raises(ValueError):
        merge(ann)


def test_main_end_to_end(tmp_path):
    rows = {  # filename: 라벨러 5명의 라벨
        "1.jpg": ["dog", "dog", "dog", "cat", "dog"],
        "2.jpg": ["cat", "cat", "fox", "fox", "rabbit"],   # 동률 → review
        "3.jpg": ["여우상", "fox", "fox", "", "cat"],
    }
    ann_dir = tmp_path / "ann"
    ann_dir.mkdir()
    for k in range(5):
        with open(ann_dir / f"m{k}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["filename", "identity", "label"])
            for i, (fn, labs) in enumerate(rows.items()):
                w.writerow([fn, str(100 + i), labs[k]])
    out = tmp_path / "out"
    stats = main(["--input-dir", str(ann_dir), "--out-dir", str(out)])
    assert stats["n_final"] == 2 and stats["n_review"] == 1

    with open(out / "final_labels.csv", encoding="utf-8") as f:
        final = {r["filename"]: r for r in csv.DictReader(f)}
    assert final["1.jpg"]["label"] == "dog" and final["1.jpg"]["votes_dog"] == "4"
    assert final["3.jpg"]["label"] == "fox" and final["3.jpg"]["n_votes"] == "4"
    with open(out / "review.csv", encoding="utf-8") as f:
        review = list(csv.DictReader(f))
    assert review[0]["filename"] == "2.jpg"

    # 병합 결과가 학습용 라벨 CSV 형식으로 바로 읽혀야 한다
    from faceapp.data import read_labels_csv
    recs = read_labels_csv(out / "final_labels.csv")
    assert {r["filename"] for r in recs} == {"1.jpg", "3.jpg"} and recs[0]["votes"] is not None


def test_per_class_kappa_and_pair_confusion():
    from scripts.merge_labels import pairwise_confusion, per_class_kappa
    # 강아지는 모두 일치, 토끼/여우는 서로 엇갈림
    m = [[3, 0, 0, 0], [3, 0, 0, 0], [0, 0, 2, 1], [0, 0, 1, 2]]
    k = per_class_kappa(m)
    assert k["dog"] == pytest.approx(1.0) and k["rabbit"] < 0.5 and k["cat"] != k["cat"]  # cat: NaN
    ann = {"a": {"x": ("1", "rabbit"), "y": ("2", "dog")}, "b": {"x": ("1", "fox"), "y": ("2", "dog")}}
    pc = pairwise_confusion(ann)
    assert pc[0][0] == 1 and pc[2][3] == 1 and pc[3][2] == 1
