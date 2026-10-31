import pytest

from faceapp import CLASSES
from scripts.data_scaling import balanced_nested_subsets, summarize, to_markdown


def recs(counts):
    return [{"filename": f"{c}_{i}.jpg", "identity": f"{c}{i}", "label": c}
            for c, n in zip(CLASSES, counts) for i in range(n)]


def test_balanced_subsets_equal_per_class_and_nested():
    subs = balanced_nested_subsets(recs([40, 20, 30, 25]), [0.25, 0.5, 1.0], seed=0)
    for f, n_per in [(0.25, 5), (0.5, 10), (1.0, 20)]:
        labels = [r["label"] for r in subs[f]]
        assert all(labels.count(c) == n_per for c in CLASSES)
    small, mid, full = ({r["filename"] for r in subs[f]} for f in (0.25, 0.5, 1.0))
    assert small <= mid <= full


def test_stratified_keeps_ratio():
    subs = balanced_nested_subsets(recs([40, 20, 40, 20]), [0.5], seed=0, balanced=False)
    labels = [r["label"] for r in subs[0.5]]
    assert [labels.count(c) for c in CLASSES] == [20, 10, 20, 10]


def test_invalid_fraction():
    with pytest.raises(ValueError):
        balanced_nested_subsets(recs([4, 4, 4, 4]), [0], seed=0)


def test_summary_mean_std():
    rows = [{"fraction": 0.5, "seed": s, "n_train": 10, "accuracy": a, "macro_precision": a, "macro_recall": a,
             "macro_f1": a, "best_val_loss": 1.0} for s, a in [(1, 0.4), (2, 0.6)]]
    s = summarize(rows)[0]
    assert s["accuracy_mean"] == pytest.approx(0.5) and s["accuracy_std"] == pytest.approx(0.1414, abs=1e-3)
    assert "50%" in to_markdown([s])
