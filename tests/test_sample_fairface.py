import csv

import numpy as np
import pytest
from PIL import Image, ImageFilter

from scripts.sample_fairface_candidates import filter_rows, main, sample_rows


def row(i, race="East Asian", age="20-29", gender="Female"):
    return {"file": f"train/{i}.jpg", "age": age, "gender": gender, "race": race}


def test_filter_race_age_gender():
    rows = [row(1), row(2, race="White"), row(3, age="50-59"), row(4, gender="Male")]
    assert [r["file"] for r in filter_rows(rows)] == ["train/1.jpg", "train/3.jpg", "train/4.jpg"]
    assert [r["file"] for r in filter_rows(rows, ages=["20-29"], genders=["female"])] == ["train/1.jpg"]
    with pytest.raises(ValueError):
        filter_rows(rows, ages=["20대"])


def test_sample_reproducible_and_gender_balanced():
    rows = [row(i, gender="Male" if i < 30 else "Female") for i in range(100)]
    assert sample_rows(rows, 10, seed=1) == sample_rows(rows, 10, seed=1)
    bal = sample_rows(rows, 20, seed=1, balance_gender=True)
    assert len(bal) == 20 and sum(r["gender"] == "Male" for r in bal) == 10


def test_main_end_to_end_with_duplicates(tmp_path):
    root = tmp_path / "fairface"
    img_dir = root / "fairface-img-margin025-trainval"
    (img_dir / "train").mkdir(parents=True)
    rng = np.random.RandomState(0)
    with open(root / "fairface_label_train.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "age", "gender", "race", "service_test"])
        for i in range(12):
            race = "East Asian" if i < 10 else "White"
            im = Image.fromarray(rng.randint(0, 255, (64, 64, 3), dtype=np.uint8)).filter(ImageFilter.GaussianBlur(3))
            im.save(img_dir / "train" / f"{i}.jpg", quality=95)
            w.writerow([f"train/{i}.jpg", "20-29", "Female", race, "True"])
        # 0번과 같은 이미지(크기만 다름) 추가
        Image.open(img_dir / "train" / "0.jpg").resize((50, 50)).save(img_dir / "train" / "dup.jpg", quality=95)
        w.writerow(["train/dup.jpg", "20-29", "Female", "East Asian", "True"])

    out = tmp_path / "labels"
    sample = main(["--fairface-root", str(root), "--n", "100", "--out-dir", str(out),
                   "--annotators", "김주환", "박정원", "--copy-to", str(tmp_path / "copy")])
    files = {r["file"] for r in sample}
    assert len(files) == 10 and not ({"train/0.jpg", "train/dup.jpg"} <= files)  # 중복 중 하나만
    assert all("White" not in f for f in files)
    with open(out / "annotators" / "김주환.csv", encoding="utf-8") as f:
        sheet = list(csv.DictReader(f))
    assert len(sheet) == 10 and sheet[0]["identity"] == sheet[0]["filename"] and sheet[0]["label"] == ""
    assert (tmp_path / "copy" / sheet[0]["filename"]).exists()
