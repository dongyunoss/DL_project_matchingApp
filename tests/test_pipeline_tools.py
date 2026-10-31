import json

import pytest
from PIL import Image

from faceapp.config import load_config
from faceapp.data import build_eval_transforms
from faceapp.pipeline import resolve_photos
from faceapp.report import write_html_report
from scripts.compare_matching_time import main as time_main
from scripts.compare_matching_time import summarize_timing
from scripts.compare_models import parse_run_spec, to_markdown


# ------------------------------------------------------------ eval transform
def test_eval_transform_follows_checkpoint_image_size():
    cfg = load_config()  # default 224
    img = Image.new("RGB", (178, 218))
    assert build_eval_transforms(cfg)(img).shape == (3, 224, 224)
    ckpt = {"config": {"data": {"image_size": 96}}}
    assert build_eval_transforms(cfg, ckpt)(img).shape == (3, 96, 96)
    assert cfg["data"]["image_size"] == 224  # 원본 cfg는 변경되지 않음


# ------------------------------------------------------------ time comparison
def rows(*items):
    return [{"trial_id": t, "method": m, "n": n, "minutes": mi} for t, m, n, mi in items]


def test_summarize_timing_manual_vs_ai_assisted():
    res = summarize_timing(rows(("1", "manual", 20, 100.0), ("1", "ai_assisted", 20, 10.0),
                                ("2", "manual", 10, 40.0), ("2", "ai_assisted", 10, 6.0)))
    assert res["by_method"]["manual"]["minutes_per_participant"] == pytest.approx(140 / 30)
    assert res["overall"]["reduction"] == pytest.approx(1 - (16 / 30) / (140 / 30))
    assert [t["reduction"] for t in res["per_trial"]] == pytest.approx([0.9, 0.85])


def test_summarize_timing_adds_compute_time_and_ai_only():
    res = summarize_timing(rows(("1", "manual", 20, 100.0), ("1", "ai_assisted", 20, 10.0)), ai_sec_per_participant=3)
    assert res["per_trial"][0]["ai_min"] == pytest.approx(10 + 3 * 20 / 60)
    only = summarize_timing(rows(("1", "manual", 20, 100.0)), ai_sec_per_participant=3)
    assert only["overall"]["compare"] == "manual vs ai_only"
    assert only["per_trial"][0]["ai_min"] == pytest.approx(1.0)


def test_time_cli_validates_method(tmp_path):
    p = tmp_path / "t.csv"
    p.write_text("trial_id,method,n_participants,minutes\n1,robot,10,5\n", encoding="utf-8")
    with pytest.raises(ValueError, match="method"):
        time_main(["--timing", str(p)])
    p.write_text("trial_id,method,n_participants,minutes\n1,manual,10,50\n", encoding="utf-8")
    s = tmp_path / "summary.json"
    s.write_text(json.dumps({"ai_matching": {"seconds_per_participant": 0.5}}), encoding="utf-8")
    res = time_main(["--timing", str(p), "--pipeline-summary", str(s)])
    assert res["overall"]["reduction"] > 0.99 and (tmp_path / "time_comparison.json").exists()


# ------------------------------------------------------------ compare_models
def test_parse_run_spec():
    assert parse_run_spec("cnn:model.name=simple_cnn train.epochs=5") == ("cnn", ["model.name=simple_cnn", "train.epochs=5"])
    assert parse_run_spec("base:") == ("base", [])
    for bad in ("noname", ":model.name=x", "a/b:x=1", "ok:novalue"):
        with pytest.raises(ValueError):
            parse_run_spec(bad)


def test_to_markdown_table():
    md = to_markdown([{"run": "a", "test_macro_f1": 0.5}])
    assert md.splitlines()[0].startswith("| run |") and "0.5000" in md


# ------------------------------------------------------------ pipeline / report
def test_resolve_photos_reports_missing(tmp_path):
    (tmp_path / "a.jpg").write_bytes(b"x")
    assert resolve_photos([{"id": "a", "photo": "sub/a.jpg"}], tmp_path) == [tmp_path / "a.jpg"]
    with pytest.raises(FileNotFoundError, match="b.jpg"):
        resolve_photos([{"id": "a", "photo": "a.jpg"}, {"id": "b", "photo": "b.jpg"}], tmp_path)


def test_html_report_escapes_and_lists_candidates(tmp_path):
    people = [
        {"id": "p<1>", "gender": "F", "age": 22, "photo": "p1.jpg", "hobbies": ["<script>"],
         "face_probs": [0.7, 0.1, 0.1, 0.1], "preference": {"face_type": {"dog": 1}}},
        {"id": "p2", "gender": "M", "age": 23, "photo": "p2.jpg", "face_probs": [0.1, 0.7, 0.1, 0.1]},
    ]
    recs = {"p<1>": [{"id": "p2", "score": 0.81234, "detail": {"parts_a_to_b": {"face": 0.5, "age": None}}}],
            "p2": []}
    out = write_html_report(recs, people, tmp_path / "photos", tmp_path / "r" / "report.html")
    html = out.read_text(encoding="utf-8")
    assert "<script>" not in html and "&lt;script&gt;" in html and "p&lt;1&gt;" in html
    assert "0.812" in html and "조건에 맞는 상대 없음" in html
    assert "../photos/p2.jpg" in html  # 사진은 상대 경로로만 참조
