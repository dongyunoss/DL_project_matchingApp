import csv
import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest
from PIL import Image

from scripts.label_tool import LabelSheet, make_handler
from scripts.merge_labels import main as merge_main
from scripts.merge_labels import normalize_label


def write_sheet(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["filename", "identity", "label"])
        w.writerows(rows)


def read_labels(path):
    with open(path, encoding="utf-8") as f:
        return [r["label"] for r in csv.DictReader(f)]


@pytest.fixture
def sheet_path(tmp_path):
    p = tmp_path / "member1.csv"
    write_sheet(p, [["a.jpg", "1", ""], ["b.jpg", "2", "고양이상"], ["c.jpg", "3", ""]])
    return p


def test_sheet_load_set_and_persist(sheet_path):
    s = LabelSheet(sheet_path)
    assert len(s) == 3 and s.n_labeled == 1 and s.first_unlabeled() == 0
    s.set_label(0, "dog")
    s.set_label(2, "skip")  # 모름(기권)도 완료로 저장
    assert read_labels(sheet_path) == ["dog", "cat", "skip"]
    s2 = LabelSheet(sheet_path)  # 재시작해도 이어짐
    assert s2.n_labeled == 3
    s2.set_label(1, "")
    assert s2.first_unlabeled() == 1 and read_labels(sheet_path) == ["dog", "", "skip"]
    with pytest.raises(ValueError):
        s2.set_label(0, "hamster")
    with pytest.raises(IndexError):
        s2.set_label(9, "dog")


def test_skip_is_abstain_in_merge(tmp_path):
    assert normalize_label("skip") is None and normalize_label("모름") is None
    d = tmp_path / "ann"
    d.mkdir()
    for k, labs in enumerate([["dog"], ["dog"], ["dog"], ["skip"], ["cat"]]):
        write_sheet(d / f"m{k}.csv", [["a.jpg", "1", labs[0]]])
    stats = merge_main(["--input-dir", str(d), "--out-dir", str(tmp_path / "out")])
    assert stats["n_final"] == 1 and stats["label_distribution"] == {"dog": 1}


@pytest.fixture
def server(sheet_path, tmp_path):
    img_dir = tmp_path / "imgs"
    img_dir.mkdir()
    Image.new("RGB", (178, 218), (200, 150, 120)).save(img_dir / "a.jpg")
    (tmp_path / "secret.txt").write_text("nope")
    sheet = LabelSheet(sheet_path)
    srv = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(sheet, img_dir))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}", sheet
    srv.shutdown()
    srv.server_close()


def get(url):
    with urllib.request.urlopen(url) as r:
        return r.status, r.read()


def post(url, obj):
    req = urllib.request.Request(url, json.dumps(obj).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())


def test_server_flow(server, sheet_path):
    base, sheet = server
    assert b"<title>" in get(base + "/")[1]
    state = json.loads(get(base + "/api/state")[1])
    assert state["n"] == 3 and state["start"] == 0 and state["annotator"] == "member1"
    assert json.loads(get(base + "/api/item/1")[1])["label"] == "cat"
    status, body = get(base + "/img/0")
    assert status == 200 and body[:2] == b"\xff\xd8"  # JPEG
    r = post(base + "/api/label", {"i": 0, "label": "fox"})
    assert r == {"labeled": 2, "next": 2}
    assert read_labels(sheet_path) == ["fox", "cat", ""]


@pytest.mark.parametrize("path", ["/img/1", "/img/99", "/img/-1", "/img/..%2Fsecret.txt", "/api/item/x", "/etc/passwd"])
def test_server_rejects_bad_paths(server, path):
    base, _ = server
    with pytest.raises(urllib.error.HTTPError) as e:
        get(base + path)
    assert e.value.code == 404


def test_server_rejects_bad_label(server):
    base, _ = server
    with pytest.raises(urllib.error.HTTPError) as e:
        post(base + "/api/label", {"i": 0, "label": "hamster"})
    assert e.value.code == 400


def test_sheet_with_path_traversal_filename_is_not_served(tmp_path):
    p = tmp_path / "s.csv"
    write_sheet(p, [["../secret.txt", "1", ""]])
    (tmp_path / "imgs").mkdir()
    (tmp_path / "imgs" / "secret.txt").write_text("in folder")  # name만 쓰므로 폴더 안 파일만 대상
    (tmp_path / "secret.txt").write_text("outside")
    srv = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(LabelSheet(p), tmp_path / "imgs"))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        _, body = get(f"http://127.0.0.1:{srv.server_address[1]}/img/0")
        assert body == b"in folder"
    finally:
        srv.shutdown()
        srv.server_close()
