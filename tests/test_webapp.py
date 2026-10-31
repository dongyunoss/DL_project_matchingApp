import io
import json
import threading
import urllib.error
import urllib.request

import pytest
from PIL import Image

from faceapp import CLASSES
from faceapp.config import load_config
from faceapp.models import FaceClassifier, save_checkpoint
from faceapp.webapp import make_server, validate_participant


@pytest.fixture
def server(tmp_path):
    cfg = load_config(overrides=[f"output_dir={tmp_path.as_posix()}", "data.image_size=64", "model.name=simple_cnn"])
    save_checkpoint(tmp_path / "best.pt", FaceClassifier("simple_cnn", pretrained=False).eval(), cfg)
    seed = tmp_path / "seed.json"
    seed.write_text(json.dumps([
        {"id": "s1", "gender": "M", "age": 23, "photo": "s1.jpg", "hobbies": ["영화"],
         "preference": {"face_type": {"dog": 1}, "gender": "F"}},
        {"id": "s2", "gender": "M", "age": 30, "photo": "s2.jpg", "preference": {"gender": "M"}}]), encoding="utf-8")
    preds = tmp_path / "pred.csv"
    preds.write_text("filename,pred," + ",".join(f"prob_{c}" for c in CLASSES) + "\n"
                     "s1.jpg,dog,0.7,0.1,0.1,0.1\ns2.jpg,cat,0.1,0.7,0.1,0.1\n", encoding="utf-8")
    srv, store = make_server(cfg, port=0, store_path=tmp_path / "store.json", seed_participants=seed,
                             seed_predictions=preds)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}", tmp_path
    srv.shutdown()
    srv.server_close()


def req(url, data=None, method=None, ctype="application/json"):
    body = json.dumps(data).encode() if isinstance(data, dict) else data
    r = urllib.request.Request(url, body, {"Content-Type": ctype} if body is not None else {}, method=method)
    with urllib.request.urlopen(r) as resp:
        raw = resp.read()
        return json.loads(raw) if resp.headers.get_content_type() == "application/json" else raw


def jpeg():
    buf = io.BytesIO()
    Image.new("RGB", (224, 224), (180, 140, 120)).save(buf, "JPEG")
    return buf.getvalue()


def test_config_and_static(server):
    base, _ = server
    cfg = req(base + "/api/config")
    assert cfg["classes"] == CLASSES and cfg["model"] == "simple_cnn"
    assert b"getUserMedia" in req(base + "/static/app.js") and b"<title>" in req(base + "/")
    for bad in ("/static/../webapp.py", "/vendor/../../train.py", "/nope"):
        with pytest.raises(urllib.error.HTTPError) as e:
            req(base + bad)
        assert e.value.code == 404


def test_predict_register_match_delete(server):
    base, tmp = server
    r = req(base + "/api/predict", jpeg(), ctype="image/jpeg")
    assert len(r["probs"]) == len(CLASSES) and abs(sum(r["probs"]) - 1) < 1e-4 and r["pred"] in CLASSES
    me = {"id": "나", "gender": "F", "age": 22, "hobbies": "영화, 러닝", "face_probs": r["probs"],
          "preference": {"face_type": {c: 25 for c in CLASSES}, "age_min": 20, "age_max": 27, "gender": "M"}}
    assert req(base + "/api/participants", me)["hobbies"] == ["영화", "러닝"]
    assert len(req(base + "/api/participants")) == 3
    matches = req(base + "/api/matches?id=%EB%82%98")
    assert [m["id"] for m in matches] == ["s1"]  # s2는 남성을 원하므로 상호 조건 불충족
    assert "photo" not in matches[0]["person"]
    saved = json.loads((tmp / "store.json").read_text(encoding="utf-8"))
    assert [p["id"] for p in saved] == ["나"] and "photo" not in saved[0]  # 사진·seed는 저장 안 함
    assert req(base + "/api/participants?id=%EB%82%98", method="DELETE")["deleted"] is True
    assert json.loads((tmp / "store.json").read_text(encoding="utf-8")) == []


def test_bad_requests(server):
    base, _ = server
    with pytest.raises(urllib.error.HTTPError) as e:
        req(base + "/api/predict", b"not an image", ctype="image/jpeg")
    assert e.value.code == 400
    with pytest.raises(urllib.error.HTTPError) as e:
        req(base + "/api/participants", {"id": "s1", "face_probs": [0.25] * len(CLASSES)})  # seed id 중복
    assert e.value.code == 400
    with pytest.raises(urllib.error.HTTPError) as e:
        req(base + "/api/matches?id=nobody")
    assert e.value.code == 404


@pytest.mark.parametrize("bad,msg", [
    ({"id": "", "face_probs": [0.25] * 4}, "이름"),
    ({"id": "a"}, "얼굴"),
    ({"id": "a", "face_probs": [0.5, 0.5, 0.5, 0.5]}, "얼굴"),
    ({"id": "a", "face_probs": [0.25] * 4, "age": 5}, "나이"),
    ({"id": "a", "face_probs": [0.25] * 4, "preference": {"face_type": {"hamster": 1}}}, "hamster"),
])
def test_validate_participant(bad, msg):
    with pytest.raises(ValueError, match=msg):
        validate_participant(bad)
