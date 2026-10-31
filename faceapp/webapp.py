"""웹캠 데모 (제안서 3-3, 3-4): 카메라 → 얼굴 검출·정렬 → ResNet18 확률 → 선호 조건 → Top-N.

브라우저: MediaDevices.getUserMedia(카메라) + HTMLVideoElement(미리보기) + Canvas(프레임 캡처·얼굴 크롭)
          + MediaPipe Tasks Vision Face Landmarker(얼굴 특징점 → 얼굴 영역·기울기 보정)
서버(이 파일, 표준 라이브러리 http.server): 크롭된 얼굴 → ResNet18 → 보정된 확률, 참가자 등록, 매칭.

  python -m scripts.fetch_mediapipe            # 처음 한 번 (MediaPipe를 로컬에 받기, 없으면 CDN 사용)
  python -m faceapp.webapp                     # → http://127.0.0.1:8000
  python -m faceapp.webapp --seed-participants outputs/smoke/participants.json `
      --seed-predictions outputs/smoke/run/predictions.csv   # 매칭 상대 풀을 미리 채워 시연

개인정보: 사진은 저장하지 않는다. 확률값과 입력한 프로필만 --store JSON(기본 data/webapp/participants.json)에 남는다.
카메라는 보안 컨텍스트에서만 동작하므로 localhost로 접속한다 (다른 기기에서 접속하려면 HTTPS가 필요).
"""
import argparse
import io
import json
import mimetypes
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch
from PIL import Image

from . import CLASS_KO, CLASSES
from .config import add_config_args, config_from_args
from .data import build_eval_transforms
from .matching import attach_face_probs, load_participants, preference_vector, recommend_top_n
from .models import load_checkpoint
from .utils import get_device

WEB_DIR = Path(__file__).resolve().parent / "web"
VENDOR_DIR = WEB_DIR / "vendor"
MAX_UPLOAD = 5 * 1024 * 1024
mimetypes.add_type("text/javascript", ".mjs")
mimetypes.add_type("application/wasm", ".wasm")


class FacePredictor:
    """크롭된 얼굴 이미지 → 보정된 클래스 확률 (TTA 좌우 반전 평균)."""

    def __init__(self, cfg: dict, ckpt_path=None):
        self.device = get_device()
        self.model, ckpt = load_checkpoint(ckpt_path or Path(cfg["output_dir"]) / "best.pt", self.device)
        self.temperature = float(ckpt.get("temperature", 1.0))
        self.tf = build_eval_transforms(cfg, ckpt)
        self.tta = cfg.get("predict", {}).get("tta", True)
        self.model_name = ckpt["model_name"]
        self.lock = threading.Lock()

    @torch.no_grad()
    def predict(self, img: Image.Image) -> list[float]:
        x = self.tf(img.convert("RGB")).unsqueeze(0).to(self.device)
        with self.lock:
            logits = self.model(x)
            if self.tta:
                logits = (logits + self.model(torch.flip(x, dims=[3]))) / 2
        return (logits / self.temperature).softmax(1)[0].cpu().tolist()


class ParticipantStore:
    """참가자 풀 (메모리 + JSON 파일). seed 참가자는 파일에 쓰지 않는다."""

    def __init__(self, path, seed: list[dict] | None = None):
        self.path = Path(path)
        self.lock = threading.Lock()
        self.seed = {p["id"]: {**p, "source": "seed"} for p in seed or []}
        self.people = {}
        if self.path.exists():
            self.people = {p["id"]: p for p in json.loads(self.path.read_text(encoding="utf-8"))}

    def all(self) -> list[dict]:
        with self.lock:
            return [*self.seed.values(), *self.people.values()]

    def upsert(self, p: dict) -> dict:
        p = validate_participant(p)
        with self.lock:
            if p["id"] in self.seed:
                raise ValueError(f"이미 있는 시연용 참가자 id입니다: {p['id']}")
            self.people[p["id"]] = p
            self._save()
        return p

    def delete(self, pid: str) -> bool:
        with self.lock:
            found = self.people.pop(pid, None) is not None
            self._save()
        return found

    def _save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(list(self.people.values()), ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.path)


def validate_participant(p: dict) -> dict:
    pid = str(p.get("id", "")).strip()
    if not pid or len(pid) > 40:
        raise ValueError("이름(id)을 1~40자로 입력하세요")
    probs = p.get("face_probs")
    if not isinstance(probs, list) or len(probs) != len(CLASSES) or abs(sum(probs) - 1) > 1e-3:
        raise ValueError("얼굴 사진을 먼저 분석하세요 (face_probs 없음)")
    age = p.get("age")
    if age is not None and (not isinstance(age, (int, float)) or not 15 <= age <= 100):
        raise ValueError("나이는 15~100 사이 숫자로 입력하세요")
    pref = dict(p.get("preference") or {})
    preference_vector(pref.get("face_type"))  # 형식 검사 (잘못되면 ValueError)
    for k in ("age_min", "age_max"):
        if pref.get(k) in ("", None):
            pref.pop(k, None)
    if not pref.get("gender"):
        pref.pop("gender", None)
    hobbies = p.get("hobbies") or []
    if isinstance(hobbies, str):
        hobbies = [h for h in hobbies.replace("，", ",").split(",")]
    return {"id": pid, "gender": p.get("gender") or None, "age": age,
            "hobbies": [h.strip() for h in hobbies if str(h).strip()][:20],
            "preference": pref, "face_probs": [float(v) for v in probs]}


def public_view(p: dict) -> dict:
    return {k: p.get(k) for k in ("id", "gender", "age", "hobbies", "face_probs", "source")}


def make_handler(predictor: FacePredictor, store: ParticipantStore, cfg: dict):
    mcfg = cfg.get("matching", {})

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send(self, code, body: bytes, ctype="application/json; charset=utf-8", cache=False):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "max-age=3600" if cache else "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code=200):
            self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"))

        def _file(self, base: Path, rel: str, cache=False):
            f = (base / rel).resolve()
            if base.resolve() not in f.parents or not f.is_file():
                return self._send(404, b"not found", "text/plain")
            return self._send(200, f.read_bytes(), mimetypes.guess_type(f.name)[0] or "application/octet-stream", cache)

        def _query(self) -> dict:
            from urllib.parse import parse_qs, urlsplit
            return {k: v[0] for k, v in parse_qs(urlsplit(self.path).query).items()}

        def _body(self) -> bytes:
            n = int(self.headers.get("Content-Length", 0))
            if n > MAX_UPLOAD:
                raise ValueError("요청이 너무 큽니다 (최대 5MB)")
            return self.rfile.read(n)

        def do_GET(self):
            path = self.path.split("?", 1)[0]
            if path == "/":
                return self._file(WEB_DIR, "index.html")
            if path.startswith("/static/"):
                return self._file(WEB_DIR, path[len("/static/"):])
            if path.startswith("/vendor/"):
                return self._file(VENDOR_DIR, path[len("/vendor/"):], cache=True)
            if path == "/api/config":
                return self._json({
                    "classes": CLASSES, "ko": CLASS_KO, "model": predictor.model_name,
                    "temperature": predictor.temperature, "top_n": mcfg.get("top_n", 5),
                    "weights": mcfg.get("weights"), "crop_scale": cfg.get("webapp", {}).get("crop_scale", 1.6),
                    "local_mediapipe": (VENDOR_DIR / "tasks-vision" / "vision_bundle.mjs").exists()
                    and (VENDOR_DIR / "models" / "face_landmarker.task").exists(),
                })
            if path == "/api/participants":
                return self._json([public_view(p) for p in store.all()])
            if path == "/api/matches":
                q = self._query()
                people = store.all()
                if q.get("id") not in {p["id"] for p in people}:
                    return self._json({"error": "등록되지 않은 id입니다"}, 404)
                top_n = int(q.get("top_n") or mcfg.get("top_n", 5))
                recs = recommend_top_n(people, mcfg, top_n)[q["id"]]
                by_id = {p["id"]: p for p in people}
                return self._json([{**c, "person": public_view(by_id[c["id"]])} for c in recs])
            return self._send(404, b"not found", "text/plain")

        def do_POST(self):
            try:
                if self.path == "/api/predict":
                    img = Image.open(io.BytesIO(self._body()))
                    img.load()
                    probs = predictor.predict(img)
                    top = max(range(len(probs)), key=probs.__getitem__)
                    return self._json({"classes": CLASSES, "probs": probs, "pred": CLASSES[top]})
                if self.path == "/api/participants":
                    p = store.upsert(json.loads(self._body() or b"{}"))
                    return self._json(public_view(p))
            except (ValueError, OSError, json.JSONDecodeError) as e:
                return self._json({"error": str(e)}, 400)
            return self._send(404, b"not found", "text/plain")

        def do_DELETE(self):
            if self.path.split("?", 1)[0] == "/api/participants":
                return self._json({"deleted": store.delete(self._query().get("id", ""))})
            return self._send(404, b"not found", "text/plain")

    return Handler


def load_seed(participants_path, predictions_path) -> list[dict]:
    if not participants_path:
        return []
    from .predict import read_predictions
    people = load_participants(participants_path)
    if predictions_path:
        people = attach_face_probs(people, read_predictions(predictions_path), strict=False)
    return [p for p in people if p.get("face_probs")]


def make_server(cfg, host="127.0.0.1", port=8000, store_path="data/webapp/participants.json",
                seed_participants=None, seed_predictions=None, ckpt_path=None):
    predictor = FacePredictor(cfg, ckpt_path)
    store = ParticipantStore(store_path, load_seed(seed_participants, seed_predictions))
    return ThreadingHTTPServer((host, port), make_handler(predictor, store, cfg)), store


def main():
    p = add_config_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--checkpoint", default=None, help="기본: <output_dir>/best.pt")
    p.add_argument("--store", default="data/webapp/participants.json", help="등록 참가자 저장 파일 (사진은 저장 안 함)")
    p.add_argument("--seed-participants", default=None, help="시연용 상대 풀 participants.json")
    p.add_argument("--seed-predictions", default=None, help="시연용 상대 풀의 predictions.csv (face_probs)")
    p.add_argument("--no-browser", action="store_true")
    args = p.parse_args()
    cfg = config_from_args(args)
    server, store = make_server(cfg, args.host, args.port, args.store, args.seed_participants,
                                args.seed_predictions, args.checkpoint)
    url = f"http://{'127.0.0.1' if args.host in ('0.0.0.0', '') else args.host}:{server.server_address[1]}/"
    print(f"[webapp] {url}  (참가자 풀 {len(store.all())}명, 종료: Ctrl+C)")
    if not args.no_browser:
        import webbrowser
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
