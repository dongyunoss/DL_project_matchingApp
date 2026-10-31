"""웹캠 데모용 MediaPipe Tasks Vision(Face Landmarker)을 받아 로컬에 둔다 (처음 한 번, 인터넷 필요).

  python -m scripts.fetch_mediapipe

받는 것 (버전·해시 고정):
  - npm @mediapipe/tasks-vision@0.10.35 (Apache-2.0): vision_bundle.mjs + wasm/*  → faceapp/web/vendor/tasks-vision/
  - face_landmarker.task (float16/1)                                               → faceapp/web/vendor/models/
받아 두면 시연 장소의 인터넷과 관계없이 동작한다. 받지 않으면 웹앱이 CDN(jsDelivr)에서 불러온다.
faceapp/web/vendor/는 .gitignore로 제외된다.
"""
import base64
import hashlib
import io
import sys
import tarfile
import urllib.request
from pathlib import Path

VERSION = "0.10.35"
TARBALL = f"https://registry.npmjs.org/@mediapipe/tasks-vision/-/tasks-vision-{VERSION}.tgz"
TARBALL_SHA512 = "HOvadwVRE6JC+45nyYhmnywnr5h/J8KZvOeUNVOG9q/0875pZgItznFB9bRTvLc264YSJqiZ1NsIpCStJw/egg=="
MODEL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/"
         "face_landmarker.task")
MODEL_SHA256 = "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"
VENDOR = Path(__file__).resolve().parent.parent / "faceapp" / "web" / "vendor"


def download(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=120) as r:
        return r.read()


def main(argv=None):
    tv_dir, model_dir = VENDOR / "tasks-vision", VENDOR / "models"
    data = download(TARBALL)
    got = base64.b64encode(hashlib.sha512(data).digest()).decode()
    if got != TARBALL_SHA512:
        sys.exit(f"tasks-vision 해시 불일치 — 받은 파일을 쓰지 않습니다 ({got[:16]}…)")
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        for m in tar.getmembers():
            name = m.name.removeprefix("package/")
            if m.isfile() and (name == "vision_bundle.mjs" or (name.startswith("wasm/") and "/" not in name[5:])):
                dst = tv_dir / name
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_bytes(tar.extractfile(m).read())
    print(f"[mediapipe] tasks-vision {VERSION} → {tv_dir}")

    model = download(MODEL)
    if hashlib.sha256(model).hexdigest() != MODEL_SHA256:
        sys.exit("face_landmarker.task 해시 불일치 — 받은 파일을 쓰지 않습니다")
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "face_landmarker.task").write_bytes(model)
    print(f"[mediapipe] face_landmarker.task → {model_dir}")


if __name__ == "__main__":
    main()
