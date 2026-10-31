"""딥러닝 얼굴 유형 분류 기반 소개팅 매칭 프로토타입.

클래스 정의는 configs/classes.yaml 하나만 기준으로 한다 (파일럿 후 확정 시 그 파일만 수정).
"""
from pathlib import Path

import yaml

CLASSES_FILE = Path(__file__).resolve().parent.parent / "configs" / "classes.yaml"


def _load_classes(path=CLASSES_FILE):
    with open(path, encoding="utf-8") as f:
        items = yaml.safe_load(f)["classes"]
    names = [str(c["name"]).strip().lower() for c in items]
    if len(names) < 2 or len(set(names)) != len(names):
        raise ValueError(f"{path}: 클래스는 2개 이상, 이름 중복 없이 정의해야 합니다")
    return names, {str(c["name"]).strip().lower(): str(c.get("ko", c["name"])) for c in items}


CLASSES, CLASS_KO = _load_classes()
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
NUM_CLASSES = len(CLASSES)
