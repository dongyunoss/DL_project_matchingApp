"""YAML 설정 로드 + `--set key.sub=value` 형태의 CLI override."""
import argparse
from pathlib import Path

import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"


def load_config(path=None, overrides=None) -> dict:
    with open(path or DEFAULT_CONFIG, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for item in overrides or []:
        apply_override(cfg, item)
    return cfg


def apply_override(cfg: dict, item: str) -> None:
    """'train.epochs=5' 같은 문자열을 cfg에 반영한다. 값은 YAML 문법으로 파싱한다."""
    if "=" not in item:
        raise ValueError(f"override 형식은 key.sub=value 이어야 합니다: {item}")
    key, raw = item.split("=", 1)
    node = cfg
    parts = key.strip().split(".")
    for p in parts[:-1]:
        node = node.setdefault(p, {})
    node[parts[-1]] = yaml.safe_load(raw)


def add_config_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="YAML 설정 파일 경로")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                        help="설정 덮어쓰기, 예: --set train.epochs=5 model.name=simple_cnn")
    return parser


def config_from_args(args) -> dict:
    return load_config(args.config, args.set)
