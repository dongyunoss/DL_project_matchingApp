"""매칭 점수 계산과 Top-N 추천.

방향 점수 (viewer → candidate):
    S = Σ w_i · s_i / Σ w_i       (정보가 없는 항목은 제외하고 가중치 재정규화)
    s_face  = cos(candidate.face_probs, viewer.preference.face_type)    # 모델이 낸 보정 확률 vs 선호 유형 벡터
    s_age   = 선호 나이 범위 적합도 (범위 안 1, 범위 밖은 age_tolerance 년에 걸쳐 0까지 선형 감소)
    s_hobby = Jaccard(viewer.hobbies, candidate.hobbies)
양방향(bidirectional)이면 최종 점수 = (S(a→b) + S(b→a)) / 2  — 서로의 선호를 모두 반영.

참가자 입력 형식 (잠정, TODO(미정): 선호 조건 입력 형식 확정)
[
  {"id": "p01", "gender": "F", "age": 22, "photo": "p01.jpg", "hobbies": ["영화", "러닝"],
   "preference": {"face_type": {"dog": 0.7, "cat": 0.3}, "age_min": 21, "age_max": 26, "gender": "M"}}
]
- face_type: 클래스별 선호 가중치(합이 1일 필요 없음). 비우면 얼굴 항목 제외.
- gender: 지정 시 하드 필터.

    python -m faceapp.matching --participants data/participants.json --predictions outputs/run1/predictions.csv
"""
import argparse
import json
from pathlib import Path

import numpy as np

from . import CLASS_KO, CLASS_TO_IDX, CLASSES, NUM_CLASSES

DEFAULT_WEIGHTS = {"face": 0.6, "age": 0.25, "hobby": 0.15}  # TODO(미정): 가중치 확정


# ------------------------------------------------------------------------ components
def cosine_similarity(a, b) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.clip(a @ b / (na * nb), -1.0, 1.0))


def preference_vector(face_type) -> np.ndarray | None:
    """{'dog': 0.7, 'cat': 0.3} 또는 길이 4 리스트 → 정규화된 벡터. 선호 없음 → None."""
    if face_type is None:
        return None
    v = np.zeros(NUM_CLASSES)
    if isinstance(face_type, dict):
        for k, w in face_type.items():
            key = k.strip().lower()
            if key not in CLASS_TO_IDX:
                raise ValueError(f"알 수 없는 얼굴 유형 '{k}' (허용: {CLASSES})")
            v[CLASS_TO_IDX[key]] = float(w)
    else:
        v = np.asarray(face_type, dtype=float)
        if v.shape != (NUM_CLASSES,):
            raise ValueError(f"face_type 리스트 길이는 {NUM_CLASSES}이어야 합니다")
    if (v < 0).any():
        raise ValueError("face_type 가중치는 0 이상이어야 합니다")
    return None if v.sum() == 0 else v / v.sum()


def preference_from_reference_probs(prob_list) -> np.ndarray:
    """(확장) 이상형 참고 사진들의 모델 확률 평균 → 선호 유형 벡터."""
    arr = np.asarray(prob_list, dtype=float)
    return arr.mean(axis=0) / arr.mean(axis=0).sum()


def age_score(age, age_min=None, age_max=None, tolerance: float = 3.0) -> float | None:
    if age is None or (age_min is None and age_max is None):
        return None
    lo = -np.inf if age_min is None else age_min
    hi = np.inf if age_max is None else age_max
    if lo <= age <= hi:
        return 1.0
    gap = (lo - age) if age < lo else (age - hi)
    if tolerance <= 0:
        return 0.0
    return float(max(0.0, 1.0 - gap / tolerance))


def _norm_hobbies(h) -> set[str]:
    return {s.strip().lower() for s in (h or []) if s and s.strip()}


def hobby_score(a, b) -> float | None:
    sa, sb = _norm_hobbies(a), _norm_hobbies(b)
    if not sa or not sb:
        return None
    return len(sa & sb) / len(sa | sb)


# ------------------------------------------------------------------------- scoring
def is_eligible(viewer: dict, candidate: dict) -> bool:
    if viewer["id"] == candidate["id"]:
        return False
    want = (viewer.get("preference") or {}).get("gender")
    if want and candidate.get("gender") and candidate["gender"] != want:
        return False
    return True


def directional_score(viewer: dict, candidate: dict, mcfg: dict | None = None) -> tuple[float, dict]:
    mcfg = mcfg or {}
    weights = {**DEFAULT_WEIGHTS, **(mcfg.get("weights") or {})}
    pref = viewer.get("preference") or {}

    pv = preference_vector(pref.get("face_type"))
    probs = candidate.get("face_probs")
    parts = {
        "face": cosine_similarity(probs, pv) if (pv is not None and probs is not None) else None,
        "age": age_score(candidate.get("age"), pref.get("age_min"), pref.get("age_max"),
                         mcfg.get("age_tolerance", 3)),
        "hobby": hobby_score(viewer.get("hobbies"), candidate.get("hobbies")),
    }
    used = {k: v for k, v in parts.items() if v is not None and weights.get(k, 0) > 0}
    wsum = sum(weights[k] for k in used)
    score = sum(weights[k] * v for k, v in used.items()) / wsum if wsum > 0 else 0.0
    return float(score), parts


def match_score(a: dict, b: dict, mcfg: dict | None = None) -> tuple[float, dict]:
    mcfg = mcfg or {}
    s_ab, p_ab = directional_score(a, b, mcfg)
    if not mcfg.get("bidirectional", True):
        return s_ab, {"a_to_b": s_ab, "parts_a_to_b": p_ab}
    s_ba, p_ba = directional_score(b, a, mcfg)
    return (s_ab + s_ba) / 2, {"a_to_b": s_ab, "b_to_a": s_ba, "parts_a_to_b": p_ab, "parts_b_to_a": p_ba}


def recommend_top_n(participants: list[dict], mcfg: dict | None = None, top_n: int | None = None) -> dict:
    """각 참가자별 Top-N 후보: {id: [{"id", "score", "detail"}, ...]} (점수 내림차순, 동점은 id 순)."""
    mcfg = mcfg or {}
    n = top_n or mcfg.get("top_n", 5)
    bidir = mcfg.get("bidirectional", True)
    ids = [p["id"] for p in participants]
    if len(set(ids)) != len(ids):
        raise ValueError("참가자 id가 중복됩니다")
    result = {}
    for a in participants:
        cands = []
        for b in participants:
            if not is_eligible(a, b) or (bidir and not is_eligible(b, a)):
                continue
            s, detail = match_score(a, b, mcfg)
            cands.append({"id": b["id"], "score": round(s, 6), "detail": detail})
        cands.sort(key=lambda c: (-c["score"], c["id"]))
        result[a["id"]] = cands[:n]
    return result


# --------------------------------------------------------------------------- I/O
def load_participants(path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def attach_face_probs(participants: list[dict], predictions: dict[str, list[float]], strict: bool = True) -> list[dict]:
    """predict.py 결과({filename: probs})를 참가자의 photo 파일명으로 연결."""
    out, missing = [], []
    for p in participants:
        q = dict(p)
        photo = Path(p.get("photo", "")).name
        if photo in predictions:
            q["face_probs"] = list(predictions[photo])
        elif p.get("photo"):
            missing.append(photo)
        out.append(q)
    if missing and strict:
        raise KeyError(f"예측 결과에 없는 사진: {missing[:10]}")
    return out


def format_recommendations(recs: dict, participants: list[dict]) -> str:
    by_id = {p["id"]: p for p in participants}
    lines = []
    for pid, cands in recs.items():
        lines.append(f"■ {pid}")
        for rank, c in enumerate(cands, 1):
            fp = by_id[c["id"]].get("face_probs")
            ft = f"{CLASS_KO[CLASSES[int(np.argmax(fp))]]}" if fp is not None else "-"
            lines.append(f"   {rank}. {c['id']:<8} score={c['score']:.3f}  ({ft})")
    return "\n".join(lines)


def main():
    from .config import add_config_args, config_from_args
    from .predict import read_predictions
    from .utils import save_json

    p = add_config_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    p.add_argument("--participants", required=True)
    p.add_argument("--predictions", default=None, help="기본: <output_dir>/predictions.csv")
    p.add_argument("--top-n", type=int, default=None)
    p.add_argument("--out", default=None, help="기본: <output_dir>/matches.json")
    args = p.parse_args()
    cfg = config_from_args(args)
    preds = read_predictions(args.predictions or Path(cfg["output_dir"]) / "predictions.csv")
    people = attach_face_probs(load_participants(args.participants), preds)
    recs = recommend_top_n(people, cfg.get("matching", {}), args.top_n)
    out = args.out or Path(cfg["output_dir"]) / "matches.json"
    save_json(recs, out)
    print(format_recommendations(recs, people))
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
