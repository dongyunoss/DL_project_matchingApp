"""수작업 매칭 vs AI 보조 매칭 소요 시간 비교 (제안서 평가 항목).

입력 CSV (잠정 형식 — TODO(미정): 측정 프로토콜 확정):
  trial_id,method,n_participants,minutes,notes
  1,manual,20,95,팀원 2명이 사진·선호 조건을 보고 수작업 매칭
  1,ai_assisted,20,10,AI Top-5 추천을 사람이 검토·확정한 시간
- method: manual | ai_assisted
- ai_assisted의 minutes는 사람이 쓴 시간만 적는다. --pipeline-summary를 주면 모델 계산 시간
  (pipeline_summary.json의 예측+매칭 초)을 참가자 수에 비례해 더한다.
- ai_assisted 행이 없으면 계산 시간만으로 'ai_only'를 만든다 (완전 자동화 가정).

  python -m scripts.compare_matching_time --timing data/timing/matching_time.csv `
      --pipeline-summary outputs/run1/pipeline_summary.json
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

METHODS = ("manual", "ai_assisted")


def read_timing_csv(path) -> list[dict]:
    rows = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        for i, r in enumerate(csv.DictReader(f), start=2):
            method = r["method"].strip().lower()
            if method not in METHODS:
                raise ValueError(f"{path}:{i} method는 {METHODS} 중 하나여야 합니다: {r['method']}")
            n, minutes = int(r["n_participants"]), float(r["minutes"])
            if n <= 0 or minutes < 0:
                raise ValueError(f"{path}:{i} n_participants>0, minutes>=0 이어야 합니다")
            rows.append({"trial_id": str(r["trial_id"]).strip(), "method": method, "n": n, "minutes": minutes})
    return rows


def summarize_timing(rows: list[dict], ai_sec_per_participant: float | None = None) -> dict:
    """방법별 참가자당 분, 시도(trial)별 절감률을 계산한다."""
    rows = [dict(r) for r in rows]
    if ai_sec_per_participant is not None:
        for r in rows:
            if r["method"] == "ai_assisted":
                r["minutes"] += ai_sec_per_participant * r["n"] / 60
        if not any(r["method"] == "ai_assisted" for r in rows):
            rows += [{"trial_id": r["trial_id"], "method": "ai_only", "n": r["n"],
                      "minutes": ai_sec_per_participant * r["n"] / 60} for r in rows if r["method"] == "manual"]

    by_method = defaultdict(lambda: {"trials": 0, "participants": 0, "minutes": 0.0})
    for r in rows:
        m = by_method[r["method"]]
        m["trials"] += 1
        m["participants"] += r["n"]
        m["minutes"] += r["minutes"]
    for m in by_method.values():
        m["minutes_per_participant"] = m["minutes"] / m["participants"]

    per_trial = defaultdict(dict)
    for r in rows:
        per_trial[r["trial_id"]][r["method"]] = r
    trials = []
    for tid, d in sorted(per_trial.items()):
        ai = d.get("ai_assisted") or d.get("ai_only")
        if "manual" in d and ai:
            man = d["manual"]["minutes"]
            trials.append({"trial_id": tid, "n": d["manual"]["n"], "manual_min": man, "ai_min": ai["minutes"],
                           "reduction": (1 - ai["minutes"] / man) if man > 0 else None})

    ai_key = "ai_assisted" if "ai_assisted" in by_method else ("ai_only" if "ai_only" in by_method else None)
    overall = None
    if "manual" in by_method and ai_key:
        man, ai = by_method["manual"]["minutes_per_participant"], by_method[ai_key]["minutes_per_participant"]
        overall = {"compare": f"manual vs {ai_key}", "manual_min_per_participant": man,
                   "ai_min_per_participant": ai, "reduction": (1 - ai / man) if man > 0 else None,
                   "speedup": (man / ai) if ai > 0 else None}
    return {"by_method": dict(by_method), "per_trial": trials, "overall": overall}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--timing", required=True, help="측정 기록 CSV")
    ap.add_argument("--pipeline-summary", default=None, help="faceapp.pipeline이 만든 pipeline_summary.json")
    ap.add_argument("--out", default=None, help="결과 JSON (기본: timing CSV 옆 time_comparison.json)")
    args = ap.parse_args(argv)

    ai_sec = None
    if args.pipeline_summary:
        with open(args.pipeline_summary, encoding="utf-8") as f:
            ai_sec = json.load(f)["ai_matching"]["seconds_per_participant"]
    res = summarize_timing(read_timing_csv(args.timing), ai_sec)
    res["ai_compute_sec_per_participant"] = ai_sec

    print(f"{'method':<12}{'trials':>7}{'people':>8}{'min/person':>12}")
    for k, m in res["by_method"].items():
        print(f"{k:<12}{m['trials']:>7}{m['participants']:>8}{m['minutes_per_participant']:>12.3f}")
    for t in res["per_trial"]:
        red = f"{t['reduction'] * 100:.1f}%" if t["reduction"] is not None else "-"
        print(f"  trial {t['trial_id']}: n={t['n']} 수작업 {t['manual_min']:.1f}분 → AI {t['ai_min']:.2f}분 (절감 {red})")
    if res["overall"] and res["overall"]["reduction"] is not None:
        o = res["overall"]
        print(f"[overall] {o['compare']}: 참가자당 {o['manual_min_per_participant']:.2f}분 → "
              f"{o['ai_min_per_participant']:.3f}분, 절감 {o['reduction'] * 100:.1f}%")

    out = Path(args.out) if args.out else Path(args.timing).with_name("time_comparison.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[saved] {out}")
    return res


if __name__ == "__main__":
    main()
