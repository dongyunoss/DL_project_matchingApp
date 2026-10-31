"""팀원별 라벨 CSV를 다수결로 합친다.

입력: 팀원별 CSV (filename, identity, label). label은 dog/cat/rabbit/fox 또는 강아지상/고양이상/토끼상/여우상.
      빈 칸은 기권으로 처리.
출력:
  final_labels.csv  filename, identity, label, votes_dog, votes_cat, votes_rabbit, votes_fox, n_votes, agreement
  review.csv        최다 득표 동률 또는 투표 수 부족 → 재검토 목록
  agreement.json    라벨러 간 일치도 (Fleiss' kappa, 평균 합의율)

  python -m scripts.merge_labels --inputs data/labels/annotators/*.csv --out-dir data/labels
  (PowerShell에서 glob이 안 펼쳐지면 --input-dir data/labels/annotators 사용)
"""
import argparse
import csv
import glob
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from faceapp import CLASS_KO, CLASSES  # noqa: E402

ALIASES = {**{c: c for c in CLASSES}, **{ko: en for en, ko in CLASS_KO.items()},
           **{ko.replace("상", ""): en for en, ko in CLASS_KO.items()}}


def normalize_label(raw: str) -> str | None:
    s = (raw or "").strip().lower()
    if not s:
        return None
    if s not in ALIASES:
        raise ValueError(f"알 수 없는 라벨 '{raw}' (허용: {CLASSES} 또는 한글 이름)")
    return ALIASES[s]


def read_annotator_csv(path) -> dict[str, tuple[str, str | None]]:
    """{filename: (identity, label|None)}"""
    out = {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        for i, row in enumerate(csv.DictReader(f), start=2):
            try:
                out[row["filename"].strip()] = (str(row["identity"]).strip(), normalize_label(row.get("label")))
            except ValueError as e:
                raise ValueError(f"{path}:{i}: {e}") from None
    return out


def majority_vote(labels: list[str | None], min_votes: int = 1) -> tuple[str | None, Counter, str]:
    """반환: (최종 라벨 또는 None, 득표 Counter, 사유). 동률·투표 부족이면 None."""
    votes = Counter(l for l in labels if l is not None)
    n = sum(votes.values())
    if n < min_votes:
        return None, votes, f"insufficient_votes({n}<{min_votes})"
    top = votes.most_common()
    if len(top) > 1 and top[0][1] == top[1][1]:
        return None, votes, "tie:" + "/".join(sorted(c for c, k in top if k == top[0][1]))
    return top[0][0], votes, "ok"


def fleiss_kappa(count_matrix) -> float:
    """count_matrix: 항목별 [클래스별 득표수]. 모든 항목의 라벨러 수가 같아야 한다."""
    rows = [r for r in count_matrix if sum(r) > 0]
    if not rows:
        return float("nan")
    n = sum(rows[0])
    if n < 2 or any(sum(r) != n for r in rows):
        return float("nan")
    N, k = len(rows), len(rows[0])
    p_j = [sum(r[j] for r in rows) / (N * n) for j in range(k)]
    P_i = [(sum(x * x for x in r) - n) / (n * (n - 1)) for r in rows]
    P_bar, P_e = sum(P_i) / N, sum(p * p for p in p_j)
    return float("nan") if P_e == 1 else (P_bar - P_e) / (1 - P_e)


def merge(annotations: dict[str, dict], min_votes: int = 1):
    """annotations: {annotator: {filename: (identity, label)}} → (final_rows, review_rows, stats)"""
    filenames = sorted({f for a in annotations.values() for f in a})
    final, review, full_matrix = [], [], []
    for fname in filenames:
        ids = {a[fname][0] for a in annotations.values() if fname in a}
        if len(ids) != 1:
            raise ValueError(f"{fname}: 라벨러마다 identity가 다릅니다 {sorted(ids)}")
        ident = ids.pop()
        labels = [a[fname][1] if fname in a else None for a in annotations.values()]
        label, votes, reason = majority_vote(labels, min_votes)
        counts = [votes.get(c, 0) for c in CLASSES]
        n = sum(counts)
        row = {"filename": fname, "identity": ident, **{f"votes_{c}": v for c, v in zip(CLASSES, counts)},
               "n_votes": n}
        if n == len(annotations):
            full_matrix.append(counts)
        if label is None:
            review.append({**row, "reason": reason})
        else:
            final.append({**row, "label": label, "agreement": round(votes[label] / n, 4)})
    stats = {
        "annotators": list(annotations),
        "n_items": len(filenames), "n_final": len(final), "n_review": len(review),
        "fleiss_kappa(all_annotators_voted)": fleiss_kappa(full_matrix),
        "n_items_all_voted": len(full_matrix),
        "mean_agreement": (sum(r["agreement"] for r in final) / len(final)) if final else None,
        "label_distribution": dict(Counter(r["label"] for r in final)),
    }
    return final, review, stats


def write_csv(rows, path, cols):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inputs", nargs="*", default=[], help="팀원별 CSV 경로 (glob 허용)")
    ap.add_argument("--input-dir", default=None, help="이 폴더의 *.csv 전체 사용")
    ap.add_argument("--out-dir", default="data/labels")
    ap.add_argument("--min-votes", type=int, default=3,
                    help="최소 유효 투표 수 (기본 3 = 5명 중 과반). 미달 시 review로")
    args = ap.parse_args(argv)

    paths = [Path(p) for pat in args.inputs for p in sorted(glob.glob(pat))]
    if args.input_dir:
        paths += sorted(Path(args.input_dir).glob("*.csv"))
    if not paths:
        raise SystemExit("입력 CSV가 없습니다 (--inputs 또는 --input-dir)")
    annotations = {p.stem: read_annotator_csv(p) for p in paths}

    final, review, stats = merge(annotations, args.min_votes)
    out = Path(args.out_dir)
    vote_cols = [f"votes_{c}" for c in CLASSES]
    write_csv(final, out / "final_labels.csv", ["filename", "identity", "label", *vote_cols, "n_votes", "agreement"])
    write_csv(review, out / "review.csv", ["filename", "identity", *vote_cols, "n_votes", "reason"])
    with open(out / "agreement.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"[merge] 라벨러 {len(annotations)}명, 항목 {stats['n_items']}개 → 확정 {stats['n_final']}, 재검토 {stats['n_review']}")
    print(f"        Fleiss' kappa={stats['fleiss_kappa(all_annotators_voted)']:.3f}  분포={stats['label_distribution']}")
    return stats


if __name__ == "__main__":
    main()
