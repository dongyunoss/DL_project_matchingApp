"""팀원별 라벨 CSV를 다수결로 합친다.

입력: 팀원별 CSV (filename, identity, label). label은 dog/cat/rabbit/fox 또는 강아지상/고양이상/토끼상/여우상.
      빈 칸이나 skip/모름은 기권으로 처리.
출력:
  final_labels.csv  filename, identity, label, votes_dog, votes_cat, votes_rabbit, votes_fox, n_votes, agreement
  review.csv        최다 득표 동률 또는 투표 수 부족 → 재검토 목록
  agreement.json    라벨러 간 일치도 (Fleiss' kappa 전체·클래스별, 라벨러별 다수결 일치율, 라벨러 쌍 혼동 행렬)
                    → 라벨링 파일럿 결과로 최종 클래스·기준을 정할 때 근거로 쓴다 (제안서 2)

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


ABSTAIN_ALIASES = {"skip", "모름", "?", "-", "x"}  # 기권 표시 (label_tool의 '모름'은 skip으로 저장됨)


def normalize_label(raw: str) -> str | None:
    s = (raw or "").strip().lower()
    if not s or s in ABSTAIN_ALIASES:
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


def per_class_kappa(count_matrix) -> dict[str, float]:
    """Fleiss의 클래스별 kappa: 그 클래스에 대해 라벨러들이 얼마나 일관되게 동의하는지 (낮으면 기준이 모호)."""
    rows = [r for r in count_matrix if sum(r) > 0]
    if not rows:
        return {}
    n, N = sum(rows[0]), len(rows)
    out = {}
    for j, c in enumerate(CLASSES):
        p = sum(r[j] for r in rows) / (N * n)
        if n < 2 or p in (0, 1):
            out[c] = float("nan")
            continue
        disagree = sum(r[j] * (n - r[j]) for r in rows)
        out[c] = 1 - disagree / (N * n * (n - 1) * p * (1 - p))
    return out


def pairwise_confusion(annotations: dict[str, dict]) -> list[list[int]]:
    """모든 라벨러 쌍이 같은 사진에 단 라벨의 대칭 혼동 행렬. 대각선 밖이 크면 두 클래스 구분이 모호."""
    k = len(CLASSES)
    idx = {c: i for i, c in enumerate(CLASSES)}
    m = [[0] * k for _ in range(k)]
    names = list(annotations)
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            A, B = annotations[names[a]], annotations[names[b]]
            for f in A.keys() & B.keys():
                la, lb = A[f][1], B[f][1]
                if la and lb:
                    m[idx[la]][idx[lb]] += 1
                    if la != lb:
                        m[idx[lb]][idx[la]] += 1
                    # 같은 라벨은 대각선에 한 번만
    return m


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
        "per_class_kappa": per_class_kappa(full_matrix),
        "pair_confusion": {"classes": CLASSES, "matrix": pairwise_confusion(annotations)},
        "annotator_vs_majority": {
            name: round(sum(a.get(r["filename"], (None, None))[1] == r["label"] for r in final) / len(final), 4)
            if final else None for name, a in annotations.items()},
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
    print("        클래스별 kappa: " + ", ".join(f"{CLASS_KO[c]} {v:.2f}" for c, v in stats["per_class_kappa"].items()))
    print("        라벨러별 다수결 일치율: " + ", ".join(f"{k} {v:.0%}" for k, v in stats["annotator_vs_majority"].items()
                                              if v is not None))
    m = stats["pair_confusion"]["matrix"]
    worst = sorted(((m[i][j], CLASSES[i], CLASSES[j]) for i in range(len(m)) for j in range(i + 1, len(m))), reverse=True)
    if worst and worst[0][0]:
        print("        가장 많이 엇갈린 클래스 쌍: " + ", ".join(f"{CLASS_KO[a]}↔{CLASS_KO[b]} {n}회" for n, a, b in worst[:3]))
    return stats


if __name__ == "__main__":
    main()
