"""FairFace에서 East Asian 라벨링 후보를 뽑고 팀원별 라벨링 시트를 만든다 (제안서 4-1).

필요 파일 (FairFace 공식 배포본, https://github.com/joojs/fairface):
  <root>/fairface_label_train.csv, <root>/fairface_label_val.csv   열: file, age, gender, race, service_test
  <root>/fairface-img-margin025-trainval/train/*.jpg, .../val/*.jpg  (padding 0.25 버전)

  python -m scripts.sample_fairface_candidates --fairface-root data/fairface --n 1500 `
      --annotators 김주환 박정원 김정유 이동윤 김채하 --copy-to data/labeling_images

- 라벨 CSV의 filename은 이미지 폴더 기준 상대 경로(예: train/123.jpg), identity는 파일 경로 자체다
  (FairFace에는 인물 ID가 없음). 동일·유사 이미지는 --dedup과 학습 시 data.dedup_hamming으로 묶는다.
- TODO(미정): 나이대/성별 필터 기준 (FairFace 20장 테스트에서 나이대 문제가 확인됨).
  --ages 20-29 처럼 지정할 수 있지만 기본값은 필터 없음.
"""
import argparse
import csv
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from faceapp.dedup import find_near_duplicates, image_hashes  # noqa: E402

LABEL_FILES = ("fairface_label_train.csv", "fairface_label_val.csv")
DEFAULT_IMAGE_SUBDIR = "fairface-img-margin025-trainval"
AGE_BINS = ("0-2", "3-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "more than 70")


def read_fairface_labels(root) -> list[dict]:
    rows = []
    for name in LABEL_FILES:
        path = Path(root) / name
        if not path.exists():
            continue
        with open(path, newline="", encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                rows.append({"file": r["file"].strip().replace("\\", "/"), "age": r["age"].strip(),
                             "gender": r["gender"].strip(), "race": r["race"].strip()})
    if not rows:
        raise FileNotFoundError(f"{root}에 {LABEL_FILES} 중 어느 것도 없습니다")
    return rows


def filter_rows(rows, race="East Asian", ages=None, genders=None) -> list[dict]:
    ages = set(ages or [])
    genders = {g.lower() for g in genders or []}
    unknown = ages - set(AGE_BINS)
    if unknown:
        raise ValueError(f"알 수 없는 나이대 {sorted(unknown)} (가능: {AGE_BINS})")
    return [r for r in rows if r["race"] == race and (not ages or r["age"] in ages)
            and (not genders or r["gender"].lower() in genders)]


def sample_rows(rows, n: int | None, seed: int = 42, balance_gender: bool = False) -> list[dict]:
    rng = random.Random(seed)
    rows = sorted(rows, key=lambda r: r["file"])
    if not balance_gender:
        rng.shuffle(rows)
        return rows[:n] if n else rows
    by_g = defaultdict(list)
    for r in rows:
        by_g[r["gender"]].append(r)
    for g in by_g.values():
        rng.shuffle(g)
    per = (n or len(rows)) // max(len(by_g), 1)
    per = min([per] + [len(g) for g in by_g.values()])
    out = [r for g in sorted(by_g) for r in by_g[g][:per]]
    rng.shuffle(out)
    return out


def drop_near_duplicates(rows, image_dir, threshold: int = 5):
    """샘플 안의 동일·유사 이미지 중 하나만 남긴다. 반환: (남은 rows, 제거 쌍)"""
    hashes = [image_hashes(Path(image_dir) / r["file"]) for r in rows]
    drop, removed = set(), []
    for i, j, d in find_near_duplicates(hashes, threshold):
        if i not in drop and j not in drop:
            drop.add(j)
            removed.append((rows[i]["file"], rows[j]["file"], d))
    return [r for k, r in enumerate(rows) if k not in drop], removed


def write_sheet(rows, path, with_label=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["filename", "identity", "label"] if with_label else ["filename", "identity", "age", "gender"])
        for r in rows:
            w.writerow([r["file"], r["file"], ""] if with_label else [r["file"], r["file"], r["age"], r["gender"]])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fairface-root", required=True, help="fairface_label_*.csv가 있는 폴더")
    ap.add_argument("--image-dir", default=None, help=f"이미지 폴더 (기본: <root>/{DEFAULT_IMAGE_SUBDIR})")
    ap.add_argument("--race", default="East Asian")
    ap.add_argument("--ages", nargs="*", default=[], help=f"포함할 나이대 (예: 20-29 30-39). 가능: {', '.join(AGE_BINS)}")
    ap.add_argument("--genders", nargs="*", default=[], help="Male / Female (기본: 모두)")
    ap.add_argument("--balance-gender", action="store_true", help="남녀 수를 같게 샘플링")
    ap.add_argument("--n", type=int, default=1500, help="샘플 수 (목표 1,200~1,500)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dedup", type=int, default=5, help="샘플 내 동일·유사 이미지 제거 해밍 거리 (-1이면 끔, 이미지 필요)")
    ap.add_argument("--out-dir", default="data/labels")
    ap.add_argument("--annotators", nargs="*", default=[], help="팀원 이름 → annotators/<이름>.csv 시트 생성")
    ap.add_argument("--copy-to", default=None, help="샘플 이미지를 이 폴더로 복사 (train/…, val/… 구조 유지)")
    args = ap.parse_args(argv)

    root = Path(args.fairface_root)
    image_dir = Path(args.image_dir) if args.image_dir else root / DEFAULT_IMAGE_SUBDIR
    rows = read_fairface_labels(root)
    pool = filter_rows(rows, args.race, args.ages, args.genders)
    print(f"[fairface] 전체 {len(rows)}장 → race={args.race} 등 필터 후 {len(pool)}장 "
          f"(성별 {dict(Counter(r['gender'] for r in pool))}, 나이 {dict(sorted(Counter(r['age'] for r in pool).items()))})")
    # 중복 제거로 줄어들 것을 감안해 조금 더 뽑은 뒤 n개로 자른다
    sample = sample_rows(pool, int(args.n * 1.05) + 10 if args.n else None, args.seed, args.balance_gender)
    if args.dedup >= 0:
        if image_dir.exists():
            sample, removed = drop_near_duplicates(sample, image_dir, args.dedup)
            print(f"[dedup] 동일·유사 이미지 {len(removed)}장 제외")
        else:
            print(f"[dedup] 이미지 폴더({image_dir})가 없어 중복 검사를 건너뜁니다. 학습 시 다시 검사합니다.")
    sample = sample[: args.n] if args.n else sample

    out = Path(args.out_dir)
    write_sheet(sample, out / "candidates.csv", with_label=False)
    for name in args.annotators:
        write_sheet(sample, out / "annotators" / f"{name}.csv")
    print(f"[sample] {len(sample)}장 → {out / 'candidates.csv'} "
          f"(성별 {dict(Counter(r['gender'] for r in sample))})")
    if args.annotators:
        print(f"[sheets] {out / 'annotators'}/<이름>.csv → python -m scripts.label_tool 로 라벨링")
    if args.copy_to:
        dst, missing = Path(args.copy_to), 0
        for r in sample:
            src = image_dir / r["file"]
            if src.exists():
                (dst / r["file"]).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst / r["file"])
            else:
                missing += 1
        print(f"[copy] {len(sample) - missing}장 복사 → {dst} (누락 {missing})")
    return sample


if __name__ == "__main__":
    main()
