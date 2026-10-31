"""CelebA에서 라벨링 후보를 identity당 1장씩 샘플링하고 팀원별 라벨링 시트를 만든다.

필요 파일 (CelebA 공식 배포본):
  <celeba_root>/identity_CelebA.txt     "000001.jpg 2880" 형식
  <celeba_root>/list_attr_celeba.txt    (선택) --exclude-attr / --require-attr 사용 시

예시:
  python -m scripts.sample_celeba_candidates --celeba-root data/celeba --n 1500 `
      --annotators member1 member2 member3 member4 member5 --exclude-attr Eyeglasses Blurry

TODO(미정): CelebA에는 인종 속성이 없어 'Asian 필터'를 걸 수 없음. 나이대/각도/안경/조도 필터 기준도 미정이므로
기본값은 필터 없음. --exclude-attr/--require-attr 는 기준이 정해지면 쓰기 위한 장치다.
"""
import argparse
import csv
import random
import shutil
from collections import defaultdict
from pathlib import Path


def read_identity_file(path) -> dict[str, str]:
    """{filename: identity}"""
    mapping = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) == 2:
                mapping[parts[0]] = parts[1]
    return mapping


def read_attr_file(path) -> dict[str, dict[str, int]]:
    """list_attr_celeba.txt → {filename: {attr: +1/-1}} (1행: 개수, 2행: 속성명)."""
    with open(path, encoding="utf-8") as f:
        f.readline()
        names = f.readline().split()
        attrs = {}
        for line in f:
            parts = line.split()
            if len(parts) == len(names) + 1:
                attrs[parts[0]] = dict(zip(names, map(int, parts[1:])))
    return attrs


def sample_one_per_identity(id_map: dict[str, str], n: int | None, seed: int = 42,
                            keep=lambda fname: True) -> list[tuple[str, str]]:
    """필터(keep)를 통과한 이미지 중 identity당 1장, 그중 n개 identity를 무작위 선택."""
    rng = random.Random(seed)
    by_id = defaultdict(list)
    for fname, ident in id_map.items():
        if keep(fname):
            by_id[ident].append(fname)
    chosen = [(rng.choice(sorted(files)), ident) for ident, files in sorted(by_id.items())]
    rng.shuffle(chosen)
    return chosen[:n] if n else chosen


def write_sheet(rows, path, with_label_col=True) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["filename", "identity", "label"] if with_label_col else ["filename", "identity"])
        for fname, ident in rows:
            w.writerow([fname, ident, ""] if with_label_col else [fname, ident])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--celeba-root", required=True, help="identity_CelebA.txt가 있는 폴더")
    ap.add_argument("--image-dir", default=None, help="이미지 폴더 (기본: <root>/img_align_celeba). 존재 여부 확인/복사용")
    ap.add_argument("--n", type=int, default=1500, help="샘플링할 identity 수 (목표 1,200~1,500)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--exclude-attr", nargs="*", default=[], help="이 속성이 +1이면 제외 (예: Eyeglasses Blurry)")
    ap.add_argument("--require-attr", nargs="*", default=[], help="이 속성이 +1인 이미지만 (예: Young)")
    ap.add_argument("--out-dir", default="data/labels")
    ap.add_argument("--annotators", nargs="*", default=[], help="팀원 이름 → annotators/<이름>.csv 시트 생성")
    ap.add_argument("--copy-to", default=None, help="샘플 이미지를 이 폴더로 복사 (라벨링 편의용, git 제외 경로 사용)")
    args = ap.parse_args()

    root = Path(args.celeba_root)
    id_map = read_identity_file(root / "identity_CelebA.txt")
    keep = lambda f: True  # noqa: E731
    if args.exclude_attr or args.require_attr:
        attrs = read_attr_file(root / "list_attr_celeba.txt")
        unknown = set(args.exclude_attr + args.require_attr) - set(next(iter(attrs.values())))
        if unknown:
            raise SystemExit(f"알 수 없는 속성: {sorted(unknown)}")

        def keep(f):
            a = attrs.get(f)
            return a is not None and all(a[x] != 1 for x in args.exclude_attr) and all(a[x] == 1 for x in args.require_attr)

    rows = sample_one_per_identity(id_map, args.n, args.seed, keep)
    out = Path(args.out_dir)
    write_sheet(rows, out / "candidates.csv", with_label_col=False)
    for name in args.annotators:
        write_sheet(rows, out / "annotators" / f"{name}.csv")
    print(f"[sample] 전체 {len(id_map)}장 / {len(set(id_map.values()))}명 → {len(rows)}명 각 1장 → {out / 'candidates.csv'}")
    if args.annotators:
        print(f"[sheets] {out / 'annotators'}/<이름>.csv 의 label 열에 dog/cat/rabbit/fox 중 하나를 입력하세요.")

    img_dir = Path(args.image_dir) if args.image_dir else root / "img_align_celeba"
    if args.copy_to:
        dst = Path(args.copy_to)
        dst.mkdir(parents=True, exist_ok=True)
        missing = 0
        for fname, _ in rows:
            src = img_dir / fname
            if src.exists():
                shutil.copy2(src, dst / fname)
            else:
                missing += 1
        print(f"[copy] {len(rows) - missing}장 복사 → {dst} (누락 {missing})")


if __name__ == "__main__":
    main()
