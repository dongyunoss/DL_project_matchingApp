"""동일·매우 유사한 이미지 검출 (제안서 4-2: 여러 분할에 동시에 포함되지 않도록).

dHash(difference hash, 64bit)로 이미지를 요약하고 해밍 거리가 threshold 이하인 쌍을 같은 그룹으로 묶는다.
좌우 반전된 복사본도 잡도록 반전 이미지의 해시와도 비교한다.
FairFace에는 인물 ID가 없으므로, split은 identity 대신 이 그룹 단위로 나눈다.
"""
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


def dhash(img: Image.Image) -> int:
    """9x8 회색조로 줄인 뒤 가로로 이웃한 픽셀의 밝기 비교 64개를 비트로 만든다."""
    a = np.asarray(img.convert("L").resize((9, 8), Image.Resampling.LANCZOS), dtype=np.int16)
    return int(np.packbits((a[:, 1:] > a[:, :-1]).flatten()).view(">u8")[0])


def image_hashes(path) -> tuple[int, int]:
    """(원본 해시, 좌우 반전 해시)"""
    with Image.open(path) as im:
        im = im.convert("RGB")
        return dhash(im), dhash(ImageOps.mirror(im))


def _popcount64(x: np.ndarray) -> np.ndarray:
    return np.unpackbits(x.astype(">u8").view(np.uint8).reshape(-1, 8), axis=1).sum(axis=1)


def find_near_duplicates(hashes: list[tuple[int, int]], threshold: int = 5) -> list[tuple[int, int, int]]:
    """해밍 거리 <= threshold 인 (i, j, 거리) 쌍 목록 (i < j)."""
    if threshold is None or threshold < 0 or len(hashes) < 2:
        return []
    h = np.array([a for a, _ in hashes], dtype=np.uint64)
    hm = np.array([b for _, b in hashes], dtype=np.uint64)
    pairs = []
    for i in range(len(h) - 1):
        rest = h[i + 1:]
        d = np.minimum(_popcount64(rest ^ h[i]), _popcount64(rest ^ hm[i]))
        for k in np.nonzero(d <= threshold)[0]:
            pairs.append((i, i + 1 + int(k), int(d[k])))
    return pairs


class UnionFind:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def assign_groups(records: list[dict], image_dir, threshold: int | None = 5):
    """identity가 같거나 이미지가 거의 같은 레코드를 하나의 그룹으로 묶는다.

    반환: (records에 'group' 키를 추가한 새 리스트, 근접 중복 쌍 [(filename_a, filename_b, 거리)])
    """
    n = len(records)
    uf = UnionFind(n)
    first = {}
    for i, r in enumerate(records):
        j = first.setdefault(r["identity"], i)
        uf.union(i, j)
    dups = []
    if threshold is not None and threshold >= 0:
        hashes = [image_hashes(Path(image_dir) / r["filename"]) for r in records]
        for i, j, d in find_near_duplicates(hashes, threshold):
            uf.union(i, j)
            dups.append((records[i]["filename"], records[j]["filename"], d))
    out = [{**r, "group": f"g{uf.find(i)}"} for i, r in enumerate(records)]
    return out, dups
