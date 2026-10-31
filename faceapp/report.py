"""매칭 결과 HTML 리포트 (발표/시연용). 외부 라이브러리 없이 표준 라이브러리만 사용.

리포트는 사진을 상대 경로로 참조만 하고 복사·임베드하지 않는다. outputs/는 git 제외.
"""
import html
import os
from pathlib import Path

from . import CLASS_KO, CLASSES

PART_KO = {"face": "얼굴", "age": "나이", "hobby": "취미"}

CSS = """
body{font-family:'Malgun Gothic','Apple SD Gothic Neo',sans-serif;margin:24px;background:#fafafa;color:#222}
h1{font-size:22px} .meta{color:#666;font-size:13px;margin-bottom:18px}
.card{background:#fff;border:1px solid #e3e3e3;border-radius:10px;padding:14px;margin-bottom:14px;display:flex;gap:18px}
.me{width:190px;flex:none} .me img{width:160px;height:160px;object-fit:cover;border-radius:8px;background:#eee}
.bar{display:flex;align-items:center;font-size:12px;margin:2px 0} .bar span{width:58px}
.bar div{height:9px;background:#5b8def;border-radius:4px;margin-right:6px}
table{border-collapse:collapse;font-size:13px;flex:1} td,th{border-bottom:1px solid #eee;padding:5px 8px;text-align:left}
td img{width:56px;height:56px;object-fit:cover;border-radius:6px;background:#eee} .score{font-weight:bold}
.muted{color:#888;font-size:12px}
"""


def _img_src(photo: Path, report_dir: Path) -> str:
    try:
        return Path(os.path.relpath(photo.resolve(), report_dir.resolve())).as_posix()
    except ValueError:  # Windows에서 드라이브가 다른 경우
        return photo.resolve().as_uri()


def _prob_bars(probs) -> str:
    if probs is None:
        return '<div class="muted">예측 없음</div>'
    return "".join(
        f'<div class="bar"><span>{CLASS_KO[c]}</span><div style="width:{p * 100:.0f}px"></div>{p:.2f}</div>'
        for c, p in zip(CLASSES, probs))


def _type_label(probs) -> str:
    if probs is None:
        return "-"
    i = max(range(len(CLASSES)), key=lambda k: probs[k])
    return f"{CLASS_KO[CLASSES[i]]} ({probs[i]:.2f})"


def _pref_text(pref: dict) -> str:
    bits = []
    ft = pref.get("face_type")
    if ft:
        items = ft.items() if isinstance(ft, dict) else zip(CLASSES, ft)
        bits.append("선호 유형 " + ", ".join(f"{CLASS_KO.get(k, k)} {v}" for k, v in items if v))
    if pref.get("age_min") is not None or pref.get("age_max") is not None:
        bits.append(f"나이 {pref.get('age_min', '')}~{pref.get('age_max', '')}")
    if pref.get("gender"):
        bits.append(f"성별 {pref['gender']}")
    return " · ".join(bits) or "선호 조건 없음"


def _breakdown(detail: dict) -> str:
    out = []
    for key, label in (("parts_a_to_b", "내→상대"), ("parts_b_to_a", "상대→나")):
        parts = detail.get(key)
        if parts:
            vals = ", ".join(f"{PART_KO[k]} {v:.2f}" for k, v in parts.items() if v is not None)
            out.append(f"{label}: {vals}")
    return "<br>".join(out)


def write_html_report(recs: dict, participants: list[dict], photo_dir, out_path, meta: dict | None = None) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rdir, pdir = out_path.parent, Path(photo_dir)
    by_id = {p["id"]: p for p in participants}
    esc = html.escape

    def photo_tag(p):
        if not p.get("photo"):
            return ""
        return f'<img src="{esc(_img_src(pdir / Path(p["photo"]).name, rdir))}" alt="{esc(p["id"])}">'

    cards = []
    for pid, cands in recs.items():
        me = by_id[pid]
        rows = []
        for rank, c in enumerate(cands, 1):
            o = by_id[c["id"]]
            rows.append(
                f"<tr><td>{rank}</td><td>{photo_tag(o)}</td><td>{esc(o['id'])}<br><span class='muted'>"
                f"{esc(str(o.get('gender', '')))} · {esc(str(o.get('age', '')))}세</span></td>"
                f"<td>{esc(_type_label(o.get('face_probs')))}</td><td class='score'>{c['score']:.3f}</td>"
                f"<td class='muted'>{_breakdown(c['detail'])}</td></tr>")
        table = ("<table><tr><th>#</th><th></th><th>상대</th><th>얼굴 유형</th><th>점수</th><th>세부 점수</th></tr>"
                 + ("".join(rows) or "<tr><td colspan=6 class='muted'>조건에 맞는 상대 없음</td></tr>") + "</table>")
        hobbies = ", ".join(me.get("hobbies") or [])
        cards.append(
            f"<div class='card'><div class='me'>{photo_tag(me)}<h3>{esc(pid)}</h3>"
            f"<div class='muted'>{esc(str(me.get('gender', '')))} · {esc(str(me.get('age', '')))}세 · {esc(hobbies)}</div>"
            f"{_prob_bars(me.get('face_probs'))}<div class='muted'>{esc(_pref_text(me.get('preference') or {}))}</div>"
            f"</div>{table}</div>")

    meta_html = " · ".join(f"{esc(str(k))}: {esc(str(v))}" for k, v in (meta or {}).items())
    doc = (f"<!DOCTYPE html><html lang='ko'><head><meta charset='utf-8'><title>매칭 결과</title>"
           f"<style>{CSS}</style></head><body><h1>얼굴 유형 기반 소개팅 매칭 결과 (Top-N)</h1>"
           f"<div class='meta'>{meta_html}</div>{''.join(cards)}</body></html>")
    out_path.write_text(doc, encoding="utf-8")
    return out_path
