"""라벨링 도구: 브라우저에서 키보드로 동물상 라벨을 단다 (표준 라이브러리만 사용, 로컬 전용).

  python -m scripts.label_tool --sheet data/labels/annotators/member1.csv --images data/labeling_images

브라우저가 열리면:  1 강아지상 · 2 고양이상 · 3 토끼상 · 4 여우상 · 0 모름(기권)
                    ←/→ 이전/다음 · N 다음 미라벨 · Backspace 라벨 지우기
누를 때마다 시트 CSV에 바로 저장되므로 언제 꺼도 됩니다. 다시 켜면 첫 미라벨부터 이어집니다.
다른 팀원의 라벨은 보지 말고 각자 독립적으로 라벨링하세요 (다수결의 전제).
TODO(미정): 동물상 라벨링 기준 — 팀 기준이 정해지면 --guide 파일로 화면에 띄울 수 있다.
"""
import argparse
import csv
import json
import mimetypes
import os
import sys
import tempfile
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from faceapp import CLASS_KO, CLASSES  # noqa: E402
from scripts.merge_labels import ABSTAIN_ALIASES, normalize_label  # noqa: E402

ABSTAIN = "skip"  # 화면의 '모름'. CSV에 skip으로 저장 → merge_labels에서 기권 처리, 재시작해도 완료로 유지


def to_sheet_value(raw) -> str:
    """CSV/요청 값 → 시트 값 ('dog' 등 클래스, 'skip', 또는 미라벨 '')."""
    s = (raw or "").strip()
    if s.lower() in ABSTAIN_ALIASES:
        return ABSTAIN
    return normalize_label(s) or ""


class LabelSheet:
    """filename, identity, label CSV를 메모리에 올리고, 변경 시 원자적으로 저장한다."""

    def __init__(self, path):
        self.path = Path(path)
        with open(self.path, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            if not {"filename", "identity"} <= set(reader.fieldnames or []):
                raise ValueError(f"{path}: filename, identity 열이 필요합니다")
            self.rows = [{"filename": r["filename"].strip(), "identity": str(r["identity"]).strip(),
                          "label": to_sheet_value(r.get("label"))} for r in reader]
        self.lock = threading.Lock()

    def __len__(self):
        return len(self.rows)

    @property
    def n_labeled(self) -> int:
        """라벨 또는 기권(skip)으로 처리 완료된 수."""
        return sum(1 for r in self.rows if r["label"])

    def first_unlabeled(self, start: int = 0) -> int:
        n = len(self.rows)
        for k in range(n):
            i = (start + k) % n
            if not self.rows[i]["label"]:
                return i
        return min(start, n - 1)

    def set_label(self, i: int, label: str | None) -> None:
        if not 0 <= i < len(self.rows):
            raise IndexError(i)
        value = to_sheet_value(label)
        with self.lock:
            self.rows[i]["label"] = value
            self.save()

    def save(self) -> None:
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, suffix=".tmp")
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["filename", "identity", "label"])
            w.writeheader()
            w.writerows(self.rows)
        os.replace(tmp, self.path)


PAGE = """<!DOCTYPE html><html lang="ko"><head><meta charset="utf-8"><title>동물상 라벨링</title>
<style>
:root{--bg:#f4f3f6;--card:#fff;--ink:#1e1c23;--muted:#6a6674;--line:#e0dde6;--accent:#b02a5b}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 "Malgun Gothic","Apple SD Gothic Neo",sans-serif}
.wrap{max-width:760px;margin:0 auto;padding:20px;display:grid;gap:14px}
.top{display:flex;justify-content:space-between;align-items:baseline;gap:12px;flex-wrap:wrap}
.prog{height:8px;background:var(--line);border-radius:4px;overflow:hidden}.prog div{height:100%;background:var(--accent)}
.stage{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px;display:grid;justify-items:center;gap:8px}
.stage img{height:min(58vh,440px);max-width:100%;object-fit:contain;border-radius:8px;background:#ddd}
.meta{color:var(--muted);font-size:13px}.cur{font-weight:700;font-size:18px;min-height:27px}
.btns{display:grid;grid-template-columns:repeat(5,1fr);gap:8px}
button{font:inherit;padding:12px 6px;border:1px solid var(--line);border-radius:10px;background:var(--card);cursor:pointer}
button:hover,button:focus-visible{border-color:var(--accent);outline:none}button.on{background:var(--accent);color:#fff;border-color:var(--accent)}
kbd{display:inline-block;min-width:20px;padding:0 5px;border:1px solid var(--line);border-bottom-width:2px;border-radius:4px;font-size:12px;background:var(--bg)}
.help{color:var(--muted);font-size:13px}.guide{white-space:pre-wrap;font-size:13px;background:var(--card);border:1px solid var(--line);border-radius:8px;padding:10px}
.err{color:#b00020}
</style></head><body><div class="wrap">
<div class="top"><b id="who"></b><span id="count" class="meta"></span></div>
<div class="prog"><div id="bar" style="width:0"></div></div>
<div class="stage"><img id="img" alt="라벨링할 얼굴 사진"><div class="meta" id="fname"></div><div class="cur" id="cur"></div></div>
<div class="btns" id="btns"></div>
<div class="help"><kbd>1</kbd>~<kbd>4</kbd> 라벨 후 자동으로 다음 · <kbd>0</kbd> 모름(기권) · <kbd>←</kbd><kbd>→</kbd> 이동 · <kbd>N</kbd> 다음 미라벨 · <kbd>Backspace</kbd> 지우기 · 누를 때마다 자동 저장</div>
<div class="guide" id="guide" hidden></div><div class="err" id="err"></div>
</div><script>
let S=null,i=0,cur=null;
const $=id=>document.getElementById(id);
async function api(p,body){const r=await fetch(p,body?{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}:{});if(!r.ok)throw new Error(await r.text());return r.json();}
function opts(){return [...S.classes.map((c,k)=>({key:String(k+1),val:c,txt:S.ko[c]})),{key:'0',val:'skip',txt:'모름'}];}
function render(){
  $('count').textContent=`${S.labeled} / ${S.n} 완료 · 현재 ${i+1}번`;$('bar').style.width=(100*S.labeled/S.n)+'%';
  $('fname').textContent=`${cur.filename} · identity ${cur.identity}`;
  $('cur').textContent=cur.label==='skip'?'모름(기권)':(cur.label?('현재 라벨: '+S.ko[cur.label]):'미라벨');
  document.querySelectorAll('#btns button').forEach(b=>b.classList.toggle('on',b.dataset.val===cur.label));
}
async function go(k){i=(k+S.n)%S.n;cur=await api('/api/item/'+i);$('img').src='/img/'+i;render();}
async function label(v){try{const r=await api('/api/label',{i,label:v});S.labeled=r.labeled;$('err').textContent='';await go(r.next);}catch(e){$('err').textContent='저장 실패: '+e.message;}}
async function clearLabel(){const r=await api('/api/label',{i,label:''});S.labeled=r.labeled;await go(i);}
document.addEventListener('keydown',e=>{if(!S||e.ctrlKey||e.metaKey||e.altKey)return;const o=opts().find(o=>o.key===e.key);
  if(o){e.preventDefault();label(o.val);}else if(e.key==='ArrowRight'){go(i+1);}else if(e.key==='ArrowLeft'){go(i-1);}
  else if(e.key==='n'||e.key==='N'){api('/api/next?from='+(i+1)).then(r=>go(r.next));}else if(e.key==='Backspace'){e.preventDefault();clearLabel();}});
(async()=>{S=await api('/api/state');$('who').textContent='라벨러: '+S.annotator;
  if(S.guide){$('guide').hidden=false;$('guide').textContent=S.guide;}
  for(const o of opts()){const b=document.createElement('button');b.dataset.val=o.val;b.innerHTML=`<kbd>${o.key}</kbd> ${o.txt}`;b.onclick=()=>label(o.val);$('btns').append(b);}
  await go(S.start);})();
</script></body></html>"""


def make_handler(sheet: LabelSheet, image_dir: Path, guide: str = ""):
    image_dir = Path(image_dir).resolve()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # 콘솔 조용히
            pass

        def _send(self, code, body: bytes, ctype="application/json; charset=utf-8"):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code=200):
            self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"))

        def _index(self, s: str) -> int | None:
            try:
                i = int(s)
            except ValueError:
                return None
            return i if 0 <= i < len(sheet) else None

        def do_GET(self):
            path, _, query = self.path.partition("?")
            if path == "/":
                return self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
            if path == "/api/state":
                return self._json({"n": len(sheet), "labeled": sheet.n_labeled, "start": sheet.first_unlabeled(),
                                   "classes": CLASSES, "ko": CLASS_KO, "annotator": sheet.path.stem, "guide": guide})
            if path == "/api/next":
                start = int(query.split("=", 1)[1]) if query.startswith("from=") and query[5:].isdigit() else 0
                return self._json({"next": sheet.first_unlabeled(start % max(len(sheet), 1))})
            if path.startswith("/api/item/"):
                i = self._index(path.rsplit("/", 1)[1])
                if i is None:
                    return self._json({"error": "index"}, 404)
                return self._json({**sheet.rows[i], "i": i})
            if path.startswith("/img/"):
                i = self._index(path.rsplit("/", 1)[1])
                if i is None:
                    return self._send(404, b"not found", "text/plain")
                f = (image_dir / Path(sheet.rows[i]["filename"]).name).resolve()
                if f.parent != image_dir or not f.is_file():  # 시트에 있는 파일명만, 이미지 폴더 안에서만
                    return self._send(404, f"image not found: {f.name}".encode(), "text/plain; charset=utf-8")
                return self._send(200, f.read_bytes(), mimetypes.guess_type(f.name)[0] or "application/octet-stream")
            return self._send(404, b"not found", "text/plain")

        def do_POST(self):
            if self.path != "/api/label":
                return self._send(404, b"not found", "text/plain")
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                i, label = int(body["i"]), body.get("label", "")
                sheet.set_label(i, label)
            except (ValueError, KeyError, IndexError, TypeError) as e:
                return self._json({"error": str(e)}, 400)
            nxt = i if label == "" else sheet.first_unlabeled(i + 1)
            return self._json({"labeled": sheet.n_labeled, "next": nxt})

    return Handler


def serve(sheet_path, image_dir, port=8765, open_browser=True, guide_path=None):
    sheet = LabelSheet(sheet_path)
    image_dir = Path(image_dir)
    missing = [r["filename"] for r in sheet.rows if not (image_dir / r["filename"]).is_file()]
    if missing:
        print(f"[경고] 이미지 폴더에 없는 파일 {len(missing)}개 (예: {missing[:3]}) — --images 경로를 확인하세요")
    guide = Path(guide_path).read_text(encoding="utf-8") if guide_path else ""
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(sheet, image_dir, guide))
    url = f"http://127.0.0.1:{server.server_address[1]}/"
    print(f"[label] {sheet.path} ({sheet.n_labeled}/{len(sheet)} 완료) → {url}  (종료: Ctrl+C)")
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        print(f"[label] 저장됨: {sheet.path} ({sheet.n_labeled}/{len(sheet)})")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sheet", required=True, help="내 라벨링 시트 CSV (sample_celeba_candidates가 만든 annotators/<이름>.csv)")
    ap.add_argument("--images", required=True, help="이미지 폴더 (예: data/labeling_images 또는 data/celeba/img_align_celeba)")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--guide", default=None, help="화면에 띄울 라벨링 기준 텍스트 파일 (선택)")
    args = ap.parse_args(argv)
    serve(args.sheet, args.images, args.port, not args.no_browser, args.guide)


if __name__ == "__main__":
    main()
