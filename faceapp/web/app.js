// 웹캠 데모 프론트엔드: getUserMedia → MediaPipe Face Landmarker → Canvas 크롭·정렬 → /api/predict → 매칭
const MP_VERSION = '0.10.35';
const CDN_BASE = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MP_VERSION}`;
const CDN_MODEL = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task';
const LM = { rightEye: 33, leftEye: 263, nose: 1 };   // MediaPipe Face Mesh 특징점 번호

const $ = (id) => document.getElementById(id);
const S = { cfg: null, lm: null, mode: null, stream: null, probs: null, myId: null, lastGeo: null };

function h(tag, attrs = {}, ...kids) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === 'class') el.className = v; else if (k === 'style') el.style.cssText = v; else el.setAttribute(k, v);
  }
  for (const k of kids.flat()) el.append(k instanceof Node ? k : document.createTextNode(String(k)));
  return el;
}
function status(id, text, kind = '') { const el = $(id); el.textContent = text; el.className = 'status ' + kind; }
async function api(path, opts) {
  const r = await fetch(path, opts);
  const body = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(body.error || `${r.status} 오류`);
  return body;
}

// ---------------------------------------------------------------- MediaPipe
async function loadLandmarker() {
  const local = S.cfg.local_mediapipe;
  const base = local ? '/vendor/tasks-vision' : CDN_BASE;
  const model = local ? '/vendor/models/face_landmarker.task' : CDN_MODEL;
  const { FilesetResolver, FaceLandmarker } = await import(base + '/vision_bundle.mjs');
  const vision = await FilesetResolver.forVisionTasks(base + '/wasm');
  let last;
  for (const delegate of ['GPU', 'CPU']) {
    try {
      const lm = await FaceLandmarker.createFromOptions(vision, {
        baseOptions: { modelAssetPath: model, delegate }, runningMode: 'VIDEO', numFaces: 1 });
      S.mode = 'VIDEO';
      return { lm, source: local ? '로컬' : 'CDN', delegate };
    } catch (e) { last = e; }
  }
  throw last;
}
async function setMode(mode) { if (S.lm && S.mode !== mode) { await S.lm.setOptions({ runningMode: mode }); S.mode = mode; } }

// 특징점 → 얼굴 중심·크기·기울기(roll)·좌우 회전(yaw) 근사
function faceGeometry(lms, W, H) {
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (const p of lms) { x0 = Math.min(x0, p.x * W); x1 = Math.max(x1, p.x * W); y0 = Math.min(y0, p.y * H); y1 = Math.max(y1, p.y * H); }
  const r = { x: lms[LM.rightEye].x * W, y: lms[LM.rightEye].y * H };
  const l = { x: lms[LM.leftEye].x * W, y: lms[LM.leftEye].y * H };
  const eyeDist = Math.hypot(l.x - r.x, l.y - r.y) || 1;
  const yaw = (lms[LM.nose].x * W - (r.x + l.x) / 2) / eyeDist;
  return { cx: (x0 + x1) / 2, cy: (y0 + y1) / 2, size: Math.max(x1 - x0, y1 - y0), angle: Math.atan2(l.y - r.y, l.x - r.x),
           yaw, box: [x0, y0, x1, y1], W, H };
}
function centerGeometry(W, H) {  // 얼굴 검출을 못 쓸 때: 가운데 정사각형
  const size = Math.min(W, H) * 0.7 / S.cfg.crop_scale;
  return { cx: W / 2, cy: H / 2, size, angle: 0, yaw: 0, box: null, W, H, fallback: true };
}

// 눈이 수평이 되도록 회전하고, 얼굴 박스 × crop_scale 영역을 224×224로 잘라 그린다 (Canvas API)
function drawCrop(src, g, out) {
  const ctx = out.getContext('2d');
  const N = out.width, side = g.size * S.cfg.crop_scale;
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.fillStyle = '#000'; ctx.fillRect(0, 0, N, N);
  ctx.translate(N / 2, N / 2); ctx.rotate(-g.angle); ctx.scale(N / side, N / side); ctx.translate(-g.cx, -g.cy);
  ctx.drawImage(src, 0, 0);
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  const d = ctx.getImageData(0, 0, N, N).data;
  let sum = 0;
  for (let i = 0; i < d.length; i += 16) sum += 0.299 * d[i] + 0.587 * d[i + 1] + 0.114 * d[i + 2];
  return sum / (d.length / 16);  // 평균 밝기 (0~255)
}

// 제안서의 FairFace 테스트에서 문제였던 각도·크기·조도를 촬영 단계에서 안내
function qualityChecks(g, brightness) {
  if (!g) return [{ ok: false, text: '얼굴이 보이지 않습니다' }];
  if (g.fallback) return [{ ok: false, text: '얼굴 검출 없이 가운데 영역 사용' }];
  const c = [{ ok: true, text: '얼굴 감지됨' }];
  const rel = (g.box[2] - g.box[0]) / g.W;
  c.push(rel < 0.18 ? { ok: false, text: '카메라에 조금 더 가까이' } : { ok: true, text: '얼굴 크기 적당' });
  c.push(Math.abs(g.yaw) > 0.22 ? { ok: false, text: '정면을 봐 주세요' } : { ok: true, text: '정면' });
  const roll = Math.abs(g.angle * 180 / Math.PI);
  if (roll > 12) c.push({ ok: false, text: `고개 기울기 ${roll.toFixed(0)}° (자동 보정)` });
  if (brightness !== undefined) {
    if (brightness < 70) c.push({ ok: false, text: '조명이 어둡습니다' });
    else if (brightness > 200) c.push({ ok: false, text: '너무 밝습니다' });
    else c.push({ ok: true, text: '조명 양호' });
  }
  return c;
}
function renderChecks(list, target = 'checks') { $(target).replaceChildren(...list.map((c) => h('li', { class: c.ok ? '' : 'bad' }, c.text))); }

// ---------------------------------------------------------------- 카메라
async function startCamera() {
  if (!navigator.mediaDevices?.getUserMedia) { status('status', '이 브라우저는 카메라를 지원하지 않습니다. 사진 올리기를 사용하세요.', 'err'); return; }
  try {
    S.stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false });
  } catch (e) {
    status('status', `카메라를 열 수 없습니다 (${e.name}). 권한을 허용했는지 확인하거나 사진 올리기를 사용하세요.`, 'err');
    return;
  }
  const v = $('video');
  v.srcObject = S.stream;
  await v.play();
  $('placeholder').hidden = true;
  $('btn-capture').disabled = false;
  $('btn-camera').textContent = '카메라 끄기';
  requestAnimationFrame(loop);
}
function stopCamera() {
  S.stream?.getTracks().forEach((t) => t.stop());
  S.stream = null;
  $('video').srcObject = null;
  $('placeholder').hidden = false;
  $('btn-capture').disabled = true;
  $('btn-camera').textContent = '카메라 켜기';
  $('overlay').getContext('2d').clearRect(0, 0, $('overlay').width, $('overlay').height);
}
async function detectVideo(v) {
  if (!S.lm) return centerGeometry(v.videoWidth, v.videoHeight);
  await setMode('VIDEO');
  const res = S.lm.detectForVideo(v, performance.now());
  return res.faceLandmarks?.length ? faceGeometry(res.faceLandmarks[0], v.videoWidth, v.videoHeight) : null;
}
async function loop() {
  const v = $('video');
  if (!S.stream) return;
  if (v.readyState >= 2 && !S.busy) {
    const g = await detectVideo(v);
    S.lastGeo = g;
    const o = $('overlay');
    if (o.width !== v.videoWidth) { o.width = v.videoWidth; o.height = v.videoHeight; }
    const ctx = o.getContext('2d');
    ctx.clearRect(0, 0, o.width, o.height);
    if (g) {
      const side = g.size * S.cfg.crop_scale;
      ctx.save(); ctx.translate(g.cx, g.cy); ctx.rotate(g.angle);
      ctx.strokeStyle = g.fallback ? '#f2cf7a' : '#ee6f9b'; ctx.lineWidth = Math.max(2, o.width / 320);
      ctx.strokeRect(-side / 2, -side / 2, side, side); ctx.restore();
    }
    renderChecks(qualityChecks(g));
  }
  requestAnimationFrame(loop);
}

// ---------------------------------------------------------------- 분석
async function analyze(src, g) {
  if (!g) { status('status', '얼굴을 찾지 못했습니다. 정면에서 다시 찍어 주세요.', 'err'); return; }
  const crop = $('crop');
  const brightness = drawCrop(src, g, crop);
  renderChecks(qualityChecks(g, brightness), 'crop-checks');
  status('status', '분석 중…');
  const blob = await new Promise((res) => crop.toBlob(res, 'image/jpeg', 0.92));
  try {
    const r = await api('/api/predict', { method: 'POST', headers: { 'Content-Type': 'image/jpeg' }, body: blob });
    S.probs = r.probs;
    renderProbs(r.probs);
    $('result').hidden = false;
    $('btn-submit').disabled = false;
    status('status', '분석 완료. 오른쪽에 정보를 입력하고 등록하세요.', 'ok');
  } catch (e) { status('status', '분석 실패: ' + e.message, 'err'); }
}
function renderProbs(probs) {
  const best = probs.indexOf(Math.max(...probs));
  $('top').textContent = `${S.cfg.ko[S.cfg.classes[best]]} ${(probs[best] * 100).toFixed(0)}%`;
  $('bars').replaceChildren(...S.cfg.classes.map((c, i) => h('div', { class: 'bar' + (i === best ? ' best' : '') },
    h('span', {}, S.cfg.ko[c]), h('span', { class: 'track' }, h('span', { class: 'fill', style: `width:${(probs[i] * 100).toFixed(1)}%` })),
    h('span', { class: 'v' }, (probs[i] * 100).toFixed(1) + '%'))));
}
async function capture() {
  const v = $('video');
  S.busy = true;
  try { await analyze(v, await detectVideo(v)); } finally { S.busy = false; }
}
async function uploadFile(file) {
  if (!file) return;
  const img = new Image();
  img.src = URL.createObjectURL(file);
  try { await img.decode(); } catch { status('status', '이미지를 열 수 없습니다.', 'err'); return; }
  S.busy = true;
  try {
    let g = centerGeometry(img.naturalWidth, img.naturalHeight);
    if (S.lm) {
      await setMode('IMAGE');
      const res = S.lm.detect(img);
      g = res.faceLandmarks?.length ? faceGeometry(res.faceLandmarks[0], img.naturalWidth, img.naturalHeight) : null;
    }
    await analyze(img, g);
  } finally { S.busy = false; URL.revokeObjectURL(img.src); }
}

// ---------------------------------------------------------------- 프로필·매칭
function buildPrefs() {
  const n = S.cfg.classes.length;
  $('prefs').replaceChildren(...S.cfg.classes.map((c) => {
    const id = `pref-${c}`;
    const input = h('input', { type: 'range', id, min: '0', max: '100', value: String(Math.round(100 / n)), 'data-class': c });
    const out = h('output', { for: id });
    input.addEventListener('input', updatePrefOutputs);
    return h('div', { class: 'pref' }, h('label', { for: id }, S.cfg.ko[c]), input, out);
  }));
  updatePrefOutputs();
}
function prefValues() {
  return Object.fromEntries([...document.querySelectorAll('#prefs input')].map((i) => [i.dataset.class, Number(i.value)]));
}
function updatePrefOutputs() {
  const v = prefValues(); const sum = Object.values(v).reduce((a, b) => a + b, 0) || 1;
  document.querySelectorAll('#prefs input').forEach((i) => { i.nextElementSibling.value = `${Math.round(100 * v[i.dataset.class] / sum)}%`; });
}
async function submit(ev) {
  ev.preventDefault();
  if (!S.probs) { status('form-msg', '먼저 얼굴을 분석하세요.', 'err'); return; }
  const num = (id) => ($(id).value === '' ? null : Number($(id).value));
  const body = {
    id: $('f-id').value.trim(), gender: $('f-gender').value, age: num('f-age'),
    hobbies: $('f-hobbies').value.split(/[,，]/).map((s) => s.trim()).filter(Boolean),
    face_probs: S.probs,
    preference: { face_type: prefValues(), age_min: num('f-amin'), age_max: num('f-amax'), gender: $('f-pgender').value },
  };
  try {
    await api('/api/participants', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
    S.myId = body.id;
    $('btn-delete').hidden = false;
    status('form-msg', `${body.id} 님이 등록되었습니다.`, 'ok');
    await showMatches();
  } catch (e) { status('form-msg', '등록 실패: ' + e.message, 'err'); }
}
async function showMatches() {
  const [list, pool] = await Promise.all([
    api(`/api/matches?id=${encodeURIComponent(S.myId)}&top_n=${S.cfg.top_n}`), api('/api/participants')]);
  $('pool').textContent = `· 참가자 풀 ${pool.length}명`;
  const ko = S.cfg.ko, cls = S.cfg.classes, PART = { face: '얼굴', age: '나이', hobby: '취미' };
  const parts = (p) => Object.entries(p || {}).filter(([, v]) => v !== null).map(([k, v]) => `${PART[k]} ${v.toFixed(2)}`).join(' · ');
  if (!list.length) { $('matches').replaceChildren(h('li', { class: 'muted' }, '조건에 맞는 상대가 없습니다. 선호 성별·나이, 또는 상대의 선호 조건 때문에 제외되었을 수 있습니다.')); return; }
  $('matches').replaceChildren(...list.map((m, i) => {
    const p = m.person, fp = p.face_probs, b = fp.indexOf(Math.max(...fp));
    const g = { M: '남', F: '여' }[p.gender] || '';
    return h('li', { class: 'match' }, h('span', { class: 'rank' }, i + 1),
      h('div', { class: 'who' }, h('b', {}, p.id), ` ${g}${p.age ? ' · ' + p.age + '세' : ''} · ${ko[cls[b]]} ${(fp[b] * 100).toFixed(0)}%`,
        h('div', { class: 'parts' }, '내→상대 ' + parts(m.detail.parts_a_to_b)),
        h('div', { class: 'parts' }, '상대→나 ' + parts(m.detail.parts_b_to_a))),
      h('span', { class: 'score' }, m.score.toFixed(3)));
  }));
}
async function deleteMe() {
  if (!S.myId) return;
  await api(`/api/participants?id=${encodeURIComponent(S.myId)}`, { method: 'DELETE' });
  status('form-msg', `${S.myId} 님의 정보를 삭제했습니다.`, 'ok');
  S.myId = null; $('btn-delete').hidden = true;
  $('matches').replaceChildren(h('li', { class: 'muted' }, '등록하면 여기에 추천 상대가 표시됩니다.'));
}

// ---------------------------------------------------------------- 시작
(async () => {
  S.cfg = await api('/api/config');
  $('model-info').textContent = `모델 ${S.cfg.model} · 보정 T=${S.cfg.temperature.toFixed(2)}`;
  buildPrefs();
  $('btn-camera').addEventListener('click', () => (S.stream ? stopCamera() : startCamera()));
  $('btn-capture').addEventListener('click', capture);
  $('file').addEventListener('change', (e) => uploadFile(e.target.files[0]));
  $('form').addEventListener('submit', submit);
  $('btn-delete').addEventListener('click', deleteMe);
  $('f-gender').addEventListener('change', (e) => { if ($('f-pgender').value) $('f-pgender').value = e.target.value === 'F' ? 'M' : 'F'; });
  try {
    const r = await loadLandmarker();
    S.lm = r.lm;
    status('status', `얼굴 검출 준비 완료 (MediaPipe Face Landmarker, ${r.source}·${r.delegate})`, 'ok');
  } catch (e) {
    console.error(e);
    status('status', '얼굴 검출 모델을 불러오지 못해 가운데 영역으로 분석합니다. python -m scripts.fetch_mediapipe 로 받아 두세요.', 'err');
  }
})();
