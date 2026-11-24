// The pop-out piano roll (web/pianoroll.html, opened by midi.js). Edits one part of a MIDI song at a time on the
// song's own beat grid (beats = the song's beat index, as midi.json stores them); edits save themselves through
// PUT /api/midi/{sid}/parts/{pid}, which re-renders the part, and every clip of it in the timeline follows.
import { api, connectEvents } from "./api.js";

const q = new URLSearchParams(location.search);
const SID = q.get("sid");
const COLORS = ["#c9a227", "#4fb3bf", "#e07a5f", "#81b29a", "#6d9dc5", "#d4a373", "#a3c45a", "#e3b23c"];
const NOTE = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
const BLACK = new Set([1, 3, 6, 8, 10]);
const noteName = (p) => `${NOTE[p % 12]}${Math.floor(p / 12) - 1}`;
const GM_DRUMS = { 35: "Kick 2", 36: "Kick", 37: "Side stick", 38: "Snare", 39: "Clap", 40: "Snare 2", 41: "Low floor tom", 42: "Hat closed",
  43: "Floor tom", 44: "Hat pedal", 45: "Low tom", 46: "Hat open", 47: "Low-mid tom", 48: "Hi-mid tom", 49: "Crash", 50: "High tom",
  51: "Ride", 52: "China", 53: "Ride bell", 54: "Tambourine", 55: "Splash", 56: "Cowbell", 57: "Crash 2", 58: "Vibraslap", 59: "Ride 2",
  60: "Hi bongo", 61: "Low bongo", 62: "Mute hi conga", 63: "Hi conga", 64: "Low conga", 65: "Hi timbale", 66: "Low timbale",
  67: "Hi agogo", 68: "Low agogo", 69: "Cabasa", 70: "Maracas", 71: "Whistle", 72: "Long whistle", 73: "Guiro", 74: "Long guiro",
  75: "Claves", 76: "Hi block", 77: "Low block", 78: "Mute cuica", 79: "Cuica", 80: "Mute triangle", 81: "Triangle" };
const colorOf = (pid) => COLORS[((parseInt(String(pid).slice(1), 10) || 1) - 1) % COLORS.length];
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
const $ = (s) => document.getElementById(s);

// ---------------------------------------------------------------- state
let song = null;                 // GET /api/midi/{sid}
let pid = q.get("part") || null;
let notes = [];                  // [{s, l, p, v, sel}]
let cc = {};                     // {"pb": [[beat, value]], "1": [...]}
const hist = new Map();          // pid -> {undo: [], redo: []}
let tool = "draw", snap = 0.25, noteLen = 0.25, lane = "vel";
let ppb = 64, rowH = 14, sx = 0, sy = 0;
let cursor = 0, dirty = false, saveTimer = null, inflight = null, pending = false;
let play = null;                 // {from, to, t0 (performance.now at start), raf}
let clip = null;                 // copied notes, relative to the first start
const lastSent = {};             // pid -> JSON of the notes/cc as last saved (or loaded)
let saving = false, starting = false;
const KW = 70, RH = 24, LH = 120, GAP = 6;

const cv = $("prCanvas"), ctx = cv.getContext("2d");
let W = 0, H = 0;
const gridBottom = () => H - LH - GAP;

// ---------------------------------------------------------------- loading
async function load(keepView = false) {
  song = await api.get(`/api/midi/${encodeURIComponent(SID)}`);
  document.title = `Piano roll · ${song.title} · Re-Tex`;
  $("prTitle").textContent = song.title;
  $("prMeta").textContent = `  ${song.bpm.toFixed(1)} BPM · ${song.bpb}/4`;
  if (!pid || !song.parts.some((p) => p.id === pid)) pid = song.parts[0]?.id || null;
  fillParts();
  const sp = part(), theirs = sp ? JSON.stringify({ notes: sp.notes, cc: sp.cc || {} }) : null;
  if (keepView && theirs === lastSent[pid]) { draw(); return; }      // our own save coming back: keep what's on screen
  usePart(pid, keepView);
  lastSent[pid] = theirs;
}
function fillParts() {
  $("prPart").innerHTML = song.parts.map((p) => `<option value="${p.id}">${p.name.replace(/</g, "&lt;")} · ${p.instrument_label.replace(/</g, "&lt;")}</option>`).join("");
  $("prPart").value = pid;
}
function part() { return song.parts.find((p) => p.id === pid); }
function usePart(id, keepView = false) {
  pid = id;
  const p = part();
  notes = (p?.notes || []).map(([s, l, pt, v]) => ({ s, l, p: pt, v, sel: false }));
  cc = JSON.parse(JSON.stringify(p?.cc || {}));
  if (!hist.has(pid)) hist.set(pid, { undo: [], redo: [] });
  if (!keepView) frame();
  history(); draw();
}
function frame() {                        // scroll to where the notes are
  if (notes.length) {
    const ps = notes.map((n) => n.p).sort((a, b) => a - b);
    const mid = ps[ps.length >> 1];
    sy = clamp((127 - mid) * rowH - (gridBottom() - RH) / 2, 0, maxSy());
    const first = Math.min(...notes.map((n) => n.s));
    sx = Math.max(0, (Math.floor(first / song.bpb) * song.bpb) * ppb - 4);
  } else {
    sy = clamp((127 - 66) * rowH - (gridBottom() - RH) / 2, 0, maxSy());
    sx = Math.max(0, (song.first_beat || 0) * ppb - 4);
  }
}
const songEnd = () => Math.max(song.beats.length, ...notes.map((n) => n.s + n.l + song.bpb * 8), 16);
const maxSx = () => Math.max(0, songEnd() * ppb - (W - KW) + 40);
const maxSy = () => Math.max(0, 128 * rowH - (gridBottom() - RH));

// ---------------------------------------------------------------- undo / save
const snapshot = () => JSON.stringify({ notes: notes.map(({ s, l, p, v }) => [s, l, p, v]), cc });
function restore(js) { const o = JSON.parse(js); notes = o.notes.map(([s, l, p, v]) => ({ s, l, p, v, sel: false })); cc = o.cc; }
function change(fn) {                    // every edit goes through here: one undo step, then an autosave
  const h = hist.get(pid), before = snapshot();
  fn();
  if (snapshot() === before) { draw(); return; }
  h.undo.push(before); if (h.undo.length > 200) h.undo.shift();
  h.redo = [];
  touched();
}
function touched() { dirty = true; history(); draw(); status(); clearTimeout(saveTimer); saveTimer = setTimeout(save, 900); }
function undo() { const h = hist.get(pid); if (!h.undo.length) return; h.redo.push(snapshot()); restore(h.undo.pop()); touched(); }
function redo() { const h = hist.get(pid); if (!h.redo.length) return; h.undo.push(snapshot()); restore(h.redo.pop()); touched(); }
function history() { const h = hist.get(pid) || { undo: [], redo: [] }; $("prUndo").disabled = !h.undo.length; $("prRedo").disabled = !h.redo.length; }
function payload() {
  return { notes: notes.map(({ s, l, p, v }) => [+s.toFixed(6), +l.toFixed(6), p, v]).sort((a, b) => a[0] - b[0] || a[2] - b[2]), cc };
}
async function save() {
  clearTimeout(saveTimer);
  if (!dirty || !pid) return;
  if (inflight) { pending = true; return; }       // one render at a time: the newest edits go once it's done
  dirty = false; status();
  const id = pid, body = payload();
  lastSent[id] = JSON.stringify(body);
  saving = true;
  try {
    const r = await api.put(`/api/midi/${encodeURIComponent(SID)}/parts/${id}`, body);
    inflight = r.job;
    const p = song.parts.find((x) => x.id === id);
    if (p) { p.notes = body.notes; p.cc = body.cc; }
  } catch (e) { dirty = true; $("prStatus").textContent = e.message; }
  saving = false;
  status();
}
function status() {
  const s = $("prStatus");
  s.className = dirty || inflight ? "busy" : "";
  s.textContent = dirty ? "Edited…" : inflight ? "Playing it in…" : "Saved";
}
connectEvents((m) => {
  if (m.type !== "job" || !m.job || m.job.ref !== SID) return;
  if (m.job.id === inflight && (m.job.status === "done" || m.job.status === "error")) {
    inflight = null;
    if (m.job.status === "error") $("prStatus").textContent = m.job.error || "couldn't play it";
    if (pending) { pending = false; dirty = true; save(); } else status();
  } else if (!inflight && !saving && m.job.status === "done" && m.job.kind === "midi" && !dirty) {
    bank = null;
    load(true).then(() => padOn && ensureBank());                          // something else changed the song (instrument, a part added in the main window)
  }
});
window.addEventListener("beforeunload", () => {
  if (!dirty || !pid) return;
  fetch(`/api/midi/${encodeURIComponent(SID)}/parts/${pid}`, { method: "PUT", keepalive: true, headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload()) });
});

// ---------------------------------------------------------------- geometry
const X = (b) => KW + b * ppb - sx;
const Y = (p) => RH + (127 - p) * rowH - sy;
const beatAt = (x) => (x - KW + sx) / ppb;
const pitchAt = (y) => clamp(127 - Math.floor((y - RH + sy) / rowH), 0, 127);
const snapOn = (e) => snap > 0 && !(e && e.altKey);
const snapR = (b, e) => snapOn(e) ? Math.round(b / snap) * snap : b;
const snapF = (b, e) => snapOn(e) ? Math.floor(b / snap + 1e-9) * snap : b;
const minLen = () => snap > 0 ? Math.min(snap, 0.25) : 1 / 32;
function noteAt(x, y) {
  if (y < RH || y > gridBottom() || x < KW) return null;
  const b = beatAt(x), p = pitchAt(y);
  for (let i = notes.length - 1; i >= 0; i--) {
    const n = notes[i];
    if (n.p === p && b >= n.s && b <= n.s + n.l) return n;
  }
  return null;
}
function edgeOf(n, x) {
  const x0 = X(n.s), x1 = X(n.s + n.l), w = x1 - x0;
  if (x1 - x <= Math.min(7, w / 3)) return "end";
  if (x - x0 <= Math.min(5, w / 4)) return "start";
  return null;
}

// ---------------------------------------------------------------- drawing
function resize() {
  const r = cv.getBoundingClientRect(), dpr = devicePixelRatio || 1;
  W = r.width; H = r.height;
  cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  sx = clamp(sx, 0, maxSx()); sy = clamp(sy, 0, maxSy());
  draw();
}
let drawQueued = false;
function draw() { if (!drawQueued) { drawQueued = true; requestAnimationFrame(paint); } }
let box = null;                         // rubber band {x0, y0, x1, y1}
function paint() {
  drawQueued = false;
  if (!song) return;
  const gb = gridBottom(), p = part(), drum = !!p?.drum, col = colorOf(pid);
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = "#0c0c0f"; ctx.fillRect(0, 0, W, H);
  const b0 = Math.max(0, Math.floor(beatAt(KW))), b1 = Math.ceil(beatAt(W));
  // rows
  ctx.save(); ctx.beginPath(); ctx.rect(KW, RH, W - KW, gb - RH); ctx.clip();
  for (let pt = pitchAt(gb); pt <= pitchAt(RH); pt++) {
    const y = Y(pt);
    ctx.fillStyle = BLACK.has(pt % 12) ? "#121216" : "#17171c";
    ctx.fillRect(KW, y, W - KW, rowH);
    if (pt % 12 === 0) { ctx.fillStyle = "rgba(255,255,255,0.10)"; ctx.fillRect(KW, y + rowH - 1, W - KW, 1); }
  }
  // past the song's own grid: dimmed
  const ge = X(song.beats.length - 1);
  if (ge < W) { ctx.fillStyle = "rgba(0,0,0,0.35)"; ctx.fillRect(Math.max(KW, ge), RH, W, gb - RH); }
  // grid lines
  const sub = snap > 0 && snap * ppb >= 7 ? snap : 0;
  if (sub) {
    ctx.fillStyle = "rgba(255,255,255,0.035)";
    for (let b = Math.floor(b0 / sub) * sub; b <= b1; b += sub) ctx.fillRect(Math.round(X(b)), RH, 1, gb - RH);
  }
  for (let b = b0; b <= b1; b++) {
    const bar = (b - (song.first_beat || 0)) % song.bpb === 0;
    ctx.fillStyle = bar ? "rgba(255,255,255,0.20)" : "rgba(255,255,255,0.08)";
    ctx.fillRect(Math.round(X(b)), RH, 1, gb - RH);
  }
  // other parts, faint
  for (const o of song.parts) {
    if (o.id === pid) continue;
    ctx.fillStyle = colorOf(o.id) + "30";
    for (const [s, l, pt] of o.notes) {
      if (s + l < b0 || s > b1) continue;
      ctx.fillRect(X(s), Y(pt) + 3, Math.max(2, l * ppb), rowH - 6);
    }
  }
  // this part
  for (const n of notes) {
    if (n.s + n.l < b0 - 1 || n.s > b1) continue;
    const x = X(n.s), y = Y(n.p), w = Math.max(3, n.l * ppb - 1);
    ctx.globalAlpha = 0.45 + 0.55 * (n.v / 127);
    ctx.fillStyle = n.sel ? "#ffffff" : col;
    ctx.fillRect(x, y + 1, w, rowH - 2);
    ctx.globalAlpha = 1;
    ctx.strokeStyle = n.sel ? col : "rgba(0,0,0,0.55)"; ctx.lineWidth = 1;
    ctx.strokeRect(x + 0.5, y + 1.5, w - 1, rowH - 3);
    if (w > 34 && rowH >= 12) {
      ctx.fillStyle = n.sel ? "#000" : "rgba(0,0,0,0.75)"; ctx.font = "10px Segoe UI, sans-serif";
      ctx.fillText(drum ? (GM_DRUMS[n.p] || noteName(n.p)) : noteName(n.p), x + 3, y + rowH - 4, w - 6);
    }
  }
  if (box) {
    ctx.fillStyle = "rgba(30,215,96,0.10)"; ctx.strokeStyle = "rgba(30,215,96,0.8)";
    const x = Math.min(box.x0, box.x1), y = Math.min(box.y0, box.y1);
    ctx.fillRect(x, y, Math.abs(box.x1 - box.x0), Math.abs(box.y1 - box.y0));
    ctx.strokeRect(x + 0.5, y + 0.5, Math.abs(box.x1 - box.x0), Math.abs(box.y1 - box.y0));
  }
  ctx.restore();

  // keyboard
  ctx.save(); ctx.beginPath(); ctx.rect(0, RH, KW, gb - RH); ctx.clip();
  for (let pt = pitchAt(gb); pt <= pitchAt(RH); pt++) {
    const y = Y(pt), blk = BLACK.has(pt % 12);
    const lit = keyDown === pt || padDown.has(pt);
    ctx.fillStyle = lit ? "#1ed760" : padRange(pt) ? (blk ? "#26332b" : "#c8dccf") : blk ? "#1b1b20" : "#d9d9de";
    ctx.fillRect(0, y, KW - 1, rowH - 1);
    if (rowH >= 10 && (drum ? GM_DRUMS[pt] : pt % 12 === 0)) {
      ctx.fillStyle = blk && !lit ? "#c9c9cf" : "#222";
      ctx.font = `${Math.min(10, rowH - 3)}px Segoe UI, sans-serif`;
      ctx.fillText(drum ? GM_DRUMS[pt] : noteName(pt), 4, y + rowH - 3, KW - 8);
    }
  }
  ctx.restore();

  // ruler
  ctx.fillStyle = "#141418"; ctx.fillRect(0, 0, W, RH);
  ctx.save(); ctx.beginPath(); ctx.rect(KW, 0, W - KW, RH); ctx.clip();
  ctx.font = "11px Segoe UI, sans-serif";
  const fb = song.first_beat || 0, every = ppb * song.bpb < 40 ? 4 : 1;
  for (let b = b0; b <= b1; b++) {
    const rel = b - fb;
    if (rel % song.bpb !== 0) { ctx.fillStyle = "rgba(255,255,255,0.15)"; ctx.fillRect(Math.round(X(b)), RH - 5, 1, 5); continue; }
    const bar = Math.floor(rel / song.bpb) + 1;
    ctx.fillStyle = "rgba(255,255,255,0.35)"; ctx.fillRect(Math.round(X(b)), 8, 1, RH - 8);
    if ((bar - 1) % every === 0) { ctx.fillStyle = "#a7a7ad"; ctx.fillText(String(bar), X(b) + 4, 15); }
  }
  ctx.restore();
  ctx.fillStyle = "#141418"; ctx.fillRect(0, 0, KW, RH);
  ctx.fillStyle = "#6d6d74"; ctx.font = "10px Segoe UI, sans-serif"; ctx.fillText(drum ? "drums" : "bar", 6, 15);

  paintLane(b0, b1, col);

  // cursor + playhead
  const cx = X(cursor);
  if (cx >= KW) { ctx.fillStyle = "rgba(30,215,96,0.55)"; ctx.fillRect(Math.round(cx), 0, 1, H); ctx.beginPath(); ctx.moveTo(cx - 5, 0); ctx.lineTo(cx + 5, 0); ctx.lineTo(cx, 7); ctx.fill(); }
  if (play) {
    const px = X(playPos());
    if (px >= KW) { ctx.fillStyle = "#ffffff"; ctx.fillRect(Math.round(px), RH, 1.5, H - RH); }
  }
}

const LANE = { vel: [1, 127], pb: [-8192, 8191], 64: [0, 127] };
const laneRange = () => LANE[lane] || [0, 127];
const laneTop = () => gridBottom() + GAP;
const laneY = (v) => { const [a, z] = laneRange(); return laneTop() + LH - 6 - (v - a) / (z - a) * (LH - 12); };
const laneV = (y) => { const [a, z] = laneRange(); return Math.round(clamp(a + (laneTop() + LH - 6 - y) / (LH - 12) * (z - a), a, z)); };
function paintLane(b0, b1, col) {
  const t = laneTop();
  ctx.fillStyle = "#111115"; ctx.fillRect(0, t - GAP, W, H - t + GAP);
  ctx.fillStyle = "#16161b"; ctx.fillRect(KW, t, W - KW, LH);
  ctx.save(); ctx.beginPath(); ctx.rect(KW, t, W - KW, LH); ctx.clip();
  for (let b = b0; b <= b1; b++) {
    const bar = (b - (song.first_beat || 0)) % song.bpb === 0;
    ctx.fillStyle = bar ? "rgba(255,255,255,0.12)" : "rgba(255,255,255,0.05)";
    ctx.fillRect(Math.round(X(b)), t, 1, LH);
  }
  if (lane === "pb") { ctx.fillStyle = "rgba(255,255,255,0.15)"; ctx.fillRect(KW, laneY(0), W - KW, 1); }
  if (lane === "vel") {
    for (const n of notes) {
      if (n.s < b0 - 1 || n.s > b1) continue;
      const x = X(n.s), y = laneY(n.v);
      ctx.fillStyle = n.sel ? "#ffffff" : col;
      ctx.fillRect(Math.round(x), y, 2, t + LH - 6 - y);
      ctx.fillRect(Math.round(x) - 2, y - 2, 6, 4);
    }
  } else {
    const pts = cc[lane] || [];
    ctx.strokeStyle = col; ctx.lineWidth = 1.5; ctx.beginPath();
    let prevY = laneY(lane === "pb" ? 0 : lane === "7" || lane === "11" ? 127 : lane === "10" ? 64 : 0);
    ctx.moveTo(KW, prevY);
    for (const [b, v] of pts) { const x = X(b), y = laneY(v); ctx.lineTo(x, prevY); ctx.lineTo(x, y); prevY = y; }
    ctx.lineTo(W, prevY); ctx.stroke();
    ctx.fillStyle = col;
    for (const [b, v] of pts) { const x = X(b); if (x > KW - 4 && x < W + 4) ctx.fillRect(x - 2, laneY(v) - 2, 4, 4); }
    if (!pts.length) { ctx.fillStyle = "#6d6d74"; ctx.font = "11px Segoe UI, sans-serif"; ctx.fillText("draw here to add changes; right-drag erases", KW + 10, t + 18); }
  }
  ctx.restore();
}

// ---------------------------------------------------------------- playing (the audition voice)
const playPos = () => play ? Math.min(play.to, play.from + (performance.now() - play.t0) / 1000 * song.bpm / 60) : 0;
async function startPlay() {
  if (starting) { starting = false; await stopPlay(); return; }    // pressed again while it was getting ready
  await stopPlay();
  starting = true;
  const end = Math.max(cursor + song.bpb * 4, ...notes.map((n) => n.s + n.l));
  const from = cursor, to = Math.min(end + 1, cursor + song.bpb * 32);
  $("prPlay").classList.add("on");
  try {
    await api.post("/api/midi/audition", { sid: SID, pid, ...payload(), from, to, bpm: song.bpm, all: $("prAll").checked });
  } catch (e) { starting = false; $("prPlay").classList.remove("on"); $("prStatus").textContent = e.message; return; }
  if (!starting) { api.post("/api/midi/audition_stop").catch(() => {}); return; }   // stopped while it rendered
  starting = false;
  play = { from, to, t0: performance.now() };
  const tick = () => {
    if (!play) return;
    if (playPos() >= play.to) { stopPlay(true); return; }
    const px = X(playPos());
    if (px > W - 40) { sx = clamp(sx + (W - KW) * 0.8, 0, maxSx()); }
    draw(); play.raf = requestAnimationFrame(tick);
  };
  tick();
}
async function stopPlay(ended = false) {
  if (play?.raf) cancelAnimationFrame(play.raf);
  const was = !!play; play = null; $("prPlay").classList.remove("on"); draw();
  if (was && !ended) await api.post("/api/midi/audition_stop").catch(() => {});
}
let hearT = 0;
function hear(p, v = 100) {                // one note, for drawing and the keyboard
  const now = performance.now();
  if (now - hearT < 60) return; hearT = now;
  if (bank) {                                // the pad's ready-made sound: no wait
    api.post("/api/midi/keys/on", { bank, pitch: p, vel: v }).catch(() => {});
    setTimeout(() => bank && api.post("/api/midi/keys/off", { bank, pitch: p }).catch(() => {}), 350);
    return;
  }
  api.post("/api/midi/audition", { sid: SID, pid, notes: [[0, 0.5, p, v]], cc: {}, from: 0, to: 1, bpm: 120 }).catch(() => {});
}

// ---------------------------------------------------------------- mouse
let drag = null, keyDown = null;
const selected = () => notes.filter((n) => n.sel);
const selectOnly = (n) => notes.forEach((m) => (m.sel = m === n));
cv.addEventListener("contextmenu", (e) => e.preventDefault());
cv.addEventListener("mousedown", (e) => {
  cv.focus();
  const r = cv.getBoundingClientRect(), x = e.clientX - r.left, y = e.clientY - r.top;
  if (y < RH && x >= KW) {                        // ruler: the cursor
    cursor = Math.max(0, snapR(beatAt(x), e));
    if (play) startPlay(); else draw();
    return;
  }
  if (y > gridBottom()) { if (x >= KW) laneDown(e, x, y); return; }
  if (x < KW) {                                   // the keyboard: plays like the pad keys (and records with Rec)
    keyDown = pitchAt(y);
    if (bank || padOn) padOnNote("mouse", keyDown); else hear(keyDown, +$("prVel").value || 100);
    draw(); return;
  }
  const n = noteAt(x, y), b = beatAt(x), p = pitchAt(y);
  if (e.button === 2 || tool === "erase") {       // delete what's under the pointer, and on while dragging
    const before = snapshot();
    if (n) notes.splice(notes.indexOf(n), 1);
    drag = { kind: "erase", before };
    draw(); return;
  }
  if (n) {
    if (e.shiftKey) { n.sel = !n.sel; draw(); return; }
    if (!n.sel) selectOnly(n);
    const before = snapshot();
    let set = selected();
    if (e.ctrlKey) {                               // Ctrl+drag: copies
      const copies = set.map((m) => ({ ...m }));
      set.forEach((m) => (m.sel = false));
      notes.push(...copies); set = copies;
    }
    const edge = edgeOf(n, x);
    drag = { kind: edge === "end" ? "len" : edge === "start" ? "start" : "move", anchor: n, b0: b, p0: p, before,
      orig: set.map((m) => ({ m, s: m.s, l: m.l, p: m.p })) };
    if ($("prHear").checked && drag.kind === "move") hear(n.p, n.v);
    draw(); return;
  }
  if (tool === "draw") {
    const before = snapshot();
    const s = Math.max(0, snapF(b, e));
    const m = { s, l: noteLen, p, v: clamp(+$("prVel").value || 100, 1, 127), sel: true };
    notes.forEach((k) => (k.sel = false));
    notes.push(m);
    drag = { kind: "len", anchor: m, b0: s + noteLen, p0: p, before, drawn: true, orig: [{ m, s: m.s, l: m.l, p: m.p }] };
    if ($("prHear").checked) hear(p, m.v);
    draw(); return;
  }
  if (!e.shiftKey) notes.forEach((k) => (k.sel = false));
  box = { x0: x, y0: y, x1: x, y1: y, keep: new Set(selected()) };
  drag = { kind: "box" };
  draw();
});
window.addEventListener("mousemove", (e) => {
  if (!drag) { hover(e); return; }
  const r = cv.getBoundingClientRect(), x = e.clientX - r.left, y = e.clientY - r.top, b = beatAt(x);
  if (drag.kind === "erase") {
    const n = noteAt(x, y); if (n) { notes.splice(notes.indexOf(n), 1); draw(); }
  } else if (drag.kind === "move") {
    const a = drag.orig.find((o) => o.m === drag.anchor);
    const ns = Math.max(0, snapR(a.s + (b - drag.b0), e)), ds = ns - a.s;
    const minP = Math.min(...drag.orig.map((o) => o.p)), maxP = Math.max(...drag.orig.map((o) => o.p));
    const dp = clamp(pitchAt(y) - drag.p0, -minP, 127 - maxP);
    const minS = Math.min(...drag.orig.map((o) => o.s));
    const dsx = Math.max(ds, -minS);
    if ($("prHear").checked && dp !== drag.dp) hear(a.p + dp, drag.anchor.v);
    drag.dp = dp;
    for (const o of drag.orig) { o.m.s = o.s + dsx; o.m.p = o.p + dp; }
    draw();
  } else if (drag.kind === "len") {
    const a = drag.orig.find((o) => o.m === drag.anchor);
    const end = snapOn(e) ? Math.max(a.s + minLen(), Math.ceil((b - 1e-9) / snap) * snap) : Math.max(a.s + 1 / 64, b);
    const dl = end - (a.s + a.l);
    for (const o of drag.orig) o.m.l = Math.max(minLen() / 2, o.l + dl);
    draw();
  } else if (drag.kind === "start") {
    const a = drag.orig.find((o) => o.m === drag.anchor);
    const ns = clamp(snapR(b, e), 0, a.s + a.l - 1 / 64), ds = ns - a.s;
    for (const o of drag.orig) { const e0 = o.s + o.l; o.m.s = clamp(o.s + ds, 0, e0 - 1 / 64); o.m.l = e0 - o.m.s; }
    draw();
  } else if (drag.kind === "box") {
    box.x1 = clamp(x, KW, W); box.y1 = clamp(y, RH, gridBottom());
    const ba = beatAt(Math.min(box.x0, box.x1)), bz = beatAt(Math.max(box.x0, box.x1));
    const pa = pitchAt(Math.max(box.y0, box.y1)), pz = pitchAt(Math.min(box.y0, box.y1));
    for (const n of notes) n.sel = box.keep.has(n) || (n.s < bz && n.s + n.l > ba && n.p >= pa && n.p <= pz);
    draw();
  } else if (drag.kind === "lane") laneMove(e, x, y);
});
window.addEventListener("mouseup", () => {
  if (keyDown != null) { keyDown = null; padOffNote("mouse"); draw(); }
  if (!drag) return;
  const d = drag; drag = null; box = null;
  if (d.before !== undefined && snapshot() !== d.before) {
    const h = hist.get(pid); h.undo.push(d.before); h.redo = [];
    if (d.kind === "len" && d.drawn) noteLen = d.anchor.l, syncLen();
    else if (d.kind === "len") noteLen = d.anchor.l, syncLen();
    if (d.anchor) $("prVel").value = d.anchor.v;
    touched();
  } else draw();
});
function hover(e) {
  const r = cv.getBoundingClientRect(), x = e.clientX - r.left, y = e.clientY - r.top;
  if (e.target !== cv) return;
  let c = "default";
  if (y < RH && x >= KW) c = "text";
  else if (y > gridBottom() || x < KW) c = "pointer";
  else {
    const n = noteAt(x, y);
    if (tool === "erase") c = "not-allowed";
    else if (n) c = edgeOf(n, x) ? "ew-resize" : "move";
    else if (tool === "draw") c = "crosshair";
  }
  cv.style.cursor = c;
}

// the lane: velocities of the notes under the pointer, or a controller's changes
function laneDown(e, x, y) {
  drag = { kind: "lane", before: snapshot(), erase: e.button === 2, lastB: beatAt(x) };
  laneMove(e, x, y);
}
function laneMove(e, x, y) {
  const b = beatAt(x), v = laneV(y);
  const lo = Math.min(drag.lastB, b) - 2 / ppb, hi = Math.max(drag.lastB, b) + 2 / ppb;
  if (lane === "vel") {
    const sel = selected(), pool = sel.length ? sel : notes;
    for (const n of pool) if (n.s >= lo - 3 / ppb && n.s <= hi + 3 / ppb) n.v = clamp(v, 1, 127);
  } else {
    const pts = (cc[lane] ||= []);
    const st = snap > 0 ? Math.max(snap / 2, 1 / 32) : 1 / 32;
    const kept = pts.filter(([pb]) => pb < lo || pb > hi);
    if (!drag.erase) kept.push([Math.max(0, Math.round(b / st) * st), lane === "64" ? (v >= 64 ? 127 : 0) : v]);
    kept.sort((a, c) => a[0] - c[0]);
    const out = [];
    for (const p of kept) { if (out.length && Math.abs(out[out.length - 1][0] - p[0]) < 1e-9) out[out.length - 1] = p; else out.push(p); }
    cc[lane] = out;
    if (!out.length) delete cc[lane];
  }
  drag.lastB = b;
  draw();
}

// ---------------------------------------------------------------- wheel
cv.addEventListener("wheel", (e) => {
  e.preventDefault();
  const r = cv.getBoundingClientRect(), x = e.clientX - r.left, y = e.clientY - r.top;
  const d = e.deltaY || e.deltaX;
  if (e.ctrlKey) {
    const b = beatAt(Math.max(x, KW));
    ppb = clamp(ppb * (d > 0 ? 1 / 1.15 : 1.15), 6, 600);
    sx = clamp(b * ppb - (Math.max(x, KW) - KW), 0, maxSx());
  } else if (e.altKey) {
    const p = (y - RH + sy) / rowH;
    rowH = clamp(rowH + (d > 0 ? -1 : 1), 6, 32);
    sy = clamp(p * rowH - (y - RH), 0, maxSy());
  } else if (e.shiftKey || Math.abs(e.deltaX) > Math.abs(e.deltaY)) sx = clamp(sx + d, 0, maxSx());
  else sy = clamp(sy + d, 0, maxSy());
  draw();
}, { passive: false });

// ---------------------------------------------------------------- tools
function setTool(t) {
  tool = t;
  for (const [id, k] of [["toolDraw", "draw"], ["toolSel", "select"], ["toolErase", "erase"]]) $(id).classList.toggle("on", k === t);
}
const target = () => { const s = selected(); return s.length ? s : notes; };
function quantize() {
  const g = snap || 0.25, k = clamp((+$("prQStr").value || 100) / 100, 0, 1), ends = $("prQEnd").checked;
  change(() => {
    for (const n of target()) {
      const e0 = n.s + n.l;
      n.s = Math.max(0, n.s + (Math.round(n.s / g) * g - n.s) * k);
      if (ends) { const ge = Math.max(n.s + g, Math.round(e0 / g) * g); n.l = Math.max(1 / 64, e0 + (ge - e0) * k - n.s); }
      else n.l = Math.max(1 / 64, e0 - n.s);
    }
  });
}
function transpose(d) {
  change(() => {
    const t = target();
    if (t.some((n) => n.p + d < 0 || n.p + d > 127)) return;
    t.forEach((n) => (n.p += d));
  });
  const t = selected(); if ($("prHear").checked && t.length) hear(t[0].p, t[0].v);
}
function humanize() {
  change(() => { for (const n of target()) { n.s = Math.max(0, n.s + (Math.random() - 0.5) * 0.03); n.v = clamp(Math.round(n.v + (Math.random() - 0.5) * 16), 1, 127); } });
}
function legato() {
  change(() => {
    const t = [...target()].sort((a, b) => a.s - b.s);
    const starts = [...new Set(t.map((n) => +n.s.toFixed(6)))].sort((a, b) => a - b);
    for (const n of t) { const nx = starts.find((s) => s > n.s + 1e-6); if (nx != null) n.l = nx - n.s; }
  });
}
function samePitch() { const ps = new Set(selected().map((n) => n.p)); if (!ps.size) return; notes.forEach((n) => (n.sel = ps.has(n.p))); draw(); }
function del() { if (!selected().length) return; change(() => { notes = notes.filter((n) => !n.sel); }); }
function copy(cut = false) {
  const s = selected(); if (!s.length) return;
  const s0 = Math.min(...s.map((n) => n.s));
  clip = { notes: s.map(({ s: st, l, p, v }) => ({ s: st - s0, l, p, v })), span: Math.max(...s.map((n) => n.s + n.l)) - s0 };
  if (cut) del();
}
function paste(at) {
  if (!clip) return;
  change(() => {
    notes.forEach((n) => (n.sel = false));
    notes.push(...clip.notes.map((n) => ({ ...n, s: n.s + at, sel: true })));
  });
}
function duplicate() {
  const s = selected(); if (!s.length) return;
  copy();
  const s0 = Math.min(...s.map((n) => n.s)), g = snap || 1;
  paste(s0 + Math.ceil((clip.span - 1e-6) / g) * g);
}
function nudge(d, e) {
  const s = selected(); if (!s.length) return;
  const step = snapOn(e) ? snap : 1 / 64;
  change(() => { if (s.some((n) => n.s + d * step < 0)) return; s.forEach((n) => (n.s += d * step)); });
}

// ---------------------------------------------------------------- the computer keyboard as a MIDI pad
// A W S E D F T G Y H U J K O L P ; ' play the notes (white keys on the middle row, black keys above), Z / X move an
// octave, C / V change the velocity. The instrument is rendered ahead for these keys (mashlab/midi_live.py), so a
// key sounds at once. With Rec on, what you play goes into the part: at the playhead while it plays, or note by
// note at the cursor while stopped.
const PAD = { KeyA: 0, KeyW: 1, KeyS: 2, KeyE: 3, KeyD: 4, KeyF: 5, KeyT: 6, KeyG: 7, KeyY: 8, KeyH: 9, KeyU: 10, KeyJ: 11,
  KeyK: 12, KeyO: 13, KeyL: 14, KeyP: 15, Semicolon: 16, Quote: 17 };
let padOn = true, padOct = 4, padVel = 100, bank = null, bankReq = null, rec = false;
const padHeld = new Map();               // key code -> {pitch, s (beat), stepNote}
const padDown = new Set();               // pitches sounding from the pad (lit on the keyboard)
let stepHeld = 0;
const padBase = () => 12 * (padOct + 1);
const padRange = (pt) => padOn && pt >= padBase() && pt <= padBase() + 17;
function padInfo() {
  $("prPadInfo").textContent = padOn ? `${noteName(padBase())}-${noteName(padBase() + 17)} · vel ${padVel}${bank ? "" : bankReq ? " · getting the sound ready…" : ""}` : "";
  $("prRec").classList.toggle("on", rec);
  draw();
}
function ensureBank() {
  if (!pid || !song) return Promise.resolve(null);
  if (bankReq) return bankReq;
  const lo = padBase() - 12, hi = padBase() + 24;
  bankReq = api.post("/api/midi/keys/prepare", { sid: SID, pid, lo, hi })
    .then((r) => { bank = r.bank; return bank; })
    .catch((e) => { $("prStatus").textContent = e.message; return null; })
    .finally(() => { bankReq = null; padInfo(); });
  padInfo();
  return bankReq;
}
async function padOnNote(code, pitch) {
  padDown.add(pitch);
  const at = play ? playPos() : cursor;
  const h = { pitch, s: at, stepNote: null };
  padHeld.set(code, h);
  if (rec && !play) {                                     // step input: the note goes in now, at the cursor
    const n = { s: cursor, l: noteLen, p: pitch, v: padVel, sel: false };
    change(() => notes.push(n));
    h.stepNote = n; stepHeld++;
  }
  draw();
  let b = bank || await ensureBank();
  if (!b) return;
  try { await api.post("/api/midi/keys/on", { bank: b, pitch, vel: padVel }); }
  catch {                                                 // the server forgot the sound (restarted): once more
    bank = null; b = await ensureBank();
    if (b) api.post("/api/midi/keys/on", { bank: b, pitch, vel: padVel }).catch(() => {});
  }
}
function padOffNote(code) {
  const h = padHeld.get(code);
  if (!h) return;
  padHeld.delete(code);
  if (![...padHeld.values()].some((o) => o.pitch === h.pitch)) padDown.delete(h.pitch);
  if (bank) api.post("/api/midi/keys/off", { bank, pitch: h.pitch }).catch(() => {});
  if (rec && play) {                                      // played along: the note as long as it was held
    const end = playPos();
    change(() => notes.push({ s: Math.max(0, h.s), l: Math.max(1 / 64, end - h.s), p: h.pitch, v: padVel, sel: false }));
  } else if (h.stepNote && --stepHeld <= 0) {             // the chord is let go: the cursor moves on
    stepHeld = 0; cursor += noteLen;
    const cx = X(cursor); if (cx > W - 60) sx = clamp(sx + (W - KW) * 0.6, 0, maxSx());
  }
  draw();
}
function padAllOff() {
  for (const code of [...padHeld.keys()]) padOffNote(code);
  padDown.clear(); stepHeld = 0;
  api.post("/api/midi/keys/all_off").catch(() => {});
  draw();
}
window.addEventListener("blur", padAllOff);
window.addEventListener("keyup", (e) => { if (padHeld.has(e.code)) { e.preventDefault(); padOffNote(e.code); } });

window.addEventListener("keydown", (e) => {
  if (e.target.matches?.("input, select, textarea")) return;
  const k = (e.key || "").toLowerCase(), c = e.ctrlKey || e.metaKey;
  if (padOn && !c && !e.altKey) {
    if (e.code in PAD) {
      e.preventDefault();
      if (!e.repeat && !padHeld.has(e.code)) padOnNote(e.code, Math.min(127, padBase() + PAD[e.code]));
      return;
    }
    if (e.code === "KeyZ" || e.code === "KeyX") {
      e.preventDefault(); padAllOff();
      padOct = clamp(padOct + (e.code === "KeyZ" ? -1 : 1), -1, 8); bank = null; ensureBank(); return;
    }
    if (e.code === "KeyC" || e.code === "KeyV") {
      e.preventDefault(); padVel = clamp(padVel + (e.code === "KeyC" ? -10 : 10), 10, 127); $("prVel").value = padVel; padInfo(); return;
    }
  }
  if (!c && !e.altKey && (e.code === "Digit1" || e.code === "Digit2" || e.code === "Digit3")) {
    setTool(["draw", "select", "erase"][+e.code.slice(5) - 1]); return;
  }
  if (c && k === "z") { e.preventDefault(); e.shiftKey ? redo() : undo(); }
  else if (c && k === "y") { e.preventDefault(); redo(); }
  else if (c && k === "a") { e.preventDefault(); notes.forEach((n) => (n.sel = true)); draw(); }
  else if (c && k === "c") { e.preventDefault(); copy(); }
  else if (c && k === "x") { e.preventDefault(); copy(true); }
  else if (c && k === "v") { e.preventDefault(); paste(cursor); }
  else if (c && k === "d") { e.preventDefault(); duplicate(); }
  else if (c && k === "s") { e.preventDefault(); save(); }
  else if (c) return;
  else if (k === "delete" || k === "backspace") { e.preventDefault(); del(); }
  else if (k === " ") { e.preventDefault(); play ? stopPlay() : startPlay(); }
  else if (padOn && k === "r") { rec = !rec; padInfo(); }
  else if (k === "b" || k === "d") setTool("draw");
  else if (k === "v" || k === "s") setTool("select");
  else if (k === "e") setTool("erase");
  else if (k === "q") quantize();
  else if (k === "arrowup") { e.preventDefault(); transpose(e.shiftKey ? 12 : 1); }
  else if (k === "arrowdown") { e.preventDefault(); transpose(e.shiftKey ? -12 : -1); }
  else if (k === "arrowleft") { e.preventDefault(); nudge(-1, e); }
  else if (k === "arrowright") { e.preventDefault(); nudge(1, e); }
  else if (k === "escape") { notes.forEach((n) => (n.sel = false)); draw(); }
});

// ---------------------------------------------------------------- the bars
const SNAPS = () => [["Bar", song?.bpb || 4], ["1/4 (beat)", 1], ["1/8", 0.5], ["1/8 triplet", 1 / 3], ["1/16", 0.25],
  ["1/16 triplet", 1 / 6], ["1/32", 0.125], ["1/32 triplet", 1 / 12], ["1/64", 1 / 16], ["Off", 0]];
function fillSnaps() {
  const opts = SNAPS().map(([l, v]) => `<option value="${v}">${l}</option>`).join("");
  $("prSnap").innerHTML = opts; $("prSnap").value = String(snap);
  $("prLen").innerHTML = SNAPS().filter(([, v]) => v > 0).map(([l, v]) => `<option value="${v}">${l}</option>`).join("") + `<option value="custom" hidden>other</option>`;
  syncLen();
}
function syncLen() {
  const sel = $("prLen"), m = [...sel.options].find((o) => Math.abs(+o.value - noteLen) < 1e-6);
  if (m) sel.value = m.value;
  else { const o = sel.querySelector('option[value="custom"]'); o.textContent = `${+noteLen.toFixed(3)} beat`; o.hidden = false; sel.value = "custom"; }
}
$("prSnap").onchange = (e) => { snap = +e.target.value; draw(); };
$("prLen").onchange = (e) => { if (e.target.value !== "custom") noteLen = +e.target.value; };
$("toolDraw").onclick = () => setTool("draw");
$("toolSel").onclick = () => setTool("select");
$("toolErase").onclick = () => setTool("erase");
$("prQuant").onclick = quantize;
$("prUp1").onclick = () => transpose(1); $("prDown1").onclick = () => transpose(-1);
$("prUp12").onclick = () => transpose(12); $("prDown12").onclick = () => transpose(-12);
$("prHuman").onclick = humanize; $("prLegato").onclick = legato; $("prSame").onclick = samePitch;

// ---------------------------------------------------------------- Transform: splice parts, groove, reshape
// Works on the selected notes (or the whole part). Every change is one undo step and saves like any edit.
const XF_HELP = {
  rhythm: "The other part's timing (when notes start and how long they last) with this part's pitches in their order - a chord stays a chord. The other part is lined up bar for bar with these notes.",
  pitches: "This part's timing with the other part's pitches, in their order.",
  groove: "Measures how far the other part's notes sit off the 1/16 grid at each step of the beat (its swing and push), then moves these notes the same way. Amount = how much of it.",
  reverse: "The notes played backwards within their span.",
  invert: "High becomes low around the middle of the notes.",
  half: "Twice as long, from where the notes start.",
  double: "Twice as fast, from where the notes start.",
  shuffle: "Cuts the notes into bars and puts the bars in a random order (the seed picks the order).",
  octaves: "Moves notes up or down an octave at random (Amount = how many).",
};
const midiSongs = { list: null, cache: new Map() };
async function xfSongData(sid) {
  if (sid === SID) return song;
  if (!midiSongs.cache.has(sid)) midiSongs.cache.set(sid, await api.get(`/api/midi/${encodeURIComponent(sid)}`));
  return midiSongs.cache.get(sid);
}
async function xfFillParts() {
  const d = await xfSongData($("xfSong").value);
  $("xfPart").innerHTML = d.parts.filter((p) => !(d.id === SID && p.id === pid) || d.id !== SID)
    .map((p) => `<option value="${p.id}">${p.name.replace(/</g, "&lt;")} (${p.notes.length})</option>`).join("");
  const drum = d.parts.find((p) => p.drum && !(d.id === SID && p.id === pid));
  if ($("xfOp").value === "groove" && drum) $("xfPart").value = drum.id;
}
function xfShow() {
  const op = $("xfOp").value;
  document.querySelectorAll(".xfSrc").forEach((e) => (e.style.display = ["rhythm", "pitches", "groove"].includes(op) ? "" : "none"));
  document.querySelectorAll(".xfAmt").forEach((e) => (e.style.display = ["groove", "octaves"].includes(op) ? "" : "none"));
  document.querySelectorAll(".xfSeed").forEach((e) => (e.style.display = ["shuffle", "octaves"].includes(op) ? "" : "none"));
  $("xfHelp").textContent = XF_HELP[op];
  $("xfScope").textContent = selected().length ? `on the ${selected().length} selected notes` : "on the whole part";
}
$("prXfBtn").onclick = async () => {
  if (!midiSongs.list) {
    const all = await api.get("/api/songs").catch(() => []);
    midiSongs.list = (Array.isArray(all) ? all : all.songs || []).filter((x) => x.kind === "midi");
  }
  const others = midiSongs.list.filter((x) => x.id !== SID);
  $("xfSong").innerHTML = `<option value="${SID}">this song</option>` + others.map((x) => `<option value="${x.id}">${x.title.replace(/</g, "&lt;")}</option>`).join("");
  await xfFillParts();
  xfShow();
  $("prXf").showModal();
};
$("xfOp").onchange = () => { xfShow(); xfFillParts(); };
$("xfSong").onchange = xfFillParts;
$("xfCancel").onclick = () => $("prXf").close();
$("xfReseed").onclick = () => { $("xfSeed").value = (+$("xfSeed").value || 1) + 1; };
function rng(seed) { let x = (seed * 2654435761) >>> 0 || 1; return () => ((x = (x * 1664525 + 1013904223) >>> 0) / 4294967296); }
const onsetGroups = (ns) => {                       // notes starting together -> one group (a chord)
  const g = [];
  [...ns].sort((a, b) => a.s - b.s || a.p - b.p).forEach((n) => {
    const last = g[g.length - 1];
    if (last && Math.abs(last[0].s - n.s) < 1e-3) last.push(n); else g.push([n]);
  });
  return g;
};
const barOf = (b) => Math.floor((b - (song.first_beat || 0)) / song.bpb) * song.bpb + (song.first_beat || 0);
$("xfGo").onclick = async () => {
  const op = $("xfOp").value, amt = clamp((+$("xfAmt").value || 0) / 100, 0, 1), seed = +$("xfSeed").value || 1;
  const tg = target();
  if (!tg.length && !["rhythm"].includes(op)) { $("prXf").close(); return; }
  let src = null;
  if (["rhythm", "pitches", "groove"].includes(op)) {
    const d = await xfSongData($("xfSong").value);
    const sp = d.parts.find((p) => p.id === $("xfPart").value);
    if (!sp || !sp.notes.length) { $("xfHelp").textContent = "That part has no notes."; return; }
    src = sp.notes.map(([s0, l, p, v]) => ({ s: s0, l, p, v }));
  }
  change(() => {
    const t0 = tg.length ? Math.min(...tg.map((n) => n.s)) : cursor;
    const t1 = tg.length ? Math.max(...tg.map((n) => n.s + n.l)) : cursor;
    if (op === "reverse") tg.forEach((n) => (n.s = t0 + (t1 - (n.s + n.l))));
    else if (op === "invert") {
      const lo = Math.min(...tg.map((n) => n.p)), hi = Math.max(...tg.map((n) => n.p)), c = lo + hi;
      tg.forEach((n) => (n.p = clamp(c - n.p, 0, 127)));
    } else if (op === "half" || op === "double") {
      const k = op === "half" ? 2 : 0.5;
      tg.forEach((n) => { n.s = t0 + (n.s - t0) * k; n.l = Math.max(1 / 64, n.l * k); });
    } else if (op === "shuffle") {
      const b0 = barOf(t0), bars = Math.max(1, Math.ceil((t1 - b0) / song.bpb - 1e-6));
      const order = Array.from({ length: bars }, (_, i) => i), r = rng(seed);
      for (let i = bars - 1; i > 0; i--) { const j = Math.floor(r() * (i + 1)); [order[i], order[j]] = [order[j], order[i]]; }
      tg.forEach((n) => { const k = Math.min(bars - 1, Math.floor((n.s - b0) / song.bpb)); n.s += (order.indexOf(k) - k) * song.bpb; });
    } else if (op === "octaves") {
      const r = rng(seed);
      tg.forEach((n) => { if (r() < amt) n.p = clamp(n.p + (r() < 0.5 ? -12 : 12), 0, 127); });
    } else if (op === "groove") {
      // the source's average offset from the grid at each step of the bar (and its accents); the grid is 1/8 for
      // parts with about two notes a beat or fewer (swung 8ths are late 8ths), else 1/16 - as mashlab/midi_remix.py
      const ons = [...new Set(src.map((n) => +n.s.toFixed(4)))].sort((a, b) => a - b);
      const res = ons.length / Math.max(1, ons[ons.length - 1] - ons[0]) <= 2.2 ? 2 : 4;
      const fb = song.first_beat || 0, steps = song.bpb * res;
      const off = Array.from({ length: steps }, () => []), vel = Array.from({ length: steps }, () => []);
      src.forEach((n) => {
        const x = (n.s - fb) * res, g = Math.round(x);
        if (Math.abs(x - g) > 0.45) return;
        const st = ((g % steps) + steps) % steps; off[st].push(n.s - fb - g / res); vel[st].push(n.v);
      });
      const mean = (a) => (a.length ? a.reduce((x, y) => x + y, 0) / a.length : 0);
      const o = off.map(mean), vm = vel.map(mean), vAll = mean(src.map((n) => n.v)) || 100;
      tg.forEach((n) => {
        const x = (n.s - fb) * res, g = Math.round(x), st = ((g % steps) + steps) % steps;
        if (Math.abs(x - g) > 0.45) return;
        n.s = Math.max(0, n.s + (fb + g / res + o[st] - n.s) * amt);
        if (vel[st].length) n.v = clamp(Math.round(n.v + (n.v * vm[st] / vAll - n.v) * amt), 1, 127);
      });
    } else if (op === "pitches") {
      const want = onsetGroups(src).map((g) => g.map((n) => n.p));
      onsetGroups(tg).forEach((g, i) => {
        const ps = want[i % want.length];
        g.forEach((n, k) => (n.p = ps[k % ps.length]));
      });
    } else if (op === "rhythm") {
      const mine = onsetGroups(tg).map((g) => g.map((n) => ({ p: n.p, v: n.v })));
      if (!mine.length) return;
      let sg = onsetGroups(src);
      const sb0 = barOf(sg[0][0].s), tb0 = barOf(t0);
      if (selected().length) sg = sg.filter((g) => g[0].s - sb0 < t1 - tb0 + 1e-6);   // only as long as the selection
      const keep = new Set(tg);
      notes = notes.filter((n) => !keep.has(n));
      sg.forEach((g, i) => {
        const chord = mine[i % mine.length];
        const len = Math.max(...g.map((n) => n.l));
        chord.forEach((c) => notes.push({ s: tb0 + (g[0].s - sb0), l: len, p: c.p, v: c.v, sel: true }));
      });
    }
  });
  $("prXf").close();
};
$("prUndo").onclick = undo; $("prRedo").onclick = redo;
$("prPlay").onclick = () => (play ? stopPlay() : startPlay());
$("prLane").onchange = (e) => { lane = e.target.value; draw(); };
$("prPart").onchange = async (e) => {
  await save(); stopPlay(); padAllOff(); usePart(e.target.value);
  bank = null; padOct = part()?.drum ? 2 : 4; if (padOn) ensureBank(); padInfo();
};
$("prInst").onclick = async () => {
  await save();
  const ch = typeof BroadcastChannel !== "undefined" ? new BroadcastChannel("retex-midi") : null;
  if (!ch || !window.opener) { $("prStatus").textContent = "open Re-Tex's main window for the instrument"; return; }
  ch.postMessage({ type: "instrument", sid: SID, pid });
  try { window.opener.focus(); } catch {}
};
$("prRename").onclick = async () => {
  const p = part(); if (!p) return;
  const n = prompt("Part name", p.name);
  if (!n || n === p.name) return;
  await save();
  p.name = n; fillParts();
  await api.put(`/api/midi/${encodeURIComponent(SID)}/parts/${pid}`, { name: n }).catch((e) => ($("prStatus").textContent = e.message));
};
$("prDone").onclick = async () => { await save(); await stopPlay(); padAllOff(); window.close(); };
$("prPad").onchange = (e) => { padOn = e.target.checked; padAllOff(); if (padOn) ensureBank(); padInfo(); };
$("prRec").onclick = () => { rec = !rec; padInfo(); cv.focus(); };
try {
  const v = JSON.parse(localStorage.getItem("pianoroll") || "{}");
  if (v.snap != null) snap = v.snap;
  if (v.hear != null) $("prHear").checked = v.hear;
  if (v.pad != null) padOn = v.pad;
} catch {}
$("prPad").checked = padOn;
window.addEventListener("pagehide", () => { try { localStorage.setItem("pianoroll", JSON.stringify({ snap, hear: $("prHear").checked, pad: padOn })); } catch {} });

new ResizeObserver(resize).observe($("prWrap"));
setTool("draw");
(async () => {
  try {
    await load();
    fillSnaps(); resize(); frame(); draw();
    padOct = part()?.drum ? 2 : 4;
    if (padOn) ensureBank();
    padInfo();
  } catch (e) { $("prTitle").textContent = `Couldn't open it: ${e.message}`; }
})();
