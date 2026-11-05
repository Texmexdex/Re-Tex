// The sample editor window (/sample?eid=...): one clip's audio on a grid you pick, an edit list applied in order on
// the server (mashlab/sampler.py), heard here through Web Audio, and "Apply to the clip" bakes it into a sample that
// replaces the clip in its lane (the arrangement window swaps it in on the "sample_applied" event).
import { api } from "./api.js";

const $ = (id) => document.getElementById(id);
const eid = new URLSearchParams(location.search).get("eid");
// grid: note values (a whole note = 4 beats); the triplets are the thirds of the next value up
const GRID = [["1/2", 2], ["1/3 (half triplet)", 4 / 3], ["1/4", 1], ["1/6 (quarter triplet)", 2 / 3], ["1/8", 0.5],
  ["1/12 (eighth triplet)", 1 / 3], ["1/16", 0.25], ["1/24 (sixteenth triplet)", 1 / 6], ["1/32", 0.125],
  ["1/48 (32nd triplet)", 1 / 12], ["1/64", 0.0625]];
const ECHO = [["1/4", 1], ["1/4 dotted", 1.5], ["1/4 triplet", 2 / 3], ["1/8", 0.5], ["1/8 dotted", 0.75], ["1/8 triplet", 1 / 3],
  ["1/16", 0.25], ["1/16 dotted", 0.375], ["1/2", 2]];
const ls = (k, d) => { try { return JSON.parse(localStorage.getItem("se_" + k)) ?? d; } catch { return d; } };
const lsSet = (k, v) => { try { localStorage.setItem("se_" + k, JSON.stringify(v)); } catch {} };

let meta = null;                 // {eid, title, stem, bpm, bpb, beats}
let ops = [], redo = [];         // the edit list (applied in order), and what Undo took off
let clip = null;                 // {a, b} copied range (beats)
let buf = null;                  // AudioBuffer of the current render
let mono = null;                 // its samples, L+R averaged, for drawing
let lenBeats = 0;
let cell = ls("cell", 0.25), snap = true;
let sel = null, cursor = 0;      // selection {a, b} and cursor (beats)
let view = { t0: 0, span: 4 };   // visible window (beats)
let rendering = 0;
const ac = new (window.AudioContext || window.webkitAudioContext)();
let play = null;                 // {src, t0 (ctx time), from (beats), loop {a, b} | null}

const spb = () => 60 / meta.bpm;
const bpb = () => meta.bpb || 4;
const fmt = (b) => {
  const bar = Math.floor(b / bpb()) + 1, beat = Math.floor(b % bpb()) + 1, sub = Math.floor(((b % 1) + 1e-9) * 4) + 1;
  return `${bar}.${beat}.${sub}`;
};

// ---------------------------------------------------------------- toasts
function toast(msg, bad = false) {
  const el = document.createElement("div");
  el.className = "se-toast" + (bad ? " bad" : "");
  el.textContent = msg;
  $("toastHost").appendChild(el);
  setTimeout(() => el.remove(), bad ? 7000 : 3500);
}

// ---------------------------------------------------------------- rendering the edit list
async function render() {
  const my = ++rendering;
  $("seBusy").textContent = "rendering…";
  try {
    const r = await api.post(`/api/sample/${eid}/render`, { ops });
    const raw = await (await fetch(`/api/sample/${eid}/preview.wav?v=${r.version}`)).arrayBuffer();
    const b = await ac.decodeAudioData(raw);
    if (my !== rendering) return;                              // a newer edit is on its way
    const wasPlaying = !!play, at = playPos();
    stop();
    buf = b;
    const L = b.getChannelData(0), R = b.numberOfChannels > 1 ? b.getChannelData(1) : L;
    mono = new Float32Array(L.length);
    for (let i = 0; i < L.length; i++) mono[i] = 0.5 * (L[i] + R[i]);
    lenBeats = b.duration / spb();
    $("seBusy").textContent = "";
    peaksCache = null;
    draw();
    if (wasPlaying) start(at);
  } catch (e) {
    if (my === rendering) { $("seBusy").textContent = ""; toast(e.message, true); }
  }
  updateButtons();
}
function push(op) {
  ops.push(op); redo = [];
  listOps(); render();
}

// ---------------------------------------------------------------- playback (Web Audio)
function playPos() {
  if (!play) return cursor;
  let b = play.from + (ac.currentTime - play.t0) / spb();
  if (play.loop) { const L = play.loop.b - play.loop.a; if (b >= play.loop.b) b = play.loop.a + ((b - play.loop.a) % L); }
  return b;
}
function start(from) {
  if (!buf) return;
  stop();
  const src = ac.createBufferSource();
  src.buffer = buf;
  src.connect(ac.destination);
  const loop = $("seLoop").classList.contains("on") && sel && sel.b > sel.a ? { ...sel } : null;
  if (loop) {
    src.loop = true; src.loopStart = loop.a * spb(); src.loopEnd = loop.b * spb();
    if (from < loop.a || from >= loop.b) from = loop.a;
  }
  src.start(0, Math.max(0, from * spb()));
  src.onended = () => { if (play && play.src === src) { play = null; updatePlay(); } };
  play = { src, t0: ac.currentTime, from, loop };
  updatePlay();
  requestAnimationFrame(tick);
}
function stop() {
  if (!play) return;
  try { play.src.onended = null; play.src.stop(); } catch {}
  play = null; updatePlay(); draw();
}
function toggle() {
  if (ac.state === "suspended") ac.resume();
  if (play) stop(); else start(sel && sel.b > sel.a ? sel.a : cursor);
}
function tick() { if (!play) return; draw(); requestAnimationFrame(tick); }
function updatePlay() { $("sePlay").classList.toggle("on", !!play); }

// ---------------------------------------------------------------- drawing
const cv = $("seCanvas"), g = cv.getContext("2d");
const RULER = 22;
let peaksCache = null;
function size() {
  const r = cv.getBoundingClientRect(), dpr = devicePixelRatio || 1;
  cv.width = Math.max(1, r.width * dpr); cv.height = Math.max(1, r.height * dpr);
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  peaksCache = null;
  draw();
}
new ResizeObserver(size).observe(cv);
const W = () => cv.getBoundingClientRect().width, H = () => cv.getBoundingClientRect().height;
const bx = (b) => ((b - view.t0) / view.span) * W();
const xb = (x) => view.t0 + (x / W()) * view.span;
const snapB = (b, e) => (snap && !(e && e.altKey) ? Math.round(b / cell) * cell : b);
function clampView() {
  const total = Math.max(lenBeats, meta?.beats || 4);
  view.span = Math.min(Math.max(view.span, cell * 2, 0.0625), total * 1.05 + 1);
  view.t0 = Math.max(0, Math.min(view.t0, total - view.span * 0.5));
}
function columnPeaks() {                       // min/max of the samples under each pixel column
  const w = Math.floor(W());
  const key = `${w}|${view.t0}|${view.span}`;
  if (peaksCache?.key === key) return peaksCache;
  const lo = new Float32Array(w), hi = new Float32Array(w), sr = buf.sampleRate;
  for (let x = 0; x < w; x++) {
    const a = Math.max(0, Math.floor(xb(x) * spb() * sr)), b = Math.min(mono.length, Math.floor(xb(x + 1) * spb() * sr));
    let mn = 0, mx = 0;
    for (let i = a; i < b; i++) { const v = mono[i]; if (v < mn) mn = v; if (v > mx) mx = v; }
    if (b <= a && a < mono.length) mn = mx = mono[a];
    lo[x] = mn; hi[x] = mx;
  }
  return (peaksCache = { key, lo, hi });
}
function draw() {
  if (!meta) return;
  const w = W(), h = H();
  g.clearRect(0, 0, w, h);
  const top = RULER, mid = top + (h - top) / 2, amp = (h - top) / 2 - 6;
  // cells: every other beat a little lighter; grid lines at the cell size (when there's room)
  for (let b = Math.floor(view.t0); b < view.t0 + view.span; b++) {
    if (b % 2) { g.fillStyle = "rgba(255,255,255,0.025)"; g.fillRect(bx(b), top, bx(b + 1) - bx(b), h - top); }
  }
  const px = (cell / view.span) * w;
  const first = Math.floor(view.t0 / cell) * cell;
  for (let b = first; b < view.t0 + view.span + cell; b += cell) {
    const x = Math.round(bx(b)) + 0.5, onBar = Math.abs(b / bpb() - Math.round(b / bpb())) < 1e-6, onBeat = Math.abs(b - Math.round(b)) < 1e-6;
    if (!onBeat && px < 5) continue;
    g.fillStyle = onBar ? "rgba(255,255,255,0.32)" : onBeat ? "rgba(255,255,255,0.16)" : "rgba(255,255,255,0.07)";
    g.fillRect(x, top, 1, h - top);
  }
  // the clip's own length: anything past it is the tail (an echo ringing on)
  const endX = bx(meta.beats);
  if (endX < w) { g.fillStyle = "rgba(0,0,0,0.28)"; g.fillRect(endX, top, w - endX, h - top); }
  // selection
  if (sel && sel.b > sel.a) {
    g.fillStyle = "rgba(30,215,96,0.16)"; g.fillRect(bx(sel.a), top, bx(sel.b) - bx(sel.a), h - top);
    g.fillStyle = "rgba(30,215,96,0.85)"; g.fillRect(bx(sel.a), top, 1.5, h - top); g.fillRect(bx(sel.b) - 1.5, top, 1.5, h - top);
  }
  // waveform
  if (buf) {
    const { lo, hi } = columnPeaks();
    g.fillStyle = "#e6e6e8";
    for (let x = 0; x < lo.length; x++) {
      const y0 = mid - hi[x] * amp, y1 = mid - lo[x] * amp;
      g.fillRect(x, y0, 1, Math.max(1, y1 - y0));
    }
    g.fillStyle = "rgba(255,255,255,0.12)"; g.fillRect(0, mid, w, 1);
  }
  // ruler
  g.fillStyle = "rgba(255,255,255,0.04)"; g.fillRect(0, 0, w, RULER);
  g.font = "11px Cascadia Mono, Consolas, monospace";
  const step = view.span / bpb() > 24 ? 4 : view.span / bpb() > 8 ? 2 : 1;
  for (let bar = Math.floor(view.t0 / bpb()); bar * bpb() < view.t0 + view.span; bar++) {
    if (bar % step) continue;
    const x = bx(bar * bpb());
    g.fillStyle = "rgba(255,255,255,0.75)"; g.fillText(String(bar + 1), x + 4, 14);
    g.fillStyle = "rgba(255,255,255,0.35)"; g.fillRect(Math.round(x), 0, 1, RULER);
    if (view.span / bpb() <= 4) for (let k = 1; k < bpb(); k++) {
      const xk = bx(bar * bpb() + k);
      g.fillStyle = "rgba(255,255,255,0.45)"; g.fillText(`${bar + 1}.${k + 1}`, xk + 3, 14);
      g.fillStyle = "rgba(255,255,255,0.18)"; g.fillRect(Math.round(xk), 8, 1, RULER - 8);
    }
  }
  // cursor + playhead
  g.fillStyle = "rgba(255,255,255,0.6)"; g.fillRect(Math.round(bx(cursor)), top, 1, h - top);
  if (play) { const p = playPos(); g.fillStyle = "#1ed760"; g.fillRect(Math.round(bx(p)), 0, 2, h); $("sePos").textContent = fmt(p); }
  else $("sePos").textContent = fmt(cursor);
}

// ---------------------------------------------------------------- mouse
let drag = null;
cv.addEventListener("mousedown", (e) => {
  if (e.button !== 0 || !meta) return;
  const b = Math.max(0, xb(e.offsetX));
  drag = { from: snapB(b, e) };
  sel = null; cursor = drag.from;
  if (play) start(cursor);
  draw(); showSel();
});
addEventListener("mousemove", (e) => {
  if (!drag) return;
  const r = cv.getBoundingClientRect();
  const b = Math.max(0, xb(e.clientX - r.left));
  let a = drag.from, z = snapB(b, e);
  if (snap && !e.altKey && Math.abs(z - a) < 1e-9) z = a + Math.sign(b - a || 1) * cell;   // at least one cell
  sel = { a: Math.min(a, z), b: Math.max(a, z) };
  draw(); showSel();
});
addEventListener("mouseup", () => { drag = null; });
cv.addEventListener("dblclick", (e) => {                   // a double-click selects the cell under the mouse
  const b = Math.floor(xb(e.offsetX) / cell) * cell;
  sel = { a: Math.max(0, b), b: b + cell }; cursor = sel.a;
  draw(); showSel();
});
cv.addEventListener("wheel", (e) => {
  e.preventDefault();
  if (e.ctrlKey) {
    const at = xb(e.offsetX), f = e.deltaY < 0 ? 0.8 : 1.25;
    view.span *= f; view.t0 = at - (e.offsetX / W()) * view.span;
  } else view.t0 += (e.deltaY + e.deltaX) / W() * view.span * 0.6;
  clampView(); peaksCache = null; draw();
}, { passive: false });
function showSel() {
  const el = $("seSel");
  if (!sel || sel.b <= sel.a) { el.textContent = `Cursor at ${fmt(cursor)} - drag across the waveform to select cells.`; return; }
  const n = (sel.b - sel.a) / cell;
  el.textContent = `Selected ${fmt(sel.a)} – ${fmt(sel.b)} · ${Math.round(n * 100) / 100} cell${Math.abs(n - 1) < 1e-6 ? "" : "s"} of ${GRID.find((x) => x[1] === cell)?.[0] || cell}`;
}

// ---------------------------------------------------------------- the edits
const NAMES = { silence: "Silence", reverse: "Reverse", repeat: "Stutter", paste: "Paste", layer: "Layer", tapestop: "Tape stop",
  tapestart: "Tape start", pitch: "Pitch", gain: "Level", echo: "Echo" };
function needSel() {
  if (sel && sel.b > sel.a) return true;
  toast("Select some cells first (drag across the waveform)", true);
  return false;
}
function cellName(c) { return (GRID.find((x) => Math.abs(x[1] - c) < 1e-9)?.[0] || `${c} beat`).split(" ")[0]; }
function describe(o) {
  const at = `${fmt(o.a)}–${fmt(o.b ?? o.a)}`;
  switch (o.op) {
    case "repeat": return `Stutter ${at} · ${cellName(o.cell)}${o.roll ? " roll" : ""}${o.rise ? ` +${o.rise} st` : ""}`;
    case "paste": return `${o.mode === "mix" ? "Layered paste" : "Paste"} of ${fmt(o.from_a)}–${fmt(o.from_b)} at ${fmt(o.a)}`;
    case "layer": return `Layer ${at} · ${o.semis ? (o.semis > 0 ? "+" : "") + o.semis + " st" : "same pitch"}${o.offset ? `, ${o.offset > 0 ? "+" : ""}${Math.round(o.offset / cell * 100) / 100} cells` : ""}`;
    case "echo": return `Echo ${at} · ${ECHO.find((x) => Math.abs(x[1] - o.time) < 1e-9)?.[0] || o.time}${o.tape ? " tape" : ""}${o.out ? " out" : ""}`;
    case "pitch": return `Pitch ${at} · ${o.semis > 0 ? "+" : ""}${o.semis} st${o.mode === "vari" ? " varispeed" : ""}`;
    case "gain": return `Level ${at} · ${o.db > 0 ? "+" : ""}${o.db} dB${o.shape !== "flat" ? " " + o.shape : ""}`;
    default: return `${NAMES[o.op] || o.op} ${at}`;
  }
}
function listOps() {
  const el = $("seOps");
  el.innerHTML = ops.length ? "" : '<span class="dim">none yet - the clip as it plays in the arrangement</span>';
  ops.forEach((o, i) => {
    const chip = document.createElement("span");
    chip.className = "se-op";
    chip.innerHTML = `<span>${i + 1}. </span><span class="t"></span><button class="x" title="Take this edit out (the others stay)">×</button>`;
    chip.querySelector(".t").textContent = describe(o);
    chip.querySelector(".x").onclick = () => { redo = []; ops.splice(i, 1); listOps(); render(); };
    chip.onmouseenter = () => { sel = { a: o.a, b: o.b ?? o.a + 1 }; draw(); };
    el.appendChild(chip);
  });
  updateButtons();
}
function updateButtons() {
  $("seUndo").disabled = !ops.length; $("seRedo").disabled = !redo.length;
  $("seRevert").disabled = !ops.length;
}
const val = (id) => $(id).value;
const DO = {
  copy: () => { if (needSel()) { clip = { ...sel }; toast(`Copied ${fmt(sel.a)}–${fmt(sel.b)}`); } },
  paste: () => { if (!clip) return toast("Copy something first", true); push({ op: "paste", from_a: clip.a, from_b: clip.b, a: sel ? sel.a : cursor }); },
  pastemix: () => { if (!clip) return toast("Copy something first", true); push({ op: "paste", from_a: clip.a, from_b: clip.b, a: sel ? sel.a : cursor, mode: "mix" }); },
  dup: () => { if (needSel()) { const L = sel.b - sel.a; push({ op: "paste", from_a: sel.a, from_b: sel.b, a: sel.b }); sel = { a: sel.b, b: sel.b + L }; } },
  reverse: () => needSel() && push({ op: "reverse", a: sel.a, b: sel.b }),
  silence: () => needSel() && push({ op: "silence", a: sel.a, b: sel.b }),
  echo: () => needSel() && push({ op: "echo", a: sel.a, b: sel.b, time: +val("seEcTime"), feedback: +val("seEcFb"), mix: +val("seEcMix"),
                                 tone: +val("seEcTone"), tape: $("seEcTape").checked, out: $("seEcOut").checked }),
  tapestop: () => needSel() && push({ op: "tapestop", a: sel.a, b: sel.b }),
  tapestart: () => needSel() && push({ op: "tapestart", a: sel.a, b: sel.b }),
  pitch: () => needSel() && push({ op: "pitch", a: sel.a, b: sel.b, semis: +val("sePiSemis"), mode: val("sePiMode") }),
  gain: () => needSel() && push({ op: "gain", a: sel.a, b: sel.b, db: +val("seGaDb"), shape: val("seGaShape") }),
};
document.querySelectorAll("[data-do]").forEach((b) => { b.onclick = () => { DO[b.dataset.do](); b.blur(); }; });

// ---------------------------------------------------------------- controls
function fillSelects() {
  $("seGrid").innerHTML = GRID.map(([n, v]) => `<option value="${v}">${n}</option>`).join("");
  $("seGrid").value = String(cell);
  if (!GRID.some((x) => x[1] === cell)) { cell = 0.25; $("seGrid").value = "0.25"; }
  const cells = () => [[`the grid cell`, "grid"], ...GRID.filter((x) => x[1] <= 1).map(([n, v]) => [n.split(" ")[0], v])];
  $("seStCell").innerHTML = cells().map(([n, v]) => `<option value="${v}">${n}</option>`).join("");
  $("seEcTime").innerHTML = ECHO.map(([n, v]) => `<option value="${v}">${n}</option>`).join("");
  $("seEcTime").value = "0.75";
  $("seLaOff").innerHTML = [["none", 0], ["+1 cell", 1], ["+2 cells", 2], ["−1 cell", -1]].map(([n, v]) => `<option value="${v}">${n}</option>`).join("");
  $("sePiSemis").innerHTML = [-12, -7, -5, -3, -2, -1, 1, 2, 3, 5, 7, 12].map((v) => `<option value="${v}"${v === -5 ? " selected" : ""}>${v > 0 ? "+" : ""}${v} st</option>`).join("");
  document.querySelectorAll(".se-row input[type=range]").forEach((r) => {
    const out = r.nextElementSibling, show = () => { out.textContent = Math.round(+r.value * 100) + "%"; };
    r.oninput = show; show();
  });
}
// the stutter repeats the grid cell unless you pick a size; the layer offset counts in cells
const realStCell = () => (val("seStCell") === "grid" ? cell : +val("seStCell"));
DO.stutter = () => needSel() && push({ op: "repeat", a: sel.a, b: sel.b, cell: realStCell(), roll: $("seStRoll").checked,
                                       rise: +val("seStRise"), level: val("seStLevel") });
DO.layer = () => needSel() && push({ op: "layer", a: sel.a, b: sel.b, semis: +val("seLaSemis"), offset: +val("seLaOff") * cell, gain_db: +val("seLaGain") });

$("seGrid").onchange = (e) => { cell = +e.target.value; lsSet("cell", cell); clampView(); draw(); showSel(); e.target.blur(); };
$("seSnap").onchange = (e) => { snap = e.target.checked; e.target.blur(); };
$("sePlay").onclick = (e) => { toggle(); e.currentTarget.blur(); };
$("seLoop").onclick = (e) => { e.currentTarget.classList.toggle("on"); e.currentTarget.blur(); if (play) start(playPos()); };
$("seZoomIn").onclick = () => { view.span *= 0.7; clampView(); peaksCache = null; draw(); };
$("seZoomOut").onclick = () => { view.span /= 0.7; clampView(); peaksCache = null; draw(); };
$("seFit").onclick = () => { view = { t0: 0, span: Math.max(lenBeats, meta.beats) * 1.02 }; clampView(); peaksCache = null; draw(); };
$("seUndo").onclick = () => { if (ops.length) { redo.push(ops.pop()); listOps(); render(); } };
$("seRedo").onclick = () => { if (redo.length) { ops.push(redo.pop()); listOps(); render(); } };
$("seRevert").onclick = () => { if (ops.length && confirm("Throw away all the edits?")) { redo = []; ops = []; listOps(); render(); } };
$("seDone").onclick = () => {
  stop();
  window.close();                                   // only works for a window Re-Tex opened - else say how
  setTimeout(() => toast("Close this window with its X (or Ctrl+W)"), 300);
};
$("seApply").onclick = async () => {
  const btn = $("seApply");
  btn.disabled = true; $("seBusy").textContent = "baking…";
  try {
    const r = await api.post(`/api/sample/${eid}/apply`, { ops });
    toast(`Applied: the clip is now this sample (${Math.round(r.beats * 100) / 100} beats). Ctrl+Z in the arrangement undoes it.`);
  } catch (e) { toast(e.message, true); }
  btn.disabled = false; $("seBusy").textContent = "";
};

addEventListener("keydown", (e) => {
  if (e.target.closest("input, select, textarea")) return;
  const k = e.key.toLowerCase(), ctrl = e.ctrlKey || e.metaKey;
  if (k === " ") { e.preventDefault(); toggle(); }
  else if (ctrl && k === "z") { e.preventDefault(); $("seUndo").click(); }
  else if (ctrl && (k === "y" || (e.shiftKey && k === "z"))) { e.preventDefault(); $("seRedo").click(); }
  else if (ctrl && k === "c") { e.preventDefault(); DO.copy(); }
  else if (ctrl && k === "v") { e.preventDefault(); e.shiftKey ? DO.pastemix() : DO.paste(); }
  else if (ctrl && k === "d") { e.preventDefault(); DO.dup(); }
  else if (k === "delete" || k === "backspace") DO.silence();
  else if (k === "r" && !ctrl) DO.reverse();
  else if (k === "s" && !ctrl) DO.stutter();
  else if (k === "l" && !ctrl) $("seLoop").click();
  else if (k === "escape") { sel = null; draw(); showSel(); }
});

// ---------------------------------------------------------------- start
(async () => {
  fillSelects();
  if (!eid) { $("seTitle").textContent = "Preparing the clip…"; $("seMeta").textContent = " rendering it the way it plays in the arrangement"; return; }
  try { meta = await api.get(`/api/sample/${eid}`); }
  catch (e) { $("seTitle").textContent = "This editing session is gone"; $("seMeta").textContent = " - open the clip again from the arrangement"; return; }
  document.title = `${meta.title} (${meta.stem}) · Sample editor · Re-Tex`;
  $("seTitle").textContent = `${meta.title}`;
  $("seMeta").textContent = ` · ${meta.stem} · ${Math.round(meta.beats * 100) / 100} beats at ${Math.round(meta.bpm * 10) / 10} BPM`;
  view = { t0: 0, span: meta.beats * 1.02 };
  listOps(); showSel();
  await render();
  size();
})();
