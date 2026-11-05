// The beat grid window (/grid?sid=...): the song as recorded and a metronome on its marks, both scheduled on this
// page's one audio clock - so however late the sound reaches you (headsets, Bluetooth), the clicks and the song
// arrive together and you can judge them by ear. The picture is held back by your "audio delay" so the playhead
// moves with what you hear. Save = the song's beat marks.
//
// Editing works on bars, by the rules in gridedit.js (the user's, Oct 19): drag / Ctrl+drag / Shift+drag a bar line or a
// beat, click + Del deletes, double-click a mark = bar's 1 / plain beat, double-click elsewhere = add a beat.
import { api } from "./api.js";
import { detectedMarks } from "./timing.js";
import * as E from "./gridedit.js";

const $ = (id) => document.getElementById(id);
const sid = new URLSearchParams(location.search).get("sid");
const pid = new URLSearchParams(location.search).get("pid");      // opened from this project (it can save just for it)
const ls = (k, d) => { try { return JSON.parse(localStorage.getItem("grid_" + k)) ?? d; } catch { return d; } };
const lsSet = (k, v) => { try { localStorage.setItem("grid_" + k, JSON.stringify(v)); } catch {} };
const bus = "BroadcastChannel" in window ? new BroadcastChannel("retex") : null;

let info = null, buf = null, mono = null;
let M = null;                       // { t: [s], d: [0|1] bar's 1, s: [0|1] sure, c: [0|1] placed by you, bpb, p0 }
let sel = -1;                       // the selected mark (Del deletes it)
let undo = [], redo = [], dirty = false, savedKey = "";    // dirty = differs from what's saved
let view = { t0: 0, span: 12 };     // seconds on screen
const ac = new (window.AudioContext || window.webkitAudioContext)();
const songGain = ac.createGain(), clickGain = ac.createGain();
songGain.connect(ac.destination); clickGain.connect(ac.destination);
let play = null;                    // { src, ctx0, from (s), rate, nextClick (mark index) }
let cursor = 0;

function toast(msg, bad = false) {
  const el = document.createElement("div");
  el.className = "se-toast" + (bad ? " bad" : "");
  el.textContent = msg;
  $("toastHost").appendChild(el);
  setTimeout(() => el.remove(), bad ? 7000 : 3000);
}
const bpb = () => M.bpb || 4;
const fmtT = (t) => `${Math.floor(t / 60)}:${(t % 60).toFixed(2).padStart(5, "0")}`;

// ---------------------------------------------------------------- bars and phrases
// bars count from the song's first downbeat; 4-bar phrases count from M.p0 (a downbeat you set), else from the
// first sure downbeat. kinds() -> per mark: 0 beat, 1 bar's 1, 2 a phrase's 1 (and the bar numbers)
let kindsCache = null;
function kinds() {
  const key = M.t.length + "|" + M.d.join("") + "|" + M.p0 + "|" + M.s.join("");
  if (kindsCache?.key === key) return kindsCache;
  const D = [];
  M.d.forEach((x, i) => { if (x) D.push(i); });
  let ref = M.p0 != null && M.d[M.p0] ? D.indexOf(M.p0) : D.findIndex((i) => M.s[i]);
  if (ref < 0) ref = 0;
  const kind = new Int8Array(M.t.length), bar = new Map();
  D.forEach((i, n) => { kind[i] = ((((n - ref) % 4) + 4) % 4) === 0 ? 2 : 1; bar.set(i, n + 1); });
  return (kindsCache = { key, kind, bar, D, ref });
}

// ---------------------------------------------------------------- edits (each one undoable)
function snap() { undo.push(JSON.stringify({ M })); if (undo.length > 200) undo.shift(); redo = []; dirty = true; updateButtons(); }
function restore(s) { M = JSON.parse(s).M; sel = -1; dirty = true; updateButtons(); draw(); }
function deleteMark(i) {
  snap();
  if (!E.deleteMark(M, i)) { undo.pop(); return; }
  sel = -1;
  draw(); updateButtons();
}
function addMark(T, exact) {
  snap();
  const i = E.addMark(M, T, exact);
  if (i < 0) { undo.pop(); return; }
  sel = i;
  draw(); updateButtons();
}
function toggleDown(i) { snap(); E.toggleDown(M, i); draw(); updateButtons(); }
const unevenBars = () => E.unevenBars(M);
function evenOut() {
  const bars = unevenBars();
  if (!bars.length) { toast("Every bar's beats are already even"); return; }
  snap();
  for (const [a, b] of bars) E.fitBetween(M, a, b);
  draw(); updateButtons();
  toast(`${bars.length} bar${bars.length > 1 ? "s" : ""} evened out - listen, then Save (Ctrl+Z undoes)`);
}
function nextUneven() {
  const bars = unevenBars();
  if (!bars.length) { toast("No uneven bars"); return; }
  const from = heardPos() + 0.01;
  const [a] = bars.find(([x]) => M.t[x] > from) || bars[0];
  cursor = Math.max(0, M.t[a] - 0.5);
  view.t0 = Math.max(0, M.t[a] - view.span * 0.25);
  peaks = null;
  if (play) start(cursor); else draw();
}
function nudge(from, dt) {
  snap();
  for (let i = 0; i < M.t.length; i++) if (M.t[i] >= from - 1e-9) M.t[i] = Math.max(0, M.t[i] + dt);
  draw(); updateButtons();
}
function moveOne(from, k) {                         // the bar's 1 moves k beats (from the playhead on)
  snap();
  const first = M.t.findIndex((t) => t >= from - 1e-9);
  if (first < 0) return;
  let lastDown = -1;
  for (let i = first; i < M.t.length; i++) if (M.d[i]) { lastDown = i; break; }
  if (lastDown < 0) return;
  const start = lastDown + k;
  for (let i = first; i < M.t.length; i++) M.d[i] = (((i - start) % bpb()) + bpb()) % bpb() === 0 ? 1 : 0;
  draw(); updateButtons();
}

// ---------------------------------------------------------------- audio
function playPos() { return play ? play.from + (ac.currentTime - play.ctx0) * play.rate : cursor; }
function heardPos() { return play ? Math.max(0, playPos() - (+$("gDelay").value / 1000) * play.rate) : cursor; }
// a hard, bright click (a noise snap + a short high tone) - a plain sine gets buried under distorted guitars
let snapBuf = null;
function clickAt(when, kind) {
  if (!snapBuf) {
    snapBuf = ac.createBuffer(1, Math.round(ac.sampleRate * 0.012), ac.sampleRate);
    const ch = snapBuf.getChannelData(0);
    for (let i = 0; i < ch.length; i++) ch[i] = (Math.random() * 2 - 1) * Math.exp(-i / (ac.sampleRate * 0.0015));
  }
  const [f, lvl] = kind === 2 ? [3600, 1] : kind === 1 ? [2800, 0.85] : [2000, 0.6];
  const o = ac.createOscillator(), g = ac.createGain(), n = ac.createBufferSource(), gn = ac.createGain();
  o.frequency.value = f;
  g.gain.setValueAtTime(lvl, when); g.gain.exponentialRampToValueAtTime(0.001, when + 0.05);
  gn.gain.value = lvl * 0.6;
  n.buffer = snapBuf;
  o.connect(g).connect(clickGain); n.connect(gn).connect(clickGain);
  o.start(when); o.stop(when + 0.06); n.start(when);
}
const markKind = (i) => kinds().kind[i];
function movePhrase(k) {                            // the 4-bar phrases start k bars later
  snap();
  const { D, ref } = kinds();
  M.p0 = D[Math.max(0, Math.min(D.length - 1, ref + k))];
  draw(); updateButtons();
}
function schedule() {
  if (!play) return;
  const ahead = ac.currentTime + 0.2;
  while (play.next < M.t.length) {
    const when = play.ctx0 + (M.t[play.next] - play.from) / play.rate;
    if (when > ahead) break;
    if (when >= ac.currentTime - 0.005 && $("gClick").checked) clickAt(Math.max(when, ac.currentTime), markKind(play.next));
    play.next++;
  }
}
function start(from) {
  if (!buf) return;
  stop();
  if (ac.state === "suspended") ac.resume();
  const rate = +$("gSpeed").value;
  const src = ac.createBufferSource();
  src.buffer = buf; src.playbackRate.value = rate;
  src.connect(songGain);
  const ctx0 = ac.currentTime + 0.05;
  src.start(ctx0, Math.max(0, from));
  play = { src, ctx0, from, rate, next: M.t.findIndex((t) => t >= from - 0.001) };
  if (play.next < 0) play.next = M.t.length;
  play.timer = setInterval(schedule, 25);
  schedule();
  $("gPlay").classList.add("on");
  requestAnimationFrame(tick);
}
function stop() {
  if (!play) return;
  cursor = heardPos();
  clearInterval(play.timer);
  try { play.src.stop(); } catch {}
  play = null;
  $("gPlay").classList.remove("on");
  draw();
}
function tick() {
  if (!play) return;
  const p = heardPos();
  if ($("gFollow").checked && (p < view.t0 || p > view.t0 + view.span * 0.8)) view.t0 = Math.max(0, p - view.span * 0.2);
  draw();
  if (p >= buf.duration) { stop(); return; }
  requestAnimationFrame(tick);
}

// ---------------------------------------------------------------- drawing
const cv = $("gCanvas"), g = cv.getContext("2d");
const RULER = 24;
let peaks = null;
function size() {
  const r = cv.getBoundingClientRect(), dpr = devicePixelRatio || 1;
  cv.width = Math.max(1, r.width * dpr); cv.height = Math.max(1, r.height * dpr);
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  peaks = null; draw();
}
new ResizeObserver(size).observe(cv);
const W = () => cv.getBoundingClientRect().width, H = () => cv.getBoundingClientRect().height;
const tx = (t) => ((t - view.t0) / view.span) * W();
const xt = (x) => view.t0 + (x / W()) * view.span;
function colPeaks() {
  const w = Math.floor(W()), key = `${w}|${view.t0}|${view.span}`;
  if (peaks?.key === key) return peaks;
  const lo = new Float32Array(w), hi = new Float32Array(w), sr = buf.sampleRate;
  for (let x = 0; x < w; x++) {
    const a = Math.max(0, Math.floor(xt(x) * sr)), b = Math.min(mono.length, Math.floor(xt(x + 1) * sr));
    let mn = 0, mx = 0;
    for (let i = a; i < b; i++) { const v = mono[i]; if (v < mn) mn = v; if (v > mx) mx = v; }
    lo[x] = mn; hi[x] = mx;
  }
  return (peaks = { key, lo, hi });
}
function draw() {
  if (!M) return;
  const w = W(), h = H(), mid = RULER + (h - RULER) / 2, amp = (h - RULER) / 2 - 6;
  g.clearRect(0, 0, w, h);
  if (buf) {
    const { lo, hi } = colPeaks();
    g.fillStyle = "rgba(230,230,232,0.55)";
    for (let x = 0; x < lo.length; x++) g.fillRect(x, mid - hi[x] * amp, 1, Math.max(1, (hi[x] - lo[x]) * amp));
  }
  g.fillStyle = "rgba(255,255,255,0.04)"; g.fillRect(0, 0, w, RULER);
  const bad = unevenBars();                          // bars whose beats aren't even: a red band in the ruler
  g.fillStyle = "rgba(240,90,80,0.28)";
  for (const [a, b] of bad) { const x0 = tx(M.t[a]), x1 = tx(M.t[b]); if (x1 > 0 && x0 < w) g.fillRect(x0, RULER - 5, x1 - x0, 5); }
  $("gUneven").textContent = bad.length ? `${bad.length} bar${bad.length > 1 ? "s" : ""} with uneven beats (red in the ruler)` : "every bar's beats are even";
  const { kind: K, bar: barOf } = kinds();
  g.font = "600 11px Segoe UI, system-ui, sans-serif";
  for (let i = 0; i < M.t.length; i++) {
    const x = tx(M.t[i]);
    if (x < -20 || x > w + 20) continue;
    const kind = K[i], unsure = !M.s[i];
    const col = i === sel ? "255,255,255" : unsure ? "224,178,82" : "30,215,96";
    g.fillStyle = kind === 2 ? `rgba(${col},0.95)` : kind === 1 ? `rgba(${col},0.7)` : `rgba(${col},${i === sel ? 0.9 : 0.32})`;
    const wd = (kind === 2 ? 3 : kind ? 2 : 1) + (i === sel ? 2 : 0);
    g.fillRect(Math.round(x) - (wd > 1 ? 1 : 0), kind ? 0 : RULER, wd, kind ? h : h - RULER);
    if (kind) {
      g.fillStyle = "rgba(255,255,255,0.85)";
      g.fillText(String(barOf.get(i)), x + 4, 15);
    } else if (M.c[i]) {                               // a beat you placed yourself: a small diamond
      g.fillStyle = "#f0c060"; g.beginPath(); g.moveTo(x, RULER + 2); g.lineTo(x + 4, RULER + 6); g.lineTo(x, RULER + 10); g.lineTo(x - 4, RULER + 6); g.fill();
    }
  }
  const p = heardPos();
  g.fillStyle = "#ffffff"; g.fillRect(Math.round(tx(p)), 0, 2, h);
  $("gPos").textContent = fmtT(p);
}

// ---------------------------------------------------------------- mouse
let drag = null;
function markAt(x) {                                 // a bar line within 7 px wins over a beat within 5 px
  let best = -1, bd = 7;
  for (let i = 0; i < M.t.length; i++) if (M.d[i]) { const d = Math.abs(tx(M.t[i]) - x); if (d < bd) { bd = d; best = i; } }
  if (best >= 0) return best;
  bd = 5;
  for (let i = 0; i < M.t.length; i++) if (!M.d[i]) { const d = Math.abs(tx(M.t[i]) - x); if (d < bd) { bd = d; best = i; } }
  return best;
}
const howOf = (e) => (e.shiftKey ? "all" : e.ctrlKey || e.metaKey ? "one" : "plain");
cv.addEventListener("mousemove", (e) => {
  if (drag) return;
  const k = markAt(e.offsetX);
  cv.style.cursor = k >= 0 ? "ew-resize" : "text";
  cv.title = k < 0 ? "Click: move the playhead · double-click: add a beat (Ctrl: exactly here)"
    : M.d[k] ? "Bar line - drag: the bar before squeezes, the rest moves along · Ctrl+drag: only the bars either side · Shift+drag: everything · Del: delete · double-click: make it a beat"
    : "Beat - drag: its bar's tempo (kept even), the rest moves along · Ctrl+drag: only this beat · Shift+drag: everything · Del: delete · double-click: make it a bar's 1";
});
cv.addEventListener("mousedown", (e) => {
  if (e.button !== 0 || !M) return;
  const k = markAt(e.offsetX);
  if (k >= 0) {                                      // select it; a drag starts once the mouse moves
    sel = k;
    drag = { k, x0: e.offsetX, how: howOf(e), base: { t: M.t.slice(), c: M.c.slice() }, moved: false };
    draw();
    return;
  }
  sel = -1;
  cursor = Math.max(0, xt(e.offsetX));
  if (play) start(cursor); else draw();
});
addEventListener("mousemove", (e) => {
  if (!drag) return;
  const r = cv.getBoundingClientRect(), x = e.clientX - r.left;
  if (!drag.moved) {
    if (Math.abs(x - drag.x0) < 3) return;
    snap(); drag.moved = true;
  }
  E.dragMark(M, drag.k, drag.base.t[drag.k] + ((x - drag.x0) / W()) * view.span, drag.how, drag.base);
  draw();
});
addEventListener("mouseup", () => {
  if (!drag) return;
  const moved = drag.moved;
  drag = null; draw(); updateButtons();
  if (moved && play) { play.next = M.t.findIndex((t) => t >= playPos()); if (play.next < 0) play.next = M.t.length; }
});
cv.addEventListener("dblclick", (e) => {
  const k = markAt(e.offsetX);
  if (k >= 0) toggleDown(k);
  else addMark(Math.max(0, xt(e.offsetX)), e.ctrlKey || e.metaKey);
});
cv.addEventListener("wheel", (e) => {
  e.preventDefault();
  if (e.ctrlKey) {
    const at = xt(e.offsetX), f = e.deltaY < 0 ? 0.8 : 1.25;
    view.span = Math.min(Math.max(view.span * f, 0.5), (buf?.duration || 300) * 1.05);
    view.t0 = Math.max(0, at - (e.offsetX / W()) * view.span);
  } else view.t0 = Math.max(0, view.t0 + ((e.deltaY + e.deltaX) / W()) * view.span * 0.6);
  peaks = null; draw();
}, { passive: false });

// ---------------------------------------------------------------- controls
const marksKey = () => JSON.stringify([M.t, M.d, M.s, M.c, M.p0 ?? null]);
function updateButtons() {
  if (M) dirty = marksKey() !== savedKey;
  $("gUndo").disabled = !undo.length; $("gRedo").disabled = !redo.length;
  $("gSaved").textContent = dirty ? "not saved" : "saved";
}
document.querySelectorAll("[data-all]").forEach((b) => { b.onclick = () => { nudge(-1, +b.dataset.all); b.blur(); }; });
document.querySelectorAll("[data-from]").forEach((b) => { b.onclick = () => { nudge(heardPos(), +b.dataset.from); b.blur(); }; });
document.querySelectorAll("[data-one]").forEach((b) => { b.onclick = () => { moveOne(heardPos(), +b.dataset.one); b.blur(); }; });
document.querySelectorAll("[data-phrase]").forEach((b) => { b.onclick = () => { movePhrase(+b.dataset.phrase); b.blur(); }; });
$("gPlay").onclick = (e) => { play ? stop() : start(cursor); e.currentTarget.blur(); };
$("gSpeed").onchange = (e) => { if (play) start(heardPos()); e.target.blur(); };
$("gClickVol").oninput = (e) => { clickGain.gain.value = +e.target.value; lsSet("clickVol", +e.target.value); };
$("gSongVol").oninput = (e) => { songGain.gain.value = +e.target.value; lsSet("songVol", +e.target.value); };
$("gDelay").oninput = (e) => { $("gDelayV").textContent = `${e.target.value} ms`; lsSet("delay", +e.target.value); draw(); };
$("gZoomIn").onclick = () => { view.span = Math.max(0.5, view.span * 0.7); peaks = null; draw(); };
$("gZoomOut").onclick = () => { view.span = Math.min((buf?.duration || 300) * 1.05, view.span / 0.7); peaks = null; draw(); };
$("gFit").onclick = () => { view = { t0: 0, span: (buf?.duration || 60) * 1.02 }; peaks = null; draw(); };
$("gUndo").onclick = () => { if (undo.length) { redo.push(JSON.stringify({ M })); restore(undo.pop()); } };
$("gRedo").onclick = () => { if (redo.length) { undo.push(JSON.stringify({ M })); restore(redo.pop()); } };
$("gEven").onclick = (e) => { evenOut(); e.currentTarget.blur(); };
$("gNextUneven").onclick = (e) => { nextUneven(); e.currentTarget.blur(); };
$("gSave").onclick = save;
$("gSaveProj").onclick = saveForProject;
$("gRepair").onclick = async () => {
  const btn = $("gRepair"), msg = $("gRepairMsg");
  btn.disabled = true; msg.textContent = "re-counting the detector's beats…";
  try {
    const r = await api.post(`/api/songs/${sid}/grid/repair`);
    snap();
    M = { t: r.marks.t, d: r.marks.d, s: r.marks.s, c: r.marks.t.map(() => 0), bpb: r.marks.bpb, p0: null };
    sel = -1;
    if (play) start(heardPos());
    draw(); updateButtons();
    const rp = r.report, fix = rp.sections.filter((x) => x.ratio && x.ratio !== 1);
    msg.textContent = `${rp.after} beats on one ${Math.round(rp.pulse)} BPM pulse` +
      (fix.length ? ` · re-counted ${fix.length} stretch${fix.length > 1 ? "es" : ""} the detector heard in half / double time` : "") +
      (rp.unsure.length ? ` · ${rp.unsure.length} unsure (amber) - check by ear` : "") + " · listen, then Save";
  } catch (e) { msg.textContent = ""; toast(e.message, true); }
  btn.disabled = false;
};
$("gDone").onclick = async () => {
  if (dirty) {
    const keep = confirm("Save the grid before closing?\n\nOK = save and close, Cancel = close without saving");
    if (keep) await save();
    dirty = false;
  }
  stop();
  window.close();                                   // only works for a window Re-Tex opened - else say how
  setTimeout(() => toast("Close this window with its X (or Ctrl+W)"), 300);
};
// Save = the song's grid: projects from now on use it, saved projects keep the grid they had (the server copies it
// into them), and the project this window was opened from follows the new grid. "For this project only" = that
// project's own timing for the song (the main window applies it to the open project - it's in its undo).
const marksNow = () => ({ t: M.t, d: M.d, s: M.s, c: M.c, bpb: bpb(), p0: M.p0 ?? null });
async function save() {
  try {
    const r = await api.put(`/api/songs/${sid}/marks`, { user_marks: marksNow(), ...(pid ? { project: pid } : {}) });
    savedKey = marksKey(); updateButtons();
    $("gMeta").textContent = ` · ${M.t.length} marks (the song's grid) · ${M.bpb} beats a bar`;
    bus?.postMessage({ type: "marks", sid, frozen: r.frozen || [], old: r.old || null, pid });
    const n = (r.frozen || []).filter((x) => x !== pid).length;
    toast(`Saved as the song's grid - projects from now on use it${pid ? ", and this project" : ""}${n ? `; ${n} older project${n > 1 ? "s" : ""} keep${n > 1 ? "" : "s"} the grid ${n > 1 ? "they" : "it"} had` : ""}`);
  } catch (e) { toast(e.message, true); }
}
async function saveForProject() {
  if (!pid || !bus) return;
  const ok = await new Promise((done) => {
    const t = setTimeout(() => { bus.removeEventListener("message", hear); done(false); }, 2000);
    function hear(e) {
      if (e.data?.type === "project-marks-saved" && e.data.pid === pid && e.data.sid === sid) {
        clearTimeout(t); bus.removeEventListener("message", hear); done(true);
      }
    }
    bus.addEventListener("message", hear);
    bus.postMessage({ type: "set-project-marks", sid, pid, marks: marksNow() });
  });
  if (!ok) { toast("That project isn't open in Re-Tex - open it there, then save again", true); return; }
  savedKey = marksKey(); updateButtons();
  $("gMeta").textContent = ` · ${M.t.length} marks (this project's own timing) · ${M.bpb} beats a bar`;
  toast("Saved for this project only - the song's grid and other projects are unchanged (Ctrl+Z in Re-Tex undoes it)");
}
addEventListener("keydown", (e) => {
  if (e.target.closest?.("input, select")) return;
  const k = e.key.toLowerCase(), ctrl = e.ctrlKey || e.metaKey;
  if (k === " ") { e.preventDefault(); $("gPlay").click(); }
  else if ((k === "delete" || k === "backspace") && sel >= 0) { e.preventDefault(); deleteMark(sel); }
  else if (k === "escape" && sel >= 0) { sel = -1; draw(); }
  else if (ctrl && k === "s") { e.preventDefault(); save(); }
  else if (ctrl && k === "z") { e.preventDefault(); $("gUndo").click(); }
  else if (ctrl && k === "y") { e.preventDefault(); $("gRedo").click(); }
  else if (k === "arrowleft" || k === "arrowright") { e.preventDefault(); nudge(e.shiftKey ? heardPos() : -1, (k === "arrowleft" ? -1 : 1) * (e.altKey ? 0.002 : 0.01)); }
});
addEventListener("beforeunload", (e) => { if (dirty) { e.preventDefault(); e.returnValue = ""; } });

// ---------------------------------------------------------------- start
(async () => {
  if (!sid) { $("gTitle").textContent = "No song"; return; }
  try { info = await api.get(`/api/songs/${sid}`); } catch (e) { $("gTitle").textContent = "Couldn't open the song"; toast(e.message, true); return; }
  document.title = `${info.title} · Beat grid · Re-Tex`;
  $("gTitle").textContent = info.title;
  // opened from a project that has its own timing for the song: start from that (ask the open project first - its
  // newest state - else its saved file)
  let own = null, pname = "";
  if (pid) {
    own = await new Promise((done) => {
      if (!bus) return done(undefined);
      const t = setTimeout(() => { bus.removeEventListener("message", hear); done(undefined); }, 700);
      function hear(e) {
        if (e.data?.type === "project-marks" && e.data.pid === pid && e.data.sid === sid) {
          clearTimeout(t); bus.removeEventListener("message", hear); pname = e.data.name || ""; done(e.data.marks || null);
        }
      }
      bus.addEventListener("message", hear);
      bus.postMessage({ type: "get-project-marks", sid, pid });
    });
    if (own === undefined) {
      try { const p = await api.get(`/api/projects/${pid}`); own = p.songs?.[sid]?.marks || null; pname = p.name || ""; } catch { own = null; }
    }
    $("gSaveProj").hidden = false;
    $("gSaveProj").title = `Save these as ${pname ? `"${pname}"` : "this project"}'s own timing for the song - the song's grid and other projects stay as they are`;
  }
  const m = own || info.user_marks || detectedMarks(info);
  M = { t: [...m.t], d: m.d.map((x) => (x ? 1 : 0)), s: [...(m.s || m.t.map(() => 1))],
    c: m.c && m.c.length === m.t.length ? m.c.map((x) => (x ? 1 : 0)) : m.t.map(() => 0), bpb: m.bpb || info.bpb || 4, p0: m.p0 ?? null };
  $("gMeta").textContent = ` · ${M.t.length} marks ${own ? `(${pname ? `"${pname}"` : "this project"}'s own timing)` : info.user_marks ? "(the song's grid - yours)" : "(the song's grid - the detector's)"} · ${M.bpb} beats a bar`;
  // the delay: what the browser knows of your output, else what you set last time
  const est = Math.round(((ac.outputLatency || 0) + (ac.baseLatency || 0)) * 1000);
  $("gDelay").value = String(ls("delay", est || 0)); $("gDelayV").textContent = `${$("gDelay").value} ms`;
  $("gClickVol").value = String(ls("clickVol", 0.6)); clickGain.gain.value = +$("gClickVol").value;
  $("gSongVol").value = String(ls("songVol", 0.8)); songGain.gain.value = +$("gSongVol").value;
  savedKey = marksKey(); updateButtons();
  size();
  $("gHint").insertAdjacentText("afterbegin", "Loading the song… ");
  try {
    const raw = await (await fetch(`/api/songs/${sid}/source`)).arrayBuffer();
    buf = await ac.decodeAudioData(raw);
  } catch (e) { toast("Couldn't load the song's audio: " + e.message, true); return; }
  const L = buf.getChannelData(0), R = buf.numberOfChannels > 1 ? buf.getChannelData(1) : L;
  mono = new Float32Array(L.length);
  for (let i = 0; i < L.length; i++) mono[i] = 0.5 * (L[i] + R[i]);
  $("gHint").firstChild.remove();
  const firstSure = M.t[Math.max(0, M.s.indexOf(1))] || 0;
  view = { t0: Math.max(0, firstSure - 2), span: 12 };
  cursor = Math.max(0, firstSure - 2);
  peaks = null; draw();
})();
