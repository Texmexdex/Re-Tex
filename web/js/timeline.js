// Arrangement timeline: canvas lanes + DOM track headers.
// Overlapping clips on a track are laid out in stacked sub-rows so every layer stays visible and grabbable.
import { ICONS } from "./icons.js";
import { S, on, emit, edit, uid, clone, fullSong, STEMS, STEM_COLORS, secPerBeat, transportPos, songMult, ensureSongInProject, toast, syncNow, editProjectMarks, projectInfo, useSongGrid, makeSongGrid } from "./state.js";
import { peaks, peaksHi } from "./api.js";
import { songToolItems } from "./songtools.js";
import { landing, tempoParts, songMarks, moveMark, addMark, deleteMark, shiftMarks, barStart, setBpb, tapReplace, confirmMarks, lockActive, lockView } from "./timing.js";
import { pieces, wAtU, uAtW, stretchSpan, resetSpan, setSpanPitch } from "./warps.js";
import { seekTo } from "./transport.js";
import { showMenu, modal } from "./ui.js";
import { openSampleEditor } from "./sampleopen.js";
import { openReadHead } from "./readhead.js";

const SECLANE = 18, RULER = 46, EDGE = 8, FADE_HANDLE = 9, SCROLLBAR = 11;   // RULER = section lane + bar numbers
const SECTION_NAMES = ["intro", "verse", "pre-chorus", "chorus", "post-chorus", "bridge", "break", "drop", "instrumental", "outro"];
const SECTION_COLORS = { intro: "#5d6168", verse: "#5d82b0", "pre-chorus": "#5a9aa8", chorus: "#b06a88", "post-chorus": "#a0708a",
  bridge: "#8a7aa8", break: "#4f8f86", drop: "#c48450", instrumental: "#6f9662", outro: "#5d6168" };
const MIN_PPB = 1.5, MAX_PPB = 4000;                                   // zoom range, px per beat (4000 ~ 0.1-0.2 ms per px)
let hScale = +(localStorage.getItem("trackScale") || 1) || 1;          // Shift+wheel: taller/shorter tracks
let waveGain = +(localStorage.getItem("waveGain") || 1) || 1;
// one decimal at most (the wheel sets it smoothly, e.g. 1.749...), so the button keeps its size
const waveLabel = () => `wave ×${waveGain < 9.95 ? (Math.round(waveGain * 10) / 10).toString() : Math.round(waveGain)}`;          // Ctrl+Shift+wheel: magnify the drawn waveforms (display only)
function setWaveGain(v) {
  waveGain = Math.max(1, Math.min(16, v));
  try { localStorage.setItem("waveGain", String(waveGain)); } catch {}
  const b = document.getElementById("waveGain");
  if (b) { b.textContent = waveLabel(); b.classList.toggle("on", waveGain > 1); }
  dirty = true;
}
const baseH = () => Math.round(70 * hScale);
const laneH = () => Math.round(38 * hScale);
let lastInteract = 0;                                                   // follow-playhead waits after you touch the timeline
const canvas = document.getElementById("timeline");
const wrap = document.getElementById("tlScroll");
const headers = document.getElementById("trackHeaders");
const ctx = canvas.getContext("2d");

export const view = { pxPerBeat: 14, scrollX: 0, scrollY: 0, w: 0, h: 0 };
let dirty = true;
let drag = null;             // current mouse interaction
let hover = null;            // {clip, zone, x, y}
const waveCache = new Map(); // clip waveform columns
let snapMode = localStorage.getItem("snap") || "beat";   // bar | beat | 1/2 | 1/4 | 1/8 | off

export const redraw = () => (dirty = true);
const rateOf = (c) => c.rate || 1;

// FEAT-005: show the vocal lyrics (word timestamps from lyrics.json) under vocal-stem clips. Toggleable; off
// by default so nothing changes for a hand-built arrangement. The words come from the loaded song info
// (info.lyrics.segments[].words[] — {t, w}); absent until a song is transcribed, in which case nothing draws.
let lyricsOverlay = localStorage.getItem("lyricsOverlay") === "1";
export function toggleLyricsOverlay(on = !lyricsOverlay) {
  lyricsOverlay = on;
  try { localStorage.setItem("lyricsOverlay", lyricsOverlay ? "1" : "0"); } catch {}
  document.getElementById("lyrOverlay")?.classList.toggle("on", lyricsOverlay);
  dirty = true;
}

// ---------------------------------------------------------------- layout (tracks with stacked sub-rows)
let L = null;
function layout() {
  if (L) return L;
  const p = S.project;
  L = { top: [], h: [], lanes: [], lane: new Map(), total: 0 };
  if (!p) return L;
  let y = 0;
  p.tracks.forEach((t, i) => {
    const clips = p.clips.filter((c) => c.track === t.id).sort((a, b) => a.start - b.start || a.id.localeCompare(b.id));
    const ends = [];
    for (const c of clips) {
      let k = ends.findIndex((e) => e <= c.start + 1e-6);
      if (k < 0) { k = ends.length; ends.push(0); }
      ends[k] = c.start + c.len;
      L.lane.set(c.id, k);
    }
    const n = Math.max(1, ends.length);
    const h = n === 1 ? baseH() : Math.max(baseH(), n * laneH() + 6);
    L.top.push(y); L.h.push(h); L.lanes.push(n);
    y += h;
  });
  L.total = y;
  return L;
}
function relayout() { L = null; dirty = true; positionHeaders(); }

// ---------------------------------------------------------------- geometry
const beatToX = (b) => b * view.pxPerBeat - view.scrollX;
const xToBeat = (x) => (x + view.scrollX) / view.pxPerBeat;
const trackY = (i) => { const l = layout(); return RULER + (i < l.top.length ? l.top[i] : l.total + (i - l.top.length) * baseH()) - view.scrollY; };
const trackH = (i) => layout().h[i] ?? baseH();
function trackAt(y) {
  const l = layout(), yy = y - RULER + view.scrollY;
  for (let i = 0; i < l.top.length; i++) if (yy >= l.top[i] && yy < l.top[i] + l.h[i]) return i;
  return -1;
}
function clipRect(c, ti) {
  const l = layout();
  const n = l.lanes[ti] || 1, k = l.lane.get(c.id) || 0;
  const top = trackY(ti);
  if (n === 1) return { x0: beatToX(c.start), x1: beatToX(c.start + c.len), y: top + 3, h: baseH() - 7 };
  const lh = (l.h[ti] - 6) / n;
  return { x0: beatToX(c.start), x1: beatToX(c.start + c.len), y: top + 3 + k * lh, h: lh - 3 };
}
function bpb() { return S.project?.bpb || 4; }
const SNAP = { bar: null, beat: 1, "1/2": 0.5, "1/4": 0.25, "1/8": 0.125, off: 0 };
const SNAP_FREE_PX = 600;     // a snap step wider than this on screen is ignored (you are editing below the grid)
function snapBeat(b, ev, { forceFree = false } = {}) {
  if (forceFree || ev?.altKey) return b;                // Alt = no snapping (smooth)
  if (ev?.shiftKey) return Math.round(b / bpb()) * bpb();
  const q = snapMode === "bar" ? bpb() : SNAP[snapMode];
  if (q && q * view.pxPerBeat > SNAP_FREE_PX) return b;   // zoomed in past the snap grid: place exactly where pointed
  return q ? Math.round(b / q) * q : b;
}
function trackIndex(id) { return S.project.tracks.findIndex((t) => t.id === id); }

function clipAt(x, y) {
  if (!S.project) return null;
  const ti = trackAt(y);
  if (ti < 0) return null;
  const tid = S.project.tracks[ti].id;
  const b = xToBeat(x);
  for (let i = S.project.clips.length - 1; i >= 0; i--) {
    const c = S.project.clips[i];
    if (c.track !== tid || b < c.start || b > c.start + c.len) continue;
    const r = clipRect(c, ti);
    if (y < r.y || y > r.y + r.h) continue;
    const { x0, x1 } = r;
    // stretch grips = the small triangles in the bottom corners (inside the triangle only)
    const up = r.y + r.h - y;                                   // px above the clip's bottom edge
    const inGripL = up < 13 && (x - x0) + up < 13, inGripR = up < 13 && (x1 - x) + up < 13;
    let zone = "move";
    if (inGripL && x1 - x0 > 26) zone = "stretchL";
    else if (inGripR && x1 - x0 > 26) zone = "stretchR";
    else if (y - r.y < FADE_HANDLE + 6 && x - x0 < FADE_HANDLE + 4 + (c.fade_in || 0) * view.pxPerBeat && x - x0 > EDGE) zone = "fadein";
    else if (y - r.y < FADE_HANDLE + 6 && x1 - x < FADE_HANDLE + 4 + (c.fade_out || 0) * view.pxPerBeat && x1 - x > EDGE) zone = "fadeout";
    else if (x - x0 < EDGE) zone = "left";
    else if (x1 - x < EDGE) zone = "right";
    return { clip: c, zone, ti };
  }
  return null;
}

// ---------------------------------------------------------------- resize / headers
function resize() {
  const r = wrap.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  view.w = r.width; view.h = r.height;
  canvas.width = Math.max(1, r.width * dpr); canvas.height = Math.max(1, r.height * dpr);
  canvas.style.width = r.width + "px"; canvas.style.height = r.height + "px";
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  waveCache.clear();
  dirty = true;
  positionHeaders();
}
new ResizeObserver(resize).observe(wrap);

function dbFromSlider(v) { return v <= 0.001 ? -60 : 20 * Math.log10(v); }
function sliderFromDb(db) { return Math.pow(10, db / 20); }

export function buildHeaders() {
  headers.innerHTML = "";
  const top = document.createElement("div");
  top.className = "th-ruler";
  top.innerHTML = `<button id="addTrack" title="Add a track">+ Trk</button>
    <select id="snapSel" title="Grid snap for moving / trimming clips: bar, beat, 1/2, 1/4, 1/8 - or free (hold Alt while dragging = no snap, Shift = bar)">
      ${["bar", "beat", "1/2", "1/4", "1/8", "off"].map((s) => `<option value="${s}" ${s === snapMode ? "selected" : ""}>${s === "off" ? "free" : s}</option>`).join("")}
    </select>
    <button id="zoomOut" title="Zoom out (Ctrl+wheel)">−</button><button id="zoomIn" title="Zoom in (Ctrl+wheel)">+</button><button id="zoomFit" title="Fit arrangement (F)">Fit</button>
    <button id="hDown" title="Shorter tracks (Shift+wheel)">▁</button><button id="hUp" title="Taller tracks / bigger waveforms (Shift+wheel)">█</button>
    <button id="waveGain" class="toggle ${waveGain > 1 ? "on" : ""}" title="Magnify the waveforms so quiet parts (word onsets, breaths) show - display only, the sound doesn't change. Click: ×1 → ×2 → ×4 → ×8. Ctrl+Shift+wheel: smooth">${waveLabel()}</button>
    <button id="markMode" class="toggle ${markMode ? "on" : ""}" title="Edit beat marks (M): drag, add, delete, tap them in - the audio never moves">Marks</button>
    <button id="lyrOverlay" class="toggle ${lyricsOverlay ? "on" : ""}" title="Show the lyrics word by word under vocal clips (yours where you typed them in the song view, else Whisper's)">Lyrics</button>
    <button id="helpKeys" title="All shortcuts and mouse moves">${ICONS.keyboard}</button>`;
  headers.appendChild(top);
  top.querySelector("#addTrack").onclick = () => edit((p) => p.tracks.push({ id: uid("t"), name: `Track ${p.tracks.length + 1}`, role: "any", color: "#8e9198", gain_db: 0, pan: 0, mute: false, solo: false, hpf: 0, lpf: 0, duck: 0, pocket: 0 }), { label: "Add track" });
  top.querySelector("#snapSel").onchange = (e) => { snapMode = e.target.value; try { localStorage.setItem("snap", snapMode); } catch {} e.target.blur(); };
  top.querySelector("#zoomIn").onclick = () => zoomAt(view.w / 2, 1.4);
  top.querySelector("#zoomOut").onclick = () => zoomAt(view.w / 2, 1 / 1.4);
  top.querySelector("#zoomFit").onclick = fit;
  top.querySelector("#hDown").onclick = () => setTrackScale(hScale / 1.25);
  top.querySelector("#hUp").onclick = () => setTrackScale(hScale * 1.25);
  top.querySelector("#waveGain").onclick = () => setWaveGain(waveGain >= 8 ? 1 : [1, 2, 4, 8].find((g) => g > waveGain + 0.01));
  top.querySelector("#helpKeys").onclick = showShortcuts;
  top.querySelector("#markMode").onclick = (e) => { toggleMarkMode(); e.target.blur(); };
  top.querySelector("#lyrOverlay").onclick = (e) => { toggleLyricsOverlay(); e.target.blur(); };
  if (!S.project) return;
  S.project.tracks.forEach((t, i) => {
    const el = document.createElement("div");
    el.className = "th";
    el.dataset.i = i;
    const fxOn = t.hpf > 20 || (t.lpf > 200 && t.lpf < 20000) || t.duck > 0 || t.pocket > 0;
    el.innerHTML = `
      <div class="row"><span class="sw" style="background:${t.color}"></span>
        <input class="name" value="${escapeHtml(t.name)}" spellcheck="false" title="${escapeHtml(t.name)}">
        <button class="ms m ${t.mute ? "on" : ""}" title="Mute">M</button>
        <button class="ms s ${t.solo ? "on" : ""}" title="Solo">S</button>
        <button class="fx ${fxOn ? "on" : ""}" title="Track FX: filters, vocal pocket, ducking">fx</button>
      </div>
      <div class="row"><input type="range" class="gain" min="0" max="2" step="0.01" value="${sliderFromDb(t.gain_db)}" title="Track volume (double-click = 0 dB)">
        <span class="db">${t.gain_db >= 0 ? "+" : ""}${t.gain_db.toFixed(1)} dB</span></div>
      <div class="lanes-note"></div>`;
    el.querySelector(".name").onchange = (e) => edit((p) => (p.tracks[i].name = e.target.value), { label: "Rename track" });
    el.querySelector(".m").onclick = () => edit((p) => (p.tracks[i].mute = !p.tracks[i].mute), { label: "Mute track" });
    el.querySelector(".s").onclick = () => edit((p) => (p.tracks[i].solo = !p.tracks[i].solo), { label: "Solo track" });
    el.querySelector(".fx").onclick = (e) => trackFxMenu(i, e);
    const g = el.querySelector(".gain"), db = el.querySelector(".db");
    g.oninput = () => {
      const v = Math.round(Math.max(-60, dbFromSlider(+g.value)) * 10) / 10;
      db.textContent = `${v >= 0 ? "+" : ""}${v.toFixed(1)} dB`;
      edit((p) => (p.tracks[i].gain_db = v), { coalesce: "gain" + t.id, label: "Track volume" });
    };
    g.ondblclick = () => edit((p) => (p.tracks[i].gain_db = 0), { label: "Track volume" });
    el.oncontextmenu = (e) => {
      e.preventDefault();
      const onLane = [...new Set(S.project.clips.filter((c) => c.track === t.id).map((c) => c.song))];
      showMenu(e, [
        ...onLane.map((sid) => [`${(S.songs.get(sid)?.title || sid).slice(0, 40)} ▸`, () => showMenu(e, songToolItems(sid))]),
        ...(onLane.length ? ["-"] : []),
        ...(onLane.length ? [
          ["ReadHead: arrange the vocal on this lane…", () => openReadHead(t.id, "arrange")],
          ["ReadHead: glitch-chop this lane…", () => openReadHead(t.id, "glitch")],
          "-",
        ] : []),
        ["Rename", () => el.querySelector(".name").focus()],
        ["Delete track + its clips", () => edit((p) => { const id = p.tracks[i].id; p.tracks.splice(i, 1); p.clips = p.clips.filter((c) => c.track !== id); }, { label: "Delete track" })],
        ["Move up", () => i > 0 && edit((p) => { const [x] = p.tracks.splice(i, 1); p.tracks.splice(i - 1, 0, x); }, { label: "Move track" })],
        ["Move down", () => i < S.project.tracks.length - 1 && edit((p) => { const [x] = p.tracks.splice(i, 1); p.tracks.splice(i + 1, 0, x); }, { label: "Move track" })],
      ]);
    };
    headers.appendChild(el);
  });
  positionHeaders();
}

function trackFxMenu(i, e) {
  const t = S.project.tracks[i];
  const set = (k, v) => () => edit((p) => (p.tracks[i][k] = v), { label: "Track FX" });
  showMenu(e, [
    [`High-pass: ${t.hpf > 20 ? t.hpf + " Hz" : "off"}`, null],
    ["  off", set("hpf", 0)], ["  90 Hz (clean up low end)", set("hpf", 90)], ["  250 Hz (thin)", set("hpf", 250)], ["  600 Hz (telephone)", set("hpf", 600)],
    "-",
    [`Low-pass: ${t.lpf > 200 && t.lpf < 20000 ? t.lpf + " Hz" : "off"}`, null],
    ["  off", set("lpf", 0)], ["  8 kHz (soften)", set("lpf", 8000)], ["  2 kHz (muffled)", set("lpf", 2000)], ["  500 Hz (underwater)", set("lpf", 500)],
    "-",
    [`Vocal pocket (cut 2.8 kHz): ${t.pocket ? t.pocket + " dB" : "off"}`, null],
    ["  off", set("pocket", 0)], ["  2.5 dB", set("pocket", 2.5)], ["  5 dB", set("pocket", 5)],
    "-",
    [`Duck under vocals: ${t.duck ? t.duck + " dB" : "off"}`, null],
    ["  off", set("duck", 0)], ["  2 dB", set("duck", 2)], ["  4 dB", set("duck", 4)], ["  8 dB (strong)", set("duck", 8)],
  ]);
}

function positionHeaders() {
  const l = layout();
  headers.querySelectorAll(".th").forEach((el) => {
    const i = +el.dataset.i;
    el.style.top = trackY(i) + "px";
    el.style.height = trackH(i) + "px";
    const n = l.lanes[i] || 1;
    el.querySelector(".lanes-note").textContent = n > 1 ? `${n} layers stacked` : "";
  });
}

function escapeHtml(s) { return String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c])); }

// ---------------------------------------------------------------- zoom / scroll
function contentBeats() {
  return S.project ? Math.max(64, ...S.project.clips.map((c) => c.start + c.len)) + 32 : 128;
}
function clampScroll() {
  const len = contentBeats();
  view.scrollX = Math.max(0, Math.min(view.scrollX, len * view.pxPerBeat - view.w * 0.5));
  const th = layout().total + RULER;
  view.scrollY = Math.max(0, Math.min(view.scrollY, Math.max(0, th - view.h + 20)));
}
export function setTrackScale(v, anchorY = null) {
  const old = hScale;
  hScale = Math.max(0.6, Math.min(6, v));
  try { localStorage.setItem("trackScale", String(hScale)); } catch {}
  if (anchorY != null) view.scrollY = (view.scrollY + anchorY - RULER) * hScale / old - (anchorY - RULER);
  L = null; clampScroll(); positionHeaders(); dirty = true;
}
export function zoomAt(x, f) {
  const b = xToBeat(x);
  view.pxPerBeat = Math.max(MIN_PPB, Math.min(MAX_PPB, view.pxPerBeat * f));
  view.scrollX = b * view.pxPerBeat - x;
  clampScroll(); waveCache.clear(); dirty = true;
}
export function fit() {
  if (!S.project || !S.project.clips.length) return;
  const len = Math.max(...S.project.clips.map((c) => c.start + c.len));
  view.pxPerBeat = Math.max(MIN_PPB, Math.min(MAX_PPB, (view.w - 30) / len));
  view.scrollX = 0; waveCache.clear(); dirty = true;
}
export function ensureVisible(beat) {
  const x = beatToX(beat);
  if (x > view.w - 60) { view.scrollX += x - view.w * 0.25; clampScroll(); dirty = true; }
  else if (x < 0) { view.scrollX = Math.max(0, beat * view.pxPerBeat - 40); dirty = true; }
}

canvas.addEventListener("wheel", (e) => {
  e.preventDefault();
  // Alt+wheel over a clip = fine pitch (0.1 st; +Shift = 1 st)
  if (e.altKey) {
    const h = clipAt(e.offsetX, e.offsetY);
    if (h) {
      const d = (e.deltaY < 0 ? 1 : -1) * (e.shiftKey ? 1 : 0.1);
      const ids = S.selClips.has(h.clip.id) ? S.selClips : new Set([h.clip.id]);
      edit((p) => { for (const c of p.clips) if (ids.has(c.id)) c.pitch = Math.round(Math.max(-24, Math.min(24, (c.pitch || 0) + d)) * 100) / 100; },
        { coalesce: "wheelpitch" + [...ids].join(), label: "Pitch" });
      showHint(`pitch ${fmtSt(h.clip.pitch || 0)}`, e.offsetX, e.offsetY);
      return;
    }
  }
  lastInteract = performance.now();
  if (e.ctrlKey && e.shiftKey) { setWaveGain(waveGain * (e.deltaY < 0 ? 1.15 : 1 / 1.15)); showHint(`waveforms ×${waveGain.toFixed(1)}`, e.offsetX, e.offsetY); return; }
  if (e.ctrlKey) zoomAt(e.offsetX, e.deltaY < 0 ? 1.18 : 1 / 1.18);
  else if (e.shiftKey) { setTrackScale(hScale * ((e.deltaY || e.deltaX) < 0 ? 1.12 : 1 / 1.12), e.offsetY); return; }
  else if (e.offsetY < RULER || e.offsetY > view.h - SCROLLBAR - 2 || Math.abs(e.deltaX) > Math.abs(e.deltaY)) {
    view.scrollX += (e.deltaX || e.deltaY); clampScroll();
  }
  else {
    const th = layout().total + RULER;
    if (th > view.h) { view.scrollY += e.deltaY; }
    else view.scrollX += e.deltaY;
    clampScroll();
  }
  positionHeaders(); dirty = true;
}, { passive: false });

// ---------------------------------------------------------------- waveform for a clip
// the song as this project plays it: as recorded, or locked to its marks (timing.js lockView)
// (a song's info as this project sees it: its own timing for the song laid over the song's grid - state.js projectInfo)
const infoOf = (sid) => projectInfo(resolved.get(sid));
const isLocked = (sid) => lockActive(S.project, sid, infoOf(sid));
const songView = (sid) => { const info = infoOf(sid); return info && isLocked(sid) ? lockView(info) : info; };
// where the song's beats land (timing.js landing); null until its info has loaded
const landFor = (sid) => landing(songView(sid));
const wAt = (Lm, b) => (Lm ? Lm.at(b) : b);          // beat index -> where it lands (song beats)
const bAt = (Lm, w) => (Lm ? Lm.inv(w) : w);         // and back
// a clip's constant-speed pieces (warps.js; one piece unless part of it is stretched / contracted)
const clipPieces = (c) => pieces(c, songMult(S.project, c.song), landFor(c.song));
const beatAtU = (c, u, ps = clipPieces(c)) => bAt(landFor(c.song), wAtU(ps, u));   // clip offset -> source beat
const uAtBeat = (c, b, ps = clipPieces(c)) => uAtW(ps, wAt(landFor(c.song), b));   // source beat -> clip offset

// A clip's waveform, worked out in tiles of WAVE_TILE pixel columns, only for the tiles on screen - so it shows at any
// zoom (it used to be one strip for the whole clip, given up past 40,000 columns: a whole-song clip vanished once
// zoomed in past ~60 px per beat). -> Float32Array [lo, hi, lo, hi, ...] for columns tile*WAVE_TILE..., or null.
const WAVE_TILE = 1024;
function clipWaveTile(c, tile) {
  const mult = songMult(S.project, c.song), rate = rateOf(c);
  const hiK = c.song + "/" + c.stem, hi = view.pxPerBeat >= HI_PPB ? hiData.get(hiK) : null;
  if (view.pxPerBeat >= HI_PPB && !hi) loadHi(c);
  const key = `${c.song}|${c.stem}|${c.src}|${c.len}|${view.pxPerBeat}|${mult}|${rate}|${landFor(c.song) ? 1 : 0}|${isLocked(c.song) ? 1 : 0}|${hi ? 1 : 0}|${JSON.stringify(c.warps || [])}`;
  let entry = waveCache.get(c.id);
  if (!entry || entry.key !== key) { entry = { key, tiles: new Map() }; waveCache.set(c.id, entry); }
  const hit = entry.tiles.get(tile);
  if (hit) return hit;
  const res = resolved.get(c.song);
  const pk = peakData.get(c.song + "/" + c.stem);
  if (!res || !pk) { loadClipData(c); return null; }
  const beats = res.beats, nb = beats.length;
  const src = hi ? hi.pk : pk, pps = hi ? hi.pps : (res.peaks_per_sec || 100), full = hi ? 32767 : 127;
  const ncol = Math.max(1, Math.ceil(c.len * view.pxPerBeat));
  const i0 = tile * WAVE_TILE, i1 = Math.min(ncol, i0 + WAVE_TILE);
  if (i1 <= i0) return null;
  const Lm = landFor(c.song), ps = clipPieces(c);
  const cols = new Float32Array((i1 - i0) * 2);
  for (let i = i0; i < i1; i++) {
    const sb0 = bAt(Lm, wAtU(ps, i / view.pxPerBeat)), sb1 = bAt(Lm, wAtU(ps, (i + 1) / view.pxPerBeat));
    const t0 = beatTime(beats, nb, sb0), t1 = beatTime(beats, nb, sb1);
    let a = Math.max(0, Math.floor(t0 * pps)), b = Math.max(a + 1, Math.ceil(t1 * pps));
    let lo = 0, hi = 0;
    for (let j = a; j < b && j * 2 + 1 < src.length; j++) { lo = Math.min(lo, src[j * 2]); hi = Math.max(hi, src[j * 2 + 1]); }
    cols[(i - i0) * 2] = lo / full; cols[(i - i0) * 2 + 1] = hi / full;
  }
  if (entry.tiles.size > 64) entry.tiles.clear();         // keep memory bounded when scrolling a long way
  entry.tiles.set(tile, cols);
  return cols;
}
// offset (project beats) from a clip's start to its first source downbeat at/after the start, or null
// the song's beat marks (timing.js songMarks: yours if you've edited them, else the detector's with sure / unsure
// flags), or null until its info loads
const marksFor = (sid) => songMarks(songView(sid));
// marks inside a clip, as project beats: [{ x, down, sure, k (index in the song's marks) }] - where those beats of
// the audio actually play
function clipMarks(c) {
  const m = marksFor(c.song);
  if (!m) { if (!resolved.get(c.song)) loadClipData(c); return []; }
  const ps = clipPieces(c), w0 = wAtU(ps, 0) - 1e-6, w1 = wAtU(ps, c.len), out = [];
  let lo = 0, hi = m.w.length;
  while (lo < hi) { const k = (lo + hi) >> 1; if (m.w[k] < w0) lo = k + 1; else hi = k; }
  for (let k = lo; k < m.w.length && m.w[k] <= w1; k++) out.push({ x: c.start + uAtW(ps, m.w[k]), down: m.down[k], sure: m.sure[k], k });
  return out;
}
// downbeats snapping may use: only marks the detector is sure of (or yours) - an unsure one could be anywhere
const clipDownbeats = (c) => clipMarks(c).filter((o) => o.down && o.sure).map((o) => o.x);
// offset (project beats) from a clip's start to its first sure downbeat, or null
function downbeatOffset(c) {
  const d = clipDownbeats(c);
  return d.length ? d[0] - c.start : null;
}
// offset from a clip's start to its sure downbeat nearest project beat `at` (where you grabbed it), or null
function nearestDownbeatOffset(c, at) {
  const d = clipDownbeats(c);
  if (!d.length) return null;
  let best = d[0];
  for (const x of d) if (Math.abs(x - at) < Math.abs(best - at)) best = x;
  return best - c.start;
}
// the audio moment (seconds) at project beat `beat` inside clip c
function timeAt(c, beat) {
  const L = landFor(c.song);
  return L ? L.wt(wAtU(clipPieces(c), beat - c.start)) : null;
}
// the clip's mark nearest to canvas x (within `px`), or null
function markAt(c, x, px = 5) {
  let best = null;
  for (const o of clipMarks(c)) {
    const d = Math.abs(beatToX(o.x) - x);
    if (d <= px && (!best || d < best.d)) best = { ...o, d };
  }
  return best;
}

// ---------------------------------------------------------------- Marks mode + tapping
// Marks mode (button / M): drag a clip's marks, double-click to add one, right-click for the marks menu, Del
// deletes the mark under the mouse, and T taps marks in while playing. Everything is undoable and saved with the
// song (every project shows it). The audio never moves.
export let markMode = false;
let hoverMark = null;                   // { clip, k, x } under the mouse in marks mode
let tap = null;                         // { info, clip, times: [], timer }
export function toggleMarkMode(on = !markMode) {
  markMode = on;
  hoverMark = null;
  document.getElementById("markMode")?.classList.toggle("on", markMode);
  if (markMode) toast("Marks mode: drag a mark to move it · double-click a clip to add one · right-click for more · Del deletes · T taps marks while playing · M or Esc = done", false, 6000);
  dirty = true;
}
const lockedNote = () => toast("This song is locked to its marks - unlock it (arrangement bar) to edit its marks; lock it again after");
function tapNow() {
  if (!S.transport.playing) { toast("Start playback, then press T on every beat"); return; }
  const beat = transportPos() / secPerBeat();
  const sel = S.project.clips.filter((c) => S.selClips.has(c.id) || c === hover?.clip);
  const c = sel.find((k) => beat >= k.start && beat <= k.start + k.len);
  if (!c) { toast("Select the clip you're tapping along to (the playhead has to be inside it)"); return; }
  const info = resolved.get(c.song), t = timeAt(c, beat);
  if (!info || t == null) return;
  if (isLocked(c.song)) { lockedNote(); return; }
  if (!tap || tap.info !== info) { finishTap(); tap = { info, clip: c, times: [] }; }
  tap.times.push(t);
  clearTimeout(tap.timer);
  tap.timer = setTimeout(finishTap, 2000);
  dirty = true;
}
function finishTap() {
  if (!tap) return;
  const t = tap; tap = null;
  clearTimeout(t.timer);
  const times = [...t.times].sort((a, b) => a - b);
  if (times.length >= 2) {
    editProjectMarks(t.info, (m) => tapReplace(m, times), { label: `Tap ${times.length} marks` });
    toast(`${times.length} tapped marks replace the marks there (Ctrl+Z undoes)`);
  } else toast("Tap at least 2 beats");
  dirty = true;
}
// keys for marks mode; true = handled
export function markKey(e) {
  const k = e.key;
  if (e.ctrlKey || e.metaKey || e.altKey) return false;
  if (k === "m" || k === "M") { toggleMarkMode(); return true; }
  if (!markMode) return false;
  if (k === "t" || k === "T") { tapNow(); return true; }
  if (k === "Escape") { finishTap(); toggleMarkMode(false); return true; }
  if (k === "Delete" || k === "Backspace") {        // in marks mode Del only ever deletes marks, never clips
    if (!hoverMark) { toast("Point at a mark to delete it (clips can't be deleted in Marks mode)"); return true; }
    if (isLocked(hoverMark.clip.song)) { lockedNote(); return true; }
    const { clip, k: mk } = hoverMark, info = resolved.get(clip.song);
    if (info) editProjectMarks(info, (m) => deleteMark(m, mk), { label: "Delete mark" });
    hoverMark = null;
    return true;
  }
  return false;
}
on("marks", () => { dirty = true; });

// parts of the clip the detector found at another tempo: [{ x0, x1 (project beats), bpm }]
function clipTempoParts(c) {
  const info = resolved.get(c.song);
  if (!info || !landFor(c.song)) return [];
  const ps = clipPieces(c), end = beatAtU(c, c.len, ps), out = [];
  for (const t of tempoParts(info)) {
    if (t.b < c.src || t.a > end) continue;
    out.push({ x0: c.start + uAtBeat(c, Math.max(t.a, c.src), ps), x1: c.start + uAtBeat(c, Math.min(t.b, end), ps), bpm: t.bpm });
  }
  return out;
}
// FEAT-005: TRANSITIONS the user can drop on a clip boundary. These mirror the transitions.py renderers
// (B.2 type set): the UI authors the event into the track's additive `automation` lane, and the backend bakes
// the matching time-varying FX clip when the project is rendered (B.4 Option B). Values are plain metadata the
// static mixer ignores, so a project with none exports byte-identically.
const TRANSITION_TYPES = [
  { type: 3, kind: "sweep", label: "Filter sweep", title: "Low-pass sweep up into the next section" },
  { type: 6, kind: "noise_sweep", label: "Noise sweep", title: "White-noise riser into the next section" },
  { type: 1, kind: "cut", label: "Hard cut", title: "Clean cut on the downbeat (no FX)" },
  { type: 2, kind: "crossfade", label: "Crossfade", title: "Short beat-aligned crossfade" },
  { type: 4, kind: "echo_out", label: "Echo out", title: "Feedback-echo tail into the change" },
  { type: 5, kind: "fill", label: "Drum fill", title: "Lift the last bar's fill, then change" },
];
// A sweep automation event -> its [x0, x1] project-beat span + label, for the overlay. An event has a `beat`
// (project beat) and optionally `beats`/`bars` length; default one bar.
function automationSpans(trk, c) {
  const lane = Array.isArray(trk?.automation) ? trk.automation : [];
  if (!lane.length) return [];
  const bpbv = bpb();
  const out = [];
  for (const ev of lane) {
    const b = ev.beat ?? ev.start;
    if (b == null) continue;
    const len = ev.beats ?? (ev.bars ? ev.bars * bpbv : bpbv);
    // only draw it over the clip it sits inside (so it rides with the clip visually)
    if (b + len < c.start || b > c.start + c.len) continue;
    const t = TRANSITION_TYPES.find((q) => q.type === ev.type) || { label: ev.kind || "sweep" };
    out.push({ x0: beatToX(b), x1: beatToX(b + len), label: t.label });
  }
  return out;
}

// FEAT-005: lyric words inside a vocal clip, as project beats. lyrics.json carries word timestamps in SONG
// seconds; map each through the song's beat landing + the clip's warps the same way marks are mapped. ->
// [{ x (project beat), w (word) }], or [] when the song has no transcript loaded / the clip isn't vocals.
function clipLyricWords(c) {
  if (c.stem !== "vocals") return [];
  const info = resolved.get(c.song);
  const segs = info?.lyrics?.segments;
  if (!segs || !landFor(c.song)) { if (!info) loadClipData(c); return []; }
  const Lm = landFor(c.song), ps = clipPieces(c);
  const w0 = wAtU(ps, 0) - 1e-6, w1 = wAtU(ps, c.len);
  const out = [];
  for (const s of segs) {
    for (const wd of (s.words || [])) {
      const t = wd.t ?? wd.start;                 // word onset, song seconds
      if (t == null || !wd.w) continue;
      const wIdx = Lm.tw(t);                       // song seconds -> song-beat landing position
      if (wIdx < w0 || wIdx > w1) continue;
      out.push({ x: c.start + uAtW(ps, wIdx), w: wd.w });
    }
  }
  return out;
}

export function alignSelectionToGrid() {
  let n = 0;
  edit((p) => {
    for (const c of p.clips) {
      if (!S.selClips.has(c.id)) continue;
      const off = downbeatOffset(c);
      if (off == null) continue;
      const d = c.start + off, target = Math.round(d / bpb()) * bpb();
      c.start = Math.max(0, c.start + (target - d)); n++;
    }
  }, { label: "Align to grid" });
  toast(n ? `Lined up ${n} clip${n > 1 ? "s" : ""}: downbeats now sit on bar lines` : "Couldn't find downbeats for the selection yet - try again in a second");
}
function beatTime(beats, nb, b) {
  const i = Math.max(0, Math.min(nb - 2, Math.floor(b)));
  return beats[i] + (b - i) * (beats[i + 1] - beats[i]);
}
const resolved = new Map(), peakData = new Map(), loading = new Set(), hiData = new Map();
const HI_PPB = 60;            // from this zoom (px per beat) the waveform uses the close-up data
function loadHi(c) {
  const k = c.song + "/" + c.stem;
  if (loading.has("hi:" + k)) return;
  loading.add("hi:" + k);
  peaksHi(c.song, c.stem).then((d) => { if (d) { hiData.set(k, d); waveCache.clear(); dirty = true; } })
    .finally(() => loading.delete("hi:" + k));
}
function loadClipData(c) {
  const k = c.song + "/" + c.stem;
  if (loading.has(k)) return;
  loading.add(k);
  Promise.all([fullSong(c.song), peaks(c.song, c.stem)]).then(([info, pk]) => {
    resolved.set(c.song, info);
    if (pk) peakData.set(k, pk);
    dirty = true;
  }).catch(() => {}).finally(() => loading.delete(k));
}
export function forgetSong(id) {
  resolved.delete(id);
  for (const k of [...peakData.keys()]) if (k.startsWith(id + "/")) peakData.delete(k);
  for (const k of [...hiData.keys()]) if (k.startsWith(id + "/")) hiData.delete(k);
  waveCache.clear();
}

// ---------------------------------------------------------------- draw
let hatch = null;
function hatchPattern() {
  if (hatch) return hatch;
  const c = document.createElement("canvas"); c.width = c.height = 10;
  const g = c.getContext("2d");
  g.strokeStyle = "rgba(255,255,255,0.13)"; g.lineWidth = 3;
  g.beginPath(); g.moveTo(0, 10); g.lineTo(10, 0); g.stroke();
  hatch = ctx.createPattern(c, "repeat");
  return hatch;
}
const fmtSt = (v) => `${v >= 0 ? "+" : "−"}${Math.abs(v).toFixed(2)} st`;
let hint = null;
function showHint(text, x, y) { hint = { text, x, y, t: performance.now() }; dirty = true; }

// ---------------------------------------------------------------- highlight: stretch / contract part of a clip
// Shift+drag on a clip highlights a span; Ctrl+click clips on other lanes to add the same span there; drag
// the highlight's right edge to stretch or contract that part of every highlighted clip (what follows in each
// clip moves, the clip's end moves with it). Snaps like everything else (Alt = smooth).
let hl = null;                          // { t0, t1, clips: Set(clip ids) }
let hlKeepPitch = true;
export function clearHighlight() { if (hl) { hl = null; dirty = true; hlBar.hidden = true; } }
// the highlighted span (Perform mode's sample pads read it): { t0, t1, clips: [clip ids] } or null
export function currentHighlight() { return hl ? { t0: hl.t0, t1: hl.t1, clips: [...hl.clips] } : null; }
const hlSnap = (b, e) => snapBeat(b, { altKey: e.altKey });           // Shift is the highlight key, not "snap to bar"
function hlSpan(c) {                   // the highlight's part of a clip, as clip offsets
  const u0 = Math.max(hl.t0, c.start) - c.start, u1 = Math.min(hl.t1, c.start + c.len) - c.start;
  return u1 - u0 > 1e-6 ? [u0, u1] : null;
}
function hlEdgeAt(x, y) {              // is (x, y) on the highlight's right edge (the stretch handle)?
  if (!hl || hl.t1 - hl.t0 < 1e-6 || Math.abs(x - beatToX(hl.t1)) > 6) return false;
  for (const id of hl.clips) {
    const c = S.project.clips.find((k) => k.id === id);
    if (!c || trackIndex(c.track) < 0) continue;
    const r = clipRect(c, trackIndex(c.track));
    if (y >= r.y && y <= r.y + r.h) return true;
  }
  return false;
}
// a different project (or the highlighted clips are gone): drop the highlight
on("project", () => {
  if (hl && !(S.project && [...hl.clips].some((id) => S.project.clips.some((c) => c.id === id)))) clearHighlight();
});
function hlApply(fn, label) {
  if (!hl) return;
  edit((p) => {
    for (const c of p.clips) {
      if (!hl.clips.has(c.id)) continue;
      const sp = hlSpan(c);
      if (sp) fn(c, songMult(p, c.song), landFor(c.song), sp[0], sp[1]);
    }
  }, { label });
}
function hlReset() {
  // keep the highlight on the same music: its end follows where that beat plays after the reset
  const first = S.project.clips.find((k) => hl.clips.has(k.id) && hlSpan(k));
  const endBeat = first ? beatAtU(first, hlSpan(first)[1]) : null;
  hlApply((c, m, Lm, u0, u1) => resetSpan(c, m, Lm, u0, u1), "Reset stretch");
  if (first && endBeat != null) {
    const c = S.project.clips.find((k) => k.id === first.id);
    hl.t1 = Math.max(hl.t0 + 0.01, c.start + uAtBeat(c, endBeat));
  }
  dirty = true;
}
function hlSetPitch(keep) {
  hlKeepPitch = keep;
  hlApply((c, m, Lm, u0, u1) => setSpanPitch(c, m, Lm, u0, u1, keep), keep ? "Stretch keeps pitch" : "Stretch: pitch follows");
  updateHlBar();
}
// the small toolbar that floats over a highlight
const hlBar = document.createElement("div");
hlBar.id = "hlBar";
hlBar.hidden = true;
hlBar.style.cssText = "position:fixed;z-index:20;display:flex;gap:4px;align-items:center;padding:3px 5px;border-radius:10px;background:rgba(32,32,38,.92);backdrop-filter:blur(16px);border:1px solid rgba(255,255,255,.12);font-size:11px;color:#e6e6e8";
hlBar.innerHTML = `<span id="hlInfo"></span><button class="small" id="hlKeep" title="Stretching keeps the pitch (time-stretch)">Keep pitch</button><button class="small" id="hlVari" title="Pitch follows the stretch (slower = lower, like a turntable)">Pitch follows</button><button class="small" id="hlReset" title="Back to normal speed in the highlight">Reset</button><button class="small" id="hlClose" title="Clear the highlight (Esc)">✕</button>`;
document.body.appendChild(hlBar);
hlBar.querySelector("#hlKeep").onclick = () => hlSetPitch(true);
hlBar.querySelector("#hlVari").onclick = () => hlSetPitch(false);
hlBar.querySelector("#hlReset").onclick = hlReset;
hlBar.querySelector("#hlClose").onclick = clearHighlight;
function updateHlBar() {
  if (!hl || hl.t1 - hl.t0 < 1e-6 || !S.project) { hlBar.hidden = true; return; }
  let top = Infinity;
  for (const id of hl.clips) {
    const c = S.project.clips.find((k) => k.id === id);
    if (c && trackIndex(c.track) >= 0) top = Math.min(top, clipRect(c, trackIndex(c.track)).y);
  }
  const r = canvas.getBoundingClientRect();
  const x = beatToX(hl.t1);
  if (!isFinite(top) || x < 0 || x > view.w) { hlBar.hidden = true; return; }
  hlBar.hidden = false;
  hlBar.querySelector("#hlInfo").textContent = `${(hl.t1 - hl.t0).toFixed(2)} beats · ${hl.clips.size} clip${hl.clips.size > 1 ? "s" : ""}`;
  hlBar.querySelector("#hlKeep").classList.toggle("on", hlKeepPitch);
  hlBar.querySelector("#hlVari").classList.toggle("on", !hlKeepPitch);
  const bw = hlBar.offsetWidth || 300;
  hlBar.style.left = Math.max(r.left, Math.min(r.left + x + 8, r.right - bw - 4)) + "px";
  hlBar.style.top = Math.max(r.top + RULER + 2, r.top + top - 26) + "px";
}

function drawHighlight() {
  if (!hl) return;
  const edge = hl.t1 - hl.t0 > 1e-6;
  for (const id of hl.clips) {
    const c = S.project.clips.find((k) => k.id === id);
    if (!c) continue;
    const ti = trackIndex(c.track);
    if (ti < 0) continue;
    const r = clipRect(c, ti);
    const xa = beatToX(Math.max(hl.t0, c.start)), xb = beatToX(Math.min(hl.t1, c.start + c.len));
    if (xb <= xa && edge) continue;
    ctx.fillStyle = "rgba(255,255,255,.14)";
    ctx.fillRect(xa, r.y, Math.max(1, xb - xa), r.h);
    ctx.strokeStyle = "rgba(255,255,255,.75)"; ctx.lineWidth = 1;
    ctx.strokeRect(Math.round(xa) + 0.5, r.y + 0.5, Math.max(1, Math.round(xb - xa) - 1), r.h - 1);
    if (edge && hl.t1 <= c.start + c.len + 1e-6) {       // the stretch handle
      ctx.fillStyle = "#e8c06e";
      ctx.fillRect(Math.round(xb) - 1, r.y, 3, r.h);
      const my = r.y + r.h / 2;
      ctx.beginPath(); ctx.moveTo(xb + 2, my - 6); ctx.lineTo(xb + 8, my); ctx.lineTo(xb + 2, my + 6); ctx.closePath(); ctx.fill();
    }
  }
}

// ---------------------------------------------------------------- your sections (project markers)
const sectionsOf = (p) => (p?.sections || []).slice().sort((a, b) => a.beat - b.beat);
const barSnap = (beat) => Math.max(0, Math.round(beat / bpb()) * bpb());      // a section always starts on a downbeat
function sectionAt(beat) {
  const L = sectionsOf(S.project);
  let k = -1;
  for (let i = 0; i < L.length; i++) if (L[i].beat <= beat + 1e-6) k = i;
  return k;
}
function sectionEdgeAt(x) {                         // index of the section whose start is under the mouse
  const L = sectionsOf(S.project);
  return L.findIndex((m) => Math.abs(beatToX(m.beat) - x) <= 5);
}
function drawSections(W, H) {
  const L = sectionsOf(S.project);
  if (!L.length) {
    ctx.fillStyle = "rgba(255,255,255,0.28)"; ctx.font = "10.5px Segoe UI, system-ui, sans-serif";
    ctx.fillText("double-click here to mark sections (intro, verse, chorus…) - they snap to bar lines", 8, 12.5);
    return;
  }
  const end = Math.max(contentBeats(), L[L.length - 1].beat + bpb());
  const seen = {};
  L.forEach((m, i) => {
    const x0 = beatToX(m.beat), x1 = beatToX(i + 1 < L.length ? L[i + 1].beat : end);
    seen[m.name] = (seen[m.name] || 0) + 1;
    if (x1 < 0 || x0 > W) return;
    const col = SECTION_COLORS[m.name] || "#5d82b0";
    ctx.fillStyle = col + "c0"; ctx.fillRect(x0 + 1, 2, Math.max(1, x1 - x0 - 2), SECLANE - 4);
    ctx.fillStyle = "rgba(255,255,255,0.85)"; ctx.fillRect(Math.round(x0), 0, 2, SECLANE);
    ctx.fillStyle = "rgba(255,255,255,0.10)"; ctx.fillRect(Math.round(x0), RULER, 1, H - RULER);   // its line through the tracks
    const label = `${m.name} ${seen[m.name]}`;
    ctx.font = "600 10.5px Segoe UI, system-ui, sans-serif"; ctx.fillStyle = "rgba(255,255,255,0.95)";
    if (x1 - x0 > ctx.measureText(label).width + 10) ctx.fillText(label, Math.max(x0, 0) + 6, 13);
  });
}
function setSections(fn, label) {
  edit((p) => { p.sections = sectionsOf(p); fn(p.sections); p.sections.sort((a, b) => a.beat - b.beat); }, { label });
}
function sectionMenu(e, beat) {
  const L = sectionsOf(S.project), k = sectionAt(beat), at = barSnap(beat);
  const bar = Math.round(at / bpb()) + 1;
  const here = L.findIndex((m) => Math.abs(m.beat - at) < 1e-6);
  const items = [[`Start a section at bar ${bar}:`, null],
    ...SECTION_NAMES.map((n) => [`  ${n}`, () => setSections((A) => { const i = A.findIndex((m) => Math.abs(m.beat - at) < 1e-6); if (i >= 0) A[i].name = n; else A.push({ beat: at, name: n }); }, "Add section")])];
  if (k >= 0) {
    const m = L[k];
    items.push("-", [`This section (${m.name}, bar ${Math.round(m.beat / bpb()) + 1}) - call it:`, null],
      ...SECTION_NAMES.filter((n) => n !== m.name).map((n) => [`  ${n}`, () => setSections((A) => { A[k].name = n; }, "Rename section")]),
      [`Remove this section's start`, () => setSections((A) => { A.splice(k, 1); }, "Remove section")]);
  }
  if (L.length) items.push("-", ["Clear all sections", () => edit((p) => { p.sections = []; }, { label: "Clear sections" })]);
  showMenu(e, items);
  void here;
}

function draw() {
  const W = view.w, H = view.h;
  ctx.clearRect(0, 0, W, H);
  const p = S.project;
  if (!p) return;
  const l = layout();
  const ppb = view.pxPerBeat, bp = p.bpb;
  const b0 = Math.floor(xToBeat(0)), b1 = Math.ceil(xToBeat(W));

  // lanes
  p.tracks.forEach((t, i) => {
    const y = trackY(i), h = trackH(i);
    if (y > H || y + h < RULER) return;
    ctx.fillStyle = i % 2 ? "rgba(255,255,255,0.012)" : "rgba(255,255,255,0.03)";
    ctx.fillRect(0, y, W, h);
    if (l.lanes[i] > 1) {                           // faint separators between stacked layers
      const lh = (h - 6) / l.lanes[i];
      ctx.fillStyle = "rgba(255,255,255,0.045)";
      for (let k = 1; k < l.lanes[i]; k++) ctx.fillRect(0, y + 3 + k * lh - 1.5, W, 1);
    }
    ctx.fillStyle = "rgba(255,255,255,0.06)"; ctx.fillRect(0, y + h - 1, W, 1);
  });

  // grid
  const barPx = ppb * bp;
  let step = 1;
  while (barPx * step < 34) step *= 2;
  for (let b = Math.max(0, b0 - (b0 % bp)); b <= b1; b++) {
    const x = Math.round(beatToX(b)) + 0.5;
    const isBar = b % bp === 0, bar = b / bp;
    if (!isBar && ppb < 7) continue;
    if (isBar && bar % step !== 0 && ppb * bp < 20) continue;
    ctx.fillStyle = isBar ? (bar % (4 * step) === 0 ? "rgba(255,255,255,0.17)" : "rgba(255,255,255,0.095)") : "rgba(255,255,255,0.055)";
    ctx.fillRect(x, RULER, 1, H - RULER);
    if (ppb >= 120) {                                  // zoomed in: sixteenths (and 64ths when very close)
      const sub = ppb >= 1200 ? 16 : 4;
      for (let k = 1; k < sub; k++) {
        const xs = Math.round(beatToX(b + k / sub)) + 0.5;
        if (xs < 0 || xs > W) continue;
        ctx.fillStyle = k % (sub / 4) === 0 ? "rgba(255,255,255,0.032)" : "rgba(255,255,255,0.016)";
        ctx.fillRect(xs, RULER, 1, H - RULER);
      }
    }
  }

  // loop region shading
  const lp = p.loop;
  if (lp && lp.end > lp.start) {
    const x0 = beatToX(lp.start), x1 = beatToX(lp.end);
    ctx.fillStyle = lp.on ? "rgba(30,215,96,0.07)" : "rgba(30,215,96,0.025)";
    ctx.fillRect(x0, RULER, x1 - x0, H - RULER);
  }

  // clips
  const waiting = S.waiting || new Set(), preview = S.preview || new Set();
  for (const c of p.clips) {
    const ti = trackIndex(c.track);
    if (ti < 0) continue;
    const r = clipRect(c, ti);
    const { x0, x1, y, h } = r;
    if (x1 < 0 || x0 > W || y > H || y + h < RULER) continue;
    const col = STEM_COLORS[c.stem] || p.tracks[ti].color;
    const sel = S.selClips.has(c.id);
    const rate = rateOf(c), pitch = c.pitch || 0;
    const head = h >= 40 ? 15 : 12;
    ctx.save();
    ctx.beginPath(); roundRect(ctx, x0, y, Math.max(2, x1 - x0), h, 5); ctx.clip();
    ctx.fillStyle = hexA(col, c.mute ? 0.04 : sel ? 0.2 : 0.11); ctx.fillRect(x0, y, x1 - x0, h);
    ctx.fillStyle = hexA(col, c.mute ? 0.12 : 0.4); ctx.fillRect(x0, y, x1 - x0, head);
    // waveform (only the tiles on screen)
    {
      const mid = y + head + (h - head) / 2, amp = (h - head - 4) / 2 * Math.pow(10, (c.gain_db || 0) / 40) * waveGain;
      ctx.fillStyle = hexA(col, c.mute ? 0.14 : 0.42);
      const ncol = Math.max(1, Math.ceil(c.len * ppb));
      const i0 = Math.max(0, Math.floor(-x0)), i1 = Math.min(ncol, Math.ceil(W - x0));
      for (let tile = Math.floor(i0 / WAVE_TILE); tile * WAVE_TILE < i1; tile++) {
        const cols = clipWaveTile(c, tile);
        if (!cols) break;
        const base = tile * WAVE_TILE, a = Math.max(i0, base), b = Math.min(i1, base + cols.length / 2);
        for (let i = a; i < b; i++) {
          const lo = cols[(i - base) * 2], hi = cols[(i - base) * 2 + 1];
          ctx.fillRect(x0 + i, mid - hi * amp, 1, Math.max(1, (hi - lo) * amp));
        }
        if (ppb >= 40 && !c.mute) {                     // zoomed in: a crisp outline so onsets stand out
          ctx.fillStyle = hexA(col, 0.95);
          for (let i = a; i < b; i++) {
            const lo = cols[(i - base) * 2], hi = cols[(i - base) * 2 + 1];
            ctx.fillRect(x0 + i, mid - hi * amp - 0.5, 1, 1.25);
            ctx.fillRect(x0 + i, mid - lo * amp - 0.75, 1, 1.25);
          }
          ctx.fillStyle = hexA(col, c.mute ? 0.14 : 0.42);
        }
      }
    }
    // the song's own beats where they actually are in the audio: a short tick per beat and a tick + faint line
    // per downbeat. White = on the project grid, red = off it. Where the detector isn't sure, the marks are amber
    // and a dashed amber strip says so - check those by ear (Marks to fix them).
    const mk = clipMarks(c);
    const dense = mk.length > 1 && (mk[1].x - mk[0].x) * ppb > 6;
    let run = null;
    const endRun = (xe) => {
      if (run == null) return;
      const xa = Math.max(beatToX(run), x0), xb = Math.min(beatToX(xe), x1);
      if (xb > xa) {
        ctx.fillStyle = "rgba(255,190,70,.10)"; ctx.fillRect(xa, y + head, xb - xa, h - head);
        ctx.fillStyle = "rgba(255,190,70,.9)";
        for (let xx = xa; xx < xb; xx += 6) ctx.fillRect(xx, y + head, Math.min(3, xb - xx), 2);
        if (h - head > 30 && xb - xa > 110) { ctx.font = "600 9px Segoe UI, system-ui, sans-serif"; ctx.fillText("detector unsure - check by ear", xa + 4, y + head + 22); }
      }
      run = null;
    };
    for (let i = 0; i < mk.length; i++) {
      const o = mk[i];
      if (!o.sure && run == null) run = o.x;
      if (o.sure && run != null) endRun(o.x);
      const show = dense || markMode || (o.down && mk.length > 1);
      if (!show) continue;
      const x = Math.round(beatToX(o.x)) + 0.5;
      if (x < x0 || x > x1) continue;
      const onGrid = o.down ? Math.abs(o.x / bp - Math.round(o.x / bp)) < 0.02 : Math.abs(o.x - Math.round(o.x)) < 0.02;
      const col = !o.sure ? "255,190,70" : onGrid ? "110,240,160" : "255,78,96";
      const hot = markMode && hoverMark && hoverMark.clip === c && hoverMark.k === o.k;
      const off = o.sure && !onGrid, wd = off ? 2 : 1, xo = off ? x - 0.5 : x;
      if (o.down) {                                    // downbeat: a tick at the top + a line down the clip
        ctx.fillStyle = `rgba(${col},${off ? 1 : o.sure ? 0.95 : 0.85})`; ctx.fillRect(xo, y + head, wd, off ? 10 : 7);
        ctx.fillStyle = `rgba(${col},${off ? 0.6 : o.sure ? 0.26 : 0.3})`; ctx.fillRect(xo, y + head + (off ? 10 : 7), wd, h - head - (off ? 10 : 7));
        if (off) { ctx.beginPath(); ctx.moveTo(xo - 3, y + h); ctx.lineTo(xo + wd + 3, y + h); ctx.lineTo(xo + wd / 2, y + h - 5); ctx.closePath(); ctx.fill(); }
      } else if (dense || markMode) {                  // beat: a short tick (longer + a faint line when it's off)
        ctx.fillStyle = `rgba(${col},${off ? 1 : o.sure ? 0.7 : 0.7})`; ctx.fillRect(xo, y + head, wd, off ? 7 : 4);
        if (off) { ctx.fillStyle = `rgba(${col},0.3)`; ctx.fillRect(xo, y + head + 7, wd, h - head - 7); }
      }
      if (markMode) {                                   // marks mode: every mark is a grabbable line + handle
        ctx.fillStyle = hot ? "#e8c06e" : `rgba(${col},.35)`;
        ctx.fillRect(x, y + head, 1, h - head);
        ctx.fillRect(x - (hot ? 2 : 1), y + h - 7, hot ? 5 : 3, 6);
      }
    }
    if (run != null) endRun(mk[mk.length - 1].x + 1);
    // beats you're tapping right now
    if (tap && tap.info === resolved.get(c.song)) {
      const L = landFor(c.song), ps = clipPieces(c);
      ctx.fillStyle = "#5fe0ff";
      for (const t of tap.times) { const xx = beatToX(c.start + uAtW(ps, L.tw(t))); if (xx >= x0 && xx <= x1) ctx.fillRect(Math.round(xx) - 1, y + head, 3, h - head); }
    }
    // parts the detector found at another tempo (they play as recorded, like the rest): a thin strip + label
    for (const t of clipTempoParts(c)) {
      const xa = beatToX(t.x0), xb = beatToX(t.x1);
      if (xb < x0 || xa > x1) continue;
      ctx.fillStyle = "rgba(120,200,255,.55)";
      ctx.fillRect(Math.max(xa, x0), y + h - 3, Math.max(2, Math.min(xb, x1) - Math.max(xa, x0)), 3);
      if (h - head > 24 && xb - xa > 60) {
        ctx.font = "600 9px Segoe UI, system-ui, sans-serif";
        ctx.fillText(`detected ${t.bpm.toFixed(1)} BPM`, Math.max(xa, x0) + 4, y + h - 6);
      }
    }
    // parts you stretched / contracted (highlight + drag): a violet band with the factor
    if (c.warps) {
      for (const pc of clipPieces(c)) {
        if (Math.abs(pc.r - 1) < 1e-6) continue;
        const xa = beatToX(c.start + pc.u0), xb = beatToX(c.start + pc.u1);
        if (xb < x0 || xa > x1) continue;
        ctx.fillStyle = "rgba(150,170,200,.17)";
        ctx.fillRect(Math.max(xa, x0), y + head, Math.max(1, Math.min(xb, x1) - Math.max(xa, x0)), h - head);
        ctx.fillStyle = "rgba(150,170,200,1)";
        ctx.fillRect(Math.round(xa), y + head, 2, h - head);
        ctx.fillRect(Math.round(xb) - 1, y + head, 1, h - head);
        if (h - head > 30 && xb - xa > 40) {
          const kp = pc.kp ?? (c.keep_pitch !== false);
          ctx.font = "600 9px Segoe UI, system-ui, sans-serif";
          ctx.fillText(`${1 / pc.r >= 1 ? "stretched" : "contracted"} ×${(1 / pc.r).toFixed(2)} · ${kp ? "pitch kept" : "pitch follows"}`, Math.round(xa) + 4, y + h - 5);
        }
      }
    }
    // lyric words under a vocal clip (toggle: Lyrics) - each word at its onset beat
    if (lyricsOverlay && c.stem === "vocals" && h - head > 18) {
      const words = clipLyricWords(c);
      if (words.length) {
        ctx.font = "10px Segoe UI, system-ui, sans-serif";
        ctx.fillStyle = "rgba(255,255,255,.82)";
        const ly = y + h - 4;
        let lastEnd = -1e9;
        for (const o of words) {
          const wx = beatToX(o.x);
          if (wx < x0 - 2 || wx > x1) continue;
          if (wx < lastEnd + 2) continue;                   // skip words that would overlap at this zoom
          ctx.fillStyle = "rgba(0,0,0,.45)"; ctx.fillRect(wx, ly - 9, Math.min(x1 - wx, ctx.measureText(o.w).width + 3), 11);
          ctx.fillStyle = "rgba(255,255,255,.9)"; ctx.fillText(o.w, wx + 1.5, ly);
          lastEnd = wx + ctx.measureText(o.w).width + 2;
        }
      }
    }
    // automation sweeps on this track (project.track.automation; design D / B.4) - a tinted band + label at
    // each sweep event so the pre-rendered time-varying FX are visible on the timeline
    {
      const trk = p.tracks[ti];
      for (const ev of automationSpans(trk, c)) {
        if (ev.x1 < x0 || ev.x0 > x1) continue;
        const xa = Math.max(ev.x0, x0), xb = Math.min(ev.x1, x1);
        ctx.fillStyle = "rgba(95,224,255,.14)"; ctx.fillRect(xa, y + head, Math.max(2, xb - xa), h - head);
        ctx.fillStyle = "rgba(95,224,255,.9)"; ctx.fillRect(Math.round(ev.x0), y + head, 2, h - head);
        if (h - head > 22 && xb - xa > 44) {
          ctx.font = "600 9px Segoe UI, system-ui, sans-serif";
          ctx.fillText(ev.label, xa + 4, y + head + 11);
        }
      }
    }
    // fades
    ctx.strokeStyle = "rgba(255,255,255,.55)"; ctx.lineWidth = 1;
    if (c.fade_in > 0.02) { const fx = c.fade_in * ppb; ctx.beginPath(); ctx.moveTo(x0, y + h); ctx.quadraticCurveTo(x0 + fx * 0.1, y + head + 1, x0 + fx, y + head); ctx.stroke(); }
    if (c.fade_out > 0.06) { const fx = c.fade_out * ppb; ctx.beginPath(); ctx.moveTo(x1, y + h); ctx.quadraticCurveTo(x1 - fx * 0.1, y + head + 1, x1 - fx, y + head); ctx.stroke(); }
    if (waiting.has(c.id)) { ctx.fillStyle = hatchPattern(); ctx.fillRect(x0, y, x1 - x0, h); }
    // label
    ctx.fillStyle = "rgba(255,255,255,.93)";
    ctx.font = `600 ${head > 12 ? 11 : 10}px Segoe UI, system-ui, sans-serif`;
    const name = S.songs.get(c.song)?.title || c.song;
    const tags = [];
    if (Math.abs(rate - 1) > 1e-4) tags.push(`×${rate.toFixed(3)}${c.keep_pitch === false ? " vari" : ""}`);
    if (Math.abs(pitch) > 0.004) tags.push(`♪${fmtSt(pitch)}`);
    if (c.gain_db) tags.push(`${c.gain_db > 0 ? "+" : ""}${c.gain_db}dB`);
    if (isLocked(c.song)) tags.push("locked");
    else if (S.project.songs[c.song]?.marks) tags.push("this project's timing");
    else if (resolved.get(c.song)?.user_marks) tags.push("your grid");
    if (waiting.has(c.id)) tags.push("warping… (playing preview)");
    else if (preview.has(c.id)) tags.push("HQ…");
    ctx.fillText(`${c.sec ? c.sec + " · " : ""}${c.stem} · ${name}${tags.length ? "   " + tags.join("  ") : ""}`, Math.max(x0, 0) + 5, y + head - 3.5);
    ctx.restore();
    // outline + handles
    ctx.strokeStyle = sel ? "#fff" : hexA(col, 0.55);
    ctx.lineWidth = sel ? 2 : 1;
    ctx.beginPath(); roundRect(ctx, x0 + 0.5, y + 0.5, Math.max(2, x1 - x0 - 1), h - 1, 5); ctx.stroke();
    const hot = hover && hover.clip === c;
    if (hot || sel) {
      // fade squares (top corners) + stretch grips (bottom corners)
      ctx.fillStyle = "#fff";
      ctx.fillRect(x0 + (c.fade_in || 0) * ppb + 2, y + 2, 6, 6);
      ctx.fillRect(x1 - (c.fade_out || 0) * ppb - 8, y + 2, 6, 6);
      if (x1 - x0 > 16) {
        for (const [gx, dir] of [[x0, 1], [x1, -1]]) {
          const on = hot && (hover.zone === (dir > 0 ? "stretchL" : "stretchR"));
          ctx.fillStyle = on ? "#e8c06e" : "rgba(255,255,255,.8)";
          ctx.beginPath();
          ctx.moveTo(gx + dir * 1, y + h - 1); ctx.lineTo(gx + dir * 12, y + h - 1); ctx.lineTo(gx + dir * 1, y + h - 12);
          ctx.closePath(); ctx.fill();
        }
      }
    }
  }

  // drop ghost (dragging a section in from the song view)
  if (S.drag && S.drag.over) {
    const d = S.drag, g = d.over;
    const mult = p.songs[d.payload.song] ? songMult(p, d.payload.song) || 1 : guessMult(d.payload.song);
    const pcs = d.payload.pieces;
    const stems = STEMS.filter((st) => pcs.some((q) => q.stems.includes(st)));
    stems.forEach((st, k) => {
      const ti = stems.length === 1 ? g.track : roleTrack(st);
      const y = ti >= 0 ? trackY(ti) + 3 : trackY(p.tracks.length + k) + 3;
      ctx.fillStyle = hexA(STEM_COLORS[st], 0.3);
      ctx.strokeStyle = STEM_COLORS[st];
      ctx.setLineDash([5, 4]);
      for (const q of pcs) {
        if (!q.stems.includes(st)) continue;
        const x = beatToX(g.beat + q.off * mult), wd = q.len * mult * ppb;
        ctx.fillRect(x, y, wd, baseH() - 7);
        ctx.strokeRect(x + 0.5, y + 0.5, wd - 1, baseH() - 8);
      }
      ctx.setLineDash([]);
    });
  }

  drawHighlight();
  updateHlBar();

  // rubber band
  if (drag && drag.type === "band") {
    ctx.fillStyle = "rgba(30,215,96,.12)"; ctx.strokeStyle = "rgba(30,215,96,.7)";
    const x = Math.min(drag.x0, drag.x), y = Math.min(drag.y0, drag.y);
    ctx.fillRect(x, y, Math.abs(drag.x - drag.x0), Math.abs(drag.y - drag.y0));
    ctx.strokeRect(x + 0.5, y + 0.5, Math.abs(drag.x - drag.x0), Math.abs(drag.y - drag.y0));
  }

  // section lane (your sections: double-click / right-click / drag) + ruler
  ctx.fillStyle = "rgba(255,255,255,0.03)"; ctx.fillRect(0, 0, W, RULER);
  ctx.fillStyle = "rgba(255,255,255,0.05)"; ctx.fillRect(0, SECLANE - 1, W, 1);
  ctx.fillStyle = "rgba(255,255,255,0.08)"; ctx.fillRect(0, RULER - 1, W, 1);
  drawSections(W, H);
  ctx.save(); ctx.translate(0, SECLANE);
  if (lp && lp.end > lp.start) {
    const x0 = beatToX(lp.start), x1 = beatToX(lp.end);
    ctx.fillStyle = lp.on ? "rgba(30,215,96,.55)" : "rgba(30,215,96,.2)";
    ctx.fillRect(x0, 3, x1 - x0, 7);
  }
  ctx.font = "11px Cascadia Mono, Consolas, monospace";
  for (let bar = Math.max(0, Math.floor(b0 / bp)); bar * bp <= b1; bar++) {
    if (bar % step) continue;
    const x = Math.round(beatToX(bar * bp)) + 0.5;
    ctx.fillStyle = "rgba(255,255,255,0.22)"; ctx.fillRect(x, 14, 1, RULER - SECLANE - 14);
    ctx.fillStyle = "#a7a7ad"; ctx.fillText(String(bar + 1), x + 3, 24);
  }
  if (ppb >= 45) {                                     // zoomed in: every beat numbered bar.beat, plus finer ticks
    ctx.font = "10px Cascadia Mono, Consolas, monospace";
    for (let b = Math.max(0, Math.floor(b0)); b <= b1; b++) {
      const x = Math.round(beatToX(b)) + 0.5;
      if (b % bp) {
        ctx.fillStyle = "rgba(255,255,255,0.16)"; ctx.fillRect(x, 18, 1, RULER - SECLANE - 18);
        ctx.fillStyle = "#77777e"; ctx.fillText(`${Math.floor(b / bp) + 1}.${(b % bp) + 1}`, x + 3, 24);
      }
      if (ppb >= 120) {
        const sub = ppb >= 1200 ? 16 : 4;
        ctx.fillStyle = "rgba(255,255,255,0.12)";
        for (let k = 1; k < sub; k++) { const xs = Math.round(beatToX(b + k / sub)) + 0.5; ctx.fillRect(xs, k % (sub / 4) ? 24 : 21, 1, RULER - SECLANE - (k % (sub / 4) ? 24 : 21)); }
      }
    }
    ctx.font = "11px Cascadia Mono, Consolas, monospace";
  }
  ctx.restore();

  // horizontal scrollbar
  const total = contentBeats() * ppb;
  if (total > W) {
    const tw = Math.max(40, W * W / total), tx = (view.scrollX / Math.max(1, total - W)) * (W - tw);
    ctx.fillStyle = "rgba(0,0,0,.22)"; ctx.fillRect(0, H - SCROLLBAR, W, SCROLLBAR);
    ctx.fillStyle = drag && drag.type === "scroll" ? "rgba(255,255,255,.34)" : "rgba(255,255,255,.18)";
    ctx.beginPath(); roundRect(ctx, tx + 1, H - SCROLLBAR + 2, tw - 2, SCROLLBAR - 4, 4); ctx.fill();
  }

  // playhead
  const pb = transportPos() / secPerBeat();
  const px = beatToX(pb);
  if (px >= -1 && px <= W + 1) {
    ctx.fillStyle = "#ffffff"; ctx.fillRect(Math.round(px), 0, 1.5, H);
    ctx.beginPath(); ctx.moveTo(px - 5, 0); ctx.lineTo(px + 6, 0); ctx.lineTo(px + 0.5, 8); ctx.closePath(); ctx.fill();
  }

  // floating hint (stretch amount, pitch, ...)
  if (hint && performance.now() - hint.t < 1400) {
    ctx.font = "600 12px Segoe UI, system-ui";
    const w = ctx.measureText(hint.text).width + 14;
    const hx = Math.min(W - w - 4, hint.x + 14), hy = Math.max(RULER + 4, hint.y - 30);
    ctx.fillStyle = "rgba(32,32,38,.95)"; ctx.strokeStyle = "rgba(255,255,255,.22)";
    ctx.beginPath(); roundRect(ctx, hx, hy, w, 22, 7); ctx.fill(); ctx.stroke();
    ctx.fillStyle = "#fff"; ctx.fillText(hint.text, hx + 7, hy + 15);
    dirty = true;
  }
}

function guessMult(sid) {
  const info = S.songs.get(sid);
  if (!info || !S.project) return 1;
  return [0.5, 1, 2].reduce((b, m) => Math.abs(Math.log(S.project.bpm / (info.bpm * m))) < Math.abs(Math.log(S.project.bpm / (info.bpm * b))) ? m : b, 1);
}
function roleTrack(stem) {
  return S.project.tracks.findIndex((t) => t.role === stem);
}
function roundRect(g, x, y, w, h, r) {
  r = Math.min(r, w / 2, h / 2);
  g.moveTo(x + r, y); g.arcTo(x + w, y, x + w, y + h, r); g.arcTo(x + w, y + h, x, y + h, r);
  g.arcTo(x, y + h, x, y, r); g.arcTo(x, y, x + w, y, r); g.closePath();
}
function hexA(hex, a) {
  const n = parseInt(hex.slice(1), 16);
  return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a})`;
}

function loop() {
  if (S.transport.playing) { ensureVisibleWhilePlaying(); dirty = true; }
  if (dirty) { dirty = false; draw(); }
  requestAnimationFrame(loop);
}
function ensureVisibleWhilePlaying() {
  if (drag || !S.follow || performance.now() - lastInteract < 3000) return;
  const pb = transportPos() / secPerBeat();
  const x = beatToX(pb);
  if (x > view.w * 0.85 || x < 0) { view.scrollX = Math.max(0, pb * view.pxPerBeat - view.w * 0.15); positionHeaders(); }
}
requestAnimationFrame(loop);

// ---------------------------------------------------------------- mouse
const CURSORS = { left: "w-resize", right: "e-resize", stretchL: "ew-resize", stretchR: "ew-resize", fadein: "col-resize", fadeout: "col-resize", move: "grab" };
canvas.addEventListener("mousemove", (e) => {
  if (drag) return;
  const h = e.offsetY > RULER ? clipAt(e.offsetX, e.offsetY) : null;
  if (markMode) {
    const m = h ? markAt(h.clip, e.offsetX) : null;
    const hm = m ? { clip: h.clip, k: m.k, x: m.x } : null;
    if ((hm?.k) !== (hoverMark?.k) || (hm?.clip) !== (hoverMark?.clip)) { hoverMark = hm; dirty = true; }
    if ((h?.clip) !== (hover?.clip)) { hover = h; dirty = true; }
    canvas.style.cursor = hm ? "ew-resize" : h ? "copy" : "default";
    canvas.title = hm ? "Drag to move this mark · Del deletes it · right-click for more" : h ? "Double-click to add a mark · right-click for the marks menu · T taps marks while playing" : "";
    return;
  }
  if (e.offsetY > RULER && hlEdgeAt(e.offsetX, e.offsetY)) {
    canvas.style.cursor = "ew-resize";
    canvas.title = "Drag to stretch / contract the highlighted part (snaps to the grid; hold Alt for smooth)";
    return;
  }
  canvas.style.cursor = e.offsetY < SECLANE ? (sectionEdgeAt(e.offsetX) >= 0 ? "ew-resize" : "pointer")
    : !h ? (e.offsetY < RULER ? "text" : "default") : CURSORS[h.zone];
  canvas.title = !h ? "" : h.zone.startsWith("stretch") ? "Drag to STRETCH (changes speed; Shift = snap end to grid)"
    : h.zone === "left" || h.zone === "right" ? "Drag to TRIM" : h.zone.startsWith("fade") ? "Drag to set the fade"
    : hl ? "Ctrl+click = add this clip to the highlight · Shift+drag = new highlight · Esc = clear"
    : "Drag to move · Shift+drag = highlight part of it (to stretch / contract) · Ctrl+drag copies · Alt = no snap · double-click for settings · keyboard button = all shortcuts";
  if ((h?.clip) !== (hover?.clip) || h?.zone !== hover?.zone) { hover = h; dirty = true; }
});
canvas.addEventListener("mouseleave", () => { hover = null; dirty = true; });

const LABELS = { move: "Move clip", left: "Trim clip", right: "Trim clip", stretchL: "Stretch clip", stretchR: "Stretch clip", fadein: "Fade in", fadeout: "Fade out", ruler: "Set loop", hlEdge: "Stretch part of a clip", section: "Move section" };

canvas.addEventListener("mousedown", (e) => {
  if (e.button !== 0 || !S.project) return;
  const x = e.offsetX, y = e.offsetY;
  lastInteract = performance.now();
  if (y > view.h - SCROLLBAR && contentBeats() * view.pxPerBeat > view.w) {
    drag = { type: "scroll", x0: x, s0: view.scrollX, moved: true };
    window.addEventListener("mousemove", onDrag);
    window.addEventListener("mouseup", onUp, { once: true });
    return;
  }
  if (markMode && y >= RULER) {
    const h = clipAt(x, y);
    if (h) {
      if (!S.selClips.has(h.clip.id)) { S.selClips.clear(); S.selClips.add(h.clip.id); emit("selection"); }
      const m = markAt(h.clip, x);
      if (m && isLocked(h.clip.song)) { lockedNote(); return; }
      if (m) {
        drag = { type: "mark", clip: h.clip.id, k: m.k, x0: x, y0: y, moved: false };
        window.addEventListener("mousemove", onDrag);
        window.addEventListener("mouseup", onUp, { once: true });
      }
      dirty = true;
      return;
    }
  }
  if (y < SECLANE) {
    const k = sectionEdgeAt(x);
    if (k >= 0) drag = { type: "section", k, x0: x, b0: sectionsOf(S.project)[k].beat, moved: false };
    else return;
  } else if (y < RULER) {
    drag = { type: "ruler", x0: x, b0: xToBeat(x), moved: false };
  } else if (hlEdgeAt(x, y)) {
    // drag the highlight's right edge: stretch / contract that part of every highlighted clip
    const orig = S.project.clips.filter((k) => hl.clips.has(k.id)).map(clone);
    drag = { type: "hlEdge", x0: x, y0: y, t0: hl.t0, t1: hl.t1, orig, b0: xToBeat(x), moved: false };
  } else if (hl && e.ctrlKey) {
    // Ctrl+click: the same span on another clip / lane joins the highlight (click again to take it out)
    const h = clipAt(x, y);
    const overlaps = (c) => c.start < hl.t1 && c.start + c.len > hl.t0;
    if (h && overlaps(h.clip)) {
      if (hl.clips.has(h.clip.id)) { if (hl.clips.size > 1) hl.clips.delete(h.clip.id); } else hl.clips.add(h.clip.id);
    } else if (!h) {
      const ti = trackAt(y);
      if (ti >= 0) for (const c of S.project.clips) if (c.track === S.project.tracks[ti].id && overlaps(c)) hl.clips.add(c.id);
    }
    dirty = true;
    return;
  } else {
    const h = clipAt(x, y);
    if (h && e.shiftKey && !e.ctrlKey && h.zone === "move") {
      // Shift+drag on a clip: highlight a span of it
      const b = Math.max(h.clip.start, Math.min(h.clip.start + h.clip.len, hlSnap(xToBeat(x), e)));
      hl = { t0: b, t1: b, clips: new Set([h.clip.id]) };
      drag = { type: "hl", clip: h.clip.id, x0: x, y0: y, b0: b, moved: false };
      dirty = true;
      window.addEventListener("mousemove", onDrag);
      window.addEventListener("mouseup", onUp, { once: true });
      return;
    }
    clearHighlight();
    if (h) {
      const c = h.clip;
      if (e.ctrlKey && !S.selClips.has(c.id)) S.selClips.add(c.id);
      else if (e.ctrlKey && h.zone === "move") { /* ctrl-drag duplicates */ }
      else if (!S.selClips.has(c.id)) { S.selClips.clear(); S.selClips.add(c.id); }
      emit("selection");
      const orig = S.project.clips.filter((k) => S.selClips.has(k.id)).map(clone);
      drag = { type: h.zone, clip: c.id, x0: x, y0: y, t0: h.ti, orig, b0: xToBeat(x), moved: false, dup: e.ctrlKey && h.zone === "move" };
    } else {
      if (!e.ctrlKey) { S.selClips.clear(); emit("selection"); }
      drag = { type: "band", x0: x, y0: y, x, y, moved: false };
    }
  }
  dirty = true;
  window.addEventListener("mousemove", onDrag);
  window.addEventListener("mouseup", onUp, { once: true });
});

function localXY(e) {
  const r = canvas.getBoundingClientRect();
  return [e.clientX - r.left, e.clientY - r.top];
}

function onDrag(e) {
  if (!drag) return;
  const [x, y] = localXY(e);
  if (Math.abs(x - drag.x0) > 3 || Math.abs(y - (drag.y0 ?? y)) > 3) drag.moved = true;
  lastInteract = performance.now();
  if (drag.type === "scroll") {
    const total = contentBeats() * view.pxPerBeat, tw = Math.max(40, view.w * view.w / total);
    view.scrollX = drag.s0 + (x - drag.x0) * (total - view.w) / Math.max(1, view.w - tw);
    clampScroll(); dirty = true;
    return;
  }
  if (!drag.moved) return;
  if (drag.type === "mark") {
    const c = S.project.clips.find((k) => k.id === drag.clip), info = c && resolved.get(c.song);
    const t = c ? timeAt(c, Math.max(c.start, Math.min(c.start + c.len, xToBeat(x)))) : null;
    if (info && t != null) editProjectMarks(info, (m) => moveMark(m, drag.k, t), { label: "Move mark", coalesce: "mark" + dragSerial });
    if (t != null) showHint(`mark at ${t.toFixed(3)} s in the song`, x, y);
    dirty = true;
    return;
  }
  S.interactive = true;                               // server: preview only while dragging
  const p = S.project;
  const db = xToBeat(x) - drag.b0;
  if (drag.type === "hl") {
    const c = p.clips.find((k) => k.id === drag.clip);
    if (c) {
      const b = Math.max(c.start, Math.min(c.start + c.len, hlSnap(xToBeat(x), e)));
      hl.t0 = Math.min(drag.b0, b); hl.t1 = Math.max(drag.b0, b);
      showHint(`${(hl.t1 - hl.t0).toFixed(2)} beats highlighted · drag its right edge to stretch · Ctrl+click other lanes to add them`, x, y);
    }
    dirty = true;
    return;
  }
  if (drag.type === "hlEdge") {
    const t1 = Math.max(drag.t0 + 0.05, snapBeat(xToBeat(x), e));
    const f = (t1 - drag.t0) / (drag.t1 - drag.t0);
    doEdit((pp) => {
      for (const o of drag.orig) {
        const i = pp.clips.findIndex((k) => k.id === o.id);
        if (i < 0) continue;
        const c = clone(o);
        const u0 = Math.max(drag.t0, c.start) - c.start, u1 = Math.min(drag.t1, c.start + c.len) - c.start;
        if (u1 - u0 > 1e-6) stretchSpan(c, songMult(pp, c.song), landFor(c.song), u0, u1, (u1 - u0) * f, hlKeepPitch);
        pp.clips[i] = c;
      }
    });
    hl.t1 = t1;
    showHint(`${(t1 - drag.t0).toFixed(2)} beats (×${f.toFixed(3)}) · ${hlKeepPitch ? "pitch kept" : "pitch follows"}`, x, y);
    dirty = true;
    return;
  }
  if (drag.type === "section") {
    const L = sectionsOf(p), k = drag.k;
    const lo = k > 0 ? L[k - 1].beat + bpb() : 0, hi = k + 1 < L.length ? L[k + 1].beat - bpb() : Infinity;
    const nb = Math.min(hi, Math.max(lo, barSnap(drag.b0 + (x - drag.x0) / view.pxPerBeat)));
    doEdit((pp) => { pp.sections = sectionsOf(pp); pp.sections[k].beat = nb; });
    showHint(`bar ${Math.round(nb / bpb()) + 1}`, x, y + 20);
  } else if (drag.type === "ruler") {
    const a = snapBeat(Math.min(drag.b0, xToBeat(x)), e), b = snapBeat(Math.max(drag.b0, xToBeat(x)), e);
    if (b > a) doEdit((pp) => { pp.loop = { on: true, start: a, end: b }; });
  } else if (drag.type === "band") {
    drag.x = x; drag.y = y;
    const ba = xToBeat(Math.min(drag.x0, x)), bb = xToBeat(Math.max(drag.x0, x));
    const ya = Math.min(drag.y0, y), yb = Math.max(drag.y0, y);
    S.selClips.clear();
    for (const c of p.clips) {
      const ti = trackIndex(c.track);
      if (ti < 0) continue;
      const r = clipRect(c, ti);
      if (r.y < yb && r.y + r.h > ya && c.start < bb && c.start + c.len > ba) S.selClips.add(c.id);
    }
    emit("selection");
  } else if (drag.type === "move") {
    // snap the grabbed clip's start (not the mouse delta), so off-grid clips land on the grid
    const lead = drag.orig.find((o) => o.id === drag.clip) || drag.orig[0];
    // snap the song downbeat nearest to where you grabbed the clip to the grid (not the clip's edge, which may
    // be mid-bar, nor its first downbeat, which may be in a part that drifts) - so any part can be lined up
    if (drag.phase === undefined) drag.phase = nearestDownbeatOffset(lead, drag.b0) ?? 0;
    const phase = e.altKey || snapMode === "off" ? 0 : drag.phase;
    const d = snapBeat(lead.start + phase + db, e) - phase - lead.start;
    const tAt = trackAt(y);
    const tDelta = tAt >= 0 ? tAt - drag.t0 : 0;
    doEdit((pp) => {
      if (drag.dup && !drag.dupDone) {
        const copies = drag.orig.map((o) => ({ ...clone(o), id: uid("c") }));
        pp.clips.push(...copies);
        drag.orig = copies.map(clone);
        drag.clip = copies[0].id;
        S.selClips = new Set(copies.map((c) => c.id));
        drag.dupDone = true;
      }
      for (const o of drag.orig) {
        const c = pp.clips.find((k) => k.id === o.id);
        if (!c) continue;
        c.start = Math.max(0, o.start + d);
        const ti = Math.max(0, Math.min(pp.tracks.length - 1, trackIndex(o.track) + tDelta));
        c.track = pp.tracks[ti].id;
      }
    });
  } else if (drag.type === "left" || drag.type === "right") {
    doEdit((pp) => {
      for (const o of drag.orig) {
        const c = pp.clips.find((k) => k.id === o.id);
        if (!c) continue;
        const k = rateOf(o) / songMult(pp, c.song);         // source beats per project beat
        if (drag.type === "right") {
          const end = snapBeat(o.start + o.len + db, e);
          c.len = Math.max(0.05, end - o.start);
        } else {
          let st = snapBeat(o.start + db, e);
          st = Math.min(st, o.start + o.len - 0.05);
          const Lm = landFor(c.song), ws = wAt(Lm, o.src);
          st = Math.max(st, o.start - (ws - wAt(Lm, 0)) / k, 0);   // can't reveal before the song starts
          c.start = st; c.len = o.len - (st - o.start);
          c.src = Lm || o.warps ? Math.round(beatAtU(o, st - o.start) * 10000) / 10000 : o.src + (st - o.start) * k;
        }
      }
    });
  } else if (drag.type === "stretchL" || drag.type === "stretchR") {
    // stretch: same source material, new length -> new playback rate. Smooth unless Shift.
    // Every selected clip gets the grabbed clip's factor, each around its own anchor (start for the right
    // grip, end for the left one).
    const o = drag.orig.find((k) => k.id === drag.clip);
    if (!o) return;
    const free = !e.shiftKey;
    const right = drag.type === "stretchR";
    let f;
    if (right) f = (snapBeat(o.start + o.len + db, e, { forceFree: free }) - o.start) / o.len;
    else f = (o.start + o.len - Math.max(0, snapBeat(o.start + db, e, { forceFree: free }))) / o.len;
    let fMin = 0, fMax = Infinity;                             // keep every clip within 0.25x..4x speed
    for (const k of drag.orig) {
      const r0 = rateOf(k);
      fMin = Math.max(fMin, r0 / 4); fMax = Math.min(fMax, r0 / 0.25);
      if (!right && k.len > 0) fMax = Math.min(fMax, (k.start + k.len) / k.len);   // can't start before 0
    }
    f = Math.max(fMin, Math.min(fMax, f));
    doEdit((pp) => {
      for (const k of drag.orig) {
        const c = pp.clips.find((q) => q.id === k.id);
        if (!c) continue;
        const r0 = rateOf(k), newLen = k.len * f;
        c.len = newLen;
        c.start = right ? k.start : k.start + k.len - newLen;
        c.rate = Math.round(r0 / f * 100000) / 100000;
        if (Math.abs(c.rate - 1) < 0.0005) c.rate = 1;        // magnetic at the original speed
        c.fade_in = Math.min(k.fade_in || 0, newLen / 2);
        c.fade_out = Math.min(k.fade_out || 0, newLen / 2);
      }
    });
    const c = S.project.clips.find((k) => k.id === drag.clip);
    const many = drag.orig.length > 1 ? ` · ${drag.orig.length} clips` : "";
    if (c) showHint(`×${rateOf(c).toFixed(3)} speed · ${(60 / S.project.bpm * c.len).toFixed(2)} s${c.keep_pitch === false ? " · pitch follows" : " · pitch kept"}${many}`, x, y);
  } else if (drag.type === "fadein" || drag.type === "fadeout") {
    const o = drag.orig.find((k) => k.id === drag.clip);
    doEdit((pp) => {
      const c = pp.clips.find((k) => k.id === drag.clip);
      if (!c) return;
      const q = (v) => e.altKey || snapMode === "off" ? v : Math.round(v * 4) / 4;
      if (drag.type === "fadein") c.fade_in = Math.max(0, Math.min(c.len - (c.fade_out || 0), q((o.fade_in || 0) + db)));
      else c.fade_out = Math.max(0, Math.min(c.len - (c.fade_in || 0), q((o.fade_out || 0) - db)));
    });
  }
  dirty = true;
}
function doEdit(fn) {
  edit(fn, { coalesce: "drag" + dragSerial, label: LABELS[drag.type] || "Edit" });
}
let dragSerial = 0;

function onUp(e) {
  window.removeEventListener("mousemove", onDrag);
  const d = drag;
  drag = null;
  dragSerial++;
  lastInteract = performance.now();
  if (S.interactive) { S.interactive = false; syncNow(); }   // now render the HQ stretch/pitch
  if (!d) return;
  if (d.type === "hl" && (!d.moved || !hl || hl.t1 - hl.t0 < 1e-6)) clearHighlight();   // a Shift+click, not a drag
  if (d.type === "ruler" && !d.moved) seekTo(Math.max(0, snapBeat(d.b0, e)) * secPerBeat());
  relayout();
}

canvas.addEventListener("dblclick", (e) => {
  if (e.offsetY < SECLANE && S.project) { sectionMenu(e, xToBeat(e.offsetX)); return; }
  if (e.offsetY < RULER) { if (S.project.loop) edit((p) => (p.loop.on = !p.loop.on), { label: "Loop on/off" }); return; }
  const h = clipAt(e.offsetX, e.offsetY);
  if (h && markMode) {
    if (isLocked(h.clip.song)) { lockedNote(); return; }
    const info = resolved.get(h.clip.song), t = timeAt(h.clip, xToBeat(e.offsetX));
    if (info && t != null && !markAt(h.clip, e.offsetX, 3)) editProjectMarks(info, (m) => addMark(m, t), { label: "Add mark" });
    return;
  }
  if (h) emit("openClip", h.clip.id);
});

canvas.addEventListener("contextmenu", (e) => {
  e.preventDefault();
  if (e.offsetY < SECLANE && S.project) { sectionMenu(e, xToBeat(e.offsetX)); return; }
  const h = clipAt(e.offsetX, e.offsetY);
  const beat = xToBeat(e.offsetX);
  if (!h) {
    showMenu(e, [
      ["Play from here", () => { seekTo(snapBeat(beat, null) * secPerBeat()); }],
      ["Set loop: this bar ×4", () => { const a = Math.floor(beat / bpb()) * bpb(); edit((p) => (p.loop = { on: true, start: a, end: a + 4 * bpb() }), { label: "Set loop" }); }],
    ]);
    return;
  }
  if (!S.selClips.has(h.clip.id)) { S.selClips.clear(); S.selClips.add(h.clip.id); emit("selection"); }
  const c = h.clip;
  const forSel = (fn, label) => () => edit((p) => { for (const k of p.clips) if (S.selClips.has(k.id)) fn(k); }, { label });
  const inHl = hl && hl.clips.has(c.id) && beat >= hl.t0 && beat <= hl.t1;
  const hlItems = !inHl ? [] : [
    [hlKeepPitch ? "Highlight: stretching keeps pitch ✓ (switch to pitch follows)" : "Highlight: pitch follows the stretch ✓ (switch to keep pitch)", () => hlSetPitch(!hlKeepPitch)],
    ["Reset the stretch in the highlight", hlReset],
    ["Clear the highlight", clearHighlight, "Esc"],
    "-",
  ];
  // beat marks: fix what the detector got wrong - marks only (the audio never moves); saved with the song, so
  // every project shows them; Ctrl+Z works
  const info = resolved.get(c.song), Lm = landFor(c.song), mk = marksFor(c.song);
  const markItems = [];
  if (info && Lm && mk && isLocked(c.song)) markItems.push(["Locked to its marks - unlock it in the arrangement bar to edit them"]);
  else if (info && Lm && mk) {
    const t = timeAt(c, beat), p = 60 / info.bpm;
    const em = (fn, label) => editProjectMarks(info, fn, { label });
    const near = markAt(c, e.offsetX, 8);
    const nextBpb = { 3: 4, 4: 6, 6: 3 }[mk.bpb] || 4;
    const t0 = timeAt(c, c.start), t1 = timeAt(c, c.start + c.len);
    markItems.push(["Beat marks - they never move the audio:"]);
    if (near) markItems.push(
      ["Delete this mark", () => em((m) => deleteMark(m, near.k), "Delete mark"), markMode ? "Del" : ""],
      ["Bars start at this mark", () => em((m) => barStart(m, near.k), "Bars start here")],
    );
    markItems.push(
      ["Shift marks from here on: +½ beat", () => em((m) => shiftMarks(m, t, 0.5 * p, false, p), "Shift marks")],
      ["Shift marks from here on: −½ beat", () => em((m) => shiftMarks(m, t, -0.5 * p, false, p), "Shift marks")],
      ["Shift marks before here: +½ beat", () => em((m) => shiftMarks(m, t, 0.5 * p, true, p), "Shift marks")],
      ["Shift marks before here: −½ beat", () => em((m) => shiftMarks(m, t, -0.5 * p, true, p), "Shift marks")],
      ["Shift marks from here on by…", () => {
        const sv = parseFloat(prompt("Shift the marks from here on by how many beats? (e.g. 0.25, -0.25, 1)", "0.25"));
        if (isFinite(sv) && sv) em((m) => shiftMarks(m, t, sv * p, false, p), "Shift marks");
      }],
      [`Beats per bar: ${mk.bpb} → ${nextBpb}`, () => em((m) => setBpb(m, nextBpb), "Beats per bar")],
      ["The marks in this clip are right (no longer unsure)", () => em((m) => confirmMarks(m, Math.min(t0, t1), Math.max(t0, t1)), "Confirm marks")],
      [markMode ? "Tap them in: play, press T on every beat" : "Edit / tap marks (Marks mode)…", () => { if (!markMode) toggleMarkMode(true); }, "M"],
    );
    if (S.project.songs[c.song]?.marks) markItems.push("-",
      ["Use the song's own grid again (drop this project's timing)", () => useSongGrid(c.song)],
      ["Make this project's timing the song's grid (projects from now on)", () => makeSongGrid(info)]);
  }
  if (markMode && markItems.length) { showMenu(e, markItems); return; }
  // transition palette: snap to this clip's nearer boundary and offer the transitions.py type set
  const boundary = Math.abs(beat - c.start) < Math.abs(beat - (c.start + c.len)) ? c.start : c.start + c.len;
  const transItems = TRANSITION_TYPES.map((t) =>
    [`  ${t.label}`, () => addTransition(c, boundary, t), ""]);
  const tools = songToolItems(c.song, { head: false });
  showMenu(e, [
    ...hlItems,
    [`${(S.songs.get(c.song)?.title || c.song).slice(0, 40)}: all its tools ▸`, () => showMenu(e, songToolItems(c.song))],
    ...tools.slice(0, 2),
    ...(markItems.length ? [["Fix beat marks here (this project) ▸", () => showMenu(e, markItems)]] : []),
    "-",
    [`Transition at ${Math.abs(beat - c.start) < Math.abs(beat - (c.start + c.len)) ? "clip start" : "clip end"} ▸`,
      () => showMenu(e, [["Add a transition (baked when rendered):"], ...transItems,
        ...(trackHasTransition(c.track, boundary) ? ["-", ["  Clear the transition here", () => clearTransition(c, boundary)]] : [])])],
    "-",
    ["Edit in sample editor…", () => openSampleEditor(c.id), "pop-out"],
    ["Split here", () => splitAt(snapBeat(beat, e)), "S"],
    ["Duplicate after", duplicateSel, "Ctrl+D"],
    ["Delete", deleteSel, "Del"],
    "-",
    [c.mute ? "Unmute clip" : "Mute clip", forSel((k) => (k.mute = !c.mute), "Mute clip")],
    ["Gain +3 dB", () => gainSel(3)], ["Gain −3 dB", () => gainSel(-3)],
    ["Fade in 1 bar", forSel((k) => (k.fade_in = bpb()), "Fade in")],
    ["Fade out 1 bar", forSel((k) => (k.fade_out = bpb()), "Fade out")],
    ["Clear fades", forSel((k) => { k.fade_in = 0; k.fade_out = 0.05; }, "Clear fades")],
    "-",
    [`Pitch +1 st`, forSel((k) => (k.pitch = Math.round(((k.pitch || 0) + 1) * 100) / 100), "Pitch"), "Alt+wheel"],
    [`Pitch −1 st`, forSel((k) => (k.pitch = Math.round(((k.pitch || 0) - 1) * 100) / 100), "Pitch")],
    [c.keep_pitch === false ? "Stretch: keep pitch (currently: pitch follows speed)" : "Stretch: pitch follows speed (currently: keep pitch)",
      forSel((k) => (k.keep_pitch = c.keep_pitch === false), "Keep pitch")],
    ["Reset speed & pitch", forSel((k) => {
      const r = rateOf(k); k.len = k.len * r; k.rate = 1; k.pitch = 0;
    }, "Reset stretch")],
    "-",
    ["Align downbeat to nearest bar", alignSelectionToGrid, "A"],
    ["Loop this clip", () => edit((p) => (p.loop = { on: true, start: c.start, end: c.start + c.len }), { label: "Set loop" })],
    ["Clip settings…", () => emit("openClip", c.id)],
  ]);
});

// ---------------------------------------------------------------- shortcuts / how-to
const SHORTCUTS = [
  ["Stretch / contract part of a clip (no slicing)", [
    ["Shift + drag on a clip", "highlight a span of it (snaps to the grid; hold Alt too for smooth)"],
    ["Ctrl + click a clip on another lane", "highlight the same span there too (Ctrl+click again to take it out); Ctrl+click an empty spot in a lane = every clip of that lane under the span"],
    ["Drag the highlight's yellow right edge", "stretch / contract the highlighted part of every highlighted clip; what follows in each clip moves, the clip's end moves with it (snaps; Alt = smooth, Shift = bars)"],
    ["Keep pitch / Pitch follows (toolbar by the highlight)", "time-stretch, or turntable-style (slower = lower); also changes parts you already stretched"],
    ["Reset (toolbar) / right-click in the highlight", "back to normal speed inside the highlight"],
    ["Esc / ✕", "clear the highlight"],
    ["Violet band on a clip", "a part you stretched / contracted (factor + pitch mode shown)"],
  ]],
  ["Beat marks on clips", [
    ["Songs play as recorded", "one fixed speed per song (project tempo ÷ the song's tempo); nothing is stretched beat by beat"],
    ["Short tick / tick + line", "a beat / a downbeat of the song, where it really is in the audio"],
    ["White / red", "on the project grid / off it - where a song drifts from the grid, the marks turn red"],
    ["Amber marks + dashed amber strip", "the beat detector isn't sure there, the marks are a guess (e.g. over a silent intro), or MuScriptor's notes disagree with them (library right-click → MuScriptor second opinion) - check by ear; snapping ignores those marks"],
    ["Thin blue strip", "a part the detector found at another tempo (shown in BPM); it plays as recorded like the rest"],
    ["Right-click → Fix beat marks here ▸", "shift the marks from here on / before here by ½ beat (or any amount), bars start at a mark, beats per bar, 'these marks are right', back to the detector's"],
    ["M  or  the Marks button", "marks mode: drag a mark to move it, double-click a clip to add one, Del deletes the mark under the mouse, right-click = marks menu, M / Esc = done"],
    ["T (in marks mode, while playing)", "tap along on every beat of the selected clip's song: your taps replace the marks they cover (2 s without a tap = done)"],
    ["Your marks", "saved with the song, so every project shows them (the clip says 'your marks'); they never move the audio; Ctrl+Z undoes every change"],
    ["Lock button next to a song (arrangement bar)", "lock it to its marks: its stems are warped once so every mark sits on a steady grid, then the lock button toggles locked / as recorded instantly (the clip says 'locked'); for songs whose own timing drifts"],
  ]],
  ["Clips", [
    ["Drag", "move: the song downbeat nearest to where you grabbed it snaps to the grid (Alt = no snap, Shift = bars)"],
    ["Ctrl + drag", "copy"], ["Ctrl + click", "add to the selection"], ["Drag on empty space", "select with a box"],
    ["Drag a clip's left / right edge", "trim"], ["Drag a bottom corner triangle", "stretch the whole clip (changes its speed); with several clips selected, all of them by the same amount"],
    ["Drag a top corner square", "fade in / out"], ["Double-click", "clip settings (speed, pitch, fades, slip)"],
    ["Alt + wheel over a clip", "fine pitch (0.1 st; + Shift = 1 st)"], ["Right-click", "clip menu"],
    ["Del / Backspace", "delete selected"], ["Ctrl + D", "duplicate after"], ["Ctrl + A", "select all"],
    ["S", "split selected clips at the playhead"], ["A", "align the selection's downbeats to the nearest bar"],
    ["← / →", "nudge selected clips 1 beat (+ Shift = 1 bar)"],
  ]],
  ["Playback & view", [
    ["Space", "play / stop"], ["Home", "back to the start"], ["Click the ruler", "jump there"],
    ["Drag on the ruler", "set the loop"], ["L", "loop on / off"], ["P", "follow the playhead on / off"],
    ["Ctrl + wheel", "zoom"], ["+ / −", "zoom"], ["F", "fit the arrangement"], ["Shift + wheel", "taller / shorter tracks"], ["Ctrl + Shift + wheel", "magnify the waveforms (display only) - quiet word onsets show; or the wave ×1 button"], ["Zoomed far in", "snapping steps aside when its grid is wider than the screen detail, so splits land exactly where you point (Alt also turns snapping off)"],
    ["Wheel", "scroll"],
  ]],
  ["Project", [["Ctrl + Z / Ctrl + Y", "undo / redo"], ["Ctrl + S", "save"]]],
  ["Library & song view", [
    ["Ctrl / Shift + click songs", "select several (e.g. to put them in a folder)"],
    ["Drop audio files anywhere", "add songs"], ["Right-click a song", "re-analyze, re-separate, delete, …"],
    ["Drag on the song's ruler", "select bars (Shift + drag over the stems too)"], ["Click a selection", "hear it"],
    ["Drag a selection to the timeline", "add it as clips"], ["Ctrl + wheel", "zoom the song view"],
    ["Whole song: all / drums / bass / other / vocals", "drag (or click: at the playhead) - the whole stem, one clip per section"],
  ]],
];
function showShortcuts() {
  const esc = (t) => String(t).replace(/[&<>]/g, (m) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[m]));
  modal("Shortcuts & mouse moves", SHORTCUTS.map(([title, rows]) => `<h4 style="margin:12px 0 6px">${esc(title)}</h4>
    <table style="width:100%;border-collapse:collapse">${rows.map(([k, v]) => `<tr><td style="padding:3px 12px 3px 0;white-space:nowrap;vertical-align:top;color:var(--accent,#1ed760)">${esc(k)}</td><td style="padding:3px 0;color:var(--dim,#9a9da4)">${esc(v)}</td></tr>`).join("")}</table>`).join(""), { wide: true });
}

// ---------------------------------------------------------------- clip commands (also used by shortcuts)
// ---------------------------------------------------------------- transition palette (FEAT-005)
// Author a transition event into the clip's track `automation` lane (the ADDITIVE project field). The backend
// transitions.py renderer bakes the matching time-varying FX clip (sweep / noise sweep) when the project is
// rendered; hard cut / crossfade / echo / fill are metadata the planner / mixer act on. One event per
// (type-less) boundary per track: a new one replaces whatever was there.
const SNAP_EPS = 1e-4;
function trackHasTransition(trackId, boundary) {
  const t = S.project?.tracks.find((k) => k.id === trackId);
  return !!(t && Array.isArray(t.automation) && t.automation.some((ev) => Math.abs((ev.beat ?? -1) - boundary) < SNAP_EPS));
}
function addTransition(c, boundary, t) {
  const at = snapBeat(boundary, null);
  edit((p) => {
    const trk = p.tracks.find((k) => k.id === c.track);
    if (!trk) return;
    const lane = (trk.automation ||= []);
    const i = lane.findIndex((ev) => Math.abs((ev.beat ?? -1) - at) < SNAP_EPS);
    const ev = { kind: t.kind, type: t.type, beat: at, bars: 1 };
    if (i >= 0) lane[i] = ev; else lane.push(ev);
    lane.sort((a, b) => (a.beat ?? 0) - (b.beat ?? 0));
  }, { label: `Transition: ${t.label}` });
  toast(`${t.label} at bar ${(at / bpb() + 1).toFixed(1)} - baked into the render`);
}
function clearTransition(c, boundary) {
  edit((p) => {
    const trk = p.tracks.find((k) => k.id === c.track);
    if (!trk || !Array.isArray(trk.automation)) return;
    trk.automation = trk.automation.filter((ev) => Math.abs((ev.beat ?? -1) - boundary) >= SNAP_EPS);
    if (!trk.automation.length) delete trk.automation;      // keep it absent -> byte-identical when empty
  }, { label: "Clear transition" });
}

export function deleteSel() {
  if (!S.selClips.size) return;
  edit((p) => { p.clips = p.clips.filter((c) => !S.selClips.has(c.id)); }, { label: `Delete ${S.selClips.size > 1 ? S.selClips.size + " clips" : "clip"}` });
  S.selClips.clear(); emit("selection");
}
export function duplicateSel() {
  if (!S.selClips.size) return;
  edit((p) => {
    const sel = p.clips.filter((c) => S.selClips.has(c.id));
    const a = Math.min(...sel.map((c) => c.start)), b = Math.max(...sel.map((c) => c.start + c.len));
    const span = Math.ceil((b - a) / bpb()) * bpb();
    const copies = sel.map((c) => ({ ...clone(c), id: uid("c"), start: c.start + span }));
    p.clips.push(...copies);
    S.selClips = new Set(copies.map((c) => c.id));
  }, { label: "Duplicate" });
  emit("selection");
}
export function splitAt(beat) {
  edit((p) => {
    const add = [];
    for (const c of p.clips) {
      if (!S.selClips.has(c.id) || beat <= c.start + 0.01 || beat >= c.start + c.len - 0.01) continue;
      const k = rateOf(c) / songMult(p, c.song);
      const src = landFor(c.song) || c.warps ? Math.round(beatAtU(c, beat - c.start) * 10000) / 10000 : c.src + (beat - c.start) * k;
      const right = { ...clone(c), id: uid("c"), start: beat, len: c.start + c.len - beat, src, fade_in: 0 };
      c.len = beat - c.start; c.fade_out = 0.05;
      add.push(right);
    }
    p.clips.push(...add);
  }, { label: "Split" });
}
export function splitAtPlayhead() { splitAt(transportPos() / secPerBeat()); }
function gainSel(d) {
  edit((p) => { for (const c of p.clips) if (S.selClips.has(c.id)) c.gain_db = Math.round(((c.gain_db || 0) + d) * 10) / 10; }, { label: "Clip gain" });
}
export function nudgeSel(beats) {
  if (!S.selClips.size) return;
  edit((p) => { for (const c of p.clips) if (S.selClips.has(c.id)) c.start = Math.max(0, c.start + beats); }, { coalesce: "nudge", label: "Nudge" });
}

// ---------------------------------------------------------------- drop from the song view
export function dragOver(clientX, clientY) {
  const r = canvas.getBoundingClientRect();
  const x = clientX - r.left, y = clientY - r.top;
  if (!S.drag) return false;
  if (x < 0 || y < RULER || x > r.width || y > r.height) { if (S.drag.over) { S.drag.over = null; dirty = true; } return false; }
  const b = xToBeat(x - (S.drag.grabDx || 0));
  S.drag.over = { beat: Math.max(0, S.drag.fine ? snapBeat(b, null) : Math.round(b / bpb()) * bpb()), track: trackAt(y) };
  dirty = true;
  return true;
}
export function dropPayload() {
  const d = S.drag;
  if (!d || !d.over) return false;
  // pieces: [{src, off, len, name, stems}] - src = the song's beat index, off / len in the song's beats as it plays
  // (landing): one section, or a whole song laid in section by section, each section its own clip
  const { song, pieces: secs } = d.payload;
  const g = d.over;
  const stems = STEMS.filter((st) => secs.some((q) => q.stems.includes(st)));
  const ids = [];
  edit((p) => {
    ensureSongInProject(p, song);
    const mult = songMult(p, song);
    stems.forEach((st) => {
      let ti = stems.length === 1 && g.track >= 0 ? g.track : p.tracks.findIndex((t) => t.role === st);
      if (ti < 0) {
        p.tracks.push({ id: uid("t"), name: st[0].toUpperCase() + st.slice(1), role: st, color: STEM_COLORS[st], gain_db: 0, pan: 0, mute: false, solo: false, hpf: 0, lpf: 0, duck: 0, pocket: 0 });
        ti = p.tracks.length - 1;
      }
      const mine = secs.filter((q) => q.stems.includes(st));
      mine.forEach((q, i) => {
        // the next section carries straight on: no fade at the join (the mixer plays it seamlessly)
        const joined = i + 1 < mine.length && Math.abs(mine[i + 1].off - (q.off + q.len)) < 1e-9;
        const c = { id: uid("c"), track: p.tracks[ti].id, song, stem: st, start: g.beat + q.off * mult, src: q.src, len: q.len * mult, gain_db: 0, fade_in: 0, fade_out: joined ? 0 : 0.05, mute: false, rate: 1, pitch: 0, keep_pitch: true };
        if (q.name) c.sec = q.name;
        p.clips.push(c); ids.push(c.id);
      });
    });
    S.selClips = new Set(ids);
  }, { label: secs.length > 1 ? `Add ${secs.length} sections` : "Add clip" });
  if (S.project.clips.length === ids.length) fit();
  emit("selection");
  return true;
}

// the same, at the playhead (rounded to the nearest bar): clicking a "Whole song" button instead of dragging it
export function dropAtPlayhead(payload) {
  if (!S.project) return false;
  const b = transportPos() / secPerBeat();
  S.drag = { payload, over: { beat: Math.max(0, Math.round(b / bpb()) * bpb()), track: -1 }, fine: false };
  try { return dropPayload(); } finally { S.drag = null; dirty = true; }
}

// rebuild header DOM only when the track list itself changes; otherwise update values in place
let headerSig = "";
function syncHeaders() {
  const p = S.project;
  const sig = p ? p.tracks.map((t) => t.id + t.name + t.color).join("|") : "";
  if (sig !== headerSig || !headers.querySelector(".th-ruler")) { headerSig = sig; buildHeaders(); return; }
  headers.querySelectorAll(".th").forEach((el) => {
    const t = p.tracks[+el.dataset.i];
    if (!t) return;
    el.querySelector(".m").classList.toggle("on", !!t.mute);
    el.querySelector(".s").classList.toggle("on", !!t.solo);
    el.querySelector(".fx").classList.toggle("on", t.hpf > 20 || (t.lpf > 200 && t.lpf < 20000) || t.duck > 0 || t.pocket > 0);
    const g = el.querySelector(".gain");
    if (document.activeElement !== g) g.value = sliderFromDb(t.gain_db);
    el.querySelector(".db").textContent = `${t.gain_db >= 0 ? "+" : ""}${t.gain_db.toFixed(1)} dB`;
  });
}
on("project", () => {
  // keep the stack layout frozen while dragging so the grabbed clip doesn't jump rows under the mouse
  if (!drag) L = null;
  syncHeaders(); positionHeaders(); clampScroll(); dirty = true;
});
on("selection", () => (dirty = true));
on("renders", () => (dirty = true));
on("transport", () => (dirty = true));
// a song was re-fetched (e.g. lyrics transcribed, beat_content back-filled): drop its cached info/peaks so
// the next draw re-reads lyrics / beat_content, then redraw
on("songUpdated", (id) => { forgetSong(id); dirty = true; });
on("markMode", (v) => { if (v !== markMode) toggleMarkMode(v); });
