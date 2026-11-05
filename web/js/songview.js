// Bottom inspector: song view (stems + sections; click = audition, drag = place on timeline) and clip editor.
// Song view: Ctrl+wheel zooms, wheel scrolls, scrollbar at the bottom; drag on the bar ruler (or Shift+drag
// anywhere) to select any bar range, which then behaves like a section (click = hear it, drag = place it).
import { api, peaks } from "./api.js";
import { S, on, emit, edit, fullSong, STEMS, STEM_COLORS, songMult, toast, invalidateSong, syncNow, editSongMarks, projectInfo } from "./state.js";
import { dragOver, dropPayload, dropAtPlayhead, redraw as redrawTimeline } from "./timeline.js";
import { landing, tempoVariation, tempoParts, detectedMarks, barStart, lockActive, lockView } from "./timing.js";
import { showMenu, modal } from "./ui.js";
import { openSampleEditor } from "./sampleopen.js";
import { songToolItems, openGrid } from "./songtools.js";

// "varies ±x %" only when the song's own tempo really moves (> ±1 % over 16-beat stretches - the beat
// detector's jitter averages out); "+ other tempos" when the detector found parts at another tempo
function tempoNote(song) {
  const v = tempoVariation(song), parts = tempoParts(song).length;
  const t = [];
  if (v > 1) t.push(`varies ±${v.toFixed(1)}%`);
  if (parts) t.push(`${parts} part${parts > 1 ? "s" : ""} at another tempo`);
  if (song.user_marks) t.push("your own beat marks");
  if (song.mus_check) t.push(`MuScriptor's notes agree with ${Math.round(song.mus_check.agree)}% of the marks, disagree with ${Math.round(song.mus_check.disagree)}%`);
  else if (song.beat_conf) {
    const s = detectedMarks(song).s, unsure = s.length ? s.filter((x) => !x).length / s.length : 0;
    if (unsure > 0.02) t.push(`detector unsure of ${Math.round(unsure * 100)}% of the beats`);
  }
  return t.length ? ` (${t.join(", ")})` : "";
}

const view = document.getElementById("songView");
const empty = document.getElementById("inspEmpty");
const clipView = document.getElementById("clipView");
const canvas = document.getElementById("songCanvas");
const ctx = canvas.getContext("2d");
const SEC_H = 24, RULER_H = 16, SCROLL_H = 13, PAD = 8;
// the song's form (songform.py: intro / verse / chorus ..., your edits win), else the engine's sections
const secs = () => song.form || song.sections;
const secName = (s) => (s.n != null ? `${s.name} ${s.n}` : `${s.name} ${s.label}`);
const TOP = SEC_H + RULER_H;
const SEC_COLORS = { intro: "#5d6168", verse: "#5d82b0", "pre-chorus": "#5a9aa8", chorus: "#b06a88", "post-chorus": "#a0708a", drop: "#c48450", break: "#4f8f86", bridge: "#8a7aa8", instrumental: "#6f9662", outro: "#5d6168", silence: "#2a2c30", section: "#5d82b0" };

let song = null;          // full info
let pk = {};              // stem -> Int8Array
let hover = null;         // {sec, stem, inSel}
let aud = null;           // {start, end, t, id}
let down = null;
let vt0 = 0, vspan = 1;   // visible window (seconds)
let sel = null;           // {a, b} selected bar range (bar indices, b exclusive)

export async function openSong(id) {
  if (!id) { song = null; showPanel(null); return; }
  const s = S.songs.get(id);
  if (!s || s.status !== "ready") {
    showPanel("song");
    song = null; pk = {};
    document.getElementById("svTitle").textContent = s?.title || "";
    document.getElementById("svMeta").textContent = s ? `${s.status}…` : "";
    draw();
    return;
  }
  showPanel("song");
  const prev = song?.id;
  song = await fullSong(id);
  pk = {};
  if (prev !== id) { vt0 = 0; vspan = song.duration; sel = null; }
  const k = song.key;
  document.getElementById("svTitle").textContent = song.title;
  document.getElementById("svMeta").textContent =
    `${song.bpm.toFixed(2)} BPM${tempoNote(song)} · ${meterLabel(song)} · ${k.name} (${k.camelot}) ${song.tuning_st == null ? "" : `· tuning ${song.tuning_st >= 0 ? "+" : ""}${Math.round(song.tuning_st * 100)}¢ from A440 `}· ${song.bars.length} bars`;
  meterHint(song);
  heardLine();
  draw();
  await Promise.all(STEMS.map(async (st) => { pk[st] = await peaks(id, st); draw(); }));
}
// what Music Flamingo heard in the song (library → right-click → Listen and describe)
function heardLine() {
  const el = document.getElementById("svHeard");
  const h = song && S.meta?.[song.id]?.heard;
  el.hidden = !h;
  if (!h) return;
  el.textContent = `Heard: ${[h.genre, h.subgenre].filter(Boolean).join(" / ")} - ${h.sound}`;
  el.title = "Music Flamingo listened to 30 s of this song. The DJ brain and the YuE2 style prompt use this line.\n\n" + el.textContent;
}
on("event:meta", () => heardLine());
const meterLabel = (s) => { const b = s.user_marks?.bpb || s.bpb; return (b === 6 ? "6/8" : b + "/4") + (s.user_marks?.bpb ? " (your marks)" : ""); };
// MuScriptor's drums repeat at another bar length than the marks' (muscheck.meter_check): offer it, never apply it
function meterHint(s) {
  const el = document.getElementById("svMeterHint");
  el.innerHTML = "";
  const m = s.mus_check?.meter, cur = s.user_marks?.bpb || s.bpb;
  if (!m || m.bpb === cur) return;
  const txt = document.createElement("span");
  txt.textContent = ` · MuScriptor's drums repeat every ${m.bpb} beats, the marks have ${cur} per bar `;
  const btn = document.createElement("button");
  btn.className = "small";
  btn.textContent = `Use ${m.bpb === 6 ? "6/8" : m.bpb + "/4"}`;
  btn.title = "Bars of " + m.bpb + " beats, starting on the kick MuScriptor found - marks only, Ctrl+Z undoes it";
  btn.onclick = () => {
    editSongMarks(s, (mk) => {
      mk.bpb = m.bpb;
      mk.d = mk.d.map((x) => (x === 2 ? 1 : x));             // one bar start: the kick
      let k = 0;
      for (let i = 1; i < mk.t.length; i++) if (Math.abs(mk.t[i] - m.start) < Math.abs(mk.t[k] - m.start)) k = i;
      barStart(mk, k);
    }, { label: `Use ${m.bpb} beats per bar` });
    openSong(s.id);
  };
  el.append(txt, btn);
}

function showPanel(which) {
  empty.hidden = which !== null;
  view.hidden = which !== "song";
  clipView.hidden = which !== "clip";
  if (which === "song") resize();
}

// ---------------------------------------------------------------- geometry
function resize() {
  const r = canvas.getBoundingClientRect();
  const dpr = devicePixelRatio || 1;
  canvas.width = Math.max(1, r.width * dpr); canvas.height = Math.max(1, r.height * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  draw();
}
new ResizeObserver(() => { if (!view.hidden) resize(); }).observe(canvas);

const W = () => canvas.getBoundingClientRect().width;
const H = () => canvas.getBoundingClientRect().height;
const tToX = (t) => PAD + ((t - vt0) / vspan) * (W() - 2 * PAD);
const xToT = (x) => vt0 + ((x - PAD) / (W() - 2 * PAD)) * vspan;
function beatT(b) {
  const bt = song.beats, n = bt.length;
  const i = Math.max(0, Math.min(n - 2, Math.floor(b)));
  return bt[i] + (b - i) * (bt[i + 1] - bt[i]);
}
function barBeat(bi) { return bi < song.bars.length ? song.bars[bi] : song.beats.length - 1; }

// What goes onto the timeline: bar ranges [{a, b (bar indices, b exclusive), name, stems}] -> pieces
// [{src (song beat), off, len (song beats as this project plays it - as recorded, or locked), name, stems}],
// laid out exactly as they follow each other in the song. Ranges without stems are left out (their gap stays).
function piecesFor(ranges) {
  const pi = projectInfo(song);                        // (this project's timing for the song, if it has its own)
  const v = S.project && lockActive(S.project, song.id, pi) ? lockView(pi) : song;
  const L = landing(v), at = (b) => (L ? L.at(b) : b);
  const rs = ranges.filter((r) => r.stems.length);
  if (!rs.length) return [];
  const w0 = at(barBeat(rs[0].a));
  return rs.map((r) => ({ src: barBeat(r.a), off: at(barBeat(r.a)) - w0, len: at(barBeat(r.b)) - at(barBeat(r.a)), name: r.name || "", stems: r.stems }));
}
const active = (sec, st) => (sec?.stems?.[st] ?? 1) > 0.05;
// the whole song, one clip per section: one stem, or every stem ("all") - each only where it plays
function wholeSong(stem) {
  return piecesFor(secs().map((s) => ({ a: s.start, b: s.end, name: secName(s),
    stems: (stem === "all" ? STEMS : [stem]).filter((st) => active(s, st)) })));
}
const barT = (bi) => beatT(barBeat(bi));
function barAt(t) {                         // bar index whose start is nearest at/under t
  const bars = song.bars;
  let lo = 0, hi = bars.length - 1;
  while (lo < hi) { const m = (lo + hi + 1) >> 1; if (barT(m) <= t) lo = m; else hi = m - 1; }
  return lo;
}
function rowGeom() {
  const h = H() - TOP - SCROLL_H - 2;
  const rh = h / STEMS.length;
  return STEMS.map((st, i) => ({ stem: st, y: TOP + i * rh, h: rh }));
}
function clampView() {
  if (!song) return;
  vspan = Math.max(2, Math.min(song.duration, vspan));
  vt0 = Math.max(0, Math.min(song.duration - vspan, vt0));
}
export function zoomSong(f, atX = null) {
  if (!song) return;
  const t = atX == null ? vt0 + vspan / 2 : xToT(atX);
  const frac = (t - vt0) / vspan;
  vspan = vspan * f;
  clampView();
  vt0 = t - frac * vspan;
  clampView(); draw();
}
function fitSong() { if (song) { vt0 = 0; vspan = song.duration; draw(); } }

// ---------------------------------------------------------------- drawing
function draw() {
  const w = W(), h = H();
  ctx.clearRect(0, 0, w, h);
  if (!song) {
    ctx.fillStyle = "#a7a7ad"; ctx.font = "13px Segoe UI";
    ctx.fillText("This song is still being processed - it will appear here when ready.", 16, 30);
    return;
  }
  const rows = rowGeom();
  const bodyB = h - SCROLL_H;
  const pxPerBar = (w - 2 * PAD) / vspan * (60 / song.bpm) * song.bpb;
  let every = 1;
  while (pxPerBar * every < 26) every *= 2;
  // bar lines + ruler numbers
  ctx.fillStyle = "rgba(255,255,255,0.03)"; ctx.fillRect(0, SEC_H, w, RULER_H);
  ctx.font = "10px Cascadia Mono, Consolas";
  const b0 = Math.max(0, barAt(vt0) - 1), b1 = Math.min(song.bars.length - 1, barAt(vt0 + vspan) + 1);
  for (let i = b0; i <= b1; i++) {
    const x = Math.round(tToX(song.beats[song.bars[i]])) + 0.5;
    if (pxPerBar > 3 || i % 4 === 0) {
      ctx.fillStyle = i % 4 === 0 ? "rgba(255,255,255,0.14)" : "rgba(255,255,255,0.05)";
      ctx.fillRect(x, TOP, 1, bodyB - TOP);
    }
    if (i % every === 0) {
      ctx.fillStyle = "rgba(255,255,255,0.22)"; ctx.fillRect(x, SEC_H + 8, 1, RULER_H - 8);
      ctx.fillStyle = "#a7a7ad"; ctx.fillText(String(i + 1), x + 3, SEC_H + 12);
    }
  }
  // stem rows
  rows.forEach((r) => {
    const data = pk[r.stem];
    ctx.fillStyle = "rgba(255,255,255,0.06)"; ctx.fillRect(0, r.y + r.h - 1, w, 1);
    ctx.fillStyle = "#8a8a92"; ctx.font = "600 10px Segoe UI"; ctx.fillText(r.stem, 10, r.y + 11);
    if (!data) return;
    const pps = song.peaks_per_sec, n = data.length / 2;
    const mid = r.y + r.h / 2, amp = r.h / 2 - 2;
    ctx.fillStyle = STEM_COLORS[r.stem] + (hover && hover.stem && hover.stem !== r.stem ? "70" : "d0");
    for (let x = PAD; x < w - PAD; x++) {
      const a = Math.max(0, Math.floor(xToT(x) * pps)), b = Math.max(a + 1, Math.floor(xToT(x + 1) * pps));
      let lo = 0, hi = 0;
      for (let j = a; j < b && j < n; j++) { lo = Math.min(lo, data[j * 2]); hi = Math.max(hi, data[j * 2 + 1]); }
      ctx.fillRect(x, mid - (hi / 127) * amp, 1, Math.max(1, ((hi - lo) / 127) * amp));
    }
  });
  // sections
  secs().forEach((s, i) => {
    const x0 = tToX(barT(s.start)), x1 = tToX(barT(s.end));
    if (x1 < 0 || x0 > w) return;
    const hot = hover && hover.sec === i && !hover.inSel;
    const col = SEC_COLORS[s.name] || "#5d82b0";
    ctx.fillStyle = col + (hot ? "ff" : "b0");
    ctx.fillRect(x0 + 1, 2, x1 - x0 - 2, SEC_H - 4);
    if (hot) { ctx.fillStyle = col + "22"; ctx.fillRect(x0, TOP, x1 - x0, bodyB - TOP); }
    ctx.fillStyle = "#0f1012"; ctx.font = "600 11px Segoe UI";
    const label = secName(s);
    const lx = Math.max(x0, 0) + 5;
    if (x1 - lx > ctx.measureText(label).width + 4) ctx.fillText(label, lx, 16);
    else if (x1 - lx > 30) ctx.fillText(s.name.slice(0, 3), lx, 16);
  });
  // hovered stem block within the section / selection
  if (hover && hover.stem) {
    const [a, b] = hover.inSel ? [sel.a, sel.b] : hover.sec != null ? [secs()[hover.sec].start, secs()[hover.sec].end] : [0, 0];
    if (b > a) {
      const r = rows.find((q) => q.stem === hover.stem);
      const x0 = tToX(barT(a)), x1 = tToX(barT(b));
      ctx.strokeStyle = STEM_COLORS[hover.stem]; ctx.lineWidth = 2;
      ctx.strokeRect(x0 + 1, r.y + 1, x1 - x0 - 2, r.h - 2);
    }
  }
  // bar-range selection
  if (sel && sel.b > sel.a) {
    const x0 = tToX(barT(sel.a)), x1 = tToX(barT(sel.b));
    ctx.fillStyle = "rgba(30,215,96,.10)"; ctx.fillRect(x0, TOP, x1 - x0, bodyB - TOP);
    ctx.fillStyle = "rgba(30,215,96,.9)"; ctx.fillRect(x0, SEC_H, x1 - x0, 4);
    ctx.fillRect(x0, TOP, 1.5, bodyB - TOP); ctx.fillRect(x1 - 1.5, TOP, 1.5, bodyB - TOP);
    ctx.font = "600 11px Segoe UI";
    const txt = `bars ${sel.a + 1}–${sel.b} (${sel.b - sel.a}) · click to hear · drag to timeline`;
    const tw = ctx.measureText(txt).width + 10;
    const tx = Math.max(2, Math.min(w - tw - 2, x0 + 2));
    ctx.fillStyle = "rgba(24,24,30,.9)"; ctx.fillRect(tx, TOP + 2, tw, 17);
    ctx.fillStyle = "#3be477"; ctx.fillText(txt, tx + 5, TOP + 14);
  }
  // scrollbar with time labels
  ctx.fillStyle = "rgba(0,0,0,0.2)"; ctx.fillRect(0, bodyB, w, SCROLL_H);
  if (vspan < song.duration - 0.01) {
    const tw = Math.max(30, (w - 2 * PAD) * vspan / song.duration);
    const tx = PAD + (w - 2 * PAD - tw) * (vt0 / (song.duration - vspan));
    ctx.fillStyle = down && down.type === "scroll" ? "rgba(255,255,255,.34)" : "rgba(255,255,255,.16)";
    ctx.fillRect(tx, bodyB + 2, tw, SCROLL_H - 4);
  }
  ctx.fillStyle = "#8a8a92"; ctx.font = "9px Cascadia Mono, Consolas";
  const tstep = [1, 2, 5, 10, 15, 30, 60, 120].find((s) => (w - 2 * PAD) / vspan * s > 60) || 120;
  for (let t = Math.ceil(vt0 / tstep) * tstep; t < vt0 + vspan; t += tstep) {
    const x = tToX(t);
    ctx.fillText(`${Math.floor(t / 60)}:${String(Math.floor(t % 60)).padStart(2, "0")}`, x + 2, bodyB + 10);
  }
  // audition cursor
  if (aud) {
    const now = performance.now() / 1000;
    let t = aud.start + (now - aud.t0);
    if (aud.loop) t = aud.start + ((t - aud.start) % (aud.end - aud.start));
    if (t <= aud.end) {
      ctx.fillStyle = "rgba(30,215,96,.10)";
      ctx.fillRect(tToX(aud.start), TOP, tToX(aud.end) - tToX(aud.start), bodyB - TOP);
      ctx.fillStyle = "#fff"; ctx.fillRect(tToX(t), TOP, 1.5, bodyB - TOP);
      // keep the cursor in view while listening
      if (t > vt0 + vspan * 0.95 || t < vt0) { vt0 = t - vspan * 0.1; clampView(); }
    }
  }
}
function anim() { if (aud && !view.hidden) draw(); requestAnimationFrame(anim); }
requestAnimationFrame(anim);

// ---------------------------------------------------------------- hit testing
function inSelX(x) { return sel && sel.b > sel.a && x >= tToX(barT(sel.a)) && x <= tToX(barT(sel.b)); }
function hit(x, y) {
  if (!song || y >= H() - SCROLL_H) return null;
  const t = xToT(x);
  const sec = secs().findIndex((s) => barT(s.start) <= t && t < barT(s.end));
  const r = rowGeom().find((q) => y >= q.y && y < q.y + q.h);
  const stem = y >= TOP && r ? r.stem : null;
  if (inSelX(x) && y >= SEC_H) return { sec: null, stem, inSel: true };
  if (y >= SEC_H && y < TOP) return { ruler: true };
  if (sec < 0) return null;
  return { sec, stem, inSel: false };
}
function phraseRange(x) {
  // Ctrl: the 4-bar phrase under the mouse instead of the whole section
  const bi = barAt(xToT(x));
  const secStart = secs().find((s) => s.start <= bi && bi < s.end)?.start ?? 0;
  const a = secStart + Math.floor((bi - secStart) / 4) * 4;
  return [a, Math.min(a + 4, song.bars.length)];
}

canvas.addEventListener("mousemove", (e) => {
  if (down) return;
  const h = hit(e.offsetX, e.offsetY);
  canvas.style.cursor = !h ? "default" : h.ruler ? "text" : "pointer";
  canvas.title = h?.ruler ? "Drag here to select any range of bars (or Shift+drag over the stems)" :
    h ? "Click to hear · drag onto the timeline · Ctrl = just the 4-bar phrase · Ctrl+wheel = zoom, wheel = scroll" : "";
  if (JSON.stringify(h) !== JSON.stringify(hover)) { hover = h; draw(); }
});
canvas.addEventListener("mouseleave", () => { if (!down) { hover = null; draw(); } });

// right-click the section strip: rename / split / merge sections (saved as your own form for the song)
const FORM_NAMES = ["intro", "verse", "pre-chorus", "chorus", "post-chorus", "bridge", "break", "drop", "instrumental", "outro"];
async function saveForm(list) {
  try {
    const r = await api.put(`/api/songs/${song.id}/form`, list ? { sections: list.map(({ start, end, name, label, energy, stems }) => ({ start, end, name, label, energy, stems })) } : { reset: true });
    song.form = r.form; invalidateSong(song.id); draw();
  } catch (e) { toast(e.message, true); }
}
canvas.addEventListener("contextmenu", (e) => {
  if (!song || !song.form || e.offsetY >= SEC_H) return;
  e.preventDefault();
  const t = xToT(e.offsetX), bi = barAt(t);
  const list = secs().map((s) => ({ ...s }));
  const i = list.findIndex((s) => s.start <= bi && bi < s.end);
  if (i < 0) return;
  const s = list[i];
  const markSel = sel && sel.b > sel.a ? [  // the bars picked on the ruler become one section
    [`Mark bars ${sel.a + 1}-${sel.b} as:`, null],
    ...FORM_NAMES.map((n) => [`  ${n}`, () => {
      const a = sel.a, b = sel.b, out = [];
      for (const x of list) {
        if (x.end <= a || x.start >= b) { out.push(x); continue; }
        if (x.start < a) out.push({ ...x, end: a });
        if (x.end > b) out.push({ ...x, start: b });
      }
      out.push({ start: a, end: b, name: n, label: "", energy: 0, stems: {} });
      out.sort((p, q) => p.start - q.start);
      saveForm(out);
    }]),
    "-",
  ] : [];
  showMenu(e, [
    ["Beat grid (the song's grid)…", () => openGrid(song.id, S.project?.clips.some((c) => c.song === song.id) ? S.project.id : null), "window"],
    ["All its tools ▸", () => showMenu(e, songToolItems(song.id))],
    "-",
    ...markSel,
    [`${secName(s)} · bars ${s.start + 1}-${s.end}`, null],
    ...FORM_NAMES.filter((n) => n !== s.name).map((n) => [`  Call it ${n}`, () => { s.name = n; saveForm(list); }]),
    "-",
    ...(bi > s.start ? [[`Split at bar ${bi + 1}`, () => { list.splice(i + 1, 0, { ...s, start: bi }); list[i].end = bi; saveForm(list); }]] : []),
    ...(i + 1 < list.length ? [["Merge with the next section", () => { list[i].end = list[i + 1].end; list.splice(i + 1, 1); saveForm(list); }]] : []),
    ...(song.form.some((x) => x.user) ? [["Back to the detected sections", () => saveForm(null)]] : []),
  ]);
});

canvas.addEventListener("wheel", (e) => {
  if (!song) return;
  e.preventDefault();
  if (e.ctrlKey) zoomSong(e.deltaY < 0 ? 1 / 1.25 : 1.25, e.offsetX);
  else {
    vt0 += (e.deltaX || e.deltaY) / 100 * vspan * 0.12;
    clampView(); draw();
  }
}, { passive: false });

canvas.addEventListener("mousedown", (e) => {
  if (e.button !== 0 || !song) return;
  const x = e.offsetX, y = e.offsetY;
  // scrollbar
  if (y >= H() - SCROLL_H) {
    if (vspan >= song.duration - 0.01) return;
    down = { type: "scroll", x0: e.clientX, t0: vt0 };
    const tw = Math.max(30, (W() - 2 * PAD) * vspan / song.duration);
    const mv = (ev) => { vt0 = down.t0 + (ev.clientX - down.x0) * (song.duration - vspan) / Math.max(1, W() - 2 * PAD - tw); clampView(); draw(); };
    window.addEventListener("mousemove", mv);
    window.addEventListener("mouseup", () => { window.removeEventListener("mousemove", mv); down = null; draw(); }, { once: true });
    return;
  }
  const h = hit(x, y);
  // range selection: drag on the ruler, or Shift+drag anywhere
  if (h?.ruler || e.shiftKey) {
    const a0 = barAt(xToT(x));
    down = { type: "select", x0: x, moved: false };
    const mv = (ev) => {
      const r = canvas.getBoundingClientRect();
      const xx = ev.clientX - r.left;
      if (Math.abs(xx - down.x0) > 3) down.moved = true;
      if (!down.moved) return;
      const t = xToT(xx), bi = barAt(t);
      sel = t >= barT(a0) ? { a: a0, b: Math.min(song.bars.length, Math.max(a0 + 1, bi + 1)) } : { a: bi, b: a0 + 1 };
      draw();
    };
    window.addEventListener("mousemove", mv);
    window.addEventListener("mouseup", () => {
      window.removeEventListener("mousemove", mv);
      if (!down.moved) sel = null;             // plain click on the ruler clears the selection
      down = null; draw();
    }, { once: true });
    return;
  }
  if (!h) return;
  let ba, bb, label;
  if (h.inSel) { ba = sel.a; bb = sel.b; label = `bars ${ba + 1}–${bb}`; }
  else {
    const sec = secs()[h.sec];
    [ba, bb] = e.ctrlKey ? phraseRange(x) : [sec.start, sec.end];
    label = secName(sec);
  }
  const secForStems = h.inSel ? null : secs()[h.sec];
  const srcBeat = barBeat(ba), endBeat = barBeat(bb);
  down = { type: "clip", x: e.clientX, y: e.clientY, h, srcBeat, endBeat, dragging: false };
  const move = (ev) => {
    if (!down.dragging && Math.hypot(ev.clientX - down.x, ev.clientY - down.y) > 6) {
      down.dragging = true;
      const stems = down.h.stem ? [down.h.stem] : STEMS.filter((st) => active(secForStems, st));
      // length in the song's own beats as it plays (as recorded): a part at another tempo takes the time it takes
      S.drag = { payload: { song: song.id, pieces: piecesFor([{ a: ba, b: bb, name: h.inSel ? "" : label, stems }]) }, over: null, fine: false };
      ghost.hidden = false;
      ghost.textContent = `${down.h.stem || "all stems"} · ${label} (${bb - ba} bars)`;
    }
    if (down.dragging) {
      ghost.style.left = ev.clientX + 12 + "px"; ghost.style.top = ev.clientY + 10 + "px";
      S.drag.fine = ev.altKey;
      dragOver(ev.clientX, ev.clientY);
    }
  };
  const up = () => {
    window.removeEventListener("mousemove", move);
    if (down.dragging) {
      if (!S.project) toast("Open or create a project first (Projects → New)", true);
      else if (!dropPayload()) toast("Drop it on the timeline above");
      S.drag = null; ghost.hidden = true; redrawTimeline();
    } else {
      audition(down.h.stem ? [down.h.stem] : STEMS, down.srcBeat, down.endBeat, `${song.id}:${ba}:${bb}:${down.h.stem}`);
    }
    down = null;
  };
  window.addEventListener("mousemove", move);
  window.addEventListener("mouseup", up, { once: true });
});

// "Whole song" buttons: drag one onto the timeline (or click it: at the playhead) - that stem (or all of them) for
// the whole song, one clip per section, each named after its section, so the ones you don't want can be deleted
document.querySelectorAll("#svWhole .wh").forEach((btn) => {
  btn.addEventListener("mousedown", (e) => {
    if (e.button !== 0 || !song) return;
    e.preventDefault();
    const stem = btn.dataset.stem;
    const pcs = wholeSong(stem);
    const st = { x: e.clientX, y: e.clientY, dragging: false };
    const move = (ev) => {
      if (!st.dragging && Math.hypot(ev.clientX - st.x, ev.clientY - st.y) > 6) {
        st.dragging = true;
        S.drag = { payload: { song: song.id, pieces: pcs }, over: null, fine: false };
        ghost.hidden = false;
        ghost.textContent = `${stem === "all" ? "all stems" : stem} · whole song (${pcs.length} sections)`;
      }
      if (st.dragging) {
        ghost.style.left = ev.clientX + 12 + "px"; ghost.style.top = ev.clientY + 10 + "px";
        S.drag.fine = ev.altKey;
        dragOver(ev.clientX, ev.clientY);
      }
    };
    const up = () => {
      window.removeEventListener("mousemove", move);
      if (!S.project) toast("Open or create a project first (Projects → New)", true);
      else if (!pcs.length) toast(`This song has no ${stem} in any section`, true);
      else if (st.dragging) { if (!dropPayload()) toast("Drop it on the timeline above"); }
      else if (dropAtPlayhead({ song: song.id, pieces: pcs })) toast(`Added ${stem === "all" ? "every stem" : stem} at the playhead: ${pcs.length} sections`);
      S.drag = null; ghost.hidden = true; redrawTimeline();
    };
    window.addEventListener("mousemove", move);
    window.addEventListener("mouseup", up, { once: true });
  });
});

document.getElementById("svZoomIn").onclick = () => zoomSong(1 / 1.5);
document.getElementById("svZoomOut").onclick = () => zoomSong(1.5);
document.getElementById("svFit").onclick = fitSong;

const ghost = document.createElement("div");
ghost.style.cssText = "position:fixed;z-index:80;pointer-events:none;background:rgba(32,32,38,.92);border:1px solid rgba(30,215,96,.6);border-radius:8px;padding:3px 8px;font-size:12px;";
ghost.hidden = true;
document.body.appendChild(ghost);

async function audition(stems, a, b, id) {
  if (aud && aud.id === id) { stopAudition(); return; }
  try {
    const r = await api.post("/api/audition", { song: song.id, stems, start_beat: a, end_beat: b, id, loop: false });
    aud = { id, start: r.start, end: r.end, t0: performance.now() / 1000, loop: false };
  } catch (e) { toast(e.message, true); }
}
export function stopAudition() { aud = null; api.post("/api/audition/stop"); draw(); }
document.getElementById("svStop").onclick = stopAudition;
on("event:audition", (m) => { if (!m.playing && aud && m.id === aud.id) { aud = null; draw(); } });

async function shiftDownbeat(k) {
  if (!song) return;
  await api.patch(`/api/songs/${song.id}`, { shift_downbeat: k });
  invalidateSong(song.id);
  openSong(song.id);
  toast(`Bar lines moved ${k > 0 ? "later" : "earlier"} by one beat`);
}
document.getElementById("svDownPrev").onclick = () => shiftDownbeat(-1);
document.getElementById("svMatch").onclick = () => song && emit("findMatches", song.id);

// ---------------------------------------------------------------- your lyrics (userlyrics.py)
// Typed once per song, lined up with its vocal stem (forced alignment: the words are yours, only their timing is
// found) and saved with the song. Whisper's words fill the box the first time, to correct rather than retype.
document.getElementById("svGrid").onclick = () => {
  if (song) openGrid(song.id, S.project?.clips.some((c) => c.song === song.id) ? S.project.id : null);
};
document.getElementById("svLyrics").onclick = async () => {
  if (!song) return;
  const sid = song.id;
  let cur;
  try { cur = await api.get(`/api/songs/${sid}/lyrics_user`); } catch (e) { toast(e.message, true); return; }
  const note = cur.yours
    ? `Your lyrics, ${cur.lines} lines lined up with the vocals${cur.low ? ` (${cur.low} hard to hear - check those)` : ""}.`
      + (cur.stale ? " <b>The vocal stem changed since - save again to line them up anew.</b>" : "")
    : cur.text ? "Whisper's guess, to correct - no tags needed, one sung line per line." : "One sung line per line, no tags needed.";
  const m = modal(`Lyrics: ${song.title}`, `
    <p class="dim" style="margin:0 0 8px">${note}</p>
    <textarea id="ulText" rows="16" spellcheck="false" style="width:100%;box-sizing:border-box"></textarea>
    <div class="row-gap" style="margin-top:10px;align-items:center">
      <button id="ulSave" class="small primary">Save &amp; line up</button>
      ${cur.yours ? `<button id="ulForget" class="small" title="Forget your lyrics: Whisper's transcript is used again">Use Whisper's</button>` : ""}
      <span id="ulMsg" class="dim"></span>
    </div>`);
  const ta = m.body.querySelector("#ulText"), msg = m.body.querySelector("#ulMsg");
  ta.value = cur.text || "";
  const send = async (text) => {
    m.body.querySelectorAll("button").forEach((b) => (b.disabled = true));
    msg.innerHTML = '<span class="spinner"></span> lining the words up with the vocals…';
    // listen before asking: a quick job (forgetting the lyrics) can end before the request returns its id
    let jobId = null, over = false;                      // (listeners stay registered: ignore after the end)
    const early = new Map();
    const end = (j) => {
      if (over || !j || (j.status !== "done" && j.status !== "error" && j.status !== "cancelled")) return;
      over = true;
      if (j.status === "done") { toast(text.trim() ? "Lyrics saved and lined up" : "Using Whisper's transcript again"); m.close(); if (song?.id === sid) openSong(sid); }
      else { msg.textContent = ""; toast(j.error || "Lining up the lyrics failed", true); m.body.querySelectorAll("button").forEach((b) => (b.disabled = false)); }
    };
    on("event:job", (ev) => {
      const j = ev?.job;
      if (!j || over) return;
      if (jobId == null) early.set(j.id, j);
      else if (j.id === jobId) end(j);
    });
    try { jobId = (await api.put(`/api/songs/${sid}/lyrics_user`, { text })).id; }
    catch (e) { over = true; msg.textContent = ""; toast(e.message, true); m.body.querySelectorAll("button").forEach((b) => (b.disabled = false)); return; }
    end(early.get(jobId));
  };
  m.body.querySelector("#ulSave").onclick = () => send(ta.value);
  m.body.querySelector("#ulForget")?.addEventListener("click", () => send(""));
  ta.focus();
};
document.getElementById("svDownNext").onclick = () => shiftDownbeat(1);

// ---------------------------------------------------------------- clip editor
let openClipId = null;
const r3 = (v) => Math.round(v * 1000) / 1000;

function openClip(id) {
  const c = S.project?.clips.find((k) => k.id === id);
  if (!c) return;
  openClipId = id;
  showPanel("clip");
  const info = S.songs.get(c.song) || {};
  const bpb = S.project.bpb;
  const mult = songMult(S.project, c.song);
  const rate = c.rate || 1, pitch = c.pitch || 0, keep = c.keep_pitch !== false;
  const fmtBar = (b) => `${Math.floor(b / bpb) + 1}.${r3(b % bpb) + 1}`;
  clipView.innerHTML = `
    <div class="cv-group" style="min-width:210px"><div class="cv-title"></div>
      <div style="color:var(--dim)">${c.stem} · starts at bar ${fmtBar(c.start)} · ${(c.len / bpb).toFixed(2)} bars</div>
      <div style="color:var(--dim)">source: beat ${c.src.toFixed(2)} of the original${mult !== 1 ? ` · ${mult === 2 ? "half" : "double"}-time` : ""}</div>
      <div class="row-gap" style="margin-top:6px"><button id="cvBack" class="small">← back to song</button><button id="cvSample" class="small" title="Open this clip in the sample editor (its own window): chop, stutter, layer, echo / tape delay on a grid down to 1/64">Sample editor…</button><button id="cvDel" class="small danger">Delete clip</button></div>
    </div>
    <div class="cv-group wide"><h4>Speed / stretch</h4>
      <div class="cv-row"><input type="range" class="long" id="cvRate" min="-2" max="2" step="0.001" value="${Math.log2(rate)}" title="Drag for a smooth speed change (double-click = back to 1×)">
        <input type="number" id="cvRateN" min="0.25" max="4" step="0.001" value="${r3(rate)}" title="Exact speed factor"><span class="meta">×</span></div>
      <div class="cv-row"><span class="seg"><button id="cvKeep" class="${keep ? "on" : ""}" title="Time-stretch: speed changes, pitch stays">Keep pitch</button><button id="cvVari" class="${keep ? "" : "on"}" title="Varispeed: faster = higher, like a turntable">Pitch follows speed</button></span></div>
      <div class="cv-row"><label>Fit to</label><input type="number" id="cvFitN" min="0.25" step="0.25" value="${(c.len / bpb).toFixed(2)}"><span class="meta">bars</span><button id="cvFit" class="small">Fit</button><button id="cvRateReset" class="small">1×</button></div>
      <div style="color:var(--faint);font-size:11px">Or drag the ◢ grips at a clip's bottom corners (Shift = snap to grid).</div>
    </div>
    <div class="cv-group wide"><h4>Pitch</h4>
      <div class="cv-row"><input type="range" class="long" id="cvPitch" min="-12" max="12" step="0.01" value="${pitch}" title="Drag for smooth pitch (double-click = reset)">
        <input type="number" id="cvPitchN" min="-24" max="24" step="0.01" value="${pitch}"><span class="meta">st</span></div>
      <div class="cv-row"><button class="small" data-p="-1">−1</button><button class="small" data-p="-0.1">−0.1</button>
        <button class="small" data-p="0.1">+0.1</button><button class="small" data-p="1">+1</button><button class="small" data-p="0">reset</button>
        <span class="val" id="cvPitchV"></span></div>
      <div style="color:var(--faint);font-size:11px">On top of the song's transpose. Alt+wheel over a clip = fine pitch.</div>
    </div>
    <div class="cv-group"><h4>Level</h4>
      <div class="cv-row"><label>Gain</label><input type="range" id="cvGain" min="-24" max="12" step="0.5" value="${c.gain_db || 0}"><span id="cvGainV" class="meta">${(c.gain_db || 0).toFixed(1)} dB</span></div>
      <div class="cv-row"><label>Fade in</label><input type="number" id="cvFi" min="0" step="0.25" value="${r3(c.fade_in || 0)}"><span class="meta">beats</span></div>
      <div class="cv-row"><label>Fade out</label><input type="number" id="cvFo" min="0" step="0.25" value="${r3(c.fade_out || 0)}"><span class="meta">beats</span></div>
    </div>
    <div class="cv-group"><h4>Slip (which part plays)</h4>
      <div class="row-gap"><button class="small" data-slip="-${bpb}">−1 bar</button><button class="small" data-slip="-1">−1 beat</button>
        <button class="small" data-slip="1">+1 beat</button><button class="small" data-slip="${bpb}">+1 bar</button></div>
      <div class="cv-row" style="margin-top:4px"><label>Source beat</label><input type="number" id="cvSrc" step="0.25" value="${r3(c.src)}"></div>
      <h4 style="margin-top:6px">Stem</h4>
      <div class="row-gap">${STEMS.map((st) => `<button class="small ${st === c.stem ? "toggle on" : ""}" data-stem="${st}">${st}</button>`).join("")}</div>
    </div>`;
  clipView.querySelector(".cv-title").textContent = info.title || c.song;
  const Q = (s) => clipView.querySelector(s);
  const upd = (fn, label, tag) => edit((p) => { const k = p.clips.find((q) => q.id === id); if (k) fn(k); }, { coalesce: tag, label });
  // sliders: varispeed preview while dragging, HQ render on release; one sweep = one undo step
  const live = (el, onMove, name) => {
    el.addEventListener("pointerdown", () => (el._tag = name + Date.now()));
    el.oninput = () => { S.interactive = true; onMove(); };
    el.onchange = () => { S.interactive = false; syncNow(); el.blur(); };
  };
  const setRate = (r, tag) => upd((k) => {
    r = Math.max(0.25, Math.min(4, r));
    k.len = k.len * (k.rate || 1) / r;          // same source material, new length
    k.rate = Math.abs(r - 1) < 0.0005 ? 1 : Math.round(r * 100000) / 100000;
  }, "Stretch clip", tag);
  const rs = Q("#cvRate");
  live(rs, () => { const r = Math.pow(2, +rs.value); Q("#cvRateN").value = r3(r); setRate(r, rs._tag); }, "rate");
  rs.ondblclick = () => setRate(1);
  Q("#cvRateN").onchange = (e) => setRate(+e.target.value || 1);
  Q("#cvRateReset").onclick = () => setRate(1);
  Q("#cvFit").onclick = () => {
    const bars = Math.max(0.25, +Q("#cvFitN").value || 1);
    const cur = S.project.clips.find((k) => k.id === id);
    setRate((cur.len * (cur.rate || 1)) / (bars * bpb));
  };
  Q("#cvKeep").onclick = () => upd((k) => (k.keep_pitch = true), "Keep pitch");
  Q("#cvVari").onclick = () => upd((k) => (k.keep_pitch = false), "Pitch follows speed");
  const setPitch = (v, tag) => upd((k) => (k.pitch = Math.round(Math.max(-24, Math.min(24, v)) * 100) / 100), "Pitch", tag);
  const showP = (v) => {
    const cents = Math.round(v * 100), st = Math.trunc(cents / 100), ct = cents - st * 100;
    Q("#cvPitchV").textContent = v === 0 ? "original pitch" : `${v < 0 ? "−" : "+"}${Math.abs(st)} st${ct ? ` ${Math.abs(ct)}¢` : ""}`;
  };
  showP(pitch);
  const ps = Q("#cvPitch");
  live(ps, () => { Q("#cvPitchN").value = ps.value; showP(+ps.value); setPitch(+ps.value, ps._tag); }, "pitch");
  ps.ondblclick = () => setPitch(0);
  Q("#cvPitchN").onchange = (e) => setPitch(+e.target.value || 0);
  clipView.querySelectorAll("[data-p]").forEach((b) => b.onclick = () => {
    const cur = S.project.clips.find((k) => k.id === id);
    setPitch(+b.dataset.p === 0 ? 0 : (cur.pitch || 0) + +b.dataset.p);
  });
  const g = Q("#cvGain");
  g.oninput = () => { Q("#cvGainV").textContent = `${(+g.value).toFixed(1)} dB`; upd((k) => (k.gain_db = +g.value), "Clip gain", "cvgain" + id); };
  Q("#cvFi").onchange = (e) => upd((k) => (k.fade_in = Math.max(0, +e.target.value)), "Fade in");
  Q("#cvFo").onchange = (e) => upd((k) => (k.fade_out = Math.max(0, +e.target.value)), "Fade out");
  Q("#cvSrc").onchange = (e) => upd((k) => (k.src = Math.max(0, +e.target.value)), "Slip");
  clipView.querySelectorAll("[data-slip]").forEach((b) => b.onclick = () => upd((k) => (k.src = Math.max(0, k.src + +b.dataset.slip)), "Slip"));
  clipView.querySelectorAll("[data-stem]").forEach((b) => b.onclick = () => upd((k) => (k.stem = b.dataset.stem), "Swap stem"));
  Q("#cvDel").onclick = () => { edit((p) => (p.clips = p.clips.filter((k) => k.id !== id)), { label: "Delete clip" }); S.selClips.delete(id); emit("selection"); openSong(S.selSong); };
  Q("#cvBack").onclick = () => openSong(c.song);
  Q("#cvSample").onclick = () => openSampleEditor(id);
}

on("songSelected", (id) => openSong(id));
on("songUpdated", (id) => { if (id === S.selSong) { invalidateSong(id); openSong(id); } });
on("openClip", openClip);
on("selection", () => {
  if (S.selClips.size === 1 && !clipView.hidden) openClip([...S.selClips][0]);
});
// keep the clip panel in step with timeline edits and undo - but never rebuild under a control in use
on("project", () => {
  if (clipView.hidden || !openClipId) return;
  const a = document.activeElement;
  if (a && a !== document.body && clipView.contains(a) && (a.type === "range" || a.type === "number")) return;
  if (!S.project?.clips.some((k) => k.id === openClipId)) { openClipId = null; if (S.selSong) openSong(S.selSong); else showPanel(null); return; }
  openClip(openClipId);
});
