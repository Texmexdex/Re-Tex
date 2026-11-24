"""One self-contained page per song (opens from the folder by double-click, no server): the tempo map drawn over
the song, a player with a click on the map or on the current grid, and where each grid sits against the music."""
import html
import json

PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>__TITLE__ - tempo map</title>
<style>
:root{--bg:#0d0f0e;--panel:rgba(255,255,255,.035);--line:rgba(255,255,255,.08);--text:#e8ece9;--dim:#8a938d;
--green:#1ed760;--green2:rgba(30,215,96,.16);--grey:#6b726d;--amber:#f2a33a}
*{box-sizing:border-box}html{background:var(--bg)}body{margin:0;background:radial-gradient(1200px 600px at 10% -10%,#16241b 0,var(--bg) 55%) fixed;
color:var(--text);font:14px/1.45 "Segoe UI",system-ui,sans-serif}
main{max-width:1500px;margin:0 auto;padding:22px 26px 60px}
h1{font-size:21px;margin:0 0 4px;font-weight:600}.sub{color:var(--dim);margin:0 0 16px}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:14px 16px;margin:12px 0;
backdrop-filter:blur(8px)}
.row{display:flex;gap:12px;align-items:center;flex-wrap:wrap}
button,select{background:rgba(255,255,255,.06);color:var(--text);border:1px solid var(--line);border-radius:999px;
padding:7px 16px;font:inherit;cursor:pointer}button:hover,select:hover{border-color:rgba(255,255,255,.2)}
button.primary{background:var(--green);color:#06120a;border-color:var(--green);font-weight:600;min-width:86px}
.time{font-variant-numeric:tabular-nums;color:var(--dim);min-width:120px}
label{color:var(--dim);display:flex;gap:6px;align-items:center}
canvas{display:block;width:100%;border-radius:10px;background:rgba(0,0,0,.25);cursor:crosshair}
#zoom{margin-top:10px;cursor:grab}
.chips{display:flex;gap:8px;flex-wrap:wrap;margin:6px 0 0}.chip{border:1px solid var(--line);border-radius:999px;
padding:3px 11px;color:var(--dim);font-size:12.5px}.chip b{color:var(--text);font-weight:600}
table{border-collapse:collapse;width:100%;font-size:13px}th,td{text-align:left;padding:6px 10px;border-bottom:1px solid var(--line)}
th{color:var(--dim);font-weight:500}td.n{font-variant-numeric:tabular-nums}tr:hover td{background:rgba(255,255,255,.025)}
tr.seek{cursor:pointer}.dim{color:var(--dim)}.warn{color:var(--amber)}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:12px}@media(max-width:1000px){.grid2{grid-template-columns:1fr}}
.legend{display:flex;gap:16px;flex-wrap:wrap;color:var(--dim);font-size:12.5px;margin-top:8px}
.sw{display:inline-block;width:14px;height:3px;border-radius:2px;margin-right:6px;vertical-align:middle}
p.note{color:var(--dim);margin:6px 0}
</style></head><body><main>
<h1 id="title"></h1><p class="sub" id="subt"></p>
<div class="panel" id="retex" hidden><button class="primary" id="toGrid">Open in the Beat grid window</button>
 <span class="note">&nbsp;Loads this map's beats and bar lines there as a proposal, to listen, fix and save as the song's grid (or for
 this project only). Nothing changes until you press Save there.</span></div>
<div class="panel">
 <div class="row">
  <button class="primary" id="play">Play</button><span class="time" id="time">0:00.0</span>
  <label>Click on <select id="src"></select></label>
  <label><input type="checkbox" id="follow" checked> follow</label>
  <span class="dim" style="margin-left:auto">Space plays. Scroll on the lower view to zoom, drag to move, click to jump.</span>
 </div>
 <canvas id="over" data-h="118" style="margin-top:12px;height:118px"></canvas>
 <canvas id="zoom" data-h="250" style="height:250px"></canvas>
 <div class="legend">
  <span><i class="sw" style="background:var(--green)"></i>tempo map: bar lines (tall), beats, subdivisions</span>
  <span><i class="sw" style="background:var(--grey)"></i>current grid (what Re-Tex has now)</span>
  <span id="lgs" hidden><i class="sw" style="background:#e0a040"></i>one speed at the stored BPM</span>
  <span id="lgm" hidden><i class="sw" style="background:#7fb7ff"></i>your saved marks</span>
  <span><i class="sw" style="background:#4a524d"></i>drum hits</span>
  <span>drift lane: where the music's pulse sits against each grid, ms (flat near 0 = the grid fits)</span>
 </div>
 <audio id="audio" preload="auto"></audio>
</div>
<div class="grid2">
 <div class="panel"><b>Tempo sections</b><table id="tsec"><thead><tr><th>from</th><th>to</th><th>BPM</th><th>how</th><th>fit to hits</th></tr></thead><tbody></tbody></table>
 <p class="note">"kept": the pulse of the section next to it is still there (the grouping changed, not the tempo; "regrouped" = the detector
counted it another way, e.g. 5:3 = five of its beats to three of the song's). "own": a different pulse.
 "no pulse": nothing steady to measure - the grid next to it carried on. "follows the drift": no one tempo
holds (a take with no click, or one that speeds up) - the grid follows the drums, slowly; BPM = typical, with its range. Fit = how far the hits sit from the grid (rms).</p></div>
 <div class="panel"><b>Bars</b><table id="tmet"><thead><tr><th>from</th><th>to</th><th>bar</th><th>subdivisions</th><th>sure</th></tr></thead><tbody></tbody></table>
 <p class="note">Bar length = how often the rhythm repeats; bar 1 = where kick, crash, bass and chord changes land hardest.
 Amber = not sure (under 0.45): listen there first.</p></div>
</div>
<div class="panel"><b>Gridded by rule, not heard</b><table id="tspan"><thead><tr><th>from</th><th>to</th><th>what</th><th>note</th></tr></thead><tbody></tbody></table>
<p class="note">"no beat": nothing to keep time to (an ambient start or end, a break) - the grid carries on evenly at the tempo around it,
so a lock plays it at one speed. "assumed": a short or unsure stretch in another meter, gridded in the song's main meter. "unknown": 4 bars
or more that sound like another meter but the bar finder isn't sure - gridded the same way, check it. These marks are amber in the Beat grid
window. <span id="steadyNote"></span></p></div>
<div class="panel"><b>How well each grid sits on the music</b><table id="tfit"><thead><tr><th>grid</th><th>pulse strength</th><th>drift (middle half of offsets)</th><th>steady</th></tr></thead><tbody></tbody></table>
<p class="note">Pulse strength: how much of the drum hits keep landing at one place in the grid's 16ths (or triplets), 8 s at a
time (chance is about 0.1; dense or loose playing lowers it for every grid alike). Drift: how much that place moves over the song -
a grid at the right tempo stays within a few ms; at the wrong tempo it slides and wraps. Steady: the share of the song within 15 ms
of the typical offset.</p></div>
</main>
<script>
const D = __DATA__;
const $ = (id) => document.getElementById(id);
const fmt = (s) => { s = Math.max(0, s); const m = Math.floor(s / 60); return `${m}:${(s - m * 60).toFixed(1).padStart(4, "0")}`; };
$("title").textContent = D.title;
$("subt").textContent = D.subtitle;
// opened from Re-Tex (its sidecar route): offer the Beat grid window with this map as a proposal
if (location.pathname.startsWith("/tempomap/")) {
  const pid = new URLSearchParams(location.search).get("pid");
  $("retex").hidden = false;
  $("toGrid").onclick = () => window.open(`/grid?sid=${encodeURIComponent(D.id)}${pid ? `&pid=${encodeURIComponent(pid)}` : ""}&proposal=tempomap`,
    "retexGrid", "width=1400,height=760");
}
const audio = $("audio"), src = $("src");
for (const [k, label] of D.sources) { const o = document.createElement("option"); o.value = k; o.textContent = label; src.append(o); }
if (D.marks) $("lgm").hidden = false;
if (D.lock.steady) $("lgs").hidden = false;
audio.src = D.files[src.value];
src.onchange = () => {
  const t = audio.currentTime, playing = !audio.paused;
  audio.src = D.files[src.value];
  audio.addEventListener("loadedmetadata", () => { audio.currentTime = t; if (playing) audio.play(); }, { once: true });
};
$("play").onclick = () => (audio.paused ? audio.play() : audio.pause());
audio.onplay = () => { $("play").textContent = "Pause"; loop(); };
audio.onpause = () => { $("play").textContent = "Play"; draw(); };
document.addEventListener("keydown", (e) => { if (e.code === "Space" && e.target.tagName !== "SELECT") { e.preventDefault(); $("play").click(); } });

// ---------------------------------------------------------------- drawing
const over = $("over"), zoom = $("zoom");
let view = { a: 0, b: Math.min(D.duration, 40) };
function fit(c) { const h = +c.dataset.h, d = window.devicePixelRatio || 1; c.style.height = h + "px"; const r = c.getBoundingClientRect(); c.width = Math.round(r.width * d); c.height = Math.round(h * d); const g = c.getContext("2d"); g.setTransform(d, 0, 0, d, 0, 0); return [g, r.width, h]; }
const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const shades = ["rgba(30,215,96,.20)", "rgba(30,215,96,.09)"];
function lanes(g, w, x, y, h, items, text, fill) {
  items.forEach((it, i) => {
    const x0 = x(it.start), x1 = x(it.end);
    if (x1 < 0 || x0 > w) return;
    g.fillStyle = fill(it, i); g.fillRect(x0, y, Math.max(1, x1 - x0 - 1), h);
    const t = text(it); g.fillStyle = it.confidence !== undefined && it.confidence < 0.45 ? css("--amber") : css("--text");
    g.font = "12px Segoe UI"; if (x1 - x0 > g.measureText(t).width + 8) g.fillText(t, Math.max(x0, 0) + 5, y + h - 5);
  });
}
function driftLane(g, w, x, y, h, lk, color, range = 80) {
  if (!lk || !lk.t.length) return;
  g.fillStyle = color;
  lk.t.forEach((t, i) => {
    const o = Math.max(-range, Math.min(range, lk.offset_ms[i]));
    const yy = y + h / 2 - (o / range) * (h / 2 - 2);
    g.globalAlpha = Math.max(0.15, Math.min(1, lk.strength[i] * 3));
    g.fillRect(x(t) - 1, yy - 1, 2.5, 2.5);
  });
  g.globalAlpha = 1;
}
function drawOver() {
  const [g, w] = fit(over), x = (t) => (t / D.duration) * w;
  g.clearRect(0, 0, w, 118);
  lanes(g, w, x, 4, 24, D.sections, (s) => `${s.bpm.toFixed(2)}`, (s, i) => (s.how === "none" ? "rgba(255,255,255,.05)" : shades[i % 2]));
  lanes(g, w, x, 32, 22, D.meters, (m) => m.label, (m, i) => (m.confidence < 0.45 ? "rgba(242,163,58,.14)" : shades[(i + 1) % 2]));
  spanLane(g, x, 32, 22);
  g.strokeStyle = "rgba(255,255,255,.08)"; g.beginPath(); g.moveTo(0, 86); g.lineTo(w, 86); g.stroke();
  driftLane(g, w, x, 58, 56, D.lock.grid, css("--grey"));
  if (D.lock.steady) driftLane(g, w, x, 58, 56, D.lock.steady, "#e0a040");
  if (D.lock.marks) driftLane(g, w, x, 58, 56, D.lock.marks, "#7fb7ff");
  driftLane(g, w, x, 58, 56, D.lock.map, css("--green"));
  g.strokeStyle = "rgba(255,255,255,.55)"; g.lineWidth = 1; g.strokeRect(x(view.a), 1, Math.max(2, x(view.b) - x(view.a)), 116);
  g.fillStyle = "#fff"; g.fillRect(x(audio.currentTime), 0, 1.5, 118);
}
function spanLane(g, x, y, h) {          // regular.py's spans over the bar lane: amber hatching + what it is
  for (const s of D.spans || []) {
    const a = x(s.start), b = x(s.end);
    if (b < 0) continue;
    g.fillStyle = "rgba(242,163,58,.22)"; g.fillRect(a, y, Math.max(2, b - a), h);
    g.fillStyle = "#f2a33a"; g.font = "11px Segoe UI"; if (b - a > 40) g.fillText(s.kind, a + 4, y + h - 6);
  }
}
function ticks(g, x, w, times, y0, y1, color, width = 1) {
  g.strokeStyle = color; g.lineWidth = width; g.beginPath();
  for (const t of times) { if (t < view.a - 1 || t > view.b + 1) continue; const xx = Math.round(x(t)) + 0.5; g.moveTo(xx, y0); g.lineTo(xx, y1); }
  g.stroke();
}
function drawZoom() {
  const [g, w] = fit(zoom), span = view.b - view.a, x = (t) => ((t - view.a) / span) * w;
  g.clearRect(0, 0, w, 250);
  lanes(g, w, x, 2, 20, D.sections, (s) => `${s.bpm.toFixed(3)} BPM (${s.label})`, (s, i) => (s.how === "none" ? "rgba(255,255,255,.05)" : shades[i % 2]));
  lanes(g, w, x, 24, 20, D.meters, (m) => `${m.label}`, (m, i) => (m.confidence < 0.45 ? "rgba(242,163,58,.14)" : shades[(i + 1) % 2]));
  spanLane(g, x, 24, 20);
  // drum hits
  const e = D.env, f = e.fps, i0 = Math.max(0, Math.floor(view.a * f)), i1 = Math.min(e.hits.length, Math.ceil(view.b * f));
  g.fillStyle = "#4a524d";
  const step = Math.max(1, Math.floor((i1 - i0) / (w * 2)));
  for (let i = i0; i < i1; i += step) { let m = 0; for (let k = i; k < Math.min(i1, i + step); k++) m = Math.max(m, e.hits[k]); const hh = Math.min(1, m) * 70; g.fillRect(x(i / f), 230 - hh, Math.max(1, w / ((i1 - i0) / step)), hh); }
  // the map
  const pxPerSec = w / span;
  if (pxPerSec > 25) ticks(g, x, w, D.tatums, 118, 130, "rgba(30,215,96,.45)");
  ticks(g, x, w, D.beats, 104, 140, "rgba(30,215,96,.85)", 1);
  ticks(g, x, w, D.bars, 50, 150, css("--green"), 2);
  // the current grid / your marks
  ticks(g, x, w, D.grid.beats, 160, 178, css("--grey"));
  ticks(g, x, w, D.grid.bars, 154, 184, css("--grey"), 2);
  if (D.marks) { ticks(g, x, w, D.marks.beats, 186, 198, "#7fb7ff"); ticks(g, x, w, D.marks.bars, 184, 202, "#7fb7ff", 2); }
  g.fillStyle = css("--dim"); g.font = "11px Segoe UI";
  g.fillText("tempo map", 6, 64); g.fillText("current grid", 6, 172); if (D.marks) g.fillText("your marks", 6, 196);
  const tt = audio.currentTime; if (tt >= view.a && tt <= view.b) { g.fillStyle = "#fff"; g.fillRect(x(tt), 0, 1.5, 250); }
  // time ruler
  g.fillStyle = css("--dim"); const st = span > 120 ? 30 : span > 50 ? 10 : span > 15 ? 5 : 1;
  for (let s = Math.ceil(view.a / st) * st; s <= view.b; s += st) g.fillText(fmt(s).replace(/\.\d$/, ""), x(s) + 3, 246);
}
function draw() { drawOver(); drawZoom(); $("time").textContent = fmt(audio.currentTime) + " / " + fmt(D.duration); }
function loop() {
  if ($("follow").checked) { const t = audio.currentTime, span = view.b - view.a; if (t > view.b - span * 0.15 || t < view.a) { view.a = Math.max(0, t - span * 0.2); view.b = view.a + span; } }
  draw(); if (!audio.paused) requestAnimationFrame(loop);
}
// overview: click jumps and centres the view
over.onclick = (ev) => { const r = over.getBoundingClientRect(), t = ((ev.clientX - r.left) / r.width) * D.duration, span = view.b - view.a;
  view.a = Math.max(0, Math.min(D.duration - span, t - span / 2)); view.b = view.a + span; audio.currentTime = t; draw(); };
// zoom view: wheel zooms, drag pans, click jumps
zoom.onwheel = (ev) => { ev.preventDefault(); const r = zoom.getBoundingClientRect(), fx = (ev.clientX - r.left) / r.width, t = view.a + fx * (view.b - view.a);
  let span = (view.b - view.a) * (ev.deltaY > 0 ? 1.25 : 0.8); span = Math.max(2, Math.min(D.duration, span));
  view.a = Math.max(0, t - fx * span); view.b = Math.min(D.duration, view.a + span); draw(); };
let drag = null;
zoom.onmousedown = (ev) => { drag = { x: ev.clientX, a: view.a, b: view.b, moved: false }; zoom.style.cursor = "grabbing"; };
window.onmousemove = (ev) => { if (!drag) return; const r = zoom.getBoundingClientRect(), dt = ((ev.clientX - drag.x) / r.width) * (drag.b - drag.a);
  if (Math.abs(ev.clientX - drag.x) > 3) drag.moved = true; const span = drag.b - drag.a; view.a = Math.max(0, Math.min(D.duration - span, drag.a - dt)); view.b = view.a + span; draw(); };
window.onmouseup = (ev) => { if (drag && !drag.moved) { const r = zoom.getBoundingClientRect(); audio.currentTime = view.a + ((ev.clientX - r.left) / r.width) * (view.b - view.a); draw(); } drag = null; zoom.style.cursor = "grab"; };
window.onresize = draw;
// tables
const tb = (id) => $(id).querySelector("tbody");
const seekRow = (tr, t) => { tr.className = "seek"; tr.onclick = () => { audio.currentTime = t; const span = view.b - view.a; view.a = Math.max(0, t - span * 0.1); view.b = view.a + span; draw(); }; };
for (const s of D.sections) { const tr = document.createElement("tr");
  tr.innerHTML = `<td class="n">${fmt(s.start)}</td><td class="n">${fmt(s.end)}</td><td class="n">${s.bpm.toFixed(3)}${s.bpm_range ? `<br><span class="dim">${s.bpm_range[0].toFixed(1)}-${s.bpm_range[1].toFixed(1)}</span>` : ""}</td><td>${s.label}${(s.feels || []).filter((f) => f.feel !== "1:1").map((f) => `<br><span class="dim">regrouped ${fmt(f.start)} to ${fmt(f.end)} (the detector counted ${f.feel})</span>`).join("")}</td><td class="n">${s.rms_ms == null ? "-" : s.rms_ms.toFixed(1) + " ms"}</td>`;
  seekRow(tr, s.start); tb("tsec").append(tr); }
for (const m of D.meters) { const tr = document.createElement("tr");
  tr.innerHTML = `<td class="n">${fmt(m.start)}</td><td class="n">${fmt(m.end)}</td><td><b>${m.label}</b></td><td class="n">${m.length}${m.sub === 3 ? " (triplets)" : ""}</td><td class="n ${m.confidence < 0.45 ? "warn" : ""}">${m.confidence.toFixed(2)}</td>`;
  seekRow(tr, m.start); tb("tmet").append(tr); }
for (const sp of D.spans || []) { const tr = document.createElement("tr");
  tr.innerHTML = `<td class="n">${fmt(sp.start)}</td><td class="n">${fmt(sp.end)}</td><td class="warn"><b>${sp.kind}</b></td><td></td>`;
  tr.lastChild.textContent = sp.note; seekRow(tr, sp.start); tb("tspan").append(tr); }
if (!(D.spans || []).length) tb("tspan").innerHTML = '<tr><td colspan="4" class="dim">nothing - every part of the grid was heard</td></tr>';
if ((D.steady || []).length) $("steadyNote").textContent = "Made steady (one tempo, the drums fit it as well as the followed beats): " +
  D.steady.map((s) => `${fmt(s.start)}-${fmt(s.end)} at ${s.bpm.toFixed(2)} BPM (hits ${s.hit_ms} ms off it, ${s.followed_ms} ms off the followed beats)`).join("; ") + ".";
for (const [name, s] of D.fit) { const tr = document.createElement("tr");
  tr.innerHTML = `<td>${name}</td><td class="n">${s.strength.toFixed(3)}</td><td class="n">${s.spread_ms == null ? "-" : s.spread_ms.toFixed(1) + " ms"}</td><td class="n">${s.steady == null ? "-" : Math.round(s.steady * 100) + " %"}</td>`;
  tb("tfit").append(tr); }
draw();
</script></body></html>
"""

HOW = {"main": "longest", "same": "kept", "own": "own", "none": "no pulse", "follows": "follows the drift"}


def page(m, extra):
    """m: the tempo map; extra: dict(grid={beats, bars}, marks={beats, bars} or None, env={fps, hits}, lock={map,
    grid, marks}, fit=[(name, summary)], files={key: file}, sources=[(key, label)], subtitle)."""
    data = dict(
        id=m.get("id"), title=m["title"], duration=m["duration"], subtitle=extra["subtitle"],
        sections=[dict(s, label=HOW.get(s["how"], s["how"])) for s in m["sections"]],
        meters=m["meters"], bars=m["bar_times"], beats=m["beats"], tatums=m["tatums"],
        spans=m.get("spans") or [], steady=m.get("steady") or [],
        grid=extra["grid"], marks=extra.get("marks"), env=extra["env"], lock=extra["lock"],
        fit=extra["fit"], files=extra["files"], sources=extra["sources"],
    )
    js = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    return PAGE.replace("__TITLE__", html.escape(m["title"] or "")).replace("__DATA__", js)


INDEX = r"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Tempo maps</title>
<style>body{margin:0;background:#0d0f0e;color:#e8ece9;font:14px/1.5 "Segoe UI",system-ui,sans-serif}main{max-width:1100px;margin:0 auto;padding:26px}
h1{font-weight:600;font-size:21px}a{color:#1ed760;text-decoration:none}a:hover{text-decoration:underline}
table{border-collapse:collapse;width:100%}th,td{text-align:left;padding:7px 10px;border-bottom:1px solid rgba(255,255,255,.08)}th{color:#8a938d;font-weight:500}
.d{color:#8a938d}</style></head><body><main><h1>Tempo maps</h1>
<p class="d">One page per song: the map drawn over the song, with the click on the map or on the grid Re-Tex has now. Nothing here changes Re-Tex or your library.</p>
<table><thead><tr><th>song</th><th>tempo</th><th>bars</th><th>made</th></tr></thead><tbody>__ROWS__</tbody></table></main></body></html>"""


def index(rows):
    body = "".join(f'<tr><td><a href="{html.escape(r["href"])}">{html.escape(r["title"])}</a></td><td>{html.escape(r["tempo"])}</td>'
                   f'<td>{html.escape(r["bars"])}</td><td class="d">{html.escape(r["made"])}</td></tr>' for r in rows)
    return INDEX.replace("__ROWS__", body)
