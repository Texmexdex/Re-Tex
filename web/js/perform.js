// Perform mode: play chops and samples from the keyboard while the song plays (mashlab/perform_live.py plays them
// live; on release each take becomes ordinary clips, applied as one undo step).
//   Off (default): every key in Re-Tex does exactly what it always did. Nothing here runs.
//   On (the Perform button in the top bar): a pad strip appears and, only while it's on,
//     A S D F G H J K L  effect pads on the chosen lane (loop 2 / 1 / ½ / ¼ beat, bar back, bar ahead, roll, chop, drop out)
//     Z X C V B N M      sample pads: Shift+drag across a clip to highlight a span, press the key to put it there;
//                        hold the key to play it on top, looping, on a layer lane of its own
//     1-9                pick the lane the effect pads play on
//   Space still plays / stops; all other keys stay Re-Tex's.
import { api } from "./api.js";
import { S, on, edit, uid, toast } from "./state.js";
import { currentHighlight, clearHighlight } from "./timeline.js";

const PADS = [
  { key: "a", label: "loop 2", desc: "loop the last 2 beats" },
  { key: "s", label: "loop 1", desc: "loop the last beat" },
  { key: "d", label: "loop ½", desc: "loop the last half beat" },
  { key: "f", label: "loop ¼", desc: "loop the last quarter beat" },
  { key: "g", label: "bar back", desc: "play from one bar earlier" },
  { key: "h", label: "bar ahead", desc: "play from one bar later" },
  { key: "j", label: "roll", desc: "repeat the hit, speeding up 1/4 > 1/8 > 1/16 > 1/32" },
  { key: "k", label: "chop", desc: "1/16 on, 1/16 off" },
  { key: "l", label: "drop out", desc: "silence" },
];
const PAD_KEYS = new Set(PADS.map((p) => p.key));
const SAMPLE_KEYS = ["z", "x", "c", "v", "b", "n", "m"];
const KEY_OF = (e) => { const m = /^Key([A-Z])$/.exec(e.code || ""); return m ? m[1].toLowerCase() : (e.key || "").toLowerCase(); };

let active = false;
let st = { quant: 0.25, gate: 1, triplet: false, sampleQuant: "auto", lane: null };
try { st = { ...st, ...JSON.parse(localStorage.getItem("perform") || "{}"), lane: null }; } catch {}
const persist = () => { try { const { lane, ...keep } = st; localStorage.setItem("perform", JSON.stringify(keep)); } catch {} };

const held = new Map();      // key -> { req: Promise<{id}>, track, sample?, layer? }
const down = new Set();      // keys physically held right now (updated at once, before any network round trip)
let effectStack = [];        // effect keys held, oldest first (the newest one plays)
let bar = null, btn = null;

const esc = (t) => String(t ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const samples = () => S.project?.perform_samples || {};
const trackName = (id) => S.project?.tracks.find((t) => t.id === id)?.name || "?";
function fmtLen(b) {
  const bpb = S.project?.bpb || 4;
  if (b >= bpb && Math.abs(b / bpb - Math.round(b / bpb)) < 1e-6) { const n = Math.round(b / bpb); return `${n} bar${n > 1 ? "s" : ""}`; }
  const named = { 1: "1 beat", 0.5: "½ beat", 0.25: "¼ beat", 0.125: "⅛ beat", 2: "2 beats", 3: "3 beats" };
  return named[Math.round(b * 1000) / 1000] || `${Math.round(b * 1000) / 1000} beats`;
}

// ---------------------------------------------------------------- the lane
function laneId() {
  const p = S.project;
  if (!p?.tracks.length) return null;
  if (st.lane && p.tracks.some((t) => t.id === st.lane)) return st.lane;
  const sel = [...(S.selClips || [])].map((id) => p.clips.find((c) => c.id === id)).find(Boolean);
  st.lane = sel?.track || p.tracks[0].id;
  return st.lane;
}

// ---------------------------------------------------------------- keys (called from main.js only while Perform is on)
export function performKey(e) {
  if (!active || e.ctrlKey || e.metaKey || e.altKey) return false;
  const k = KEY_OF(e);
  if (/^[1-9]$/.test(e.key)) {
    const t = S.project?.tracks[Number(e.key) - 1];
    if (t) { st.lane = t.id; render(); }
    return true;
  }
  if (!PAD_KEYS.has(k) && !SAMPLE_KEYS.includes(k)) return false;
  if (e.repeat) return true;
  down.add(k);
  if (SAMPLE_KEYS.includes(k)) {
    const h = currentHighlight();
    if (h && h.clips.length && h.t1 - h.t0 > 1e-6) { assign(k, h); return true; }
  }
  if (!S.transport?.playing) { toast("Perform pads play while the song plays: press Space first"); return true; }
  if (PAD_KEYS.has(k)) pressEffect(k); else pressSample(k);
  return true;
}
window.addEventListener("keyup", (e) => {
  if (!active) return;
  const k = KEY_OF(e);
  down.delete(k);
  if (PAD_KEYS.has(k) && effectStack.includes(k)) releaseEffect(k);
  else if (held.has(k)) releaseKey(k);
});
window.addEventListener("blur", () => { down.clear(); releaseAll(); });
on("transport", () => { if (!S.transport?.playing) { held.clear(); down.clear(); effectStack = []; render(); } });

function opts() { return { quant: st.quant, gate: st.gate, triplet: st.triplet }; }

function press(k, body, extra = {}) {
  const req = api.post("/api/perform/press", { key: k, opts: opts(), ...body }).catch((e) => { toast(`Perform: ${e.message}`, true); return null; });
  held.set(k, { req, ...extra });
  render();
  return req;
}
let takes = Promise.resolve();      // takes become clips one at a time, each against the project as it is then
async function releaseKey(k) {
  const h = held.get(k);
  held.delete(k);
  render();
  if (!h) return;
  const r = await h.req;
  if (!r) return;
  let rel;
  try { rel = await api.post("/api/perform/release", { id: r.id, quant: st.quant }); }   // the end is fixed now
  catch (e) { toast(`Perform: ${e.message}`, true); return; }
  if (!rel || rel.gone) return;
  takes = takes.then(async () => {
    try {
      const res = await api.post("/api/perform/take", { id: r.id, project: S.project, layer_track: h.layer?.id ?? null });
      if (res && !res.gone && res.take) applyTake(res, h);
    } catch (e) { toast(`Perform: ${e.message}`, true); }
  });
  return rel;                       // callers wait for the end to be fixed, not for the clips
}

// effect pads: one at a time per lane; a key held over another takes over, letting go falls back to the one under it
async function pressEffect(k) {
  const track = laneId();
  if (!track) return;
  const top = effectStack[effectStack.length - 1];
  effectStack.push(k);
  if (top && held.has(top)) await releaseKey(top);
  if (!down.has(k) || !effectStack.includes(k)) return;        // let go while the old take was being closed
  press(k, { track }, { track });
}
async function releaseEffect(k) {
  const wasTop = effectStack[effectStack.length - 1] === k;
  effectStack = effectStack.filter((x) => x !== k);
  await releaseKey(k);
  const under = effectStack[effectStack.length - 1];
  // fall back to the key still held underneath - only if it really still is (it may have gone up meanwhile)
  if (wasTop && under && down.has(under) && !held.has(under) && S.transport?.playing) press(under, { track: laneId() }, { track: laneId() });
}

// sample pads
function pressSample(k) {
  const smp = samples()[k];
  if (!smp) { toast(`${k.toUpperCase()} is empty: Shift+drag across a clip to highlight a span, then press ${k.toUpperCase()}`); return; }
  if (!S.project.tracks.some((t) => t.id === smp.track)) { toast(`${k.toUpperCase()}'s lane is gone - highlight a new span`, true); return; }
  const layer = S.project.tracks.find((t) => t.perform_layer?.key === k && t.perform_layer?.src === smp.track) || { id: uid("t"), fresh: true };
  press(k, { sample: { track: smp.track, start: smp.start, len: smp.len, quant: st.sampleQuant === "auto" || st.sampleQuant === "bar" ? st.sampleQuant : Number(st.sampleQuant) } },
    { track: smp.track, sample: k, layer });
}
function assign(k, h) {
  const c0 = S.project.clips.find((c) => h.clips.includes(c.id));
  if (!c0) return;
  edit((p) => { p.perform_samples = { ...(p.perform_samples || {}), [k]: { track: c0.track, start: h.t0, len: h.t1 - h.t0 } }; },
    { label: `Perform: put a sample on ${k.toUpperCase()}` });
  clearHighlight();
  toast(`${k.toUpperCase()} = ${fmtLen(h.t1 - h.t0)} of ${trackName(c0.track)}. Hold ${k.toUpperCase()} while it plays.`);
  render();
}

function releaseAll() {
  down.clear();
  for (const k of [...new Set([...held.keys(), ...effectStack])]) (PAD_KEYS.has(k) ? releaseEffect(k) : releaseKey(k));
}

// ---------------------------------------------------------------- a take -> clips (one undo step)
function applyTake(res, h) {
  const take = res.take;
  const label = h.sample ? `Perform: sample ${h.sample.toUpperCase()}` : `Perform: ${PADS.find((p) => p.key === res.key)?.label || res.key}`;
  edit((p) => {
    if (h.sample && !p.tracks.some((t) => t.id === take.replace.track)) {
      // the layer lane: a copy of the sample's lane settings, right under it (and its other layers)
      const src = p.tracks.find((t) => t.id === h.track);
      if (!src) return;
      const lt = { ...JSON.parse(JSON.stringify(src)), id: take.replace.track, name: `${h.sample.toUpperCase()} · ${src.name}`, solo: false,
        perform_layer: { key: h.sample, src: src.id } };
      let at = p.tracks.indexOf(src) + 1;
      while (at < p.tracks.length && p.tracks[at].perform_layer?.src === src.id) at++;
      p.tracks.splice(at, 0, lt);
    }
    applyTakeClips(p, take);
    p.perform_seen = Math.max(p.perform_seen || 0, res.id);
  }, { label });
}

// readhead.js applyResult, except a left part keeps a seamless join where the take carries straight on (left_fade 0)
function applyTakeClips(p, r) {
  const trims = new Map((r.replace.trims || []).map((t) => [t.id, t]));
  const kept = [];
  for (const c of p.clips) {
    const t = trims.get(c.id);
    if (!t) { kept.push(c); continue; }
    if (t.left_len != null && t.left_len > 1e-6) kept.push({ ...c, len: t.left_len, fade_out: t.left_fade ?? 0.03 });
    if (t.right) kept.push({ ...t.right, id: uid("c"), track: c.track });
  }
  for (const n of r.clips) kept.push({ ...n, id: uid("c"), track: r.replace.track });
  p.clips = kept;
}

// ---------------------------------------------------------------- UI: the Perform button and the pad strip
function setActive(on) {
  active = on;
  if (!on) releaseAll();
  btn.classList.toggle("on", on);
  bar.hidden = !on;
  if (on) toast("Perform on: A-L effect pads, Z-M sample pads, 1-9 picks the lane. Turn it off to get Re-Tex's keys back.", false, 5000);
  render();
}
function render() {
  if (!bar || bar.hidden) return;
  const p = S.project;
  const lane = laneId();
  const smp = samples();
  bar.querySelector(".pf-lane").innerHTML = (p?.tracks || []).map((t, i) => `<option value="${esc(t.id)}" ${t.id === lane ? "selected" : ""}>${i < 9 ? `${i + 1} · ` : ""}${esc(t.name)}</option>`).join("");
  bar.querySelectorAll(".pf-pad").forEach((el) => {
    const k = el.dataset.k;
    el.classList.toggle("held", held.has(k));
    if (el.dataset.s) {
      const m = smp[k];
      el.classList.toggle("empty", !m);
      el.querySelector("span").textContent = m ? `${fmtLen(m.len)} · ${trackName(m.track).slice(0, 9)}` : "empty";
      el.title = m ? `${fmtLen(m.len)} of ${trackName(m.track)} - hold ${k.toUpperCase()} while it plays` : `empty - Shift+drag across a clip, then press ${k.toUpperCase()}`;
    }
  });
}
on("project", render);

function build() {
  const style = document.createElement("style");
  style.textContent = `
    #performBar { position: fixed; left: 50%; bottom: 14px; transform: translateX(-50%); z-index: 60; display: flex; flex-direction: column; gap: 6px;
      padding: 8px 10px; border-radius: 12px; background: rgba(18,18,20,.94); border: 1px solid var(--line2); box-shadow: 0 10px 30px rgba(0,0,0,.5); backdrop-filter: blur(8px); }
    #performBar[hidden] { display: none; }
    #performBar .pf-row { display: flex; align-items: center; gap: 5px; flex-wrap: wrap; }
    #performBar .pf-t { font-size: 10px; letter-spacing: .08em; color: var(--dim); font-weight: 700; width: 58px; }
    #performBar .pf-pad { width: 60px; text-align: center; padding: 3px 2px 2px; border-radius: 7px; border: 1px solid var(--line2); background: rgba(255,255,255,.04);
      line-height: 1.15; user-select: none; cursor: pointer; border-bottom: 2px solid var(--accent); }
    #performBar .pf-pad b { display: block; font-size: 13px; color: var(--text); }
    #performBar .pf-pad span { font-size: 10px; color: var(--dim); white-space: nowrap; overflow: hidden; display: block; }
    #performBar .pf-pad.s { width: 84px; border-bottom-color: #63e6be; }
    #performBar .pf-pad.empty { opacity: .45; border-bottom-color: var(--line2); }
    #performBar .pf-pad.held { background: var(--accent); }
    #performBar .pf-pad.s.held { background: #63e6be; }
    #performBar .pf-pad.held b, #performBar .pf-pad.held span { color: #071a0e; }
    #performBar label { font-size: 11px; color: var(--dim); display: flex; align-items: center; gap: 4px; margin-left: 6px; }
    #performBar select { font-size: 11px; }`;
  document.head.appendChild(style);

  btn = document.createElement("button");
  btn.id = "btnPerform";
  btn.className = "toggle";
  btn.title = "Perform mode: play chops (A-L) and samples (Z-M) from the keyboard while the song plays; each take becomes clips (Ctrl+Z undoes it). While it's on those keys are pads; turn it off and every Re-Tex key is back.";
  btn.innerHTML = '<svg class="ic" viewBox="0 0 16 16" aria-hidden="true"><rect x="2" y="3" width="5" height="4.2" rx="1" fill="none" stroke="currentColor" stroke-width="1.3"/><rect x="9" y="3" width="5" height="4.2" rx="1" fill="none" stroke="currentColor" stroke-width="1.3"/><rect x="2" y="8.8" width="5" height="4.2" rx="1" fill="none" stroke="currentColor" stroke-width="1.3"/><rect x="9" y="8.8" width="5" height="4.2" rx="1" fill="currentColor"/></svg><span>Perform</span>';
  btn.onclick = () => setActive(!active);
  const follow = document.getElementById("btnFollow");
  follow.parentNode.insertBefore(btn, follow.nextSibling);

  bar = document.createElement("div");
  bar.id = "performBar";
  bar.hidden = true;
  bar.innerHTML = `
    <div class="pf-row"><span class="pf-t">PADS</span>
      ${PADS.map((p) => `<div class="pf-pad" data-k="${p.key}" title="${esc(p.desc)} (hold ${p.key.toUpperCase()})"><b>${p.key.toUpperCase()}</b><span>${esc(p.label)}</span></div>`).join("")}
      <label title="The lane the effect pads play on (1-9)">Lane <select class="pf-lane"></select></label>
      <label title="Effects start and stop on this grid">Quantize <select class="pf-q"><option value="1">beat</option><option value="0.5">1/8</option><option value="0.25">1/16</option><option value="0.125">1/32</option></select></label>
      <label title="Space between repeats: 100 = a continuous loop, lower cuts each repeat short">Gap <input class="pf-gate" type="range" min="10" max="100" step="5" style="width:70px"></label>
      <label title="Loop pads in triplets (½ becomes ⅓, ¼ becomes ⅙)"><input class="pf-trip" type="checkbox">Triplets</label>
    </div>
    <div class="pf-row"><span class="pf-t">SAMPLES</span>
      ${SAMPLE_KEYS.map((k) => `<div class="pf-pad s" data-k="${k}" data-s="1"><b>${k.toUpperCase()}</b><span>empty</span></div>`).join("")}
      <label title="Grid a sample starts on. Auto = its own length up to a bar, so a 4-bar loop lands on the bar and a 1-beat hit on the beat.">Start on <select class="pf-sq"><option value="auto">auto</option><option value="bar">bar</option><option value="1">beat</option><option value="0.5">1/8</option><option value="0.25">1/16</option></select></label>
      <span style="font-size:11px;color:var(--dim);margin-left:6px">Shift+drag a clip to highlight, then press Z-M</span>
    </div>`;
  document.body.appendChild(bar);
  const $ = (s) => bar.querySelector(s);
  $(".pf-q").value = String(st.quant); $(".pf-gate").value = String(Math.round(st.gate * 100)); $(".pf-trip").checked = st.triplet; $(".pf-sq").value = String(st.sampleQuant);
  $(".pf-q").onchange = (e) => { st.quant = Number(e.target.value); persist(); e.target.blur(); };
  $(".pf-gate").oninput = (e) => { st.gate = Number(e.target.value) / 100; persist(); };
  $(".pf-gate").onchange = (e) => e.target.blur();
  $(".pf-trip").onchange = (e) => { st.triplet = e.target.checked; persist(); e.target.blur(); };
  $(".pf-sq").onchange = (e) => { st.sampleQuant = e.target.value; persist(); e.target.blur(); };
  $(".pf-lane").onchange = (e) => { st.lane = e.target.value; e.target.blur(); };
  bar.querySelectorAll(".pf-pad").forEach((el) => {
    const k = el.dataset.k;
    el.onpointerdown = (e) => {
      e.preventDefault();
      if (el.dataset.s) { const h = currentHighlight(); if (h && h.clips.length && h.t1 - h.t0 > 1e-6) { assign(k, h); return; } }
      if (!S.transport?.playing) { toast("Perform pads play while the song plays: press Space first"); return; }
      el.setPointerCapture(e.pointerId);
      down.add(k);
      if (el.dataset.s) pressSample(k); else pressEffect(k);
    };
    el.onpointerup = el.onpointercancel = () => { down.delete(k); if (el.dataset.s ? held.has(k) : effectStack.includes(k)) (el.dataset.s ? releaseKey(k) : releaseEffect(k)); };
  });
}
build();
