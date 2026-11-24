// MIDI as instructions for the project's lanes: glitch scores (mashlab/glitch.py), arrangement maps, chord following
// and groove transplant (mashlab/midi_remix.py). Each dialog sends the project as it is, gets the whole new clip list
// back and applies it as one undo step; "Again" re-does it from the lanes as they were before (another seed / setting).
import { api } from "./api.js";
import { S, edit, toast } from "./state.js";
import { modal, showMenu } from "./ui.js";

const esc = (t) => String(t ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

export function remixMenu(e, sid) {
  showMenu(e, [
    ["Use this MIDI on the project's lanes"],
    ["Glitch score… (its notes play glitch effects)", () => glitchDialog(sid)],
    ["Arrangement map… (its parts bring lanes in and out)", () => arrangeDialog(sid)],
    ["Chord follow… (lanes follow its chords)", () => chordDialog(sid)],
    ["Groove… (lanes take its swing)", () => grooveDialog(sid)],
  ]);
}

// ---------------------------------------------------------------- shared bits
const bpb = () => S.project?.bpb || 4;
const projBars = () => {
  const end = Math.max(0, ...(S.project?.clips || []).map((c) => c.start + c.len));
  return Math.max(1, Math.ceil(end / bpb()));
};
const laneList = () => (S.project?.tracks || []).filter((t) => (S.project.clips || []).some((c) => c.track === t.id));

function rangeFields(midiBars) {
  return `
    <div class="row-gap" style="flex-wrap:wrap;margin-top:8px">
      <label>From project bar <input data-f="start_bar" type="number" min="1" value="1" style="width:64px"></label>
      <label>for <input data-f="bars" type="number" min="1" value="${projBars()}" style="width:64px"> bars</label>
      <label title="Which bar of the MIDI to start from (empty = its first bar with notes)">MIDI from bar <input data-f="midi_bar" type="number" min="1" placeholder="auto" style="width:64px"></label>
      <label title="How many bars of the MIDI to use; with 'repeat' they loop to fill the range">MIDI bars <input data-f="midi_bars" type="number" min="1" value="${midiBars}" style="width:64px"></label>
      <label><input data-f="loop" type="checkbox" checked> repeat</label>
    </div>`;
}
function lanesField(pick) {
  const ls = laneList();
  if (!ls.length) return `<p style="color:#e0a050">The project has no clips yet - put some stems on lanes first.</p>`;
  return `<div style="margin-top:8px"><span class="dim">Lanes</span>
    <div style="display:flex;flex-wrap:wrap;gap:4px 12px;margin-top:4px">${ls.map((t) => `<label><input type="checkbox" data-lane="${esc(t.id)}" ${pick(t) ? "checked" : ""}> ${esc(t.name)}</label>`).join("")}</div></div>`;
}
function readFields(box) {
  const o = {};
  box.querySelectorAll("[data-f]").forEach((el) => {
    o[el.dataset.f] = el.type === "checkbox" ? el.checked : el.value === "" ? null : +el.value;
  });
  o.tracks = [...box.querySelectorAll("[data-lane]:checked")].map((el) => el.dataset.lane);
  return o;
}
async function midiInfo(sid) {
  const d = await api.get(`/api/midi/${encodeURIComponent(sid)}`);
  const last = Math.max(0, ...d.parts.flatMap((p) => p.notes.map((n) => n[0] + n[1])));
  const first = Math.min(...d.parts.flatMap((p) => p.notes.map((n) => n[0])), last);
  d.noteBars = Math.max(1, Math.ceil((last - Math.floor((first - (d.first_beat || 0)) / d.bpb) * d.bpb - (d.first_beat || 0)) / d.bpb));
  return d;
}
// apply a result; "again" re-runs on the clips as they were before the first apply
function runner(label, url, buildBody, describe) {
  let before = null;
  return async (again = false) => {
    if (!S.project) { toast("Open a project first", true); return; }
    const base = again && before ? { ...S.project, clips: before } : S.project;
    let r;
    try { r = await api.post(url, { project: base, ...buildBody() }); }
    catch (e) { toast(e.message, true); return null; }
    if (!again || !before) before = JSON.parse(JSON.stringify(S.project.clips));
    edit((p) => { p.clips = r.clips; }, { label });
    toast(describe(r), false, 6000);
    return r;
  };
}

// ---------------------------------------------------------------- glitch score
const PRESETS = {
  tasteful: null,                                    // the built-in weights
  busy: { st16: 5, st32: 5, st8: 4, re16: 5, gate16: 5, gate32: 4, pat3: 3, pat2: 3, st8t: 3, st16t: 3, echo: 2, dive: 2, scram: 3 },
  stutter: { st16: 5, st32: 4, st8: 3, st4: 1, re16: 4, re8: 2, st8t: 2, st16t: 2 },
  chops: { gate16: 5, gate32: 4, gate8: 3, pat3: 4, pat2: 4, scram: 2 },
  chaos: "all",
};
export async function glitchDialog(sid) {
  const [d, cat] = await Promise.all([midiInfo(sid), api.get("/api/glitch/effects")]);
  const drum = d.parts.find((p) => p.drum) || d.parts[0];
  const m = modal(`Glitch score - ${d.title}`, `
    <p class="lead">Every note of the part plays a glitch effect on the lanes, at the same place and for as long as the note.
      Its busiest notes get the busiest effects; roll, drop out, bar back and the other rare ones wait out a cool-down.</p>
    <div class="row-gap" style="flex-wrap:wrap">
      <label>Part <select id="glPart">${d.parts.map((p) => `<option value="${esc(p.id)}">${esc(p.name)} (${p.notes.length})</option>`).join("")}</select></label>
      <label title="How many of the notes play an effect">Notes used <input id="glDen" type="number" min="5" max="100" step="5" value="60" style="width:60px"> %</label>
      <label>Effects <select id="glAssign"><option value="mix">by note, with variety</option><option value="note">by note (same note, same effect)</option><option value="chance">by chance (weighted)</option></select></label>
      <label title="Rare effects can't come again within this many bars">Rare every <input id="glCool" type="number" min="0" max="16" value="2" style="width:50px"> bars</label>
      <label>Seed <input id="glSeed" type="number" min="1" value="1" style="width:56px"></label>
    </div>
    ${rangeFields(Math.min(d.noteBars, 8))}
    ${lanesField((t) => true)}
    <div class="row-gap" style="margin-top:10px;flex-wrap:wrap"><span class="dim">Style</span>
      ${Object.keys(PRESETS).map((k) => `<button class="small" data-pre="${k}">${k}</button>`).join("")}
      <span class="dim" style="font-size:12px">weights: 0 = never, 5 = most</span></div>
    <div id="glW" style="display:grid;grid-template-columns:repeat(auto-fill,minmax(205px,1fr));gap:3px 12px;margin-top:6px;max-height:190px;overflow:auto">
      ${cat.effects.map((e) => `<label style="display:flex;justify-content:space-between;gap:6px"><span>${esc(e.label)}${e.rare ? ' <span class="dim">(rare)</span>' : ""}</span>
        <input type="number" min="0" max="5" data-w="${e.id}" value="${e.weight}" style="width:44px"></label>`).join("")}
    </div>
    <div class="row-gap" style="margin-top:12px"><span id="glOut" class="dim" style="flex:1"></span>
      <button id="glAgain" class="small" disabled title="Undo-free re-roll: the same lanes as before, another seed">Again (new seed)</button>
      <button id="glGo" class="primary">Glitch it</button></div>`, { wide: true });
  const B = m.body, $ = (s) => B.querySelector(s);
  $("#glPart").value = drum.id;
  B.querySelectorAll("[data-pre]").forEach((b) => (b.onclick = () => {
    const pre = PRESETS[b.dataset.pre];
    B.querySelectorAll("[data-w]").forEach((el) => {
      const e = cat.effects.find((x) => x.id === el.dataset.w);
      el.value = pre === null ? e.weight : pre === "all" ? 3 : (pre[e.id] ?? 0);
    });
    if (b.dataset.pre === "chaos") $("#glCool").value = 1;
  }));
  const body = () => ({
    midi: sid, part: $("#glPart").value, ...readFields(B),
    opts: { density: +$("#glDen").value, assign: $("#glAssign").value, cooldown_bars: +$("#glCool").value, seed: +$("#glSeed").value,
      weights: Object.fromEntries([...B.querySelectorAll("[data-w]")].map((el) => [el.dataset.w, +el.value || 0])) },
  });
  const run = runner("Glitch score", "/api/glitch/apply", body, (r) =>
    `Glitched: ${r.count} effects (${Object.entries(r.effects).sort((a, b) => b[1] - a[1]).slice(0, 6).map(([k, v]) => `${v} ${k}`).join(", ")}${Object.keys(r.effects).length > 6 ? "…" : ""})`);
  const show = (r) => { if (r) { $("#glOut").textContent = `${r.count} effects placed - Ctrl+Z undoes it`; $("#glAgain").disabled = false; } };
  $("#glGo").onclick = async () => show(await run(false));
  $("#glAgain").onclick = async () => { $("#glSeed").value = +$("#glSeed").value + 1; show(await run(true)); };
}

// ---------------------------------------------------------------- arrangement map
function guessLane(p, lanes) {
  const by = (role) => lanes.find((t) => t.role === role || new RegExp(role, "i").test(t.name));
  if (p.drum) return by("drums");
  if (p.program >= 32 && p.program <= 39) return by("bass");
  if (/vox|vocal|voice|choir|вокал/i.test(p.name) || (p.program >= 52 && p.program <= 54)) return by("vocals");
  return null;
}
export async function arrangeDialog(sid) {
  const d = await midiInfo(sid);
  const lanes = laneList();
  const m = modal(`Arrangement map - ${d.title}`, `
    <p class="lead">Tie the MIDI's parts to lanes. Wherever a part is silent for a bar, its lane drops out there - the MIDI
      song's arrangement shapes yours.</p>
    <div style="display:grid;grid-template-columns:1fr auto;gap:4px 12px;align-items:center;max-height:240px;overflow:auto">
      ${d.parts.map((p) => {
        const g = guessLane(p, lanes);
        return `<span>${esc(p.name)} <span class="dim">· ${esc(p.instrument_label)} · ${p.notes.length} notes</span></span>
          <select data-map="${esc(p.id)}"><option value="">-</option>${lanes.map((t) => `<option value="${esc(t.id)}" ${g?.id === t.id ? "selected" : ""}>${esc(t.name)}</option>`).join("")}</select>`;
      }).join("")}
    </div>
    ${rangeFields(d.noteBars)}
    <div class="row-gap" style="margin-top:6px"><label title="A bar counts as playing if the part has at least this many notes sounding in it">Playing = at least <input data-f="min_notes" type="number" min="1" value="1" style="width:50px"> note(s) in the bar</label></div>
    <div class="row-gap" style="margin-top:12px"><span id="arOut" class="dim" style="flex:1"></span><button id="arGo" class="primary">Arrange it</button></div>`, { wide: true });
  const B = m.body;
  const body = () => ({ midi: sid, ...readFields(B), map: Object.fromEntries([...B.querySelectorAll("[data-map]")].map((s) => [s.dataset.map, s.value]).filter(([, v]) => v)) });
  const run = runner("Arrangement map", "/api/midi/arrange", body, (r) =>
    "Arranged: " + Object.entries(r.dropped).map(([t, n]) => `${(S.project.tracks.find((x) => x.id === t) || {}).name || t} out for ${n} bars`).join(", "));
  B.querySelector("#arGo").onclick = async () => {
    if (!Object.keys(body().map).length) { toast("Tie at least one part to a lane", true); return; }
    const r = await run(false);
    if (r) B.querySelector("#arOut").textContent = "Done - Ctrl+Z undoes it";
  };
}

// ---------------------------------------------------------------- chord follow
export async function chordDialog(sid) {
  const d = await midiInfo(sid);
  const m = modal(`Chord follow - ${d.title}`, `
    <p class="lead">Reads the MIDI's chords bar by bar and shifts the lanes so their own chord lands on it (never more than
      6 semitones). Works best on lanes that play chords or riffs: bass, guitar, keys, pads.</p>
    <div><span class="dim">Chords from</span>
      <div style="display:flex;flex-wrap:wrap;gap:4px 12px;margin-top:4px">${d.parts.map((p) => `<label><input type="checkbox" data-part="${esc(p.id)}" ${p.drum ? "" : "checked"} ${p.drum ? "disabled" : ""}> ${esc(p.name)}</label>`).join("")}</div></div>
    ${rangeFields(d.noteBars)}
    ${lanesField((t) => /bass|other|guitar|key|pad/i.test(t.role + " " + t.name))}
    <div class="row-gap" style="margin-top:8px"><label>A lane's own chord from <select data-f2="mode">
      <option value="audio">its audio, bar by bar</option><option value="key">its song's key (steadier for one-note riffs)</option></select></label></div>
    <div id="cfOut" class="dim" style="margin-top:8px;font-size:12px"></div>
    <div class="row-gap" style="margin-top:10px"><span style="flex:1"></span><button id="cfGo" class="primary">Follow the chords</button></div>`, { wide: true });
  const B = m.body;
  const body = () => ({ midi: sid, ...readFields(B), mode: B.querySelector("[data-f2=mode]").value,
    parts: [...B.querySelectorAll("[data-part]:checked")].map((el) => el.dataset.part) });
  const run = runner("Chord follow", "/api/midi/chord_follow", body, (r) => `Following ${r.chords.filter(Boolean).length} bars of chords`);
  B.querySelector("#cfGo").onclick = async () => {
    const r = await run(false);
    if (!r) return;
    const sh = Object.values(r.shifts)[0] || [];
    B.querySelector("#cfOut").innerHTML = "Chords: " + r.chords.map((c, i) => `<b>${c || "-"}</b>${sh[i] != null && sh[i] !== 0 ? ` <span class="dim">(${sh[i] > 0 ? "+" : ""}${sh[i]})</span>` : ""}`).join(" · ") + " - Ctrl+Z undoes it";
  };
}

// ---------------------------------------------------------------- groove
export async function grooveDialog(sid) {
  const d = await midiInfo(sid);
  const drum = d.parts.find((p) => p.drum) || d.parts[0];
  const m = modal(`Groove - ${d.title}`, `
    <p class="lead">Measures how the part's notes push and lag against the 1/16 grid at each step of the bar (its swing and
      feel), then moves the hits of the lanes the same way - nothing is stretched, each hit's slice just moves.</p>
    <div class="row-gap" style="flex-wrap:wrap">
      <label>Groove of <select id="grPart">${d.parts.map((p) => `<option value="${esc(p.id)}">${esc(p.name)} (${p.notes.length})</option>`).join("")}</select></label>
      <label>Amount <input data-f="amount" type="number" min="0" max="100" step="10" value="100" style="width:60px"> %</label>
      <label title="Also takes how loud each step is (accents)"><input data-f="levels" type="checkbox"> accents too</label>
    </div>
    ${rangeFields(d.noteBars).replace(/<label title="Which bar[\s\S]*?<\/label>\s*<label title="How many bars[\s\S]*?<\/label>\s*<label><input data-f="loop"[^<]*<\/label>/, "")}
    ${lanesField((t) => /drum/i.test(t.role + " " + t.name))}
    <div id="grOut" class="dim" style="margin-top:8px;font-size:12px"></div>
    <div class="row-gap" style="margin-top:10px"><span style="flex:1"></span><button id="grGo" class="primary">Take the groove</button></div>`, { wide: true });
  const B = m.body;
  B.querySelector("#grPart").value = drum.id;
  const body = () => ({ midi: sid, part: B.querySelector("#grPart").value, ...readFields(B) });
  const run = runner("Groove", "/api/midi/groove_audio", body, () => "Grooved");
  B.querySelector("#grGo").onclick = async () => {
    const r = await run(false);
    if (r) B.querySelector("#grOut").textContent = `Its feel per ${r.grid} step (ms, + = late): ${r.groove.join(" ")} - Ctrl+Z undoes it`;
  };
}
