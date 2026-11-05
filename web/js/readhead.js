// ReadHead: automatic beat-aligned chops for a lane (POST /api/readhead/run, mashlab/readhead.py).
//   Arrange vocal  re-arranges the vocal song on this lane against the rest of the project (Flow: reads forward
//                  with loops / alternations / stammers / ratchets / hook loops, the singer's bar phase kept, cuts
//                  only between syllables - learned from a hand-made edit; Lines: whole lines per
//                  section). Your section markers, if you've set any, steer it.
//   Glitch chops   chops the lane's own audio in place: stutters, rolls, repeats, skips, gates, displacements.
// The result is ordinary clips on THIS lane only, applied as one undo step (Ctrl+Z puts the lane back).
import { api } from "./api.js";
import { S, edit, uid, toast } from "./state.js";
import { modal } from "./ui.js";

const KEY = "readhead";
let st = { mode: "arrange", style: "flow", chop: 50, space: 35, seed: 1, scope: "song", symmetry: "four" };
try { st = { ...st, ...JSON.parse(localStorage.getItem(KEY) || "{}") }; } catch {}
const persist = () => { try { localStorage.setItem(KEY, JSON.stringify(st)); } catch {} };
const esc = (t) => String(t ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

let m = null, track = null, last = null, busy = false;

export function openReadHead(trackId, mode) {
  if (m) m.close();
  track = trackId;
  if (mode) st.mode = mode;
  const t = S.project.tracks.find((x) => x.id === trackId);
  const loop = S.project.loop;
  const bpb = S.project.bpb || 4;
  m = modal(`ReadHead - ${t?.name || "lane"}`, `
    <p class="lead">Automatic chops on the beat, written as normal clips on <b>this lane only</b> (one undo step).
      <b>Arrange vocal</b> rebuilds the vocal on this lane against everything else in the project: it reads forward through
      the vocal with loops, alternations, stammers and ratchets, keeps the singer's own beat 1 and only cuts between
      syllables. Mark sections (intro, verse, chorus, drop…) in the section lane to steer it. <b>Glitch chops</b> chops
      this lane's own audio in place.</p>
    <div class="row-gap" style="flex-wrap:wrap">
      <label>Mode <select id="rhMode"><option value="arrange">Arrange vocal</option><option value="glitch">Glitch chops</option></select></label>
      <label class="rh-arr" title="Flow: your Re-Tex way (read forward, loops, ratchets). Lines: whole vocal lines per section.">Style
        <select id="rhStyle"><option value="flow">Flow</option><option value="lines">Lines per section</option></select></label>
      <label title="How much chopping">Chop <input id="rhChop" type="range" min="0" max="100" step="5" style="width:110px"> <b id="rhChopV"></b></label>
      <label class="rh-arr" title="How much room to leave (verses resting a bar, sparser breakdowns)">Space <input id="rhSpace" type="range" min="0" max="100" step="5" style="width:90px"> <b id="rhSpaceV"></b></label>
      <label class="rh-arr" title="Each kind of section gets one treatment and one variation, decided once; repeats of the same material get the same chop. This sets where the variation falls.">Pattern
        <select id="rhSym"><option value="four">A A A B (variation every 4th)</option><option value="alt">A B A B (alternate)</option><option value="same">A A A A (always the same)</option></select></label>
      <label>Where <select id="rhScope"><option value="song">Whole project</option>${loop?.on ? `<option value="loop">Loop region (bars ${Math.floor(loop.start / bpb) + 1}-${Math.ceil(loop.end / bpb)})</option>` : ""}</select></label>
      <label>Seed <input id="rhSeed" type="number" min="0" step="1" style="width:80px"></label>
      <button id="rhGo" class="primary">Make it</button>
      <span id="rhProg" style="color:var(--dim)"></span>
    </div>
    <div id="rhOut"></div>`, { wide: true, onClose: () => (m = null) });
  const B = m.body;
  const $ = (id) => B.querySelector(id);
  $("#rhMode").value = st.mode; $("#rhStyle").value = st.style; $("#rhChop").value = st.chop; $("#rhSpace").value = st.space; $("#rhSym").value = st.symmetry || "four";
  $("#rhSeed").value = st.seed; $("#rhScope").value = st.scope === "loop" && loop?.on ? "loop" : "song";
  const sync = () => {
    $("#rhChopV").textContent = st.chop; $("#rhSpaceV").textContent = st.space;
    B.querySelectorAll(".rh-arr").forEach((el) => (el.style.display = st.mode === "arrange" ? "" : "none"));
  };
  $("#rhMode").onchange = (e) => { st.mode = e.target.value; persist(); sync(); };
  $("#rhStyle").onchange = (e) => { st.style = e.target.value; persist(); };
  $("#rhSym").onchange = (e) => { st.symmetry = e.target.value; persist(); };
  $("#rhChop").oninput = (e) => { st.chop = +e.target.value; persist(); sync(); };
  $("#rhSpace").oninput = (e) => { st.space = +e.target.value; persist(); sync(); };
  $("#rhScope").onchange = (e) => { st.scope = e.target.value; persist(); };
  $("#rhSeed").onchange = (e) => { st.seed = Math.max(0, Math.floor(+e.target.value || 0)); persist(); };
  $("#rhGo").onclick = () => run();
  sync();
}

async function run() {
  if (busy || !m) return;
  const B = m.body, prog = B.querySelector("#rhProg"), out = B.querySelector("#rhOut");
  const loop = S.project.loop;
  const range = st.scope === "loop" && loop?.on ? [loop.start, loop.end] : null;
  busy = true;
  prog.innerHTML = '<span class="spinner"></span> listening to the project… (the first run on a project takes longest)';
  out.replaceChildren();
  try {
    const opts = { chop: st.chop / 100, space: st.space / 100, seed: st.seed, style: st.style, symmetry: st.symmetry || "four" };
    last = await api.post("/api/readhead/run", { project: S.project, track, mode: st.mode, opts, range });
    last.trackId = track;
    prog.textContent = "";
    show(out, last);
  } catch (e) {
    prog.textContent = e.message;
    toast(`ReadHead: ${e.message}`, true, 8000);
  } finally { busy = false; }
}

function show(out, r) {
  const s = r.summary, v = s.vocal;
  const plan = new Map((r.plan || []).filter((p) => p.actions).map((p) => [p.startBar, p.actions.join(", ")]));
  out.innerHTML = `
    <p class="dim">${s.clips} clips${v ? ` · vocal: ${v.lines} lines, ${v.syllables} syllables, its beat 1 = beat ${v.barPhase + 1}${v.hooks?.length ? `, hook = lines ${v.hooks.join(", ")}` : ""}` : ""}
      ${s.counts ? ` · ${Object.entries(s.counts).map(([k, n]) => `${k} ${n}`).join(", ")}` : ""} · ${Math.round((s.ms || 0) / 100) / 10} s</p>
    ${r.warnings.length ? `<p style="color:var(--warn,#e0a050)">${r.warnings.map(esc).join("<br>")}</p>` : ""}
    <div style="max-height:300px;overflow:auto"><table class="dj-plan"><thead><tr><th>bars</th><th>section</th><th>the music</th>${s.mode === "arrange" ? "<th>the vocal</th>" : ""}</tr></thead><tbody>
      ${r.phrases.map((p) => `<tr><td>${p.startBar + 1}-${p.endBar}</td><td class="dim">${esc(p.section || "")}</td><td>${esc(p.role)}</td>${s.mode === "arrange" ? `<td>${esc(plan.get(p.startBar) || "")}</td>` : ""}</tr>`).join("")}
    </tbody></table></div>
    <div class="row-gap" style="margin-top:10px">
      <button id="rhApply" class="primary">Put it on the lane</button>
      <button id="rhAgain">↻ Another take</button>
      <span class="dim">Replaces ${s.mode === "arrange" ? "this vocal's clips" : "the clips"} on this lane${r.replace.end < 1e9 && r.replace.start > 0 ? " in the loop region" : ""}. Ctrl+Z undoes it.</span>
    </div>`;
  out.querySelector("#rhApply").onclick = () => {
    edit((p) => applyResult(p, r), { label: s.mode === "arrange" ? "ReadHead: arrange vocal" : "ReadHead: glitch chops" });
    toast(`ReadHead: ${s.clips} clips on the lane - Ctrl+Z to undo`);
  };
  out.querySelector("#rhAgain").onclick = () => {
    st.seed = (st.seed || 0) + 1; persist();
    m.body.querySelector("#rhSeed").value = st.seed;
    run();
  };
}

/** Replace the lane's clips in the range with the result's clips. Pure on the project object (tested in node).
 *  The server lists which existing clips are affected and what of them stays outside the range (r.replace.trims). */
export function applyResult(p, r, mkId = () => uid("c")) {
  const trims = new Map((r.replace.trims || []).map((t) => [t.id, t]));
  const kept = [];
  for (const c of p.clips) {
    const t = trims.get(c.id);
    if (!t) { kept.push(c); continue; }
    if (t.left_len != null && t.left_len > 1e-6) kept.push({ ...c, len: t.left_len, fade_out: 0.03 });
    if (t.right) kept.push({ ...t.right, id: mkId(), track: c.track });
  }
  for (const n of r.clips) kept.push({ ...n, id: mkId(), track: r.replace.track });
  p.clips = kept;
}
