// DJ Mix: two songs + what you want -> a language model plans the mix (POST /api/dj/mix), the arrangement engine
// plays it in time. Remix (default): one song is the bed as recorded, the other's vocal goes on top line by line,
// optionally a third song's drums. Blend: stems swapped between the songs part by part. Shows the DJ's plan and
// reasons, then opens it like any mashup.
import { api } from "./api.js";
import { S, on, setProject, toast } from "./state.js";
import { modal } from "./ui.js";
import { fit } from "./timeline.js";
import { play } from "./transport.js";

let m = null, pendingJob = null, pendingAutoplay = null, last = null;
const saved = (() => { try { return JSON.parse(localStorage.getItem("djPicks") || "{}"); } catch { return {}; } })();
const st = Object.assign({ a: "", b: "", c: "", prompt: "", minutes: 3, planner: "claude", style: "remix" }, saved);
const persist = () => { try { localStorage.setItem("djPicks", JSON.stringify(st)); } catch {} };
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

function songPicker(el, key, none = "") {
  const ready = [...S.songs.values()].filter((s) => s.status === "ready" && s.kind !== "sample").sort((x, y) => x.title.localeCompare(y.title));
  el.innerHTML = `<input class="dj-q" placeholder="filter…" spellcheck="false"><select class="dj-sel" size="7"></select>`;
  const q = el.querySelector(".dj-q"), sel = el.querySelector(".dj-sel");
  const fill = () => {
    const f = q.value.trim().toLowerCase();
    sel.innerHTML = (none ? `<option value=""${!st[key] ? " selected" : ""}>${esc(none)}</option>` : "")
      + ready.filter((s) => !f || s.title.toLowerCase().includes(f) || s.id === st[key])
      .map((s) => `<option value="${esc(s.id)}"${s.id === st[key] ? " selected" : ""}>${esc(s.title)} · ${Math.round(s.bpm || 0)} BPM</option>`).join("");
  };
  q.oninput = fill;
  sel.onchange = () => { st[key] = sel.value; persist(); };
  fill();
}

export function openDjMix() {
  if (m) return;
  m = modal("DJ Mix", `
    <p class="lead">Pick two songs and say what you want. A language model plans the mix like a DJ and the engine
      plays it on the beat, in key, with every vocal line whole. <b>Remix</b>: one song's music plays as recorded and
      the other sings or raps on top of it - verses, looped hooks, stutters, a drop (the song that fits best as the
      bed is picked for you); a third song can bring its drums. <b>Blend</b>: the two songs' stems swapped part by
      part. You get a normal project you can edit.</p>
    <div class="dj-pick"><div><b>Song A</b><div id="djA"></div></div><div><b>Song B</b><div id="djB"></div></div>
      <div id="djCWrap"><b>Drums from</b> <span class="dim">(optional)</span><div id="djC"></div></div></div>
    <textarea id="djPrompt" rows="2" spellcheck="false" placeholder="What do you want? e.g. start with B's vocal over A's intro, slow build, big drop on the hook, end instrumental (or leave empty)"></textarea>
    <div class="row-gap">
      <label title="Remix: one song is the bed, as recorded, the other's vocal on top line by line. Blend: stems swapped between the songs part by part.">Style <select id="djStyle"><option value="remix">Remix</option><option value="blend">Blend</option></select></label>
      <label>Length <select id="djMin">${[2, 3, 4, 5, 6].map((n) => `<option value="${n}">${n} min</option>`).join("")}</select></label>
      <label title="Who plans the mix. Claude runs through Claude Code on your own Claude login (no API key; it uses your plan's usage, about a minute a mix). Gemma runs here, offline.">Planner <select id="djPlanner"><option value="claude">Claude (your Claude login)</option><option value="local">Gemma (local, offline)</option></select></label>
      <button id="djGo" class="primary">Plan &amp; build</button>
      <span id="djProg" style="color:var(--dim)"></span>
    </div>
    <div id="djOut"></div>`, { wide: true, onClose: () => (m = null) });
  const B = m.body;
  songPicker(B.querySelector("#djA"), "a");
  songPicker(B.querySelector("#djB"), "b");
  songPicker(B.querySelector("#djC"), "c", "(none - the bed's own drums)");
  const pr = B.querySelector("#djPrompt"), mn = B.querySelector("#djMin"), sy = B.querySelector("#djStyle");
  pr.value = st.prompt; mn.value = String(st.minutes);
  pr.oninput = () => { st.prompt = pr.value; persist(); };
  mn.onchange = () => { st.minutes = +mn.value; persist(); };
  const styleShow = () => { B.querySelector("#djCWrap").style.display = sy.value === "remix" ? "" : "none"; };
  sy.value = st.style === "blend" ? "blend" : "remix";
  sy.onchange = () => { st.style = sy.value; persist(); styleShow(); };
  styleShow();
  const pl = B.querySelector("#djPlanner");
  pl.value = st.planner === "local" ? "local" : "claude";
  pl.onchange = () => { st.planner = pl.value; persist(); plannerHint(); };
  const plannerHint = async () => {               // say so up front when Claude can't plan (not signed in)
    const prog = B.querySelector("#djProg");
    if (pl.value !== "claude" || pendingJob) return;
    try {
      const r = await api.get("/api/dj/planners");
      if (!r.claude.ok && pl.value === "claude") prog.textContent = r.claude.why;
      else if (prog.textContent.startsWith("Claude Code")) prog.textContent = "";
    } catch { /* the plan request reports it */ }
  };
  plannerHint();
  B.querySelector("#djGo").onclick = () => start();
  if (pendingJob) B.querySelector("#djProg").innerHTML = '<span class="spinner"></span> still planning…';
  else if (last) showPlan(last);
}

async function start(seed) {
  const prog = m?.body.querySelector("#djProg");
  if (!st.a || !st.b || st.a === st.b) { toast("Pick two different songs", true); return; }
  if (prog) prog.innerHTML = '<span class="spinner"></span> starting…';
  m?.body.querySelector("#djOut")?.replaceChildren();
  const style = st.style === "blend" ? "blend" : "remix";
  const drums = style === "remix" && st.c && st.c !== st.a && st.c !== st.b ? st.c : null;
  try {
    pendingJob = (await api.post("/api/dj/mix", { songs: [st.a, st.b], style, drums, prompt: st.prompt.trim(), minutes: st.minutes,
      seed: seed ?? Math.floor(Math.random() * 1e6), planner: st.planner === "local" ? "local" : "claude" })).id;
  } catch (e) { if (prog) prog.textContent = e.message; toast(e.message, true); }
}

function showRemix(out, res) {
  const d = res.desc, R = d.roles, T = d.titles, mmss = (s) => `${Math.floor(s / 60)}:${String(Math.round(s % 60)).padStart(2, "0")}`;
  const semis = (n) => (n ? `${n > 0 ? "+" : ""}${n} st` : "own key");
  const yes = (b) => (b ? "on" : `<span class="dim">-</span>`);
  out.innerHTML = `
    <div class="dj-head"><b>${esc(d.title)}</b> · ${d.bars} bars · ${mmss(d.length_sec)} · ${d.bpm} BPM
      <span class="dim">· planned by ${esc(d.llm)}</span></div>
    <p class="dim"><span class="dj-a">Bed: ${esc(T[R.bed])}</span> (${semis(d.transpose[R.bed])}) ·
      <span class="dj-b">on top: ${esc(T[R.guest])}</span> (${semis(d.transpose[R.guest])})${R.donor ? ` · drums: ${esc(T[R.donor])}` : ""}</p>
    <table class="dj-plan"><thead><tr><th>#</th><th>bar</th><th>bars</th><th>bed plays</th><th>music</th><th>drums</th><th>bed vocal</th><th>drop</th><th>why</th></tr></thead><tbody>
      ${d.parts.map((p) => `<tr><td>${p.n}</td><td>${p.bar}</td><td>${p.bars}</td>
        <td class="dj-a">${p.carries_on ? `<span class="dim">carries on</span> (${esc(p.section)})` : `from ${esc(p.section)} ${esc(p.name)}`}</td>
        <td>${yes(p.music)}</td><td>${esc(p.drums)}</td><td>${yes(p.bed_vocal)}</td><td>${p.drop ? "drop" : ""}</td><td>${esc(p.why)}</td></tr>`).join("")}
    </tbody></table>
    <table class="dj-plan"><thead><tr><th>part · bar</th><th>line</th><th>words</th><th>times</th><th>stutter</th><th></th></tr></thead><tbody>
      ${d.lines.map((l) => `<tr${l.placed ? "" : ' class="dim"'}><td>${l.part} · ${l.bar}</td>
        <td class="${String(l.line).startsWith(R.bed) ? "dj-a" : "dj-b"}">${esc(l.line)}</td>
        <td>${l.words ? `"${esc(l.words.slice(0, 70))}"` : '<span class="dim">(no words heard)</span>'}</td>
        <td>${l.placed}${l.repeat > 1 ? ` of ${l.repeat}, every ${l.every}` : ""}</td><td>${l.stutter || ""}</td><td class="dim">${esc(l.note || "")}</td></tr>`).join("")}
    </tbody></table>
    ${d.repairs.length ? `<div class="dj-rep"><b>Fixed by the engine:</b><ul>${d.repairs.map((r) => `<li>${esc(r)}</li>`).join("")}</ul></div>` : ""}
    <div class="row-gap"><button id="djOpen" class="primary">Open in timeline ▶</button><button id="djAgain">↻ Another take</button></div>
    <details><summary class="dim">What the DJ was told</summary><pre class="dj-brief">${esc(d.brief)}</pre></details>`;
  out.querySelector("#djOpen").onclick = () => openResult(res);
  out.querySelector("#djAgain").onclick = () => start();
}

function showPlan(res) {
  const out = m?.body.querySelector("#djOut");
  if (!out) return;
  if (res.desc.kind === "dj-remix") return showRemix(out, res);
  const d = res.desc, fitTxt = (f) => (f ? `${Math.round(f.rating)}/100` : "doesn't fit");
  out.innerHTML = `
    <div class="dj-head"><b>${esc(d.title)}</b> · ${d.measures} measures · ${Math.floor(d.length_sec / 60)}:${String(Math.round(d.length_sec % 60)).padStart(2, "0")}
      · ${d.bpm} BPM <span class="dim">· planned by ${esc(d.llm)} · fit: A's vocal on B ${fitTxt(d.fit.AB)}, B's vocal on A ${fitTxt(d.fit.BA)}</span></div>
    <table class="dj-plan"><thead><tr><th>#</th><th>at</th><th>len</th><th>drums</th><th>bass</th><th>instruments</th><th>vocal</th><th>vocal 2</th><th>comes in</th><th>why</th></tr></thead><tbody>
      ${d.parts.map((p) => `<tr><td>${p.n}</td><td>m${p.start_measure}</td><td>${p.measures}</td>
        ${["drums", "bass", "instruments", "vocal", "vocal2"].map((k) => `<td class="${p[k].startsWith("A") ? "dj-a" : p[k].startsWith("B") ? "dj-b" : "dim"}">${esc(p[k])}</td>`).join("")}
        <td>${esc(p.transition)}</td><td>${esc(p.why)}</td></tr>`).join("")}
    </tbody></table>
    ${d.repairs.length ? `<div class="dj-rep"><b>Fixed by the engine:</b><ul>${d.repairs.map((r) => `<li>${esc(r)}</li>`).join("")}</ul></div>` : ""}
    <p class="dim"><span class="dj-a">A = ${esc(d.titles.A)}</span>, <span class="dj-b">B = ${esc(d.titles.B)}</span>. % = how well that stem fits the chords it is on (engine-checked). Transitions other than cut are a one-bar blend for now.</p>
    <div class="row-gap"><button id="djOpen" class="primary">Open in timeline ▶</button><button id="djAgain">↻ Another take</button></div>
    <details><summary class="dim">What the DJ was told</summary><pre class="dj-brief">${esc(d.brief)}</pre></details>`;
  out.querySelector("#djOpen").onclick = () => openResult(res);
  out.querySelector("#djAgain").onclick = () => start();
}

function openResult(res) {
  if (S.dirty && S.project?.clips.length && !confirm("Replace the current (unsaved) arrangement?")) return;
  const p = JSON.parse(JSON.stringify(res.project));
  setProject(p);
  fit();
  m?.close();
  pendingAutoplay = p.id;
  toast("Warping stems to the mix tempo/key - playback starts when ready", false, 6000);
}

on("event:job", (msg) => {
  if (msg.job.id !== pendingJob) return;
  const prog = m?.body.querySelector("#djProg");
  if (msg.job.status === "running" && prog) prog.innerHTML = `<span class="spinner"></span> ${esc(msg.job.message || "")} <span class="pct">${Math.round(100 * (msg.job.progress || 0))}%</span>`;
  if (msg.job.status === "error") { if (prog) prog.textContent = msg.job.error; else toast(msg.job.error, true); pendingJob = null; }
  if (msg.job.status === "cancelled") { if (prog) prog.textContent = "cancelled"; pendingJob = null; }
  if (msg.job.status === "done" && msg.result) {
    pendingJob = null;
    last = msg.result;
    if (prog) prog.textContent = "";
    if (m) showPlan(last); else toast("Your DJ mix is planned - open DJ Mix to see it");
  }
});

on("renders", () => {
  if (!pendingAutoplay || S.project?.id !== pendingAutoplay) return;
  if ((S.waiting?.size || 0) === 0) { pendingAutoplay = null; play(0); }
});
