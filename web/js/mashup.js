// Auto-Mashup: pick songs (or candidates per role) -> ranked, ready-to-play arrangements.
import { api } from "./api.js";
import { S, on, setProject, toast } from "./state.js";
import { modal } from "./ui.js";
import { fit } from "./timeline.js";
import { play } from "./transport.js";
import { groups } from "./folders.js";

let pendingJob = null, m = null, pendingAutoplay = null, lastResults = null;

// selections survive closing the dialog (and reloads)
const saved = (() => { try { return JSON.parse(localStorage.getItem("mashupPicks") || "{}"); } catch { return {}; } })();
const sel = {
  quick: new Set(saved.quick || []),
  vocals: new Set(saved.vocals || []), music: new Set(saved.music || []),
  drums: new Set(saved.drums || []), bass: new Set(saved.bass || []),
};
const opts = Object.assign({ mode: "quick", vocalsOwn: false, vocalsNone: false, drumsOwn: true, bassOwn: true, density: 0.5, lyricsMode: "off" }, saved.opts || {});
const collapsed = new Set(saved.collapsed || []);
function persist() {
  try {
    localStorage.setItem("mashupPicks", JSON.stringify({
      ...Object.fromEntries(Object.entries(sel).map(([k, v]) => [k, [...v]])), opts, collapsed: [...collapsed],
    }));
  } catch {}
}

const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

/** Folder-grouped checklist. Folder checkbox = whole folder on/off (indeterminate when partly picked). */
function picker(el, set, { onlyVocals = false, key, onChange } = {}) {
  const ready = [...S.songs.values()].filter((s) => s.status === "ready" && s.kind !== "sample");
  for (const id of [...set]) if (!S.songs.has(id)) set.delete(id);
  el.innerHTML = `<div class="pk-tools"><input class="pk-search" placeholder="filter…" spellcheck="false">
    <button class="small pk-all" title="Select every song shown">all</button><button class="small pk-none" title="Clear">none</button>
    <span class="pk-count"></span></div><div class="pk-list"></div>`;
  const listEl = el.querySelector(".pk-list"), search = el.querySelector(".pk-search"), count = el.querySelector(".pk-count");
  let shown = [];
  let tmr = null;
  const changed = () => { if (onChange) { clearTimeout(tmr); tmr = setTimeout(onChange, 400); } };
  const draw = () => {
    const q = search.value.trim().toLowerCase();
    const songs = ready.filter((s) => !q || s.title.toLowerCase().includes(q));
    shown = songs.map((s) => s.id);
    listEl.innerHTML = "";
    for (const g of groups(songs)) {
      if (!g.songs.length) continue;
      const gid = (g.folder ? g.folder.id : "_unsorted") + "|" + key;
      const n = g.songs.filter((s) => set.has(s.id)).length;
      const hasFolders = S.folders.folders.length > 0;
      if (hasFolders) {
        const h = document.createElement("div");
        h.className = "pk-folder";
        h.innerHTML = `<span class="tw">${collapsed.has(gid) && !q ? "▸" : "▾"}</span><input type="checkbox"><span class="fn"></span><span class="cnt">${n}/${g.songs.length}</span>`;
        h.querySelector(".fn").textContent = g.folder ? g.folder.name : "Unsorted";
        const cb = h.querySelector("input");
        cb.checked = n === g.songs.length; cb.indeterminate = n > 0 && n < g.songs.length;
        cb.onclick = (e) => { e.stopPropagation(); const on = n < g.songs.length; g.songs.forEach((s) => on ? set.add(s.id) : set.delete(s.id)); persist(); draw(); changed(); };
        h.onclick = () => { collapsed.has(gid) ? collapsed.delete(gid) : collapsed.add(gid); persist(); draw(); };
        listEl.appendChild(h);
        if (collapsed.has(gid) && !q) continue;
      }
      for (const s of g.songs) {
        const row = document.createElement("label");
        const dim = onlyVocals && !s.has_vocals;
        row.className = "pk-song" + (set.has(s.id) ? " on" : "") + (dim ? " dim" : "") + (hasFolders ? " in" : "");
        row.title = dim ? "No clear vocal detected in this song - it will be skipped for vocals" : s.title;
        row.innerHTML = `<input type="checkbox" ${set.has(s.id) ? "checked" : ""}><span class="t"></span>
          <span class="tag">${Math.round(s.bpm)}</span><span class="tag key">${s.key?.camelot || ""}</span>${s.has_vocals ? '<span class="tag vox">vox</span>' : ""}`;
        row.querySelector(".t").textContent = s.title;
        row.querySelector("input").onchange = (e) => { e.target.checked ? set.add(s.id) : set.delete(s.id); persist(); draw(); changed(); };
        listEl.appendChild(row);
      }
    }
    if (!songs.length) listEl.innerHTML = `<div class="folder-empty">no songs</div>`;
    count.textContent = `${set.size} picked`;
  };
  search.oninput = draw;
  el.querySelector(".pk-all").onclick = () => { shown.forEach((id) => set.add(id)); persist(); draw(); changed(); };
  el.querySelector(".pk-none").onclick = () => { shown.forEach((id) => set.delete(id)); persist(); draw(); changed(); };
  draw();
}

export function openMashup() {
  const ready = [...S.songs.values()].filter((s) => s.status === "ready" && s.kind !== "sample");
  if (ready.length < 2) { toast("Add at least two songs first (one with vocals)", true); return; }
  if (!sel.quick.size) ready.forEach((s) => sel.quick.add(s.id));
  m = modal("Auto-Mashup", `
    <div class="tabs"><button data-tab="quick">Every pairing</button><button data-tab="roles">By roles</button></div>
    <div class="tab-quick">
      <p class="lead">Every song's vocal is arranged over every other song's music and the finished arrangement is rated -
        harmony note by note, tempo &amp; key changes, structure, groove, acapella quality. Duplicates, alternate versions and
        instrumentals of the same song are recognised from the audio and left out. The first run tracks every vocal melody
        (about a minute per song, spread over your CPU cores); results are kept, so later runs only rate new songs.</p>
      <details class="pk-wrap" id="pkWrap"><summary>Songs to include: <b id="pkN"></b> <span class="dim">(click to change)</span></summary>
        <div class="picker big" id="pkQuick"></div></details>
    </div>
    <div class="tab-roles">
      <p class="lead">Choose candidates for each part. The <b>music</b> song is the bed (its "other" stem + structure);
        vocals, drums and bass can each come from any of their candidates - every combination is built and rated.</p>
      <div class="roles">
        <div class="role"><h4><i class="sd vocals"></i>Vocals from</h4>
          <label class="opt"><input type="checkbox" id="oVOwn"> keep the music song's own vocals</label>
          <label class="opt"><input type="checkbox" id="oVNone"> no vocals (instrumental remix)</label>
          <div class="picker" id="pkVocals"></div></div>
        <div class="role"><h4><i class="sd other"></i>Music (bed) from</h4>
          <div class="opt-note">melody / chords / structure</div>
          <div class="picker" id="pkMusic"></div></div>
        <div class="role"><h4><i class="sd drums"></i>Drums from</h4>
          <label class="opt"><input type="checkbox" id="oDOwn"> keep the music song's own drums</label>
          <div class="picker" id="pkDrums"></div></div>
        <div class="role"><h4><i class="sd bass"></i>Bass from</h4>
          <label class="opt"><input type="checkbox" id="oBOwn"> keep the music song's own bass</label>
          <div class="picker" id="pkBass"></div></div>
      </div>
    </div>
    <div class="opts">
      <label title="How much of the arrangement carries the vocal">Vocal density
        <input type="range" id="dens" min="0" max="1" step="0.05" value="${opts.density}"><span id="densV"></span></label>
      <label title="Factor the two songs' lyrics into the rating. Resonance rewards lyrics that say the same thing; Contrast (secondary) rewards opposites. Off = music only (the default, unchanged rating). Only engages once both songs have been transcribed.">Lyrics
        <select id="lyrMode">
          <option value="off">off (music only)</option>
          <option value="resonance">resonance (same feel)</option>
          <option value="contrast">contrast (opposites)</option>
        </select></label>
      <button id="go" class="primary">Rate every pairing</button>
      <span id="prog" style="color:var(--dim)"></span>
    </div>
    <details class="auto-mashup" id="autoWrap">
      <summary>Auto-Mashup (2 songs → finished)</summary>
      <p class="lead">Pick exactly two songs above (Every-pairing list), add a style word, and the whole mashup is
        built end to end — separated, analysed, transcribed, aligned, blended and assembled — into a normal,
        editable project. Leave the style empty for an even 50/50.</p>
      <div class="row-gap">
        <input type="text" id="autoStyle" placeholder="style… e.g. remix · chopped and screwed · workout hype · contrast (or blank = 50/50)" spellcheck="false" style="flex:1">
        <button id="autoGo" class="primary">Build it</button>
      </div>
      <span id="autoProg" style="color:var(--dim)"></span>
    </details>
    <div class="variants" id="variants"></div>`, { wide: true, onClose: () => (m = null) });
  const B = m.body;
  const pkN = () => { B.querySelector("#pkN").textContent = `${sel.quick.size} of ${ready.length} songs`; };
  picker(B.querySelector("#pkQuick"), sel.quick, { key: "q", onChange: () => { pkN(); loadTable(); } });
  pkN();
  picker(B.querySelector("#pkVocals"), sel.vocals, { onlyVocals: true, key: "v" });
  picker(B.querySelector("#pkMusic"), sel.music, { key: "m" });
  picker(B.querySelector("#pkDrums"), sel.drums, { key: "d" });
  picker(B.querySelector("#pkBass"), sel.bass, { key: "b" });
  const bindOpt = (id, k) => { const c = B.querySelector(id); c.checked = !!opts[k]; c.onchange = () => { opts[k] = c.checked; persist(); }; };
  bindOpt("#oVOwn", "vocalsOwn"); bindOpt("#oVNone", "vocalsNone"); bindOpt("#oDOwn", "drumsOwn"); bindOpt("#oBOwn", "bassOwn");
  const setTab = (t) => {
    opts.mode = t; persist();
    B.querySelectorAll(".tabs button").forEach((b) => b.classList.toggle("on", b.dataset.tab === t));
    B.querySelector(".tab-quick").hidden = t !== "quick";
    B.querySelector(".tab-roles").hidden = t !== "roles";
    B.querySelector("#go").textContent = t === "quick" ? "Rate every pairing" : "Build & rate every combination";
    B.querySelector("#variants").innerHTML = "";
    if (t === "quick") loadTable(); else if (lastResults) showVariants(lastResults);
  };
  B.querySelectorAll(".tabs button").forEach((b) => b.onclick = () => setTab(b.dataset.tab));
  const dens = B.querySelector("#dens"), densV = B.querySelector("#densV");
  const showD = () => (densV.textContent = +dens.value < 0.3 ? "sparse" : +dens.value > 0.7 ? "dense" : "balanced");
  dens.oninput = () => { opts.density = +dens.value; persist(); showD(); };
  dens.onchange = () => { if (opts.mode === "quick") loadTable(); };
  showD();
  const lyr = B.querySelector("#lyrMode");
  lyr.value = opts.lyricsMode || "off";
  lyr.onchange = () => { opts.lyricsMode = lyr.value; persist(); if (opts.mode === "quick") loadTable(); };
  B.querySelector("#go").onclick = go;
  B.querySelector("#autoGo").onclick = autoMashup;
  setTab(opts.mode);
  if (pendingJob) B.querySelector("#prog").innerHTML = '<span class="spinner"></span> still working…';
}

// Auto-Mashup: two picked songs + a style word -> POST /api/mashup/auto -> a finished, editable project.
async function autoMashup() {
  const B = m.body, prog = B.querySelector("#autoProg");
  const songs = [...sel.quick];
  if (songs.length !== 2) { toast("Pick exactly two songs in the Every-pairing list for Auto-Mashup", true); return; }
  const prompt = B.querySelector("#autoStyle").value.trim();
  prog.innerHTML = '<span class="spinner"></span> starting the full build…';
  try {
    pendingJob = (await api.post("/api/mashup/auto", { songs, prompt, density: opts.density })).id;
  } catch (e) { prog.textContent = e.message; toast(e.message, true); }
}

async function go() {
  const B = m.body, prog = B.querySelector("#prog");
  if (opts.mode === "quick") {
    if (sel.quick.size < 2) { toast("Pick at least two songs", true); return; }
    prog.innerHTML = '<span class="spinner"></span> starting…';
    try { pendingJob = (await api.post("/api/mashup/rate", { songs: [...sel.quick], density: opts.density, lyrics_mode: opts.lyricsMode })).id; }
    catch (e) { prog.textContent = e.message; }
    return;
  }
  if (!sel.music.size) { toast("Pick at least one Music (bed) song", true); return; }
  const vocals = [...sel.vocals];
  if (opts.vocalsOwn) vocals.push("own");
  if (opts.vocalsNone) vocals.push("none");
  const drums = [...sel.drums]; if (opts.drumsOwn) drums.push("own");
  const bass = [...sel.bass]; if (opts.bassOwn) bass.push("own");
  if (!vocals.length) { toast("Pick vocal candidates, or tick 'keep own vocals' / 'no vocals'", true); return; }
  if (!drums.length || !bass.length) { toast("Pick drum and bass candidates, or tick 'keep own'", true); return; }
  B.querySelector("#variants").innerHTML = "";
  prog.innerHTML = '<span class="spinner"></span> searching…';
  try {
    pendingJob = (await api.post("/api/mashup/generate", { roles: { music: [...sel.music], vocals, drums, bass }, density: opts.density, max: 500, lyrics_mode: opts.lyricsMode })).id;
  } catch (e) { prog.textContent = e.message; }
}

on("event:job", (msg) => {
  if (msg.job.id !== pendingJob) return;
  // the Auto-Mashup job reports into its own progress line; everything else uses #prog
  const auto = msg.job.kind === "auto";
  const prog = m?.body.querySelector(auto ? "#autoProg" : "#prog");
  if (msg.job.status === "running" && prog) prog.innerHTML = `<span class="spinner"></span> ${esc(msg.job.message || "")} <span class="pct">${Math.round(100 * (msg.job.progress || 0))}%</span>`;
  if (msg.job.status === "error") { if (prog) prog.textContent = msg.job.error; pendingJob = null; }
  if (msg.job.status === "cancelled") { if (prog) prog.textContent = "cancelled"; pendingJob = null; }
  if (msg.job.status === "done" && msg.result) {
    pendingJob = null;
    if (auto) {                                       // a finished autonomous project: open it like a variant
      if (prog) prog.textContent = "";
      openVariant({ project: msg.result.project, desc: msg.result.desc });
      return;
    }
    if (msg.job.kind === "rate") { if (prog) prog.textContent = ""; loadTable(); return; }
    lastResults = msg.result.variants;
    if (prog) prog.textContent = lastResults.length ? `${lastResults.length} combinations - best first` : "";
    showVariants(lastResults);
  }
});

// ------------------------------------------------------------------ every-pairing results table

let table = null;                 // server table for the current selection
const tview = { sort: "rating", dir: -1, q: "", min: 0, shown: 150, open: null };
const COLS = [
  ["rating", "Rating"], ["vocal_title", "Vocal"], ["bed_title", "Music"], ["harmony", "Harmony"], ["tempo", "Tempo"],
  ["key", "Key"], ["structure", "Structure"], ["groove", "Groove"], ["acapella", "Acapella"],
];
let loadSeq = 0;

async function loadTable() {
  if (!m || opts.mode !== "quick") return;
  const box = m.body.querySelector("#variants");
  if (sel.quick.size < 2) { box.innerHTML = ""; table = null; return; }
  const seq = ++loadSeq;
  if (!table) box.innerHTML = `<div class="empty-hint"><span class="spinner"></span> loading ratings…</div>`;
  let t;
  try { t = await api.post("/api/mashup/table", { songs: [...sel.quick], density: opts.density, lyrics_mode: opts.lyricsMode }); }
  catch (e) { if (seq === loadSeq) box.innerHTML = `<div class="empty-hint">${esc(e.message)}</div>`; return; }
  if (seq !== loadSeq || !m) return;
  table = t;
  tview.shown = 150; tview.open = null;
  if (!t.rows.length) m.body.querySelector("#pkWrap")?.setAttribute("open", "");
  drawTable();
}

const colr = (x) => (x >= 85 ? "var(--good)" : x >= 65 ? "var(--warn)" : "var(--bad)");
const sgn = (x) => (x > 0 ? "+" : "") + x;

function explain(r) {
  const out = [];
  const h = r.harmony;
  const hw = h >= 90 ? "the vocal sits on these chords about as well as on its own song"
    : h >= 70 ? "mostly fits the chords - a few notes rub"
    : h >= 45 ? "noticeable clashes in places"
    : "clashes with the chords a lot";
  const j = r.judgeable ?? 1;
  out.push(`<b>Harmony ${h}%</b> - ${hw}` + (j < 0.5 ? ` (the vocal is mostly unpitched here - screamed, rapped or spoken - so key matters less; judged on the ${Math.round(100 * Math.min(1, j))}% that is sung)` : ""));
  const t = [];
  if (Math.abs(r.stretch_vocal) >= 0.1) t.push(`vocal ${sgn(r.stretch_vocal)}% speed`);
  if (Math.abs(r.stretch_bed) >= 0.1) t.push(`music ${sgn(r.stretch_bed)}% speed`);
  if (r.mult_vocal !== r.mult_bed) t.push(r.mult_vocal > r.mult_bed ? "vocal at half-time" : "vocal at double-time");
  out.push(`<b>Tempo ${r.tempo}%</b> - ${r.bpm} BPM; ${t.length ? t.join(", ") : "no stretching needed"}`);
  const k = [];
  if (r.shift_vocal) k.push(`vocal ${sgn(r.shift_vocal)} semitone${Math.abs(r.shift_vocal) > 1 ? "s" : ""}`);
  if (r.shift_bed) k.push(`music ${sgn(r.shift_bed)} semitone${Math.abs(r.shift_bed) > 1 ? "s" : ""}`);
  out.push(`<b>Key ${r.key}%</b> - ${k.length ? k.join(", ") : "no key change needed"}`);
  out.push(`<b>Structure ${r.structure}%</b> - vocal on ${r.vocal_bars} of ${r.total_bars} bars in ${r.entries} phrase${r.entries === 1 ? "" : "s"}, ` +
    `${Math.round(100 * r.hook_cover)}% of the big sections carry vocal, lines cut cleanly ${Math.round(100 * r.cuts)}%`);
  if (r.lyrics_mode && r.lyrics_mode !== "off" && (r.lyrics != null || r.lyrics_text)) {
    const lab = r.lyrics_mode === "contrast" ? "Lyrics (contrast)" : "Lyrics (resonance)";
    out.push(`<b>${lab}${r.lyrics != null ? ` ${r.lyrics}%` : ""}</b> - ${esc(r.lyrics_text || "the two lyrics compared")}`);
  }
  if (r.groove < 95) out.push(`<b>Groove ${r.groove}%</b> - the two have a different swing feel`);
  if (r.music != null && r.music < 100) out.push(`<b>Music ${r.music}%</b> - the backing is mostly drums / percussion, with little harmony under the vocal`);
  if (r.length != null && r.length < 100) out.push(`<b>Length ${r.length}%</b> - only ${r.total_bars} bars of the music are at its main tempo, so this mashup is short`);
  if (r.acapella < 85) out.push(`<b>Acapella ${r.acapella}%</b> - other instruments bleed into this vocal stem; right-click the song → Re-separate (Ultra) may clean it up`);
  if (r.also?.length) out.push(`<span style="color:var(--dim)">Same pairing with other versions of these songs: ${r.also.map(esc).join("; ")}</span>`);
  return out.map((x) => `<li>${x}</li>`).join("");
}

function drawTable() {
  if (!m || !table) return;
  const box = m.body.querySelector("#variants");
  const q = tview.q.toLowerCase();
  let rows = table.rows.filter((r) => r.rating >= tview.min && (!q || r.vocal_title.toLowerCase().includes(q) || r.bed_title.toLowerCase().includes(q)));
  const k = tview.sort, d = tview.dir;
  rows = rows.slice().sort((a, b) => (typeof a[k] === "string" ? a[k].localeCompare(b[k]) : (a[k] - b[k])) * d || (b.rating - a.rating));
  const skipped = Object.entries(table.skipped || {}).sort((a, b) => b[1].count - a[1].count);
  const skipTxt = skipped.map(([why, v]) => `${v.count} ${esc(why)}`).join(" · ");
  const groups = table.same_song_groups || [];
  const missing = Math.max(0, (table.expected || 0) - table.total);
  box.innerHTML = `
    <div class="rt-sum">
      <div><b>${table.rows.length}</b> mashups rated${skipTxt ? ` · not workable: ${skipTxt}` : ""}
        ${missing ? ` · <span style="color:var(--warn)">${missing} pairing${missing > 1 ? "s" : ""} not rated yet - press <i>Rate every pairing</i></span>` : ""}</div>
      ${groups.length ? `<details class="rt-dupes"><summary>${groups.length} group${groups.length > 1 ? "s" : ""} of songs recognised as the same song (never paired with each other)</summary>
        ${groups.map((g) => `<div>${g.map(esc).join("  =  ")}</div>`).join("")}</details>` : ""}
    </div>
    <div class="rt-tools">
      <input class="rt-q" placeholder="filter by song…" value="${esc(tview.q)}" spellcheck="false">
      <label>min rating <input type="range" class="rt-min" min="0" max="90" step="5" value="${tview.min}"><span class="rt-minv">${tview.min}</span></label>
      <span class="rt-n">${rows.length} shown · click a row for details</span>
    </div>
    <div class="rt">
      <div class="rt-row rt-head">${COLS.map(([key, lab]) => `<div class="c-${key} ${tview.sort === key ? "on" : ""}" data-k="${key}">${lab}${tview.sort === key ? (tview.dir < 0 ? " ▾" : " ▴") : ""}</div>`).join("")}</div>
      <div class="rt-body"></div>
    </div>
    ${rows.length > tview.shown ? `<button class="small rt-more">Show more (${rows.length - tview.shown} left)</button>` : ""}`;
  const body = box.querySelector(".rt-body");
  for (const r of rows.slice(0, tview.shown)) {
    const key = r.vocal + "|" + r.bed;
    const el = document.createElement("div");
    el.className = "rt-row" + (tview.open === key ? " open" : "");
    el.innerHTML = `<div class="c-rating"><span class="rt-score" style="border-color:${colr(r.rating)}">${Math.round(r.rating)}</span></div>
      <div class="c-vocal_title" title="${esc(r.vocal_title)}">${esc(r.vocal_title)}</div>
      <div class="c-bed_title" title="${esc(r.bed_title)}">${esc(r.bed_title)}</div>
      ${["harmony", "tempo", "key", "structure", "groove", "acapella"].map((c) => `<div class="c-${c}" style="color:${colr(r[c])}">${r[c]}</div>`).join("")}`;
    el.onclick = () => { tview.open = tview.open === key ? null : key; drawTable(); };
    body.appendChild(el);
    if (tview.open === key) {
      const det = document.createElement("div");
      det.className = "rt-detail";
      const alts = (r.alternatives || []).map((a) => `<button class="small alt" data-s="${a.shift_rel}" title="vocal ${sgn(a.shift_vocal)} st, music ${sgn(a.shift_bed)} st">other key (vocal ${sgn(a.shift_vocal)}, music ${sgn(a.shift_bed)}): ${Math.round(a.rating)}</button>`).join("");
      det.innerHTML = `<ul>${explain(r)}</ul><div class="row-gap"><button class="primary open">Open &amp; play</button>${alts}</div>`;
      det.querySelector(".open").onclick = (e) => { e.stopPropagation(); openPair(r, r.shift_rel); };
      det.querySelectorAll(".alt").forEach((b) => b.onclick = (e) => { e.stopPropagation(); openPair(r, +b.dataset.s); });
      body.appendChild(det);
    }
  }
  box.querySelectorAll(".rt-head > div").forEach((h) => h.onclick = () => {
    const k2 = h.dataset.k;
    if (tview.sort === k2) tview.dir *= -1; else { tview.sort = k2; tview.dir = k2.endsWith("title") ? 1 : -1; }
    drawTable();
  });
  const qi = box.querySelector(".rt-q");
  qi.oninput = () => {
    tview.q = qi.value; tview.shown = 150; drawTable();
    const v = m.body.querySelector(".rt-q"); v.focus(); v.setSelectionRange(v.value.length, v.value.length);
  };
  const mi = box.querySelector(".rt-min");
  mi.oninput = () => { box.querySelector(".rt-minv").textContent = mi.value; };
  mi.onchange = () => { tview.min = +mi.value; drawTable(); };
  box.querySelector(".rt-more")?.addEventListener("click", () => { tview.shown += 150; drawTable(); });
}

async function openPair(r, shift) {
  const prog = m?.body.querySelector("#prog");
  if (prog) prog.innerHTML = '<span class="spinner"></span> arranging…';
  try {
    const res = await api.post("/api/mashup/build", { vocal: r.vocal, bed: r.bed, shift_rel: shift, density: opts.density, lyrics_mode: opts.lyricsMode });
    if (prog) prog.textContent = "";
    openVariant({ project: res.project, desc: res.desc });
  } catch (e) { if (prog) prog.textContent = e.message; toast(e.message, true); }
}

const ICON = { music: '<i class="sd other"></i>', vocals: '<i class="sd vocals"></i>', drums: '<i class="sd drums"></i>', bass: '<i class="sd bass"></i>' };
const st = (x) => (x > 0 ? "+" : "") + x;

let vShown = 40;
function showVariants(vs, more = false) {
  if (!m) return;
  const box = m.body.querySelector("#variants");
  if (!vs.length) { box.innerHTML = `<div class="empty-hint">No workable combinations - try songs with closer tempos, or add a song with clear vocals.</div>`; return; }
  if (!more) vShown = 40;
  box.innerHTML = "";
  vs.slice(0, vShown).forEach((v) => {
    const d = v.desc;
    const col = d.rating >= 70 ? "var(--good)" : d.rating >= 45 ? "var(--warn)" : "var(--bad)";
    const el = document.createElement("div");
    el.className = "variant";
    const roles = (d.roles || []).map((r) => {
      const bits = [];
      if (!r.own) {
        if (r.shift) bits.push(`${st(r.shift)} st`);
        if (r.stretch) bits.push(`${st(r.stretch)}%`);
        if (r.mult && r.mult !== 1) bits.push(r.mult === 2 ? "half-time" : "double-time");
        if (r.harmony_pct != null) bits.push(`harmony ${r.harmony_pct}%`);
        if (r.bars != null && r.role !== "music") bits.push(`${r.bars} bars`);
      }
      return `<div class="rl"><span class="ri">${ICON[r.role] || ""}</span><span class="rn">${r.role}</span><b>${esc(r.title)}</b>${r.own ? '<span class="own">own</span>' : ""}<span class="rb">${bits.join(" · ")}</span></div>`;
    }).join("");
    el.innerHTML = `<div class="score" style="border-color:${col}" title="Overall fit">${d.rating}</div>
      <div><div class="d" style="margin:0 0 4px"><span class="tag">${d.bpm} BPM</span>
        <span class="tag">${Math.floor(d.length_sec / 60)}:${String(Math.round(d.length_sec % 60)).padStart(2, "0")}</span></div>
        <div class="roles-desc">${roles}</div></div>
      <div class="row-gap"><button class="primary open">Open &amp; play</button></div>`;
    el.querySelector(".open").onclick = () => openVariant(v);
    box.appendChild(el);
  });
  if (vs.length > vShown) {
    const b = document.createElement("button");
    b.className = "small"; b.textContent = `Show more (${vs.length - vShown} left)`;
    b.onclick = () => { vShown += 40; showVariants(vs, true); };
    box.appendChild(b);
  }
}

async function openVariant(v) {
  if (S.dirty && S.project?.clips.length && !confirm("Replace the current (unsaved) arrangement?")) return;
  if (!v.project && v.ref) {
    try { v.project = await api.get(`/api/mashup/variant/${v.ref[0]}/${v.ref[1]}`); }
    catch (e) { toast(e.message, true); return; }
  }
  const p = JSON.parse(JSON.stringify(v.project));
  setProject(p);
  fit();
  m?.close();
  pendingAutoplay = p.id;
  toast("Warping stems to the new tempo/key - playback starts when ready (usually under a minute)", false, 6000);
}

// start playback of a freshly opened mashup once its stems are rendered
on("renders", () => {
  if (!pendingAutoplay || S.project?.id !== pendingAutoplay) return;
  if ((S.waiting?.size || 0) === 0) {
    pendingAutoplay = null;
    // start at the first vocal entry minus a bar, so you hear the blend immediately
    const vox = S.project.clips.filter((c) => c.stem === "vocals").sort((a, b) => a.start - b.start)[0];
    const beat = vox ? Math.max(0, vox.start - S.project.bpb) : 0;
    play(beat * 60 / S.project.bpm);
  }
});
