// Library panel: the song list grouped by folder / artist / album / genre / ... (libgroups.js), import, delete.
import { api, dropPeaks } from "./api.js";
import { S, on, emit, toast, invalidateSong } from "./state.js";
import { modal, showMenu, showPicker } from "./ui.js";
import { forgetSong } from "./timeline.js";
import { loadFolders, groups, createFolder, renameFolder, deleteFolder, moveToFolder, toggleCollapsed, moveFolder, folderOf } from "./folders.js";
import { MODES, mode, setMode, sections, isCollapsed, toggleCollapsed as toggleSection, loadMeta, loadCompat, compareNow,
  guessGenres, listenSongs, dropOnSection, renameSection, editInfo, keyTile, artistOf, setMeta, SEPARATION, sepOf } from "./libgroups.js";
import { openStemImport, stemPicker, looksLikeStems } from "./stemimport.js";
import { songToolItems, openGrid } from "./songtools.js";

const list = document.getElementById("songList");
const search = document.getElementById("libSearch");
const fileInput = document.getElementById("fileInput");
const jobBySong = new Map();     // song id -> job
const libSel = new Set();        // multi-selection (Ctrl/Shift+click) for moving songs into folders
let lastClicked = null;
let visibleOrder = [];

export async function loadSongs() {
  const [songs] = await Promise.all([api.get("/api/songs"), loadFolders()]);
  S.songs = new Map(songs.map((s) => [s.id, s]));
  render();
  loadMeta();
  if (mode === "compat") loadCompat();
}

// group-by chips under the search box
const chips = document.getElementById("libChips");
function renderChips() {
  chips.innerHTML = MODES.map(([k, label]) => `<button class="chip${k === mode ? " on" : ""}" data-m="${k}">${label}</button>`).join("");
  chips.querySelectorAll(".chip").forEach((b) => (b.onclick = () => setMode(b.dataset.m)));
  document.getElementById("btnFolder").hidden = mode !== "folders";
  const n = document.getElementById("libCount");
  if (n) n.textContent = S.songs.size ? String(S.songs.size) : "";
}
on("libgroups", () => render());

function statusText(s) {
  const j = jobBySong.get(s.id);
  if (s.status === "error") return s.error || "failed";
  if (j && (j.status === "running" || j.status === "queued")) return `${j.message || j.status} · ${Math.round(j.progress * 100)}%`;
  return { queued: "waiting…", separating: "separating stems…", analyzing: "analyzing…" }[s.status] || s.status;
}

// ---------------------------------------------------------------- "find matches" mode
let match = null;          // {ref, matches} from /api/match, or null
let matchLoading = null;
let rateJob = null;        // {id, sid}: a "rate properly" job for the match list
on("event:job", (msg) => {
  if (!rateJob || msg.job.id !== rateJob.id) return;
  const el = list.querySelector(".match-head .rj");
  if (msg.job.status === "running" && el) el.textContent = `${msg.job.message || "rating"} ${Math.round(100 * (msg.job.progress || 0))}%`;
  if (["done", "error", "cancelled"].includes(msg.job.status)) {
    const sid = rateJob.sid; rateJob = null;
    if (msg.job.status === "error") toast(msg.job.error, true);
    if (match?.ref.id === sid) showMatches(sid);
  }
});

export async function showMatches(sid) {
  const s = S.songs.get(sid);
  if (!s || s.status !== "ready") { toast("That song isn't analyzed yet", true); return; }
  matchLoading = s.title; match = null; render();
  try {
    match = await api.get(`/api/match/${sid}`);
  } catch (e) { toast(e.message, true); }
  matchLoading = null; render();
  list.scrollTop = 0;
}
function clearMatches() { match = null; matchLoading = null; render(); }

const esc = (t) => String(t).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const ICON = { "their vocals over your music": '<i class="sd vocals"></i>', "your vocals over their music": '<i class="sd vocals"></i>', "layer the music together": '<i class="sd other"></i>', "their bass under your music": '<i class="sd bass"></i>' };

// A 'Lyrics' breakdown line from the FEAT-002 keys (lyrics / lyrics_mode / lyrics_judgeable / lyrics_text) a
// rated row carries. The lyrics score is only engaged when a resonance/contrast rating has run against a
// transcribed pair; a hand-built or lyrics-off row leaves these absent/neutral and this returns null, so the
// card simply shows no chip (honest degradation — never a fake 0%).
function lyricsLine(r) {
  if (!r) return null;
  const mode = r.lyrics_mode;
  if (!mode || mode === "off") return null;
  const pct = r.lyrics;                               // round(Lraw*100) or null when A.5-neutralized
  const text = r.lyrics_text || (pct == null ? "not transcribed" : "");
  if (pct == null && !r.lyrics_text) return null;
  const label = mode === "contrast" ? "contrast" : "resonance";
  const j = r.lyrics_judgeable;
  const tip = (j != null ? `judgeable ${Math.round(100 * j)}% · ` : "") +
    `${label} of the two lyrics` + (pct != null ? ` (${pct}%)` : "");
  return { label, pct: pct ?? null, text, tip };
}

function renderMatches() {
  list.innerHTML = "";
  const head = document.createElement("div");
  head.className = "match-head";
  if (matchLoading) {
    head.innerHTML = `<span class="spinner"></span> Finding matches for <b></b>…`;
    head.querySelector("b").textContent = matchLoading;
    list.appendChild(head);
    return;
  }
  const r = match.ref;
  const un = match.unrated || 0;
  head.innerHTML = `<div><div class="mh-t">Matches for <b></b></div>
      <div class="mh-m">${r.bpm} BPM · ${esc(r.key)}${r.has_vocals ? " · has vocals" : ""} — best first</div>
      ${un ? `<div class="mh-m">${rateJob ? `<span class="spinner"></span> <span class="rj">rating…</span>` :
        `≈ ${un} score${un > 1 ? "s are" : " is a"} quick estimate${un > 1 ? "s" : ""} <button class="small rate" title="Plan and rate every mashup with this song properly (vocal melody tracking + full arrangement)">Rate properly</button>`}</div>` : ""}</div>
    <button class="small x" title="Back to the normal library">✕</button>`;
  head.querySelector("b").textContent = r.title;
  head.querySelector(".x").onclick = clearMatches;
  head.querySelector(".rate")?.addEventListener("click", async () => {
    try {
      rateJob = { id: (await api.post("/api/mashup/rate", { involving: r.id })).id, sid: r.id };
      renderMatches();
    } catch (e) { toast(e.message, true); }
  });
  list.appendChild(head);
  const q = search.value.trim().toLowerCase();
  for (const m of match.matches) {
    if (q && !m.title.toLowerCase().includes(q)) continue;
    const s = S.songs.get(m.id);
    if (!s) continue;
    const el = card(s);
    if (m.same_song) {
      el.classList.add("matched");
      el.title = "Recognised from the audio as the same song (duplicate, alternate version or instrumental) - never paired";
      el.insertAdjacentHTML("afterbegin", `<div class="mscore" style="border-color:var(--line2);color:var(--dim)">=</div>`);
      el.insertAdjacentHTML("beforeend", `<div class="mwhy"><span class="mchip dim">same song as ${esc(r.title)} - not a mashup</span></div>`);
      list.appendChild(el);
      continue;
    }
    const col = m.score >= 70 ? "var(--good)" : m.score >= 45 ? "var(--warn)" : "var(--bad)";
    const u = m.uses[0];
    const lyr = lyricsLine(m);              // FEAT-002 lyrics keys surfaced on the match (null when neutral/off)
    const tip = [
      `Overall ${m.score}/100${m.rated ? " (full rating: arranged and scored)" : " (quick estimate - use 'Rate properly' above)"}`,
      `Tempo ${m.tempo.score}/100 — ${m.tempo.text}. They're ${m.tempo.their_bpm} BPM; you'd meet at ${m.tempo.meet_bpm}.`,
      `Key: they're in ${m.key.their} — ${m.key.relation}.`,
      ...m.uses.map((x) => `${x.kind}: harmony ${x.harmony}% — ${x.text}`),
      lyr ? `Lyrics: ${lyr.text}` : "",
      m.meter_ok ? "" : "Different meter (3 vs 4 beats per bar) — hard to combine",
    ].filter(Boolean).join("\n");
    el.title = tip;
    el.classList.add("matched");
    el.insertAdjacentHTML("afterbegin", `<div class="mscore${m.rated ? "" : " est"}" style="border-color:${col}" title="${m.rated ? "full rating" : "quick estimate"}">${m.rated ? "" : "≈"}${m.score}</div>`);
    el.insertAdjacentHTML("beforeend", `<div class="mwhy">
        <span class="mchip" title="How much stretching it takes">Tempo ${m.tempo.score} · ${esc(m.tempo.text)}</span>
        ${u ? `<span class="mchip" title="How well the notes fit, measured on the actual audio">${ICON[u.kind] || ""} ${u.harmony}% · ${esc(u.kind)} · ${esc(u.text)}</span>` : ""}
        <span class="mchip dim" title="Detected keys">Key ${esc(m.key.relation)}</span>
        ${lyr ? `<span class="mchip" title="${esc(lyr.tip)}">Lyrics ${lyr.label}${lyr.pct != null ? " " + lyr.pct + "%" : ""} · ${esc(lyr.text)}</span>` : ""}
        ${m.meter_ok ? "" : `<span class="mchip bad">Different meter</span>`}
      </div>`);
    list.appendChild(el);
  }
}

export function render() {
  renderChips();
  if (match || matchLoading) { visibleOrder = []; renderMatches(); return; }
  const q = search.value.trim().toLowerCase();
  const songs = [...S.songs.values()].filter((s) => s.kind !== "sample" && (!q || s.title.toLowerCase().includes(q)));   // baked samples live in their clips
  list.innerHTML = "";
  visibleOrder = [];
  if (!S.songs.size) {
    list.innerHTML = `<div class="empty-hint" style="padding:10px 6px">No songs yet. Click <b>+ Add</b> or drop audio files here.</div>`;
    return;
  }
  if (mode !== "folders") { renderSections(songs, q); return; }
  const gs = groups(songs);
  const anyFolders = S.folders.folders.length > 0;
  for (const g of gs) {
    if (q && !g.songs.length) continue;
    if (!g.folder && !anyFolders) { g.songs.forEach((s) => list.appendChild(card(s))); continue; }
    const head = document.createElement("div");
    head.className = "folder" + (g.folder ? "" : " unsorted");
    const collapsed = g.folder?.collapsed && !q;
    head.innerHTML = `<span class="tw">${collapsed ? "▸" : "▾"}</span><span class="fn"></span><span class="cnt">${g.songs.length}</span>`;
    head.classList.toggle("closed", !!collapsed);
    head.querySelector(".fn").textContent = g.folder ? g.folder.name : "Unsorted";
    if (g.folder) head.onclick = () => toggleCollapsed(g.folder);
    head.oncontextmenu = (e) => {
      e.preventDefault();
      if (!g.folder) { showMenu(e, [["New folder…", () => createFolder()]]); return; }
      showMenu(e, [
        ["Rename folder…", () => renameFolder(g.folder)],
        ["Move folder up", () => moveFolder(g.folder, -1)],
        ["Move folder down", () => moveFolder(g.folder, 1)],
        ...(S.listen?.ok && g.songs?.length ? [["Listen and describe its songs (Music Flamingo)", () => listenSongs(g.songs.map((s) => s.id))]] : []),
        "-",
        ["Delete folder (keeps the songs)", () => deleteFolder(g.folder)],
      ]);
    };
    // drop target for dragged song cards
    head.addEventListener("dragover", (e) => { if (e.dataTransfer.types.includes("text/x-song")) { e.preventDefault(); head.classList.add("drop"); } });
    head.addEventListener("dragleave", () => head.classList.remove("drop"));
    head.addEventListener("drop", (e) => {
      const ids = e.dataTransfer.getData("text/x-song");
      if (!ids) return;
      e.preventDefault(); e.stopPropagation(); head.classList.remove("drop");
      const arr = ids.split(",");
      moveToFolder(arr, g.folder ? g.folder.id : null);
      toast(`Moved ${arr.length} song${arr.length > 1 ? "s" : ""} to ${g.folder ? g.folder.name : "Unsorted"}`);
    });
    list.appendChild(head);
    if (!collapsed) {
      const box = document.createElement("div");
      box.className = "folder-body";
      g.songs.forEach((s) => box.appendChild(card(s)));
      if (!g.songs.length) box.innerHTML = `<div class="folder-empty">drag songs here</div>`;
      list.appendChild(box);
    }
  }
  if (q && !songs.length) list.insertAdjacentHTML("beforeend", `<div class="empty-hint" style="padding:10px 6px">No matches.</div>`);
}

function renderSections(songs, q) {
  const gs = sections(songs);
  for (const g of gs) {
    if (q && !g.songs.length) continue;
    const closed = isCollapsed(g.key) && !q;
    const head = document.createElement("div");
    head.className = "folder" + (closed ? " closed" : "") + (g.field ? " editable" : "");
    head.innerHTML = `<span class="tw">${closed ? "▸" : "▾"}</span><span class="fn"></span>${g.sub ? '<span class="fs"></span>' : ""}
      ${g.compare ? '<button class="small cmp">Compare</button>' : ""}<span class="cnt">${g.songs.length}</span>`;
    head.querySelector(".fn").textContent = g.label;
    if (g.sub) head.querySelector(".fs").textContent = g.sub;
    head.onclick = (e) => { if (!e.target.closest("button")) toggleSection(g.key); };
    head.querySelector(".cmp")?.addEventListener("click", compareNow);
    head.oncontextmenu = (e) => {
      e.preventDefault();
      const items = [];
      if (g.field && g.value) items.push([`Rename ${g.field} for these ${g.songs.length} songs…`, () => renameSection(g)]);
      if (g.field) items.push([`Edit info for these ${g.songs.length} songs…`, () => editInfo(g.songs.map((s) => s.id))]);
      if (mode === "genre") items.push(["Suggest genres for songs without one (local AI)", () => guessGenres(g.value ? null : g.songs.map((s) => s.id))]);
      if (S.listen?.ok) items.push([`Listen and describe these ${g.songs.length} songs (Music Flamingo)`, () => listenSongs(g.songs.map((s) => s.id))]);
      if (mode === "artist" && !g.value) items.push(["Use each song's folder name as its artist", () => folderAsArtist(g.songs)]);
      if (mode === "compat") items.push(["Compare new songs", compareNow]);
      if (g.sep && g.sep !== "blend") items.push([`Re-separate these ${g.songs.length} songs: SW + Ultra`, () => reseparateSW(g.songs.map((s) => s.id), "blend")]);
      if (items.length) showMenu(e, items);
    };
    if (g.field) {                                     // drop songs here = give them this artist / album / genre / year
      head.addEventListener("dragover", (e) => { if (e.dataTransfer.types.includes("text/x-song")) { e.preventDefault(); head.classList.add("drop"); } });
      head.addEventListener("dragleave", () => head.classList.remove("drop"));
      head.addEventListener("drop", (e) => {
        const ids = e.dataTransfer.getData("text/x-song");
        if (!ids) return;
        e.preventDefault(); e.stopPropagation(); head.classList.remove("drop");
        dropOnSection(g, ids.split(",")); libSel.clear();
      });
    }
    list.appendChild(head);
    if (!closed) {
      const box = document.createElement("div");
      box.className = "folder-body";
      g.songs.forEach((s) => box.appendChild(card(s)));
      if (!g.songs.length) box.innerHTML = `<div class="folder-empty">${mode === "compat" ? "loading…" : "empty"}</div>`;
      list.appendChild(box);
    }
  }
  if (q && !songs.length) list.insertAdjacentHTML("beforeend", `<div class="empty-hint" style="padding:10px 6px">No matches.</div>`);
}

function folderAsArtist(songs) {
  const by = new Map();
  for (const s of songs) {
    const f = S.folders.folders.find((x) => x.id === folderOf(s.id));
    if (!f) continue;
    if (!by.has(f.name)) by.set(f.name, []);
    by.get(f.name).push(s.id);
  }
  if (!by.size) { toast("None of these songs are in a folder", true); return; }
  for (const [name, ids] of by) setMeta(ids, { artist: name });
  toast(`Set the artist from the folder for ${[...by.values()].flat().length} songs`);
}

function card(s) {
  visibleOrder.push(s.id);
  const el = document.createElement("div");
  el.className = "song" + (s.id === S.selSong ? " sel" : "") + (libSel.has(s.id) ? " multi" : "") + (s.status === "error" ? " err" : "");
  el.dataset.id = s.id;
  el.draggable = true;
  const ready = s.status === "ready";
  const j = jobBySong.get(s.id);
  const vox = ready && s.has_vocals;
  // first item of the subtitle: whatever the section header doesn't already say
  const m = S.meta[s.id] || {};
  const artist = mode === "artist" ? (m.album || "") : mode === "album" ? (m.year || "") : artistOf(s);
  const meter = s.bpb && s.bpb !== 4 ? (s.bpb === 6 ? "6/8" : s.bpb + "/4") : "";
  const sp = SEPARATION[sepOf(s)];
  el.innerHTML = `${ready ? keyTile(s) : `<div class="kt busy"><span class="spinner"></span></div>`}
    <div class="sb"><div class="t"></div>
    ${ready ? `<div class="sub"><span class="ar"></span>${artist ? '<span class="dot">·</span>' : ""}<span class="num">${s.bpm?.toFixed(1)} BPM</span>${meter ? `<span class="dot">·</span><span class="num">${meter}</span>` : ""}<span class="dot">·</span><span class="num">${fmtDur(s.duration)}</span></div>`
      : `<div class="st"></div><div class="bar"><i style="width:${Math.round((j?.progress || 0) * 100)}%"></i></div>`}</div>
    <div class="badges">${ready ? `<span class="tag sep sep-${sepOf(s)}" title="${sp.tip}">${sp.tag}</span>` : ""}${vox ? '<i class="vdot" title="Has a usable vocal"></i>' : ""}</div>
    <button class="x" title="Delete">✕</button>`;
  el.querySelector(".t").textContent = s.title;
  el.title = [s.title, s.key?.name, vox ? "has vocals" : "instrumental", ready ? sp.name + " stems" : ""].filter(Boolean).join(" · ");
  const ar = el.querySelector(".ar");
  if (ar) ar.textContent = artist;
  if (!ready) el.querySelector(".st").textContent = statusText(s);
  el.onclick = (e) => {
    if (e.target.classList.contains("x")) return;
    if (e.ctrlKey || e.metaKey) {                       // build a multi-selection
      libSel.has(s.id) ? libSel.delete(s.id) : libSel.add(s.id);
      lastClicked = s.id; render(); return;
    }
    if (e.shiftKey && lastClicked) {
      const a = visibleOrder.indexOf(lastClicked), b = visibleOrder.indexOf(s.id);
      if (a >= 0 && b >= 0) { visibleOrder.slice(Math.min(a, b), Math.max(a, b) + 1).forEach((id) => libSel.add(id)); render(); return; }
    }
    libSel.clear(); lastClicked = s.id;
    select(s.id);
  };
  el.querySelector(".x").onclick = (e) => { e.stopPropagation(); remove(s); };
  el.addEventListener("dragstart", (e) => {
    const ids = libSel.has(s.id) && libSel.size > 1 ? [...libSel] : [s.id];
    e.dataTransfer.setData("text/x-song", ids.join(","));
    e.dataTransfer.effectAllowed = "move";
  });
  el.oncontextmenu = (e) => {
    e.preventDefault();
    const ids = libSel.has(s.id) && libSel.size > 1 ? [...libSel] : [s.id];
    const n = ids.length > 1 ? `${ids.length} songs` : "";
    const cur = folderOf(s.id);
    // folders only open when asked for: a filter box and a short scrolling list (the library has many folders)
    const pickFolder = () => showPicker(e, {
      title: `Move ${n || "it"} to folder:`, placeholder: "find a folder…",
      items: S.folders.folders.filter((f) => ids.length > 1 || f.id !== cur)
        .map((f) => [f.name, () => { moveToFolder(ids, f.id); libSel.clear(); }]),
      extra: [["New folder…", async () => { const f = await createFolder(null, ids); if (f) libSel.clear(); }],
        ...(cur || ids.length > 1 ? [["Unsorted (no folder)", () => { moveToFolder(ids, null); libSel.clear(); }]] : [])],
    });
    showMenu(e, [
      ...(ids.length === 1 ? [["Open", () => select(s.id)],
        ["Beat grid (the song's grid)…", () => openGrid(s.id, S.project?.clips.some((c) => c.song === s.id) ? S.project.id : null), "window"],
        ["All its tools ▸", () => showMenu(e, songToolItems(s.id))],
        ["Find matches for this song", () => showMatches(s.id)], ["Rename…", () => rename(s)]] : []),
      [ids.length > 1 ? `Edit info for ${ids.length} songs…` : "Edit info (artist, album, genre, year)…", () => editInfo(ids)],
      "-",
      [n ? `Move ${n} to folder` : "Move to folder", pickFolder, "▸"],
      "-",
      [ids.length > 1 ? `Re-separate ${ids.length} songs (Ultra stems)` : "Re-separate stems (Ultra quality)", () => reseparate(ids)],
      [ids.length > 1 ? `Re-separate ${ids.length} songs with BS-Roformer SW` : "Re-separate with BS-Roformer SW", () => reseparateSW(ids, "sw")],
      [ids.length > 1 ? `Re-separate ${ids.length} songs: SW + Ultra blend` : "Re-separate: SW + Ultra blend", () => reseparateSW(ids, "blend")],
      ...(S.listen?.ok ? [[ids.length > 1 ? `Listen and describe ${ids.length} songs (Music Flamingo)` : "Listen and describe (Music Flamingo)", () => listenSongs(ids)],
        ...(ids.some((i) => S.meta?.[i]?.heard) ? [[ids.length > 1 ? "Listen again (replace what it heard)" : "Listen again", () => listenSongs(ids, true)]] : [])] : []),
      ...(ids.length === 1 ? [["Re-analyze (beats, key, sections)", () => api.post(`/api/songs/${s.id}/reanalyze`).then(() => toast("Re-analyzing…"))]] : []),
      [ids.length > 1 ? `MuScriptor second opinion on ${ids.length} songs' beat marks` : "MuScriptor second opinion on the beat marks",
        () => { for (const id of ids) api.post(`/api/songs/${id}/muscheck`); toast(`MuScriptor is transcribing ${ids.length > 1 ? ids.length + " songs (one at a time)" : "the song"} - about 2 min per song. Marks where its notes disagree turn amber.`); }],
      [ids.length > 1 ? `Delete ${ids.length} songs` : "Delete", () => ids.length > 1 ? removeMany(ids) : remove(s)],
    ]);
  };
  return el;
}
const fmtDur = (d) => d ? `${Math.floor(d / 60)}:${String(Math.round(d % 60)).padStart(2, "0")}` : "";

function select(id) {
  S.selSong = id;
  render();
  emit("songSelected", id);
}
async function remove(s) {
  if (!confirm(`Delete "${s.title}" and its stems from the library?`)) return;
  await api.del(`/api/songs/${s.id}`);
}
async function removeMany(ids) {
  if (!confirm(`Delete ${ids.length} songs and their stems from the library?`)) return;
  for (const id of ids) await api.del(`/api/songs/${id}`);
  libSel.clear();
}
async function rename(s) {
  const t = prompt("Song title", s.title);
  if (t && t !== s.title) await api.patch(`/api/songs/${s.id}`, { title: t });
}

search.oninput = render;
on("findMatches", (sid) => showMatches(sid));
on("folders", render);
on("event:folders", (m) => { S.folders = m.folders; render(); });
document.getElementById("btnFolder").onclick = () => createFolder(null, [...libSel]).then(() => libSel.clear());

// ---------------------------------------------------------------- adding songs
let caps = { ultra: false, sw: false, default_quality: "best" };
api.get("/api/capabilities").then((c) => (caps = c)).catch(() => {});

async function reseparate(ids) {
  if (!caps.ultra) { toast("Ultra separation needs the BS-Roformer model (found in Ultimate Vocal Remover's models folder)", true, 6000); return; }
  const n = ids.length;
  if (!confirm(`Re-separate ${n > 1 ? n + " songs" : "this song"} with Ultra quality (BS-Roformer vocals + Demucs)?

` +
      `About 1-2 minutes per song on your GPU. Beat grids stay the same, so your arrangements keep lining up.`)) return;
  for (const id of ids) await api.post(`/api/songs/${id}/reseparate`, { quality: "ultra" });
  toast(`Re-separating ${n} song${n > 1 ? "s" : ""} - progress is in the status bar`);
  libSel.clear(); render();
}

async function reseparateSW(ids, quality) {
  if (!caps.sw) { toast("BS-Roformer SW needs its model files in mashup_lab/models (BS-Rofo-SW-Fixed.ckpt + .yaml)", true, 6000); return; }
  if (quality === "blend" && !caps.ultra) { toast("The blend needs the Ultra (BS-Roformer vocal) model too", true, 6000); return; }
  const n = ids.length, blend = quality === "blend";
  const what = blend
    ? `Both BS-Roformer SW and Ultra separate the song, and their stems are averaged (their artifacts partly cancel). About 3-5 minutes per song on your GPU, plus the analysis.`
    : `One model makes all four stems (its guitar and piano go into "other"). About a minute per song on your GPU, plus the analysis.`;
  if (!confirm(`Re-separate ${n > 1 ? n + " songs" : "this song"} ${blend ? "with the SW + Ultra blend" : "with BS-Roformer SW"}?

${what}
` +
      `This replaces ${n > 1 ? "their" : "the song's"} current stems; the rest of the library keeps its own. Beat grids stay the same, ` +
      `so your arrangements keep lining up (a locked song needs one re-lock click).`)) return;
  for (const id of ids) await api.post(`/api/songs/${id}/reseparate`, { quality });
  toast(`Re-separating ${n} song${n > 1 ? "s" : ""} ${blend ? "(SW + Ultra blend)" : "with BS-Roformer SW"} - progress is in the status bar`);
  libSel.clear(); render();
}

export function addDialog() {
  const m = modal("Add songs", `
    <p class="lead">Songs are split into drums / bass / other / vocals on your GPU (a minute or two each), then analyzed
      for beats, key and sections. You can keep working while that runs.</p>
    <div class="row-gap" style="margin-bottom:14px">
      <button class="primary" id="pickFiles">Choose audio files…</button>
      <button id="pickStems" title="Already-split stems (or just an acapella, a drum loop...) as ONE song, not separated again">Import stems of one song…</button>
      <span style="color:var(--dim)">or drop files anywhere in the window</span>
    </div>
    <div class="field"><label>…or paste a path on this PC (a song, a folder of songs, or a folder of stems)</label>
      <div class="row-gap"><input type="text" id="pathIn" placeholder="C:\\Music\\song.mp3   or   C:\\stems\\MySong" spellcheck="false" style="flex:1">
      <button id="pathGo">Import</button></div></div>
    <div class="opts"><label>Separation quality
      <select id="qual">${caps.sw && caps.ultra ? '<option value="blend">SW + Ultra blend - both separators, averaged (the default: best in your listening test, ~3-5 min/song)</option>' : ""}
        ${caps.sw ? '<option value="sw">BS-Roformer SW - one newer model for all four stems (~1 min/song)</option>' : ""}
        ${caps.ultra ? '<option value="ultra">Ultra - BS-Roformer vocals + Demucs (~2 min/song)</option>' : ""}
        <option value="best">Best - Demucs x4</option><option value="fast">Fast - Demucs x1</option></select></label></div>`);
  m.body.querySelector("#qual").value = caps.default_quality || "best";
  m.body.querySelector("#pickFiles").onclick = () => { fileInput.dataset.q = m.body.querySelector("#qual").value; fileInput.click(); m.close(); };
  m.body.querySelector("#pickStems").onclick = () => { m.close(); stemPicker(); };
  m.body.querySelector("#pathGo").onclick = async () => {
    const path = m.body.querySelector("#pathIn").value.trim();
    if (!path) return;
    try {
      const r = await api.post("/api/songs/import_path", { path, quality: m.body.querySelector("#qual").value });
      toast(`Importing ${r.length} song${r.length > 1 ? "s" : ""}…`);
      m.close();
    } catch (e) { toast(e.message, true); }
  };
}
fileInput.onchange = () => { upload([...fileInput.files], fileInput.dataset.q || ""); fileInput.value = ""; };

const AUDIO_RE = /\.(wav|flac|mp3|m4a|aac|ogg|opus|aiff?|mp4|webm|wma)$/i;

export async function upload(files, quality = "") {
  files = files.filter((f) => AUDIO_RE.test(f.name));
  if (!files.length) { toast("No audio files in that drop", true); return; }
  const fd = new FormData();
  files.forEach((f) => fd.append("files", f));
  fd.append("quality", quality);
  toast(`Uploading ${files.length} file${files.length > 1 ? "s" : ""}…`);
  try { await api.post("/api/songs/upload", fd); }
  catch (e) { toast("Upload failed: " + e.message, true); }
}

document.getElementById("btnAdd").onclick = addDialog;

// window-wide drag & drop of audio files
const overlay = document.getElementById("dropOverlay");
let dragDepth = 0;
window.addEventListener("dragenter", (e) => { if ([...e.dataTransfer.types].includes("Files")) { dragDepth++; overlay.hidden = false; e.preventDefault(); } });
window.addEventListener("dragleave", () => { if (--dragDepth <= 0) { dragDepth = 0; overlay.hidden = true; } });
window.addEventListener("dragover", (e) => { if ([...e.dataTransfer.types].includes("Files")) e.preventDefault(); });
window.addEventListener("drop", (e) => {
  if (!e.dataTransfer.files.length) return;
  e.preventDefault(); dragDepth = 0; overlay.hidden = true;
  const files = [...e.dataTransfer.files].filter((f) => AUDIO_RE.test(f.name));
  // stems (named drums / vocals / kick / acapella...): ask how to bring them in instead of separating each one
  if (files.length && looksLikeStems(files.map((f) => f.name))) openStemImport({ files, fromDrop: true, onSeparate: () => upload(files) });
  else upload([...e.dataTransfer.files]);
});

// ---------------------------------------------------------------- live updates
on("event:song", (m) => {
  const prev = S.songs.get(m.song.id);
  S.songs.set(m.song.id, m.song);
  if (m.song.status === "ready" && prev && prev.status !== "ready") toast(`"${m.song.title}" is ready`);
  if (m.song.status === "ready") { invalidateSong(m.song.id); dropPeaks(m.song.id); forgetSong(m.song.id); }
  render();
  emit("songUpdated", m.song.id);
});
on("event:song_deleted", (m) => {
  S.songs.delete(m.id);
  if (S.selSong === m.id) { S.selSong = null; emit("songSelected", null); }
  render();
});
on("event:job", (m) => {
  if (m.job.ref) {
    jobBySong.set(m.job.ref, m.job);
    const el = list.querySelector(`.song[data-id="${m.job.ref}"]`);
    if (el && el.querySelector(".bar")) {
      el.querySelector(".bar > i").style.width = Math.round(m.job.progress * 100) + "%";
      const s = S.songs.get(m.job.ref);
      if (s) el.querySelector(".st").textContent = statusText(s);
    }
  }
});
