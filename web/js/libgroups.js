// Library "group by": the song list sectioned by folder, artist, album, genre, songs that work together, key,
// tempo, meter, month added, year, vocals or how the stems were made. Artist / album / genre / year come from the files' tags (server:
// libmeta.py) and can be edited (Edit info…, or drag songs onto a section); your edits always win.
import { api } from "./api.js";
import { S, on, emit, toast } from "./state.js";
import { modal } from "./ui.js";

export const MODES = [
  ["folders", "Folders"], ["artist", "Artists"], ["album", "Albums"], ["genre", "Genres"], ["compat", "Works together"],
  ["key", "Key"], ["tempo", "Tempo"], ["meter", "Meter"], ["added", "Added"], ["year", "Year"], ["vocals", "Vocals"], ["stems", "Stems"],
];
// how a song's stems were made (song.separation; songs from before it was recorded are Demucs)
export const SEPARATION = {
  blend: { tag: "SW+U", name: "SW + Ultra", tip: "Stems: BS-Roformer SW and Ultra, averaged (the default)" },
  ultra: { tag: "ULTRA", name: "Ultra", tip: "Stems: BS-Roformer vocals + Demucs for the rest" },
  sw: { tag: "SW", name: "BS-Roformer SW", tip: "Stems: BS-Roformer SW for all four" },
  demucs: { tag: "DEMUCS", name: "Demucs", tip: "Stems: Demucs only (the oldest way) - right-click to re-separate" },
  imported: { tag: "STEMS", name: "Imported stems", tip: "Stems imported as they were - never separated here" },
};
export const sepOf = (s) => (SEPARATION[s.separation] ? s.separation : "demucs");
export const EDITABLE = { artist: "artist", album: "album", genre: "genre", year: "year" };   // mode -> meta field
const esc = (t) => String(t ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const ls = (k, d) => { try { return JSON.parse(localStorage.getItem(k)) ?? d; } catch { return d; } };
const lsSet = (k, v) => { try { localStorage.setItem(k, JSON.stringify(v)); } catch {} };

S.meta = {};
S.genres = [];
S.compat = null;
export let mode = ls("libGroup", "folders");
const collapsed = ls("libCollapsed", {});           // {mode: [group keys]}

export function setMode(m) { mode = m; lsSet("libGroup", m); if (m === "compat") loadCompat(); emit("libgroups"); }
export const isCollapsed = (key) => (collapsed[mode] || []).includes(key);
export function toggleCollapsed(key) {
  const a = new Set(collapsed[mode] || []);
  a.has(key) ? a.delete(key) : a.add(key);
  collapsed[mode] = [...a]; lsSet("libCollapsed", collapsed); emit("libgroups");
}

export async function loadMeta() {
  try { const r = await api.get("/api/library/meta"); S.meta = r.meta; S.genres = r.genres; S.listen = r.listen; } catch {}
  emit("libgroups");
}
export async function loadCompat() {
  try { S.compat = await api.get("/api/library/compat"); } catch { S.compat = null; }
  emit("libgroups");
}
on("event:meta", (m) => { Object.assign(S.meta, m.meta); emit("libgroups"); });
on("event:job", (m) => {
  if (m.job.kind === "compat" && m.job.status === "done") loadCompat();
  if (m.job.kind === "genres" && m.job.status === "done") { loadMeta(); toast(`Suggested genres for ${m.result?.guessed ?? 0} songs`); }
});

// ---------------------------------------------------------------- the sections

const monthFmt = new Intl.DateTimeFormat(undefined, { month: "long", year: "numeric" });
const metaOf = (s) => S.meta[s.id] || {};
export const artistOf = (s) => metaOf(s).artist || "";

/** [{key, label, sub?, songs, field?, value?}] for the current mode (songs already filtered). */
export function sections(songs) {
  const ready = songs.filter((s) => s.status === "ready"), busy = songs.filter((s) => s.status !== "ready");
  let out = [];
  const by = (keyFn, labelFn, order) => {
    const m = new Map();
    for (const s of ready) {
      const k = keyFn(s);
      if (!m.has(k)) m.set(k, []);
      m.get(k).push(s);
    }
    out = [...m.entries()].map(([k, a]) => ({ key: String(k), label: labelFn(k, a), songs: a }));
    out.sort(order || ((a, b) => (a.key === "" ? 1 : b.key === "" ? -1 : a.label.localeCompare(b.label))));
  };
  const words = (a, f) => {              // the spelling most songs use ("HORSE the Band" vs "HORSE the band")
    const c = new Map();
    for (const s of a) { const v = metaOf(s)[f]; if (v) c.set(v, (c.get(v) || 0) + 1); }
    return [...c.entries()].sort((x, y) => y[1] - x[1])[0]?.[0] || "";
  };
  switch (mode) {
    case "artist":
      by((s) => artistOf(s).toLowerCase(), (k, a) => k ? words(a, "artist") : "Unknown artist");
      out.forEach((g) => { g.field = "artist"; g.value = g.key ? g.label : ""; });
      break;
    case "album":
      by((s) => { const m = metaOf(s); return m.album ? `${(m.artist || "").toLowerCase()}|${m.album.toLowerCase()}` : ""; },
        (k, a) => k ? words(a, "album") : "No album");
      out.forEach((g) => { g.field = "album"; g.value = g.key ? g.label : ""; if (g.key) g.sub = words(g.songs, "artist"); });
      break;
    case "genre":
      by((s) => (metaOf(s).genre || "").toLowerCase(), (k, a) => k ? words(a, "genre") : "No genre");
      out.forEach((g) => {
        g.field = "genre"; g.value = g.key ? g.label : "";
        const guessed = g.songs.filter((s) => metaOf(s).src?.genre === "guess").length;
        if (guessed) g.sub = guessed === g.songs.length ? "suggested" : `${guessed} suggested`;
      });
      break;
    case "year":
      by((s) => metaOf(s).year || "", (k) => k || "No year", (a, b) => (a.key === "" ? 1 : b.key === "" ? -1 : b.key.localeCompare(a.key)));
      out.forEach((g) => { g.field = "year"; g.value = g.key; });
      break;
    case "key": {
      const num = (c) => parseInt(c) || 99;
      by((s) => s.key?.camelot || "", (k, a) => k ? `${k} · ${a[0].key?.name || ""}` : "Unknown key",
        (a, b) => num(a.key) - num(b.key) || a.key.localeCompare(b.key));
      break;
    }
    case "tempo":
      by((s) => Math.floor((s.bpm || 0) / 10) * 10, (k) => `${k}–${+k + 9} BPM`, (a, b) => a.key - b.key);
      break;
    case "meter":
      by((s) => s.bpb || 4, (k) => (+k === 6 ? "6/8" : `${k}/4`), (a, b) => (a.key === "4" ? -1 : b.key === "4" ? 1 : a.key - b.key));
      break;
    case "added":
      by((s) => { const d = new Date((s.created || 0) * 1000); return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}`; },
        (k) => { const [y, m] = k.split("-"); return monthFmt.format(new Date(+y, +m - 1, 1)); },
        (a, b) => b.key.localeCompare(a.key));
      break;
    case "vocals":
      by((s) => (s.has_vocals ? "v" : "i"), (k) => (k === "v" ? "With vocals" : "Instrumental"), (a, b) => (a.key === "v" ? -1 : 1));
      break;
    case "stems": {
      const order = Object.keys(SEPARATION);
      by(sepOf, (k) => SEPARATION[k].name, (a, b) => order.indexOf(a.key) - order.indexOf(b.key));
      out.forEach((g) => { g.sep = g.key; if (g.key !== "blend") g.sub = "right-click: re-separate these"; });
      break;
    }
    case "compat":
      out = compatSections(ready);
      break;
  }
  for (const g of out) if (!g.ordered) g.songs.sort((a, b) => a.title.localeCompare(b.title));   // "works together": best fit first
  if (busy.length) out.unshift({ key: "~busy", label: "Importing", songs: busy });
  return out;
}

function compatSections(ready) {
  const C = S.compat;
  const byId = new Map(ready.map((s) => [s.id, s]));
  if (!C) return [{ key: "~wait", label: "Loading…", songs: [] }];
  const out = C.groups.map((g) => ({
    key: g.ids.slice().sort().join(","), label: byId.get(g.hub)?.title || "Group", ordered: true,
    sub: `+${g.ids.length - 1} · ${g.keys.join(" ")} · ${g.bpm[0] === g.bpm[1] ? g.bpm[0] : `${g.bpm[0]}–${g.bpm[1]}`} BPM · fit ${g.fit}`,
    songs: g.ids.map((id) => byId.get(id)).filter(Boolean),
  })).filter((g) => g.songs.length);
  const loose = C.loose.map((id) => byId.get(id)).filter(Boolean);
  if (loose.length) out.push({ key: "~loose", label: "Few strong partners", sub: "nothing scores high with these yet", songs: loose });
  const missing = C.missing.map((id) => byId.get(id)).filter(Boolean);
  if (missing.length) out.unshift({ key: "~missing", label: "Not compared yet", songs: missing, compare: true,
    sub: `about ${Math.max(1, Math.round(missing.length * 1.5 / 60))} min to compare` });
  return out;
}

export async function compareNow() {
  try { await api.post("/api/library/compat"); toast("Comparing songs in the background - the groups fill in when it's done"); }
  catch (e) { toast(e.message, true); }
}
// Music Flamingo listens to 30 s of each song and writes its genre + one line on its sound (listen.py)
export async function listenSongs(ids, redo = false) {
  try {
    const r = await api.post("/api/library/listen", { ids: ids || null, redo });
    if (r.queued === 0) toast(redo ? "Nothing to listen to" : "Already heard - use \"Listen again\" to redo them");
    else toast("Music Flamingo is listening - about 20 s to start, then a few seconds a song. What it hears shows in the song view; its genres show as 'suggested'.");
  } catch (e) { toast(e.message, true); }
}

export async function guessGenres(ids) {
  try { await api.post("/api/library/genres/guess", { ids: ids || null }); toast("The local model is suggesting genres - they show as 'suggested' until you change them"); }
  catch (e) { toast(e.message, true); }
}

// ---------------------------------------------------------------- editing

export async function setMeta(ids, changes) {
  try {
    const r = await api.put("/api/library/meta", { ids, set: changes });
    Object.assign(S.meta, r.meta);
    emit("libgroups");
  } catch (e) { toast(e.message, true); }
}

/** Songs dropped on a section in an editable mode take that section's value. */
export function dropOnSection(g, ids) {
  if (!g.field) return false;
  setMeta(ids, { [g.field]: g.value || "" });
  if (g.field === "album" && g.value) {
    const artist = artistOf(g.songs[0] || {});
    if (artist) setMeta(ids, { artist });
  }
  toast(g.value ? `${ids.length} song${ids.length > 1 ? "s" : ""} → ${g.value}` : `Cleared ${g.field} for ${ids.length} song${ids.length > 1 ? "s" : ""}`);
  return true;
}

export function renameSection(g) {
  const v = prompt(`Rename "${g.label}" for its ${g.songs.length} song${g.songs.length > 1 ? "s" : ""} to`, g.value || "")?.trim();
  if (v == null || v === g.value) return;
  setMeta(g.songs.map((s) => s.id), { [g.field]: v });
}

export function editInfo(ids) {
  const songs = ids.map((id) => S.songs.get(id)).filter(Boolean);
  if (!songs.length) return;
  const one = songs.length === 1;
  const val = (f) => { const v = new Set(songs.map((s) => metaOf(s)[f] || "")); return v.size === 1 ? [...v][0] : null; };
  const genres = [...new Set([...S.genres, ...Object.values(S.meta).map((m) => m.genre).filter(Boolean)])].sort();
  const field = (f, label) => {
    const v = val(f);
    return `<label class="ei-row"><span>${label}</span><input data-f="${f}" spellcheck="false" ${f === "genre" ? 'list="eiGenres"' : ""}
      value="${esc(v ?? "")}" placeholder="${v == null ? "(several - leave empty to keep)" : ""}"></label>`;
  };
  const m = modal(one ? "Edit info" : `Edit info · ${songs.length} songs`, `
    <p class="lead">${one ? esc(songs[0].title) : "Only the fields you change are applied to all of them."}
      Values from the files' tags are filled in; yours always win.</p>
    <div class="ei">${field("artist", "Artist")}${field("album", "Album")}${field("genre", "Genre")}${field("year", "Year")}</div>
    <datalist id="eiGenres">${genres.map((g) => `<option value="${esc(g)}">`).join("")}</datalist>
    <div class="row-gap" style="margin-top:14px"><button class="primary" id="eiSave">Save</button>
      <button id="eiReset" title="Forget your edits for these songs and use what the files say">Use the files' tags</button></div>`);
  const inputs = [...m.body.querySelectorAll("input[data-f]")];
  const before = Object.fromEntries(inputs.map((i) => [i.dataset.f, i.value]));
  m.body.querySelector("#eiSave").onclick = () => {
    const ch = {};
    for (const i of inputs) if (i.value !== before[i.dataset.f] && (one || i.value)) ch[i.dataset.f] = i.value.trim();
    if (Object.keys(ch).length) setMeta(ids, ch);
    m.close();
  };
  m.body.querySelector("#eiReset").onclick = () => { setMeta(ids, { artist: "", album: "", genre: "", year: "" }); m.close(); };
  inputs[0]?.focus();
}

// Camelot key -> a colour tile (DJ convention: neighbouring keys have neighbouring colours)
export function keyTile(s) {
  const c = s.key?.camelot || "";
  const n = parseInt(c), minor = c.endsWith("A");
  if (!n) return `<div class="kt" style="background:var(--tile)">–</div>`;
  const h = (165 + (n - 1) * 30) % 360;
  return `<div class="kt" style="background:linear-gradient(145deg,hsl(${h} 46% ${minor ? 36 : 46}%),hsl(${(h + 18) % 360} 50% ${minor ? 24 : 32}%))" title="${esc(s.key?.name || "")}">${esc(c)}</div>`;
}
