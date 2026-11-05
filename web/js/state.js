// Central store: songs, the open project, selection, transport. Tiny event emitter + undo history.
import { api } from "./api.js";
import { detectedMarks } from "./timing.js";

const listeners = {};
export function on(ev, fn) { (listeners[ev] ||= []).push(fn); }
export function emit(ev, ...a) { (listeners[ev] || []).forEach((f) => f(...a)); }

export const S = {
  songs: new Map(),        // id -> summary
  full: new Map(),         // id -> full song info (beats, bars, sections...)
  fullDone: new Map(),     // id -> that info once it has loaded (for code that can't wait)
  project: null,
  dirty: false,
  selSong: null,
  selClips: new Set(),
  transport: { playing: false, pos: 0, t: 0, latency: 0, loop: null },
  clockOffset: 0,          // server time - client time (seconds)
  waiting: new Set(),      // clip ids whose warped audio is still rendering
  preview: new Set(),      // clip ids playing a varispeed preview until their HQ stretch/pitch render lands
  interactive: false,      // true while dragging: server skips HQ renders until release
  follow: (() => { try { return localStorage.getItem("follow") !== "0"; } catch { return true; } })(),
  jobs: new Map(),
};

export const STEMS = ["drums", "bass", "other", "vocals"];
export const STEM_COLORS = { drums: "#df8c55", bass: "#5b9fe0", other: "#6fbf86", vocals: "#d9779f", mix: "#a7a7ad" };

export async function fullSong(id) {
  if (!S.full.has(id)) S.full.set(id, api.get(`/api/songs/${id}`).then((x) => { S.fullDone.set(id, x); return x; }));
  try { return await S.full.get(id); } catch (e) { S.full.delete(id); throw e; }
}
export function invalidateSong(id) { S.full.delete(id); S.fullDone.delete(id); }

// ---------------------------------------------------------------- Phase-2 client-model helpers (FEAT-005)
// These are read-only views over the ADDITIVE fields autonomous output attaches. Everything round-trips
// through the project JSON for free (setProject / edit / undo deep-clone the whole project), so there is no
// extra serialization path to maintain — these just give the UI a tidy, defaulted accessor.

// A track's automation lane (design D / B.4): a list of time-varying FX events, or [] when the track has none
// (the manual flow never writes it, so a Phase-2-free project reads back as empty everywhere).
export function trackAutomation(t) { return Array.isArray(t?.automation) ? t.automation : []; }

// The autonomous-mashup description a generated project carries at project.mashup (style bundle, lyrics line,
// lyrics_mode, transitions{events}, blend{congested_sections}), or null for a hand-built project.
export function mashupDesc(p = S.project) { return (p && p.mashup) || null; }
export function projectTransitions(p = S.project) { return mashupDesc(p)?.transitions || null; }

// A song's per-beat 0..100 beat_content list, or null until it loads / is back-filled.
export function beatContent(info) { return info?.beat_content ?? null; }

// When the server back-fills a song's beat_content (POST /api/songs/{id}/backfill_beat_content -> a job with
// kind "backfill"), drop the cached song info so the next fullSong() re-reads info.beat_content. The timeline
// listens for "songUpdated" to redraw; emit it so overlays depending on beat_content refresh.
on("event:job", (m) => {
  const j = m?.job;
  if (!j || j.status !== "done" || !j.ref) return;
  if (j.kind !== "backfill" && j.kind !== "lyrics") return;   // both change the stored song info
  invalidateSong(j.ref);
  emit("songUpdated", j.ref);
});

export const uid = (p) => p + Math.random().toString(16).slice(2, 10);
export const clone = (o) => JSON.parse(JSON.stringify(o));

// ---------------------------------------------------------------- history + sync
// Undo entries are project snapshots ({ json }), or a song's beat marks ({ kind: "marks", info, before }):
// marks belong to the song (every project shows them) but share the same Ctrl+Z / Ctrl+Y.
const undo = [], redo = [];
let syncTimer = null;

export function setProject(p, { keepHistory = false } = {}) {
  S.project = p;
  if (!keepHistory) { undo.length = 0; redo.length = 0; emitHistory(); }
  S.selClips.clear();
  S.dirty = !!p?.mashup && !p.modified;       // a freshly generated mashup gets autosaved
  emit("project");
  syncNow();
  if (S.dirty) { clearTimeout(saveTimer); saveTimer = setTimeout(autosave, 1000); }
}

// Every edit goes through here: snapshot for undo, then notify + push to the audio engine.
// `coalesce`: consecutive edits with the same tag (one drag, one slider sweep) become ONE undo step.
export function edit(fn, { coalesce = null, label = "Edit" } = {}) {
  const before = JSON.stringify(S.project);
  if (!(coalesce && undo.length && undo[undo.length - 1].tag === coalesce)) {
    undo.push({ json: before, tag: coalesce, label });
    if (undo.length > 300) undo.shift();
  }
  redo.length = 0;
  fn(S.project);
  S.dirty = true;
  emit("project");
  emitHistory();
  scheduleSync();
}

function restore(json) {
  S.project = JSON.parse(json);
  // keep only selections that still exist
  S.selClips = new Set([...S.selClips].filter((id) => S.project.clips.some((c) => c.id === id)));
  S.dirty = true;
  emit("project");
  emit("selection");
  emitHistory();
  scheduleSync();
}
function swapMarks(entry) {
  const info = entry.info, now = info.user_marks ?? null;
  setMarks(info, entry.before);
  return { kind: "marks", info, before: now, label: entry.label, tag: null };
}
const marksSave = {};                                    // song id -> pending save (one per drag, not per pixel)
function setMarks(info, m) {
  info.user_marks = m;                                   // a new object: every cache keyed on it refreshes
  S.full.get(info.id)?.then((cur) => { if (cur && cur !== info) cur.user_marks = m; }).catch(() => {});
  clearTimeout(marksSave[info.id]);
  marksSave[info.id] = setTimeout(() => {
    api.put(`/api/songs/${info.id}/marks`, { user_marks: info.user_marks ?? null })
      .then((r) => applyFrozen(info.id, r))
      .catch((e) => toast("Couldn't save the marks: " + e.message, true));
  }, 300);
  emit("marks", info.id);
}

// ---------------------------------------------------------------- a song's timing in this project
// A song's grid (its marks: the beat grid window, the song view) is for projects from the moment it's saved; saved
// projects that used the old grid keep it (the server copies it into them - project.set_song_grid). A project can
// also have its own timing for a song (Marks mode): p.songs[sid].marks, laid over the song's info for everything
// that reads marks or picks a lock (projectInfo).
const overlay = new WeakMap();                           // song info -> { marks, view }
export function projectInfo(info, project = S.project) {
  const pm = info && project?.songs?.[info.id]?.marks;
  if (!pm) return info;
  const hit = overlay.get(info);
  if (hit && hit.marks === pm) return hit.view;
  const view = { ...info, user_marks: pm, project_marks: true };
  overlay.set(info, { marks: pm, view });
  return view;
}
// The server froze the open project on the old grid too (the file): do the same here, or the next autosave drops it.
export function applyFrozen(sid, r) {
  const p = S.project;
  if (!r?.frozen?.includes(p?.id) || !p.songs?.[sid] || p.songs[sid].marks) return;
  p.songs[sid].marks = r.old;
  emit("project");
  scheduleSync();
}
// Change this project's timing for a song (Marks mode): fn(m) edits m = { t, d, s, bpb } (timing.js), starting from
// the project's own timing, else the song's grid. One project undo step (coalesce: one drag).
export function editProjectMarks(info, fn, { label = "Edit this project's timing", coalesce = null } = {}) {
  const sid = info.id, own = S.project?.songs?.[sid]?.marks;
  if (!S.project?.songs?.[sid]) return;
  const m = clone(own || (info.project_marks ? null : info.user_marks) || detectedMarks(info));
  if (!m.s || m.s.length !== m.t.length) m.s = m.t.map(() => 1);
  if (!m.d || m.d.length !== m.t.length) m.d = m.t.map(() => 0);
  fn(m);
  edit((p) => { p.songs[sid].marks = m; }, { label, coalesce });
  emit("marks", sid);
}
export function useSongGrid(sid) {                       // drop this project's own timing: the song's grid again
  if (!S.project?.songs?.[sid]?.marks) return;
  edit((p) => { delete p.songs[sid].marks; }, { label: "Use the song's grid" });
  emit("marks", sid);
}
// This project's timing becomes the song's grid (for projects from now on; this one follows it).
export async function makeSongGrid(info) {
  const sid = info.id, m = S.project?.songs?.[sid]?.marks;
  if (!m) return;
  try {
    const r = await api.put(`/api/songs/${sid}/marks`, { user_marks: m, project: S.project.id });
    info.user_marks = m;
    S.full.get(sid)?.then((cur) => { if (cur && cur !== info) cur.user_marks = m; }).catch(() => {});
    edit((p) => { delete p.songs[sid].marks; }, { label: "Make it the song's grid" });
    emit("marks", sid);
    const n = r.frozen?.length || 0;
    toast(`This is the song's grid now - new projects use it${n ? `; ${n} older project${n > 1 ? "s" : ""} keep${n > 1 ? "" : "s"} the grid ${n > 1 ? "they" : "it"} had` : ""}`, false, 6000);
  } catch (e) { toast("Couldn't save the song's grid: " + e.message, true); }
}
// Change a song's beat marks: fn(m) edits m = { t, d, s, bpb } (timing.js); the first edit starts from the
// detector's marks. `coalesce`: one drag = one undo step.
export function editSongMarks(info, fn, { label = "Edit marks", coalesce = null } = {}) {
  const before = info.user_marks ?? null;
  const m = clone(before || detectedMarks(info));
  if (!m.s || m.s.length !== m.t.length) m.s = m.t.map(() => 1);
  if (!m.d || m.d.length !== m.t.length) m.d = m.t.map(() => 0);
  fn(m);
  const last = undo[undo.length - 1];
  if (!(coalesce && last && last.tag === coalesce && last.info === info)) {
    undo.push({ kind: "marks", info, before, label, tag: coalesce });
    if (undo.length > 300) undo.shift();
  }
  redo.length = 0;
  setMarks(info, m);
  emitHistory();
}
export function resetSongMarks(info) {
  if (!info.user_marks) return;
  undo.push({ kind: "marks", info, before: info.user_marks, label: "Back to the detector's marks", tag: null });
  redo.length = 0;
  setMarks(info, null);
  emitHistory();
}
export function doUndo() {
  const u = undo.pop();
  if (!u) { toast("Nothing to undo"); return; }
  if (u.kind === "marks") redo.push(swapMarks(u));
  else {
    redo.push({ json: JSON.stringify(S.project), label: u.label });
    restore(u.json);
  }
  toast(`↶ Undid: ${u.label}`, false, 1800);
  emitHistory();
}
export function doRedo() {
  const r = redo.pop();
  if (!r) { toast("Nothing to redo"); return; }
  if (r.kind === "marks") undo.push(swapMarks(r));
  else {
    undo.push({ json: JSON.stringify(S.project), label: r.label, tag: null });
    restore(r.json);
  }
  toast(`↷ Redid: ${r.label}`, false, 1800);
  emitHistory();
}
export function history() {
  return { undo: undo.length, redo: redo.length, undoLabel: undo[undo.length - 1]?.label, redoLabel: redo[redo.length - 1]?.label };
}
function emitHistory() { emit("history", history()); }

let saveTimer = null;
function scheduleSync() {
  clearTimeout(syncTimer);
  syncTimer = setTimeout(syncNow, 120);
  // autosave a few seconds after the last edit (only arrangements with something in them)
  clearTimeout(saveTimer);
  saveTimer = setTimeout(autosave, 2500);
}
export async function autosave() {
  const p = S.project;
  if (!p || !p.clips.length || !S.dirty) return;
  // what goes out now; an edit made while it's on its way stays unsaved (it used to be marked saved when this save
  // came back - and was lost: "Play as recorded" clicked during the lock's autosave, Oct 19)
  const sent = JSON.stringify(p);
  try {
    await api.put(`/api/projects/${p.id}`, JSON.parse(sent));
    if (S.project === p && JSON.stringify(p) === sent) S.dirty = false;
    else if (S.project === p) { clearTimeout(saveTimer); saveTimer = setTimeout(autosave, 1000); }
    try { localStorage.setItem("lastProject", p.id); } catch {}
    emit("saved");
  } catch (e) { console.error("autosave failed", e); }
}
export async function syncNow() {
  clearTimeout(syncTimer);
  if (!S.project) return;
  try {
    const r = await api.post(S.interactive ? "/api/active?interactive=1" : "/api/active", S.project);
    S.waiting = new Set(r.waiting);
    S.preview = new Set(r.preview || []);
    S.rendering = r.rendering;
    emit("renders");
  } catch (e) { console.error(e); }
}

// ---------------------------------------------------------------- project helpers
// tuning a song plays at in this project (semitones from A440), or null if unmeasured - twin of
// mashlab/project.py _played_tuning
export function playedTuning(p, sid, seen = new Set()) {
  const own = S.songs.get(sid)?.tuning_st;
  if (own == null) return null;
  const to = p.songs[sid]?.tune_to;
  if (!to || seen.has(sid)) return own;
  if (to === "a440") return 0;
  if (!p.songs[to]) return own;
  const t = playedTuning(p, to, new Set([...seen, sid]));
  return t == null ? own : t;
}
// semitones a song is shifted by to match the tuning picked for it (off unless you pick one)
export function tuningShift(p, sid) {
  const own = S.songs.get(sid)?.tuning_st;
  if (own == null || !p.songs[sid]?.tune_to) return 0;
  return playedTuning(p, sid, new Set()) - own;
}
export function songSemis(p, sid, stem) {          // the twin of project.song_semis (the screw drops every stem)
  const screw = p.screw && p.screw !== 1 ? 12 * Math.log2(p.screw) : 0;
  if (stem === "drums") return Math.round(screw * 1000) / 1000;
  return Math.round(((p.songs[sid]?.transpose || 0) + tuningShift(p, sid) + screw) * 1000) / 1000;
}
export function songMult(p, sid) { return (p.songs[sid] || {}).mult ?? 1; }
export async function refreshWaiting() {
  if (!S.project) return;
  const r = await api.post("/api/waiting", S.project);
  S.waiting = new Set(r.waiting);
  S.preview = new Set(r.preview || []);
  emit("renders");
}
export function projectLength(p) {
  return p.clips.reduce((m, c) => Math.max(m, c.start + c.len), 0);
}
export function ensureSongInProject(p, sid) {
  if (!p.songs[sid]) {
    const info = S.songs.get(sid);
    let mult = 1;
    if (info && info.bpm) {
      const opts = [0.5, 1, 2];
      mult = opts.reduce((b, m) => Math.abs(Math.log(p.bpm / (info.bpm * m))) < Math.abs(Math.log(p.bpm / (info.bpm * b))) ? m : b, 1);
    }
    p.songs[sid] = { transpose: 0, mult };
  }
}

// ---------------------------------------------------------------- time helpers
export const secPerBeat = () => 60 / (S.project?.bpm || 120);
export function transportPos() {
  const t = S.transport;
  if (!t.playing) return t.pos;
  const now = performance.now() / 1000 + S.clockOffset;
  let p = t.pos + (now - t.t) - (t.latency || 0);
  if (t.loop && p >= t.loop[1]) {
    const L = t.loop[1] - t.loop[0];
    p = t.loop[0] + ((p - t.loop[0]) % L);
  }
  return Math.max(0, p);
}
export function fmtTime(s) {
  const m = Math.floor(s / 60), r = s - m * 60;
  return `${m}:${r.toFixed(1).padStart(4, "0")}`;
}

export function toast(msg, err = false, ms = 3500) {
  const t = document.getElementById("toast");
  t.textContent = msg;
  t.className = err ? "err" : "";
  t.hidden = false;
  clearTimeout(toast._t);
  toast._t = setTimeout(() => (t.hidden = true), ms);
}
