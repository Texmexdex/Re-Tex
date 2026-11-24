// What you can do with a song, wherever you right-click it (a clip or a lane in the timeline, the song in the library,
// the song view) or open its Tools menu in the arrangement bar - one list, so every place offers the same tools.
//   Beat grid…      the song's grid (its beat marks, for projects from now on - saved projects keep the grid they
//                   had); opened from a project it can also save just for that project
//   Marks mode      this project's timing for the song, on its clips against the other songs (only this project)
// plus lock / unlock in this project, going back to the song's grid, opening it, finding matches, a second opinion.
import { api } from "./api.js";
import { S, emit, toast, useSongGrid, makeSongGrid, fullSong } from "./state.js";

export function openGrid(sid, pid = null, extra = "") {
  const q = `sid=${encodeURIComponent(sid)}${pid ? `&pid=${encodeURIComponent(pid)}` : ""}${extra ? `&${extra}` : ""}`;
  window.open(`/grid?${q}`, "retexGrid", "width=1400,height=760");
}

// TempoMap, the sidecar (a separate program): the song's tempo sections, beats and bars of any length, in a pop-out
// with click tracks to compare; from there its marks open in the Beat grid window as a proposal to fix and save.
export function openTempoMap(sid, pid = null) {
  const q = `sid=${encodeURIComponent(sid)}${pid ? `&pid=${encodeURIComponent(pid)}` : ""}`;
  window.open(`/tempomap?${q}`, "retexTempoMap", "width=1300,height=900");
}

/** Menu items (ui.js showMenu) for song sid. head: start with the song's name. */
export function songToolItems(sid, { head = true } = {}) {
  const s = S.songs.get(sid), title = s?.title || sid;
  const p = S.project, ps = p?.songs?.[sid], inProj = !!(ps && p.clips.some((c) => c.song === sid));
  const own = !!ps?.marks;
  const items = head ? [[title.length > 48 ? title.slice(0, 46) + "…" : title]] : [];
  items.push(
    ["Beat grid (the song's grid)…", () => openGrid(sid, inProj ? p.id : null), "window"],
    ...(String(sid).startsWith("midi_") || String(sid).startsWith("smp_") ? [] : [
      ["Tempo map (find its grid structure)…", () => openTempoMap(sid, inProj ? p.id : null), "window"],
    ]),
    ...(inProj ? [
      [`Marks mode - ${own ? "this project's own timing" : "timing for this project only"}`, () => emit("markMode", true), "M"],
      ...(own ? [
        ["Use the song's grid again (drop this project's timing)", () => useSongGrid(sid)],
        ["Make this project's timing the song's grid", async () => makeSongGrid(await fullSong(sid))],
      ] : []),
      [ps.lock ? "Unlock in this project (play it as recorded)" : "Lock to its marks in this project", () => emit("setLock", [sid, !ps.lock])],
      ...(own ? [] : [
        ["Verify grid to drum hits & lock (zero-phase)", () => emit("setLock", [sid, "verify"])],
      ]),
      // the other kind of lock (lock.py): pulled onto the grid along a smooth line through its marks - it follows a
      // drift that lasts, never one beat or bar, so marks that sit a little off don't make the song speed up and slow down
      [s?.lock_kind === "smooth" && ps.lock ? "Locked smoothly (lock it again that way)" : "Lock smoothly - follow its drift, never single beats", () => emit("setLock", [sid, "smooth"])],
    ] : []),
    // a lock moves every mark onto a steady grid: where the marks were wrong, the song plays off its recorded speed
    ...(s?.lock_ok !== undefined ? [
      [s.lock_off ? "What its lock changed… (parts play off speed)" : "What its lock changed…", () => import("./lockreport.js").then((m) => m.lockReport(sid))],
    ] : []),
    ["Locks that play off speed (all songs)…", () => import("./lockreport.js").then((m) => m.locksReview())],
    "-",
    ["Open the song (sections, stems, lyrics)", () => { S.selSong = sid; emit("songSelected", sid); }],
    ...(String(sid).startsWith("midi_") ? [
      ["Piano roll…", () => import("./midi.js").then((m) => m.openPianoRoll(sid, null)), "pop-out"],
      ["Instruments…", () => import("./midi.js").then((m) => m.instrumentDialog(sid, null))],
      ["Play it on another song's timing…", () => import("./midi.js").then((m) => m.timingDialog(sid))],
      ["Export as a .mid file", () => import("./midi.js").then((m) => m.exportMid(sid))],
      ["Glitch score… (its notes glitch the lanes)", () => import("./midiremix.js").then((m) => m.glitchDialog(sid))],
      ["Arrangement map… (its parts bring lanes in and out)", () => import("./midiremix.js").then((m) => m.arrangeDialog(sid))],
      ["Chord follow… (lanes follow its chords)", () => import("./midiremix.js").then((m) => m.chordDialog(sid))],
      ["Groove… (lanes take its swing)", () => import("./midiremix.js").then((m) => m.grooveDialog(sid))],
    ] : String(sid).startsWith("smp_") ? [] : [
      ["Turn a stem into MIDI (MuScriptor) ▸", (e) => import("./midi.js").then((m) => m.transcribeMenu(sid))],
    ]),
    ["Find songs that match it", () => emit("findMatches", sid)],
    ["MuScriptor second opinion on its beat marks", () => {
      api.post(`/api/songs/${sid}/muscheck`).then(() => toast("MuScriptor is transcribing it (about 2 min) - marks where its notes disagree turn amber"))
        .catch((e) => toast(e.message, true));
    }],
  );
  return items;
}
