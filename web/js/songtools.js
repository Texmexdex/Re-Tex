// What you can do with a song, wherever you right-click it (a clip or a lane in the timeline, the song in the library,
// the song view) or open its Tools menu in the arrangement bar - one list, so every place offers the same tools.
//   Beat grid…      the song's grid (its beat marks, for projects from now on - saved projects keep the grid they
//                   had); opened from a project it can also save just for that project
//   Marks mode      this project's timing for the song, on its clips against the other songs (only this project)
// plus lock / unlock in this project, going back to the song's grid, opening it, finding matches, a second opinion.
import { api } from "./api.js";
import { S, emit, toast, useSongGrid, makeSongGrid, fullSong } from "./state.js";

export function openGrid(sid, pid = null) {
  const q = `sid=${encodeURIComponent(sid)}${pid ? `&pid=${encodeURIComponent(pid)}` : ""}`;
  window.open(`/grid?${q}`, "retexGrid", "width=1400,height=760");
}

/** Menu items (ui.js showMenu) for song sid. head: start with the song's name. */
export function songToolItems(sid, { head = true } = {}) {
  const s = S.songs.get(sid), title = s?.title || sid;
  const p = S.project, ps = p?.songs?.[sid], inProj = !!(ps && p.clips.some((c) => c.song === sid));
  const own = !!ps?.marks;
  const items = head ? [[title.length > 48 ? title.slice(0, 46) + "…" : title]] : [];
  items.push(
    ["Beat grid (the song's grid)…", () => openGrid(sid, inProj ? p.id : null), "window"],
    ...(inProj ? [
      [`Marks mode - ${own ? "this project's own timing" : "timing for this project only"}`, () => emit("markMode", true), "M"],
      ...(own ? [
        ["Use the song's grid again (drop this project's timing)", () => useSongGrid(sid)],
        ["Make this project's timing the song's grid", async () => makeSongGrid(await fullSong(sid))],
      ] : []),
      [ps.lock ? "Unlock in this project (play it as recorded)" : "Lock to its marks in this project", () => emit("setLock", [sid, !ps.lock])],
    ] : []),
    "-",
    ["Open the song (sections, stems, lyrics)", () => { S.selSong = sid; emit("songSelected", sid); }],
    ["Find songs that match it", () => emit("findMatches", sid)],
    ["MuScriptor second opinion on its beat marks", () => {
      api.post(`/api/songs/${sid}/muscheck`).then(() => toast("MuScriptor is transcribing it (about 2 min) - marks where its notes disagree turn amber"))
        .catch((e) => toast(e.message, true));
    }],
  );
  return items;
}
