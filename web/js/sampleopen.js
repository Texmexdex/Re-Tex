// The arrangement's side of the sample editor (web/js/sampler.js in its own window): open a clip in it, and swap
// the clip for the baked sample when "Apply to the clip" comes back ("sample_applied" on the event stream - so it
// works even after this page reloads). Only that clip changes: same lane, same start; nothing else moves.
// Also the Screw (the whole mix slower, pitch down with it - drums too): project.screw, next to the BPM.
import { api } from "./api.js";
import { S, on, edit, emit, toast, invalidateSong, applyFrozen, useSongGrid } from "./state.js";

export async function openSampleEditor(clipId) {
  if (!S.project) return;
  // open the window now (a pop-up opened after a wait gets blocked), point it at the clip once it's rendered
  const w = window.open("/sample", "retexSampleEditor", "width=1320,height=820");
  if (!w) { toast("Allow pop-ups for Re-Tex to open the sample editor", true, 7000); return; }
  try {
    const meta = await api.post("/api/sample/open", { project: S.project, clip_id: clipId });
    w.location.href = `/sample?eid=${encodeURIComponent(meta.eid)}`;
    w.focus();
  } catch (e) {
    toast(`Couldn't open the clip in the sample editor: ${e.message}`, true, 7000);
    try { w.close(); } catch {}
  }
}

on("event:sample_applied", (m) => {
  if (!S.project || S.project.id !== m.project) return;
  if (!S.project.clips.some((c) => c.id === m.clip)) { toast("The sample is baked, but its clip isn't in the arrangement any more", true); return; }
  if (m.song) S.songs.set(m.sid, m.song);
  edit((p) => {
    const c = p.clips.find((x) => x.id === m.clip);
    if (!c) return;
    Object.assign(c, { song: m.sid, src: 0, len: m.beats, rate: 1, pitch: 0, keep_pitch: true });
    delete c.warps;
    p.songs[m.sid] = { transpose: 0, mult: 1 };
  }, { label: "Sample edit" });
  emit("selection");
  toast("The clip now plays your edited sample - Ctrl+Z brings the original back");
});

// The beat grid window (gridtool.js) and this page:
//   marks              it saved the song's grid: read the song again; if the open project was frozen on the old grid
//                      (the server did it to its file), do the same here; the project it was opened from follows
//                      the new grid (its own timing for the song, if it had one, goes)
//   get-project-marks  it asks the open project's own timing for a song   -> project-marks
//   set-project-marks  "save for this project only": the open project's own timing (undoable here)
//                                                                         -> project-marks-saved
if ("BroadcastChannel" in window) {
  const ch = new BroadcastChannel("retex");
  ch.onmessage = (e) => {
    const d = e.data || {}, p = S.project;
    if (d.type === "marks" && d.sid) {
      invalidateSong(d.sid);
      applyFrozen(d.sid, d);
      if (d.pid && p?.id === d.pid && p.songs?.[d.sid]?.marks) useSongGrid(d.sid);
      emit("songUpdated", d.sid);
      toast("The song's grid you saved is in");
    } else if (d.type === "get-project-marks" && p?.id === d.pid) {
      ch.postMessage({ type: "project-marks", sid: d.sid, pid: d.pid, name: p.name, marks: p.songs?.[d.sid]?.marks || null });
    } else if (d.type === "set-project-marks" && p?.id === d.pid && p.songs?.[d.sid] && d.marks?.t) {
      edit((pp) => { pp.songs[d.sid].marks = d.marks; }, { label: "Beat grid: this project's timing" });
      emit("marks", d.sid);
      ch.postMessage({ type: "project-marks-saved", sid: d.sid, pid: d.pid });
      toast("This project has its own timing for the song now (Ctrl+Z undoes)");
    }
  };
}

// ---------------------------------------------------------------- screw
const screwSel = document.getElementById("screwSel");
export function setScrew(r) {
  edit((p) => {
    const was = p.screw || 1;
    p.bpm = Math.round((p.bpm / was) * r * 1000) / 1000;
    if (Math.abs(r - 1) < 1e-9) delete p.screw; else p.screw = r;
  }, { label: r === 1 ? "Unscrew" : `Screw ${Math.round((r - 1) * 100)}%` });
}
if (screwSel) {
  screwSel.innerHTML = [1, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65].map((r) =>
    `<option value="${r}">${r === 1 ? "no screw" : `screw ${Math.round((r - 1) * 100)}%`}</option>`).join("");
  screwSel.onchange = () => { setScrew(+screwSel.value); screwSel.blur(); };
  on("project", () => { screwSel.value = String(S.project?.screw || 1); });
}
