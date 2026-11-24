// What a lock changed. A lock moves every beat mark onto a steady grid, so wherever the marks were wrong the audio
// between them plays faster or slower than it was recorded (mashlab/lock.py speed_report).
//   lockReport(sid)  one song's lock: how much of the song plays off its recorded speed, and where
//   locksReview()    every locked song, the most changed first, with the projects that have the lock switched on
import { api } from "./api.js";
import { S, emit, toast } from "./state.js";
import { modal } from "./ui.js";
import { openGrid } from "./songtools.js";

const mmss = (s) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;
const secs = (s) => (s >= 90 ? `${Math.floor(s / 60)} min ${Math.round(s % 60)} s` : `${Math.round(s)} s`);
const WRONG_SEC = 2;                                  // lock.WRONG_SEC
const LIST = 12;

// one stretch [start s, end s, speed, lowest, highest] in words
function partText([a, b, speed, lo, hi]) {
  if (lo < 0.8 && hi > 1.2) return `${mmss(a)}-${mmss(b)}: uneven - some beats as slow as ${lo.toFixed(2)}x, others as fast as ${hi.toFixed(2)}x`;
  return `${mmss(a)}-${mmss(b)}: plays at ${speed.toFixed(2)}x (${speed > 1 ? "faster" : "slower"} than recorded)`;
}

export async function lockReport(sid) {
  let r;
  try { r = await api.get(`/api/songs/${sid}/lock/report`); } catch (e) { toast(e.message, true); return; }
  const m = modal(`What the lock changed - ${r.title}`, "");
  const p = (text, cls = "") => { const e = document.createElement("p"); e.textContent = text; if (cls) e.className = cls; m.body.append(e); return e; };
  const sp = r.speed, aud = r.audit;
  if (aud && aud.active_windows > 0) {
    const sign = aud.signed_median_ms >= 0 ? "+" : "";
    const msg = aud.status === "ok"
      ? `Drum-hit verification on the locked audio: ${aud.median_offset_ms.toFixed(1)} ms median (max ${aud.max_offset_ms.toFixed(1)} ms across ${aud.active_windows} active sections).`
      : `Drum-hit verification on the locked audio: ${sign}${aud.signed_median_ms.toFixed(0)} ms off beat (median ${aud.median_offset_ms.toFixed(1)} ms, max ${aud.max_offset_ms.toFixed(1)} ms; ${aud.bad_windows} of ${aud.active_windows} sections off beat). Re-locking with "Verify grid to drum hits & re-lock" aligns the grid to zero-phase kick + snare transients and fixes this.`;
    p(msg);
  }
  if (!r.locked) p("This song has no lock.");
  else if (!sp) p("The song's beats changed after this lock was made, so its record no longer says what it did. Lock it again to get a fresh one.");
  else if (sp.sec5 < 0.5) p(`This lock changed nothing you could hear: every part of the song plays within 5% of its recorded speed (${sp.slowest.toFixed(2)}x to ${sp.fastest.toFixed(2)}x). The song was steady already.`);
  else {
    p(r.kind === "smooth"
      ? `This is a smooth lock: the song is pulled onto a steady grid along a smooth line through its marks, so it follows drift that lasts and never a single beat. Where the song's own tempo moves, the lock changes its speed to undo that. In this lock, of ${secs(sp.seconds)}:`
      : `A lock moves every beat mark onto a steady grid. Where the marks were off, the audio between them plays faster or slower than it was recorded. In this lock, of ${secs(sp.seconds)}:`);
    const ul = document.createElement("ul");
    const li = (t) => { const e = document.createElement("li"); e.textContent = t; ul.append(e); };
    li(`${secs(sp.sec20)} play more than 20% off the recorded speed`);
    li(`${secs(sp.sec5)} play more than 5% off`);
    li(`slowest ${sp.slowest.toFixed(2)}x, fastest ${sp.fastest.toFixed(2)}x`);
    m.body.append(ul);
    if (sp.parts.length) {
      p(sp.parts.length > LIST ? `The ${LIST} longest of ${sp.parts.length} stretches over 20% off:` : "Where it is more than 20% off:");
      const ul2 = document.createElement("ul");
      const longest = sp.parts.slice().sort((x, y) => (y[1] - y[0]) - (x[1] - x[0])).slice(0, LIST).sort((x, y) => x[0] - y[0]);
      for (const pt of longest) { const e = document.createElement("li"); e.textContent = partText(pt); ul2.append(e); }
      m.body.append(ul2);
    }
    if (sp.sec20 >= WRONG_SEC) p("If this is a studio recording it most likely keeps one tempo all the way, and the marks were misread in those parts. Played as recorded it keeps its own timing; turn on Click and listen. To lock it properly, fix the marks first (Repair, or line them up by ear in the Beat grid window), then lock again.");
    // what Repair + a smooth lock would leave (lock.fix_of; only for the detector's marks)
    if (r.fix && r.fix.sec20 <= 0.5 * sp.sec20) p(`Repair can fix ${r.fix.sec20 < 2 ? "it" : "most of it"}: with the miscounted stretches put back on the song's pulse and a smooth lock (one that follows drift, never a single beat), about ${secs(r.fix.sec20)} would play over 20% off instead of ${secs(sp.sec20)}. The repaired marks become this song's marks and the locked version is made again - the song's own stems are never touched.`);
    else if (sp.sec20 >= WRONG_SEC && r.yours) p("These are marks you made, so nothing here changes them: the Beat grid window has Repair, with undo.", "dim");
    if (!r.current) p("Its marks have changed since, so this lock is out of date: the song plays as recorded until it is locked again.", "dim");
  }
  m.footer.hidden = false;
  const btn = (label, cls, fn) => { const b = document.createElement("button"); b.className = cls; b.textContent = label; b.onclick = () => { m.close(); fn(); }; m.footer.append(b); };
  const inProj = !!S.project?.songs?.[sid]?.lock;
  const usedHere = !!S.project?.songs?.[sid] && S.project.clips.some((c) => c.song === sid) && !S.project.songs[sid].marks;
  if (inProj) btn("Play it as recorded (unlock here)", sp && sp.sec20 >= WRONG_SEC && (!aud || aud.status === "ok") ? "primary" : "", () => emit("setLock", [sid, false]));
  if (usedHere && ((aud && aud.status !== "ok" && aud.active_windows > 0) || (sp && sp.sec20 >= WRONG_SEC))) {
    btn("Verify grid to drum hits & re-lock", aud && aud.status === "off_beat" ? "primary" : "", () => emit("setLock", [sid, "verify"]));
  }
  if (sp && r.fix && r.fix.sec20 <= 0.5 * sp.sec20 && usedHere) btn("Repair its grid and lock again", "", () => emit("setLock", [sid, "repair"]));
  btn("Open Beat grid", "", () => openGrid(sid, S.project?.songs?.[sid] ? S.project.id : null));
  btn("All locks that play off speed…", "", locksReview);
  btn(inProj ? "Keep it locked" : "Close", "", () => {});
}

export async function locksReview() {
  const m = modal("Locks that play parts of a song off speed", `<p class="lead"><span class="spinner"></span> reading every lock…</p>`, { wide: true });
  let list;
  try { list = await api.get("/api/locks"); } catch (e) { m.body.textContent = e.message; return; }
  const bad = list.filter((r) => r.speed.sec20 >= WRONG_SEC), rest = list.length - bad.length;
  m.body.innerHTML = `<p class="lead"></p><table class="dj-plan"><thead><tr><th>song</th><th>plays over 20% off</th><th>after Repair</th><th>longest stretch</th><th>speed range</th><th>switched on in</th><th></th></tr></thead><tbody></tbody></table><p class="dim"></p>`;
  m.body.querySelector(".lead").textContent = bad.length
    ? `A lock moves every beat mark onto a steady grid, so where the marks were wrong the audio plays faster or slower than it was recorded. ${bad.length === 1 ? "This lock plays" : `These ${bad.length} locks play`} 2 s or more of the song over 20% off. "After Repair" is what would be left with its miscounted stretches put right and a smooth lock. Nothing is changed from here: open one to see where, unlock it in a project, or (with the song in the open project) repair it and lock again from "Where".`
    : "No lock plays 2 s or more of a song over 20% off its recorded speed.";
  if (!bad.length) m.body.querySelector("table").hidden = true;
  const tb = m.body.querySelector("tbody");
  for (const r of bad) {
    const sp = r.speed, long = sp.parts.slice().sort((x, y) => (y[1] - y[0]) - (x[1] - x[0]))[0];
    const tr = document.createElement("tr");
    const td = (t, cls = "") => { const e = document.createElement("td"); e.textContent = t; if (cls) e.className = cls; tr.append(e); return e; };
    td(r.title + (r.current ? "" : " (out of date)"));
    td(`${secs(sp.sec20)} of ${secs(sp.seconds)}`);
    td(r.fix ? (r.fix.sec20 <= 0.5 * sp.sec20 ? `about ${secs(r.fix.sec20)}` : "no better") : r.yours ? "your marks" : "-", r.fix && r.fix.sec20 <= 0.5 * sp.sec20 ? "" : "dim");
    td(long ? `${mmss(long[0])}-${mmss(long[1])}` + (sp.parts.length > 1 ? ` (+${sp.parts.length - 1} more)` : "") : "");
    td(`${sp.slowest.toFixed(2)}x to ${sp.fastest.toFixed(2)}x`);
    td(r.projects.length ? r.projects.join(", ") : "-", r.projects.length ? "" : "dim");
    const act = td("");
    const b1 = document.createElement("button"); b1.className = "small"; b1.textContent = "Where"; b1.onclick = () => { m.close(); lockReport(r.id); };
    const b2 = document.createElement("button"); b2.className = "small"; b2.textContent = "Beat grid"; b2.onclick = () => openGrid(r.id);
    act.append(b1, " ", b2);
    if (!r.current) tr.title = "Its marks changed since this lock was made: it is out of date, and the song plays as recorded until it is locked again";
    tb.append(tr);
  }
  m.body.querySelector("p.dim").textContent = rest ? `${rest} other lock${rest > 1 ? "s" : ""} changed little or nothing.` : "";
}
