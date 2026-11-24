// Boot, event routing, shortcuts, project/save/export dialogs, per-song controls, status bar.
import { ICONS } from "./icons.js";
import { api, connectEvents } from "./api.js";
import { S, on, emit, edit, setProject, doUndo, doRedo, toast, refreshWaiting, secPerBeat, transportPos, songMult, history, playedTuning, syncNow, fullSong, projectInfo } from "./state.js";
import { lockFor } from "./timing.js";
import { loadSongs } from "./library.js";
import { applyState, togglePlay, seekTo, toggleLoop, toggleFollow, toggleMetronome } from "./transport.js";
import { deleteSel, duplicateSel, splitAtPlayhead, nudgeSel, fit, zoomAt, view, alignSelectionToGrid, alignSongToGrid, clearHighlight, markKey } from "./timeline.js";
import { openMashup } from "./mashup.js";
import { openDjMix } from "./djmix.js";
import { openReimagine } from "./reimagine.js";
import { openSettings, outputMenu } from "./settings.js";
import "./listening.js";                                   // the Listening button (test results to hear)
import "./sampleopen.js";                                  // the sample editor's hand-off + the Screw menu
import { stopAudition } from "./songview.js";
import { performKey } from "./perform.js";                // Perform mode (off by default): pads on A-L / Z-M
import { modal, showMenu } from "./ui.js";
import { songToolItems, openGrid } from "./songtools.js";
import { showComponent, componentJob, showEarlierFallbacks, markFallbackSeen } from "./components.js";

// ---------------------------------------------------------------- events from the server
connectEvents((m) => {
  if (m.type === "hello") { applyState(m.transport); m.jobs.forEach((j) => onJob({ job: j })); showEarlierFallbacks(); return; }
  if (m.type === "transport") { applyState(m); return; }
  if (m.type === "job") onJob(m);
  if (m.type === "component_missing") showComponent(m.info);
  if (m.type === "component_fallback") { showComponent(m.info, { fallback: true }); markFallbackSeen(m.info.key, m.info.since); }
  if (m.type === "renders") refreshWaiting();
  emit("event:" + m.type, m);
});

// ---------------------------------------------------------------- status bar: running jobs
const jobsEl = document.getElementById("jobs");
function onJob(m) {
  const j = m.job;
  S.jobs.set(j.id, j);
  if (j.status === "done" || j.status === "cancelled") setTimeout(() => { S.jobs.delete(j.id); renderJobs(); }, 2500);
  componentJob(j);
  if (j.status === "error") {
    if (j.missing) showComponent(j.missing);          // what's missing, the fix, a Download button
    toast(`${j.title}: ${j.error}`, true, 8000);
    setTimeout(() => { S.jobs.delete(j.id); renderJobs(); }, 12000);
  }
  if (j.kind === "export" && j.status === "done" && m.result) exportDone(m.result);
  if (j.kind === "lock") renderProjSongs();
  renderJobs();
}
function renderJobs() {
  jobsEl.innerHTML = "";
  for (const j of S.jobs.values()) {
    const d = document.createElement("div");
    d.className = "job" + (j.status === "error" ? " error" : "");
    const running = j.status === "running" || j.status === "queued";
    d.innerHTML = `${running ? '<span class="spinner" style="width:11px;height:11px"></span>' : j.status === "done" ? "✓" : j.status === "cancelled" ? "–" : "!"}
      <span class="n"></span><div class="bar"><i style="width:${Math.round(j.progress * 100)}%"></i></div>
      ${running && j.kind !== "render" ? '<button title="Cancel">✕</button>' : ""}`;
    d.querySelector(".n").textContent = `${j.title}${j.message && (running || j.status === "cancelled") ? " - " + j.message : ""}`;
    d.querySelector("button")?.addEventListener("click", () => api.post(`/api/jobs/${j.id}/cancel`));
    jobsEl.appendChild(d);
  }
}

// The lock pulls every part of a song to its main tempo; where the marks run at another speed (lock.check) it asks
// instead of failing: play it as recorded, repair the detector's miscount, line it up by ear, or lock anyway.
// "As recorded" first (Oct 19): a studio take is usually steady - Fear Inoculum keeps one tempo for ten minutes while
// the drummer's groupings (16ths, then triplets) pull the detector's marks to "148 BPM"; every lock of those marks
// jumped around, as recorded it sat with the click and kept coming back around to it (the user, a drummer).
const mmss = (s) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;
const fmtSecs = (s) => (s >= 90 ? `${Math.floor(s / 60)} min ${Math.round(s % 60)} s` : `${Math.round(s)} s`);
function lockChoice(title, sid, c, startLock, playAsRecorded = null) {
  const aud = c.audit;
  const offBeat = !!(c.off_beat && aud && aud.active_windows > 0);
  const hasSlips = !!(c.slips && c.slips.length);
  const m = modal(offBeat && !hasSlips ? "Lock: marks are off the drum hits" : "Lock: some parts would play at another speed", "");
  const p = (text) => { const e = document.createElement("p"); e.textContent = text; m.body.append(e); return e; };
  if (offBeat) {
    const sign = aud.signed_median_ms >= 0 ? "+" : "";
    p(`Measured against "${title}"'s real drum hits, the current beat marks sit ${sign}${Math.round(aud.signed_median_ms)} ms off beat (median ${aud.median_offset_ms.toFixed(0)} ms, max ${aud.max_offset_ms.toFixed(0)} ms across ${aud.bad_windows} of ${aud.active_windows} active sections). Locking to them as they are would bake that offset into the stems.`);
  }
  if (hasSlips) {
    p(`Locking pulls every part of "${title}" to its main tempo. In these parts the marks run at a different speed, so locking would play them faster or slower than recorded:`);
    const ul = document.createElement("ul");
    const parts = c.speed?.parts || [];
    for (const [a, b, k] of c.slips) {
      const li = document.createElement("li");
      // a stretch whose beats are uneven (a bunched bar: one long beat, then short ones) plays some slow and some fast
      const pt = parts.find((q) => Math.abs(q[0] - a) < 0.02 && Math.abs(q[1] - b) < 0.02);
      li.textContent = pt && pt[3] < 0.8 && pt[4] > 1.2
        ? `${mmss(a)}-${mmss(b)}: uneven marks - some beats would play as slow as ${pt[3].toFixed(2)}x, others as fast as ${pt[4].toFixed(2)}x`
        : `${mmss(a)}-${mmss(b)}: ${k.toFixed(2)}x the main tempo (${k > 1 ? "it would play slower" : "it would play faster"})`;
      ul.append(li);
    }
    m.body.append(ul);
  }
  const sp = c.speed;
  if (hasSlips && sp && sp.sec20 >= 1) {
    const more = (sp.parts?.length || 0) - c.slips.length;
    p(`All together a lock to these marks would play ${fmtSecs(sp.sec20)} of the song more than 20% off its recorded speed (from ${sp.slowest.toFixed(2)}x to ${sp.fastest.toFixed(2)}x)${more > 0 ? `, in ${sp.parts.length} stretches - the longest are listed` : ""}.`);
  }
  if (hasSlips) {
    p("If it's a studio recording, it most likely keeps one steady tempo all the way and only the drummer's grouping changes there (16ths to triplets, odd groupings) - the marks follow the grouping, not the tempo, and a lock to them would make it speed up and slow down. Then it's best played as recorded: turn on Click and listen - it should keep coming back around to the click. A lock is for takes that really drift (live recordings).");
  }
  // Repair (lock.repaired): miscounted stretches re-counted, stretches where the detector lost the beat evened out;
  // then a smooth lock, which follows drift and never a single beat. c.repair says what that would leave.
  const rp = c.repair, was = sp?.sec20 || 0;
  const helps = !!rp && !c.repair_clears && was >= 2 && rp.sec20 <= 0.5 * was;       // not all of it, but most
  if (offBeat) {
    p("Verify & lock aligns the beat grid to the zero-phase kick + snare transients first (removing constant phase errors and 2x/0.5x glitches), saves it as the song's grid (older projects keep their old grid), then locks and verifies the rendered drums.");
  } else if (c.repair_clears) p(`Repair fixes it: it puts the stretches the detector miscounted back on the song's pulse${rp?.method === "neighbours" ? " (and gives even beats to the ones where it lost the beat)" : ""}, then locks the song smoothly - following its drift, never a single beat${rp ? (rp.sec20 < 1 ? ". Nothing would be left playing over 20% off" : `. About ${fmtSecs(rp.sec20)} would be left over 20% off`) : ""}. The repaired marks become this song's marks; the detector's stay available.`);
  else if (c.yours) p("These are marks you made. Line them up by ear in the Beat grid window (it has Verify and Repair too, with undo), or lock anyway.");
  else if (helps) p(`Repair fixes most of it: about ${fmtSecs(rp.sec20)} would still play over 20% off instead of ${fmtSecs(was)}${c.left?.length ? ` (${c.left.map(([a, b]) => `${mmss(a)}-${mmss(b)}`).slice(0, 4).join(", ")})` : ""}. What is left may be a real change of tempo - listen there with Click on. The repaired marks become this song's marks; the detector's stay available.`);
  else if (hasSlips) p("Repair can't fix these: they may be real tempo changes (a live take, an outro). Line them up by ear, or lock anyway and listen.");
  m.footer.hidden = false;
  const btn = (label, cls, fn) => { const b = document.createElement("button"); b.className = cls; b.textContent = label; b.onclick = () => { m.close(); fn(); }; m.footer.append(b); };
  if (offBeat && !S.project?.songs?.[sid]?.marks) {
    btn("Verify grid to drum hits & lock", "primary", () => startLock({ verify: true, force: true, project: S.project?.id }));
  }
  if (playAsRecorded) btn("Play it as recorded (don't lock)", offBeat ? "" : "primary", playAsRecorded);
  if (c.repair_clears) btn("Repair, then lock", "", () => startLock({ repair: true }));
  else if (helps) btn(`Repair, then lock (leaves ${fmtSecs(rp.sec20)})`, "", () => startLock({ repair: true, force: true }));
  if (!offBeat && !S.project?.songs?.[sid]?.marks) {
    btn("Verify grid to drum hits & lock", "", () => startLock({ verify: true, force: true, project: S.project?.id }));
  }
  btn("Line it up by ear", "", () => openGrid(sid, S.project?.id));
  btn("Lock anyway", "", () => startLock({ force: true }));
  btn("Cancel", "", () => {});
}

// ---------------------------------------------------------------- per-song controls (transpose / half-double)
const psEl = document.getElementById("projSongs");
const lockActions = new Map();                  // song id -> what its lock button does now (Tools / right-click menus)
const pendingLockSnap = new Set();              // song ids currently locking from this window: snap clips when done
on("setLock", ([sid, lockIt]) => lockActions.get(sid)?.[lockIt === "smooth" ? "smooth" : lockIt === "repair" ? "repair" : lockIt === "verify" ? "verify" : lockIt ? "lock" : "unlock"]());
function renderProjSongs() {
  const p = S.project;
  psEl.innerHTML = "";
  if (!p || !Object.keys(p.songs).length) {
    psEl.innerHTML = `<span class="empty">Songs in this arrangement will appear here with their transpose and tempo controls.</span>`;
    return;
  }
  const used = new Set(p.clips.map((c) => c.song));
  for (const [sid, cfg] of Object.entries(p.songs)) {
    if (!used.has(sid)) continue;
    const info = S.songs.get(sid) || { title: sid, bpm: p.bpm };
    const mult = cfg.mult ?? 1;
    const stretch = ((p.bpm / (info.bpm * mult)) - 1) * 100;
    const el = document.createElement("div");
    el.className = "psong";
    el.innerHTML = `<button class="fm" title="Find songs that match this one">${ICONS.search}</button><span class="n"></span>
      <span class="ctl" title="Transpose (drums are never pitched)"><button data-t="-1">−</button><span class="v ${cfg.transpose ? "changed" : ""}">${cfg.transpose > 0 ? "+" : ""}${cfg.transpose || 0} st</span><button data-t="1">+</button></span>
      <span class="ctl" title="Play this song at half / normal / double time"><button data-m="0.5" class="${mult === 0.5 ? "toggle on" : ""}">½×</button><button data-m="1" class="${mult === 1 ? "toggle on" : ""}">1×</button><button data-m="2" class="${mult === 2 ? "toggle on" : ""}">2×</button></span>
      <span class="v ${Math.abs(stretch) > 8 ? "changed" : ""}" title="Tempo change applied to this song">${stretch >= 0 ? "+" : ""}${stretch.toFixed(1)}%</span>
      <span class="ctl" title="Tuning: songs play at their own tuning unless you pick something to match (drums are never pitched)"><select class="tune"></select></span>
      <span class="ctl"><button class="lock"></button></span>
      <span class="ctl"><button class="tools" title="Everything you can do with this song: its beat grid, Marks mode (this project's timing), lock, ...">Tools ▾</button></span>`;
    el.querySelector(".n").textContent = info.title;
    el.querySelector(".fm").onclick = () => emit("findMatches", sid);
    el.querySelector(".tools").onclick = (e) => showMenu(e, songToolItems(sid));
    el.oncontextmenu = (e) => { e.preventDefault(); showMenu(e, songToolItems(sid)); };
    el.querySelector(".n").title = `${info.title} · ${info.bpm?.toFixed(1)} BPM · ${info.key?.name || ""}`;
    el.querySelectorAll("[data-t]").forEach((b) => b.onclick = () => edit((pp) => { pp.songs[sid].transpose = Math.max(-12, Math.min(12, (pp.songs[sid].transpose || 0) + +b.dataset.t)); }));
    // lock to marks: the song warped once so every mark sits on a steady grid (mashlab/lock.py); toggles per song
    const lb = el.querySelector(".lock");
    const job = [...S.jobs.values()].find((j) => j.kind === "lock" && j.ref === sid && (j.status === "running" || j.status === "queued"));
    // the grid this project plays the song on: its own timing (Marks mode) or the song's grid - each has its own lock
    const ownGrid = cfg.marks || null, fi = S.fullDone.get(sid);
    if (ownGrid && !fi) fullSong(sid).then(renderProjSongs).catch(() => {});
    const lockOk = ownGrid ? !!(fi && lockFor(projectInfo(fi))) : !!info.lock_ok;
    const startLock = (body) => {
      pendingLockSnap.add(sid);
      return api.post(`/api/songs/${sid}/lock`, { ...(body || {}), ...(ownGrid ? { marks: ownGrid } : {}) })
        .then(() => toast("Locking - warping its stems onto a steady grid, once (about a minute). It switches over by itself.", false, 6000))
        .catch((e) => { pendingLockSnap.delete(sid); toast(e.message, true); });
    };
    // kind "smooth": the lock follows the marks' drift, never a single beat (lock.py) - asked for from the Tools menu
    const makeLock = async (kind = null) => {
      let c = null;
      try { c = ownGrid ? await api.post(`/api/songs/${sid}/lock/check`, { marks: ownGrid }) : await api.get(`/api/songs/${sid}/lock/check`); }
      catch { /* the lock job runs the check itself */ }
      const go = (body) => startLock({ ...(kind ? { kind } : {}), ...(body || {}) });
      const slips = kind === "smooth" && c?.smooth ? c.smooth.slips : c?.slips;
      if (!slips?.length && !c?.off_beat) return go();
      lockChoice(info.title, sid, c, go, () => edit((pp) => { delete pp.songs[sid].lock; }, { label: "Play as recorded" }));
    };
    // a lock that plays 2 s or more of the song over 20 % off its recorded speed, or whose verified drum hits sit
    // off beat, warns on the button and opens the lock report on click
    const off = !ownGrid && info.lock_off;
    const aud = !ownGrid && info.lock_audit;
    const auditOff = !!(aud && aud.status === "off_beat" && aud.active_windows > 0);
    if (cfg.lock && lockOk && (off || auditOff)) {
      lb.innerHTML = ICONS.lock + "<span>locked · check</span>"; lb.className = "lock toggle on warn";
      lb.title = auditOff
        ? `Locked - but measured drum hits sit ${aud.median_offset_ms.toFixed(0)} ms off the grid (${aud.bad_windows}/${aud.active_windows} sections). Click to inspect and verify & re-lock.`
        : `Locked - but this lock plays about ${fmtSecs(off)} of the song more than 20% faster or slower than it was recorded (its marks were off there). Click to see where, and to play it as recorded.`;
      lb.onclick = () => import("./lockreport.js").then((m) => m.lockReport(sid));
    } else if (cfg.lock && lockOk) {
      const smooth = !ownGrid && info.lock_kind === "smooth";
      const audNote = aud && aud.active_windows > 0 ? ` Verified against drum hits: ${aud.median_offset_ms.toFixed(1)} ms median (max ${aud.max_offset_ms.toFixed(1)} ms).` : "";
      lb.innerHTML = ICONS.lock + `<span>${smooth ? "locked · smooth" : "locked"}</span>`; lb.className = "lock toggle on";
      lb.title = (smooth ? "Locked smoothly: plays the locked version, pulled onto a steady grid along a smooth line through its marks (it follows drift, never a single beat)."
        : "Locked to its marks: plays the locked version, every mark on a steady grid.") + audNote + " Click to play it as recorded again.";
      lb.onclick = () => edit((pp) => { delete pp.songs[sid].lock; }, { label: "Unlock song" });
    } else if (cfg.lock) {
      lb.innerHTML = ICONS.lock + (job ? `<span>locking… ${Math.round((job.progress || 0) * 100)}%</span>` : "<span>re-lock</span>");
      lb.className = "lock toggle on";
      lb.title = job ? "Making the locked version - it plays as recorded until it's done" : "Its marks (or stems) changed since it was locked: click to lock it again from the current marks. Plays as recorded meanwhile.";
      lb.onclick = () => { if (!job) makeLock(); else edit((pp) => { delete pp.songs[sid].lock; }, { label: "Unlock song" }); };
    } else {
      lb.innerHTML = ICONS.lock; lb.className = "lock";
      lb.title = "Lock to marks: warp this song so every one of its marks sits on a steady grid (made once, then toggles instantly). For songs whose own timing drifts.";
      lb.onclick = () => {
        edit((pp) => { pp.songs[sid].lock = true; }, { label: "Lock song" });
        if (lockOk) alignSongToGrid(sid);
        else makeLock();
      };
    }
    lockActions.set(sid, {
      lock: () => { if (!cfg.lock) edit((pp) => { pp.songs[sid].lock = true; }, { label: "Lock song" }); if (lockOk) alignSongToGrid(sid); else if (!job) makeLock(); },
      unlock: () => edit((pp) => { delete pp.songs[sid].lock; }, { label: "Unlock song" }),
      // made again even if a lock exists: the smooth one replaces it (same marks, same place; the stems are never touched)
      smooth: () => { if (!cfg.lock) edit((pp) => { pp.songs[sid].lock = true; }, { label: "Lock song" }); if (!job) makeLock("smooth"); },
      // the song's grid repaired (it becomes the song's marks), then locked smoothly - for a lock that plays parts off speed
      repair: () => { if (!cfg.lock) edit((pp) => { pp.songs[sid].lock = true; }, { label: "Lock song" }); if (!job && !ownGrid) startLock({ repair: true, force: true }); },
      // zero-phase drum-hit-verified grid, saved as the song's grid (freezing older projects), then locked & verified
      verify: () => { if (!cfg.lock) edit((pp) => { pp.songs[sid].lock = true; }, { label: "Lock song" }); if (!job && !ownGrid) startLock({ verify: true, force: true, project: S.project?.id }); },
    });
    // tuning: as recorded unless you pick A440 or another song of the arrangement to match
    const sel = el.querySelector(".tune"), own = info.tuning_st;
    const cents = (st) => `${st >= 0 ? "+" : "−"}${Math.abs(Math.round(st * 100))} c`;
    if (own == null) {
      sel.innerHTML = `<option>tuning not measured</option>`;
      sel.disabled = true;
    } else {
      const opts = [["", `Tuning: as recorded (${cents(own)} from A440)`], ["a440", `Tune to A440 (shift ${cents(-own)})`]];
      for (const other of Object.keys(p.songs)) {
        if (other === sid || !used.has(other)) continue;
        const t = playedTuning(p, other, new Set([sid]));
        if (t != null) opts.push([other, `Match ${S.songs.get(other)?.title || other} (shift ${cents(t - own)})`]);
      }
      sel.innerHTML = opts.map(([v, label]) => `<option value="${v}"></option>`).join("");
      [...sel.options].forEach((o, i) => (o.textContent = opts[i][1]));
      sel.value = opts.some(([v]) => v === (cfg.tune_to || "")) ? (cfg.tune_to || "") : "";
      sel.classList.toggle("changed", !!cfg.tune_to);
      sel.onchange = () => edit((pp) => { if (sel.value) pp.songs[sid].tune_to = sel.value; else delete pp.songs[sid].tune_to; }, { label: "Tuning" });
    }
    el.querySelectorAll("[data-m]").forEach((b) => b.onclick = () => {
      const nm = +b.dataset.m;
      if (nm === mult) return;
      // keep each clip covering the same source material: its length scales with the tempo multiplier
      edit((pp) => {
        pp.songs[sid].mult = nm;
        for (const c of pp.clips) if (c.song === sid) c.len = c.len * nm / mult;
      });
    });
    psEl.appendChild(el);
  }
}
on("project", renderProjSongs);
on("songUpdated", renderProjSongs);
// a lock finished (or went stale): the audio engine has to pick the other version of the song, and if this window
// started the lock we snap the song's clips onto the newly locked steady grid so they don't stay off by a fraction
on("songUpdated", (id) => {
  if (S.project?.songs?.[id]?.lock) {
    syncNow();
    if (pendingLockSnap.has(id) && S.songs.get(id)?.lock_ok) {
      pendingLockSnap.delete(id);
      alignSongToGrid(id).then((n) => {
        const aud = S.songs.get(id)?.lock_audit;
        const ms = aud && aud.active_windows > 0 ? ` (${aud.median_offset_ms.toFixed(1)} ms median vs drum hits)` : "";
        if (n > 0) toast(`Locked & snapped ${n} clip${n > 1 ? "s" : ""} onto the grid${ms}`, false, 4500);
        else if (ms) toast(`Locked & verified${ms}`, false, 4000);
      });
    }
  }
});

// ---------------------------------------------------------------- projects
const projName = document.getElementById("projName");
projName.onchange = () => edit((p) => (p.name = projName.value.trim() || "Untitled mashup"));
on("project", () => { if (S.project && document.activeElement !== projName) projName.value = S.project.name; document.title = `${S.project?.name || ""} · Re-Tex`; });

async function save() {
  if (!S.project) return;
  await api.put(`/api/projects/${S.project.id}`, S.project);
  S.dirty = false;
  localStorage.setItem("lastProject", S.project.id);
  toast("Saved");
}
document.getElementById("btnSave").onclick = save;

async function newProject() {
  setProject(await api.post("/api/projects/new", { name: "Untitled mashup", bpm: 120 }));
}

// Branch: the project as it is now stays saved under its own name; editing carries on in a copy ("<name> (branch N)").
// Autosave writes every edit, so this is the way to try changes without losing the version you have.
async function branchProject(pid) {
  const open = S.project && S.project.id === pid;
  if (open && S.dirty) await save();
  const p = await api.post(`/api/projects/${pid}/branch`, {});
  const from = p.branch_of?.name || "the original";
  setProject(p);
  localStorage.setItem("lastProject", p.id);
  fit();
  toast(`Now editing "${p.name}" - "${from}" stays as it was`, false, 6000);
}
document.getElementById("btnBranch").onclick = () => {
  if (!S.project?.clips.length) { toast("Nothing to branch yet - add something to the arrangement first", true); return; }
  branchProject(S.project.id).catch((e) => toast(e.message, true));
};

async function projectsDialog() {
  const list = await api.get("/api/projects");
  const m = modal("Projects", `
    <div class="row-gap" style="margin-bottom:12px"><button class="primary" id="pNew">+ New empty project</button>
      <span style="color:var(--dim)">or use Auto-Mashup to start from a generated arrangement</span></div>
    <div class="proj-list">${list.length ? "" : '<div class="empty-hint" style="padding:6px">No saved projects yet.</div>'}</div>`);
  const box = m.body.querySelector(".proj-list");
  for (const p of list) {
    const d = document.createElement("div");
    d.className = "proj-item";
    d.innerHTML = `<span class="t"></span><span class="meta"><span class="from"></span>${p.bpm} BPM · ${p.n_clips} clips · ${new Date(p.modified * 1000).toLocaleString()}</span>`
      + `<button class="small branch" title="Keep this one as it is and open a copy to change">Branch</button><button class="small danger">Delete</button>`;
    d.querySelector(".t").textContent = p.name;
    if (p.branch_of) d.querySelector(".from").textContent = `branch of ${p.branch_of.name} · `;
    d.onclick = async (e) => {
      if (e.target.tagName === "BUTTON") return;
      if (S.dirty && S.project?.clips.length && !confirm("Discard unsaved changes to the current arrangement?")) return;
      setProject(await api.get(`/api/projects/${p.id}`));
      localStorage.setItem("lastProject", p.id);
      fit(); m.close();
    };
    d.querySelector("button.branch").onclick = async () => {
      if (S.project?.id !== p.id && S.dirty && S.project?.clips.length && !confirm("Discard unsaved changes to the current arrangement?")) return;
      try { await branchProject(p.id); m.close(); } catch (e) { toast(e.message, true); }
    };
    d.querySelector("button.danger").onclick = async () => { if (confirm(`Delete project "${p.name}"?`)) { await api.del(`/api/projects/${p.id}`); d.remove(); } };
    box.appendChild(d);
  }
  m.body.querySelector("#pNew").onclick = async () => {
    if (S.dirty && S.project?.clips.length && !confirm("Discard unsaved changes to the current arrangement?")) return;
    await newProject(); m.close();
  };
}
document.getElementById("btnProjects").onclick = projectsDialog;
document.getElementById("btnMashup").onclick = openMashup;
document.getElementById("btnDj").onclick = openDjMix;
document.getElementById("btnReimagine").onclick = openReimagine;
document.getElementById("btnSettings").onclick = openSettings;
{ const spk = document.querySelector("#topbar .vol > span"); spk.title = "Audio output - click to switch device"; spk.onclick = (e) => { e.preventDefault(); outputMenu(e); }; }

// ---------------------------------------------------------------- export
function exportDialog() {
  if (!S.project?.clips.length) { toast("Nothing to export yet", true); return; }
  const m = modal("Export", `
    <p class="lead">Renders the whole arrangement (same engine as playback) with the master glue + limiter.</p>
    <div class="opts" style="margin-top:0">
      <label>Format <select id="fmt"><option value="wav">WAV 24-bit</option><option value="mp3">MP3 320k</option></select></label>
      <label title="Match loudness to typical streaming level"><input type="checkbox" id="norm" checked> Normalize to
        <input type="number" id="lufs" value="${S.project.master?.target_lufs ?? -12}" step="0.5" style="width:64px"> LUFS</label>
    </div>`);
  m.footer.hidden = false;
  m.footer.innerHTML = `<button id="cancel">Cancel</button><button class="primary" id="go">Export</button>`;
  m.footer.querySelector("#cancel").onclick = m.close;
  m.footer.querySelector("#go").onclick = async () => {
    const proj = JSON.parse(JSON.stringify(S.project));
    proj.master = { ...proj.master, target_lufs: +m.body.querySelector("#lufs").value };
    await api.post("/api/export", { project: proj, format: m.body.querySelector("#fmt").value, normalize: m.body.querySelector("#norm").checked });
    toast("Exporting… (see the status bar)");
    m.close();
  };
}
function exportDone(r) {
  const m = modal("Export finished", `<p>${r.file} · ${Math.floor(r.seconds / 60)}:${String(Math.round(r.seconds % 60)).padStart(2, "0")}</p>
    <div class="row-gap"><a href="/api/exports/${encodeURIComponent(r.file)}" download><button class="primary">Download</button></a>
    <button id="reveal">Show in folder</button></div>`);
  m.body.querySelector("#reveal").onclick = () => api.post("/api/reveal", { path: r.path });
}
document.getElementById("btnExport").onclick = exportDialog;

// ---------------------------------------------------------------- undo / redo buttons
const btnUndo = document.getElementById("btnUndo"), btnRedo = document.getElementById("btnRedo");
btnUndo.onclick = doUndo;
btnRedo.onclick = doRedo;
on("history", (h) => {
  btnUndo.disabled = !h.undo; btnRedo.disabled = !h.redo;
  btnUndo.title = h.undo ? `Undo: ${h.undoLabel} (Ctrl+Z) · ${h.undo} step${h.undo > 1 ? "s" : ""} back available` : "Nothing to undo";
  btnRedo.title = h.redo ? `Redo: ${h.redoLabel} (Ctrl+Y)` : "Nothing to redo";
});

// ---------------------------------------------------------------- keyboard
window.addEventListener("keydown", (e) => {
  const tag = e.target.tagName;
  const k = e.key;
  const typing = (tag === "INPUT" && !["range", "checkbox", "button"].includes(e.target.type)) || tag === "TEXTAREA";
  if (document.querySelector(".modal-back")) return;
  // undo/redo work everywhere except while typing in a text field
  if ((e.ctrlKey || e.metaKey) && !typing && (k.toLowerCase() === "z" || k.toLowerCase() === "y")) {
    e.preventDefault(); if (tag === "SELECT") e.target.blur();
    (k.toLowerCase() === "y" || e.shiftKey) ? doRedo() : doUndo();
    return;
  }
  if (typing || tag === "SELECT") return;
  if (performKey(e)) { e.preventDefault(); return; }          // only while Perform is on; otherwise does nothing
  if (markKey(e)) { e.preventDefault(); return; }              // Marks mode: M, T (tap), Del, Esc
  if (k === " ") { e.preventDefault(); togglePlay(); }
  else if (k === "Delete" || k === "Backspace") { e.preventDefault(); deleteSel(); }
  else if ((e.ctrlKey || e.metaKey) && k.toLowerCase() === "z") { e.preventDefault(); e.shiftKey ? doRedo() : doUndo(); }
  else if ((e.ctrlKey || e.metaKey) && k.toLowerCase() === "y") { e.preventDefault(); doRedo(); }
  else if ((e.ctrlKey || e.metaKey) && k.toLowerCase() === "d") { e.preventDefault(); duplicateSel(); }
  else if ((e.ctrlKey || e.metaKey) && k.toLowerCase() === "s") { e.preventDefault(); save(); }
  else if ((e.ctrlKey || e.metaKey) && k.toLowerCase() === "a") { e.preventDefault(); S.selClips = new Set(S.project?.clips.map((c) => c.id)); emit("selection"); }
  else if (k === "s" || k === "S") splitAtPlayhead();
  else if (k === "l" || k === "L") toggleLoop();
  else if (k === "p" || k === "P") toggleFollow();
  else if ((k === "k" || k === "K") && !e.ctrlKey && !e.metaKey) toggleMetronome();
  else if ((k === "a" || k === "A") && S.selClips.size) alignSelectionToGrid();
  else if (k === "Home") seekTo(0);
  else if (k === "Escape") { stopAudition(); clearHighlight(); S.selClips.clear(); emit("selection"); }
  else if (k === "ArrowLeft" && S.selClips.size) { e.preventDefault(); nudgeSel(e.shiftKey ? -S.project.bpb : -1); }
  else if (k === "ArrowRight" && S.selClips.size) { e.preventDefault(); nudgeSel(e.shiftKey ? S.project.bpb : 1); }
  else if (k === "=" || k === "+") zoomAt(view.w / 2, 1.3);
  else if (k === "-") zoomAt(view.w / 2, 1 / 1.3);
  else if (k === "f" || k === "F") fit();
});

window.addEventListener("beforeunload", (e) => { if (S.dirty && S.project?.clips.length) { e.preventDefault(); e.returnValue = ""; } });

// ---------------------------------------------------------------- splitter between timeline and inspector
const insp = document.getElementById("inspector");
try { const h = +localStorage.getItem("inspH"); if (h > 80) insp.style.height = h + "px"; } catch {}
document.getElementById("splitter").addEventListener("mousedown", (e) => {
  const y0 = e.clientY, h0 = insp.getBoundingClientRect().height;
  const mv = (ev) => { insp.style.height = Math.max(90, Math.min(innerHeight - 220, h0 - (ev.clientY - y0))) + "px"; };
  window.addEventListener("mousemove", mv);
  window.addEventListener("mouseup", () => { window.removeEventListener("mousemove", mv); try { localStorage.setItem("inspH", parseInt(insp.style.height)); } catch {} }, { once: true });
});

// handy for debugging from the browser console
window.__ml = { S, edit, doUndo, doRedo, history, setProject, view, zoomAt };

// ---------------------------------------------------------------- boot
(async () => {
  await loadSongs();
  let last = null;
  try { last = localStorage.getItem("lastProject"); } catch {}
  if (last) {
    try { setProject(await api.get(`/api/projects/${last}`)); setTimeout(fit, 50); return; } catch {}
  }
  await newProject();
})();
