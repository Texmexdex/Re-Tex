// MIDI songs in the main window (mashlab/midi.py): the part chips in the song view, the instrument dialog,
// MIDI from a stem (MuScriptor), "play it on another song's timing", .mid export, a new empty MIDI song, and
// opening the pop-out piano roll (web/pianoroll.html). A MIDI song's parts are its stems everywhere else.
import { api } from "./api.js";
import { S, on, emit, toast, syncNow, invalidateSong, stemColor, fullSong, stemsOf, stemLabel } from "./state.js";
import { modal, showMenu } from "./ui.js";

const esc = (t) => String(t ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const NOTE = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
const noteName = (p) => `${NOTE[p % 12]}${Math.floor(p / 12) - 1}`;
const chan = typeof BroadcastChannel !== "undefined" ? new BroadcastChannel("retex-midi") : null;

// ---------------------------------------------------------------- pop-out piano roll
export function openPianoRoll(sid, pid) {
  const w = window.open(`/pianoroll?sid=${encodeURIComponent(sid)}&part=${encodeURIComponent(pid || "")}`, `retexRoll_${sid}`, "width=1400,height=840");
  if (!w) toast("The browser blocked the pop-out: allow pop-ups for Re-Tex", true);
  else w.focus();
}
// the piano roll asks the main window for things it shows here (the instrument dialog), and says when it saved
chan?.addEventListener("message", (e) => {
  const m = e.data || {};
  if (m.type === "instrument") instrumentDialog(m.sid, m.pid);
});

// a MIDI job finished: the part's renders changed - re-send the project so the player picks them up
on("event:job", (m) => {
  const j = m.job;
  if (!j || (j.kind !== "midi" && j.kind !== "vst3")) return;
  if (j.status === "done") {
    if (j.kind === "midi") {
      if (j.ref) invalidateSong(j.ref);
      syncNow();
      if (m.result?.sid) {
        toast("MIDI ready: it's in the library (MIDI tag) - drag its parts onto the timeline", false, 6000);
        S.selSong = m.result.sid; emit("songSelected", m.result.sid);
      }
    }
    chan?.postMessage({ type: "rendered", sid: j.ref });
    emit("midiJobDone", j);
  } else if (j.status === "error") toast(`${j.title}: ${j.error || "failed"}`, true, 8000);
});

// ---------------------------------------------------------------- the song view's MIDI bar
export async function midiSongBar(song) {
  let bar = document.getElementById("svMidi");
  if (!bar) {
    const meta = document.getElementById("svMeta");
    if (!meta) return;
    bar = document.createElement("div");
    bar.id = "svMidi";
    bar.style.cssText = "display:flex;flex-wrap:wrap;gap:6px;align-items:center;margin:4px 0 2px;font-size:12px";
    (meta.closest(".sv-title") || meta).insertAdjacentElement("afterend", bar);
  }
  if (!song || song.kind !== "midi") { bar.hidden = true; bar.innerHTML = ""; return; }
  bar.hidden = false;
  let d;
  try { d = await api.get(`/api/midi/${song.id}`); } catch { return; }
  bar.innerHTML = d.parts.map((p) => `<button class="small" data-pid="${esc(p.id)}" title="${esc(p.name)} - ${esc(p.instrument_label)} (${p.notes.length} notes). Click for its piano roll, instrument, name…">
      <span style="display:inline-block;width:8px;height:8px;border-radius:2px;margin-right:5px;background:${stemColor(p.id, song)}"></span>${esc(p.name)}
      <span style="color:var(--dim);margin-left:4px">${esc(p.instrument_label)}</span></button>`).join("")
    + `<button class="small" data-act="add" title="Add an empty part">+ Part</button>
       <button class="small" data-act="roll" title="Open the piano roll (pop-out)">Piano roll</button>
       <button class="small" data-act="timing" title="Play this MIDI song on another song's beat grid">Timing…</button>
       <button class="small" data-act="export" title="Download as a .mid file">.mid</button>
       <button class="small" data-act="remix" title="Use this MIDI on the project's lanes: glitch score, arrangement map, chord follow, groove">Use on lanes ▾</button>`
    + (d.part_errors ? `<span style="color:#e0a050" title="${esc(Object.entries(d.part_errors).map(([k, v]) => `${k}: ${v}`).join("\n"))}">some parts couldn't play - check their instruments</span>` : "");
  bar.querySelectorAll("button[data-pid]").forEach((b) => (b.onclick = (e) => partMenu(e, song, d.parts.find((p) => p.id === b.dataset.pid))));
  bar.querySelector('[data-act="add"]').onclick = async () => {
    try { await api.post(`/api/midi/${song.id}/parts`, { name: "New part" }); toast("Added a part"); } catch (e) { toast(e.message, true); }
  };
  bar.querySelector('[data-act="roll"]').onclick = () => openPianoRoll(song.id, null);
  bar.querySelector('[data-act="timing"]').onclick = () => timingDialog(song.id);
  bar.querySelector('[data-act="export"]').onclick = () => exportMid(song.id);
  bar.querySelector('[data-act="remix"]').onclick = (e) => import("./midiremix.js").then((m) => m.remixMenu(e, song.id));
}
on("midiJobDone", (j) => { if (S.selSong && j.ref === S.selSong) emit("songSelected", S.selSong); });

function partMenu(e, song, p) {
  showMenu(e, [
    [`${p.name} · ${p.instrument_label}`],
    ["Piano roll…", () => openPianoRoll(song.id, p.id), "pop-out"],
    ["Instrument…", () => instrumentDialog(song.id, p.id)],
    ["Rename…", async () => {
      const n = prompt("Part name", p.name);
      if (n) await api.put(`/api/midi/${song.id}/parts/${p.id}`, { name: n }).catch((er) => toast(er.message, true));
    }],
    ["Duplicate (to try another instrument)", () => api.post(`/api/midi/${song.id}/parts`, { copy_of: p.id }).catch((er) => toast(er.message, true))],
    [p.drum ? "Treat as a pitched part" : "Treat as drums (never transposed)", () => api.put(`/api/midi/${song.id}/parts/${p.id}`, { drum: !p.drum })],
    "-",
    ["Delete this part", async () => {
      const used = S.project?.clips.some((c) => c.song === song.id && c.stem === p.id);
      if (used && !confirm(`"${p.name}" has clips in this project - they'll go silent. Delete it?`)) return;
      await api.del(`/api/midi/${song.id}/parts/${p.id}`).catch((er) => toast(er.message, true));
    }],
  ]);
}

// ---------------------------------------------------------------- instruments
let sf2Cache = new Map();
async function presetsFor(path) {
  if (!path) return [];
  if (!sf2Cache.has(path)) sf2Cache.set(path, api.get(`/api/instruments/sf2?path=${encodeURIComponent(path)}`).then((r) => r.presets).catch(() => []));
  return sf2Cache.get(path);
}

export async function instrumentDialog(sid, pid) {
  let d;
  try { d = await api.get(`/api/midi/${sid}`); } catch (e) { toast(e.message, true); return; }
  let part = d.parts.find((p) => p.id === pid) || d.parts[0];
  const [status, fonts, vst] = await Promise.all([api.get("/api/instruments/status"), api.get("/api/instruments/soundfonts"), api.get("/api/instruments/vst3")]);
  let inst = JSON.parse(JSON.stringify(part.instrument || {}));
  const m = modal(`Instrument - ${d.title}`, `
    <div class="row-gap" style="flex-wrap:wrap;margin-bottom:8px">
      <label>Part <select id="inPart">${d.parts.map((p) => `<option value="${esc(p.id)}">${esc(p.name)}</option>`).join("")}</select></label>
      <label>Type <select id="inType">
        <option value="sf2">SoundFont (.sf2)</option><option value="vst3">VST3 instrument</option>
        <option value="sampler">Sample - one sound across the keys</option>
        <option value="chops">Chops - a stem cut up, a chop per note</option>
        <option value="basic">Built-in synth</option></select></label>
      <label>Level <input id="inGain" type="number" step="0.5" style="width:60px"> dB</label>
      <span style="flex:1"></span>
      <button id="inFolders" class="small" title="Where Re-Tex looks for SoundFonts and VST3 plugins">Folders…</button>
    </div>
    <div id="inBody"></div>
    <div class="row-gap" style="margin-top:12px">
      <button id="inTry">▶ Try it</button><button id="inStop">■</button>
      <span style="flex:1"></span>
      <button id="inApply" class="primary">Use it</button>
    </div>
    <p class="dim" style="margin-top:8px">The part is played again with this instrument (a few seconds; VST3 can take longer). Every clip of the part
      follows. To use two instruments for the same notes, duplicate the part first.</p>`, { wide: true });
  const B = m.body, $ = (s) => B.querySelector(s);
  $("#inPart").value = part.id;

  const body = async () => {
    const t = inst.type || "basic";
    $("#inType").value = t;
    $("#inGain").value = inst.gain_db || 0;
    const el = $("#inBody");
    if (t === "sf2") {
      const path = inst.path || fonts.default || "";
      el.innerHTML = `
        <div class="row-gap" style="flex-wrap:wrap">
          <label style="flex:1;min-width:380px">SoundFont <input id="sfPath" type="text" style="width:100%" value="${esc(path)}" spellcheck="false"></label>
          <button id="sfBrowse">Browse…</button>
        </div>
        ${fonts.soundfonts.length ? `<div class="row-gap" style="margin-top:6px;flex-wrap:wrap">
          <input id="sfFilter" type="search" placeholder="filter ${fonts.soundfonts.length} soundfonts…" style="width:200px">
          <select id="sfKnown" style="flex:1;min-width:320px"></select></div>` : ""}
        <div class="row-gap" style="margin-top:8px;flex-wrap:wrap">
          <label>Preset <select id="sfPreset" style="min-width:320px"><option>loading…</option></select></label>
          <label><input id="sfRev" type="checkbox" ${inst.reverb ? "checked" : ""}> its own reverb</label>
        </div>
        ${status.fluidsynth ? "" : `<p style="color:#e0a050">FluidSynth isn't installed, so SoundFonts can't play yet (install.bat adds it).</p>`}`;
      const fill = async () => {
        const ps = await presetsFor($("#sfPath").value.trim());
        const sel = $("#sfPreset");
        sel.innerHTML = ps.length ? ps.map((p) => `<option value="${p.bank}:${p.program}">${p.bank === 128 ? "Drums" : `Bank ${p.bank}`} · ${p.program} · ${esc(p.name)}</option>`).join("")
          : `<option value="">no presets found - check the file</option>`;
        const want = `${inst.bank ?? (part.drum ? 128 : 0)}:${inst.program ?? part.program ?? 0}`;
        if ([...sel.options].some((o) => o.value === want)) sel.value = want;
        sync();
      };
      const sync = () => {
        const [bk, pr] = ($("#sfPreset").value || "0:0").split(":").map(Number);
        Object.assign(inst, { type: "sf2", path: $("#sfPath").value.trim(), bank: bk, program: pr, reverb: $("#sfRev").checked,
          name: ($("#sfPreset").selectedOptions[0]?.textContent || "").split(" · ").pop() });
      };
      $("#sfPath").onchange = fill;
      $("#sfPreset").onchange = sync; $("#sfRev").onchange = sync;
      const fillKnown = () => {
        const sel = $("#sfKnown"); if (!sel) return;
        const q = ($("#sfFilter").value || "").toLowerCase();
        const hit = fonts.soundfonts.filter((f) => !q || (f.name + " " + f.folder).toLowerCase().includes(q));
        const groups = {};
        hit.forEach((f) => (groups[f.folder] ||= []).push(f));
        sel.innerHTML = `<option value="">${hit.length} soundfonts${q ? " match" : ""} - pick one…</option>` + Object.entries(groups).map(([g, fs]) =>
          `<optgroup label="${esc(g)}">${fs.map((f) => `<option value="${esc(f.path)}">${esc(f.name)}${f.mb ? ` · ${f.mb >= 100 ? Math.round(f.mb) : f.mb} MB` : ""}</option>`).join("")}</optgroup>`).join("");
        const cur = $("#sfPath").value.trim();
        if ([...sel.options].some((o) => o.value === cur)) sel.value = cur;
      };
      if ($("#sfKnown")) {
        fillKnown();
        $("#sfFilter").oninput = fillKnown;
        $("#sfKnown").onchange = (e) => { if (e.target.value) { $("#sfPath").value = e.target.value; fill(); } };
      }
      $("#sfBrowse").onclick = async () => {
        const r = await api.post("/api/pickfile", { title: "Choose a SoundFont", types: [["SoundFont", "*.sf2 *.sf3"]], start: $("#sfPath").value });
        if (r.path) { $("#sfPath").value = r.path; fill(); }
      };
      fill();
    } else if (t === "vst3") {
      const list = (vst.plugins || []).filter((p) => p.instrument);
      el.innerHTML = `
        <div class="row-gap" style="flex-wrap:wrap">
          <label>Plugin <select id="vsList" style="min-width:320px">${list.length ? list.map((p, i) => `<option value="${i}">${esc(p.name)}</option>`).join("")
            : `<option value="">${vst.plugins ? "no VST3 instruments found" : "not scanned yet"}</option>`}</select></label>
          <button id="vsScan" title="Finds the VST3 instruments installed in ${esc((vst.folders || [])[0] || "")} (each loads once, in its own process)">${vst.plugins ? "Scan again" : "Find my plugins"}</button>
          <button id="vsEdit" title="Opens the plugin's own window on this PC. Shape the sound, close the window: the sound is kept with this part.">Open plugin window…</button>
        </div>
        <div class="row-gap" style="margin-top:8px"><label style="flex:1">or a path <input id="vsPath" type="text" style="width:100%" spellcheck="false" value="${esc(inst.type === "vst3" ? inst.path || "" : "")}"></label>
          <button id="vsBrowse">Browse…</button></div>
        <p class="dim" id="vsState">${inst.type === "vst3" && inst.state ? "Its sound settings are saved with the part." : ""}</p>`;
      const pick = () => {
        const p = list[+$("#vsList").value];
        if (p && (inst.type !== "vst3" || inst.path !== p.path || inst.plugin_name !== p.plugin_name)) {
          inst = { type: "vst3", path: p.path, plugin_name: p.plugin_name || null, name: p.name, gain_db: inst.gain_db || 0 };
          $("#vsPath").value = p.path; $("#vsState").textContent = "";
        }
      };
      if (inst.type === "vst3") {
        const i = list.findIndex((p) => p.path === inst.path && (p.plugin_name || null) === (inst.plugin_name || null));
        if (i >= 0) $("#vsList").value = String(i);
      } else if (list.length) pick();
      $("#vsList").onchange = pick;
      $("#vsPath").onchange = () => { inst = { type: "vst3", path: $("#vsPath").value.trim(), name: $("#vsPath").value.split(/[\\/]/).pop().replace(/\.vst3$/i, ""), gain_db: inst.gain_db || 0 }; };
      $("#vsBrowse").onclick = async () => {
        const r = await api.post("/api/pickfile", { title: "Choose a VST3 instrument", types: [["VST3", "*.vst3"]], start: $("#vsPath").value });
        if (r.path) { $("#vsPath").value = r.path; $("#vsPath").onchange(); }
      };
      $("#vsScan").onclick = async () => {
        const r = await api.post("/api/instruments/vst3/scan");
        toast("Looking through your VST3 plugins (each loads once)…", false, 6000);
        const off = on("midiJobDone", async (j) => {
          if (j.id !== r.job) return;
          Object.assign(vst, await api.get("/api/instruments/vst3"));
          toast(`Found ${(vst.plugins || []).filter((p) => p.instrument).length} VST3 instruments`);
          body();
        });
        void off;
      };
      $("#vsEdit").onclick = async () => {
        if (!inst.path) { toast("Pick a plugin first", true); return; }
        $("#vsState").textContent = "The plugin's window is open on your desktop - close it when the sound is right.";
        try {
          const r = await api.post("/api/instruments/vst3/editor", inst);
          inst.state = r.state;
          $("#vsState").textContent = "Its sound settings are saved with the part (press Use it).";
        } catch (e) { $("#vsState").textContent = e.message; }
      };
    } else if (t === "sampler") {
      el.innerHTML = `
        <div class="row-gap" style="flex-wrap:wrap">
          <label style="flex:1;min-width:380px">Audio file <input id="smPath" type="text" style="width:100%" spellcheck="false" value="${esc(inst.type === "sampler" ? inst.path || "" : "")}"></label>
          <button id="smBrowse">Browse…</button>
          <button id="smHl" title="Shift+drag across a clip in the timeline first">Use the highlighted span</button>
        </div>
        <div class="row-gap" style="margin-top:8px;flex-wrap:wrap">
          <label>Plays as <select id="smMode"><option value="pitched">pitched across the keys</option><option value="oneshot">one-shot (same pitch, plays whole)</option></select></label>
          <label>Its own note <select id="smRoot">${Array.from({ length: 88 }, (_, i) => i + 21).map((p) => `<option value="${p}">${noteName(p)}</option>`).join("")}</select></label>
          <label>Attack <input id="smA" type="number" min="0" step="1" style="width:60px"> ms</label>
          <label>Release <input id="smR" type="number" min="0" step="10" style="width:60px"> ms</label>
          <label><input id="smLoop" type="checkbox"> loop while held</label>
        </div>
        <p class="dim" id="smSpan">${inst.span ? `From ${esc(S.songs.get(inst.span.song)?.title || inst.span.song)} (${inst.span.stem}), ${(+inst.span.t0).toFixed(2)}-${(+inst.span.t1).toFixed(2)} s` : ""}</p>`;
      if (inst.type !== "sampler" || inst.root === undefined) inst = { type: "sampler", root: 60, mode: "pitched", attack: 0.003, release: 0.12, gain_db: inst.gain_db || 0 };
      $("#smMode").value = inst.mode || "pitched"; $("#smRoot").value = String(inst.root || 60);
      $("#smA").value = Math.round((inst.attack ?? 0.003) * 1000); $("#smR").value = Math.round((inst.release ?? 0.12) * 1000);
      $("#smLoop").checked = !!inst.loop;
      const sync = () => Object.assign(inst, { mode: $("#smMode").value, root: +$("#smRoot").value, attack: +$("#smA").value / 1000,
        release: +$("#smR").value / 1000, loop: $("#smLoop").checked });
      ["#smMode", "#smRoot", "#smA", "#smR", "#smLoop"].forEach((s) => ($(s).onchange = sync));
      $("#smPath").onchange = () => { inst.path = $("#smPath").value.trim(); inst.span = null; inst.name = "sample: " + inst.path.split(/[\\/]/).pop(); $("#smSpan").textContent = ""; };
      $("#smBrowse").onclick = async () => {
        const r = await api.post("/api/pickfile", { title: "Choose a sample", types: [["Audio", "*.wav *.flac *.mp3 *.aif *.aiff *.ogg"]], start: $("#smPath").value });
        if (r.path) { $("#smPath").value = r.path; $("#smPath").onchange(); }
      };
      $("#smHl").onclick = async () => {
        const { currentHighlight } = await import("./timeline.js");
        const h = currentHighlight();
        if (!h || !h.clips.length || h.t1 - h.t0 < 1e-3) { toast("Shift+drag across a clip in the timeline first", true); return; }
        $("#smSpan").textContent = "Taking that bit as it sounds…";
        try {
          const r = await api.post("/api/midi/sample_from_highlight", { project: S.project, clip_id: h.clips[0], t0: h.t0, t1: h.t1 });
          const t = S.songs.get(r.song)?.title || r.song;
          Object.assign(inst, { path: r.path, span: null, name: `sample: ${t} (${r.stem})` });
          $("#smPath").value = r.path;
          $("#smSpan").textContent = `${r.seconds.toFixed(2)} s of ${t} (${r.stem}), as it plays in the timeline`;
        } catch (e) { $("#smSpan").textContent = e.message; }
      };
    } else if (t === "chops") {
      await chopsBody(el, $);
    } else {
      if (inst.type !== "basic") inst = { type: "basic", wave: part.drum ? "drums" : "saw", attack: 0.005, decay: 0.25, sustain: 0.6, release: 0.15, cutoff: 5000, gain_db: inst.gain_db || 0 };
      el.innerHTML = `
        <div class="row-gap" style="flex-wrap:wrap">
          <label>Wave <select id="bsWave"><option value="saw">saw</option><option value="square">square</option><option value="triangle">triangle</option><option value="sine">sine</option><option value="drums">drum kit</option></select></label>
          <label>Attack <input id="bsA" type="number" min="0" step="1" style="width:60px"> ms</label>
          <label>Decay <input id="bsD" type="number" min="0" step="10" style="width:60px"> ms</label>
          <label>Sustain <input id="bsS" type="number" min="0" max="100" step="5" style="width:60px"> %</label>
          <label>Release <input id="bsR" type="number" min="0" step="10" style="width:60px"> ms</label>
          <label>Brightness <input id="bsC" type="number" min="200" max="18000" step="100" style="width:80px"> Hz</label>
        </div>
        <p class="dim">A simple synth that's always there - handy for sketching before you pick a real sound.</p>`;
      $("#bsWave").value = inst.wave || "saw";
      $("#bsA").value = Math.round((inst.attack ?? 0.005) * 1000); $("#bsD").value = Math.round((inst.decay ?? 0.25) * 1000);
      $("#bsS").value = Math.round((inst.sustain ?? 0.6) * 100); $("#bsR").value = Math.round((inst.release ?? 0.15) * 1000); $("#bsC").value = inst.cutoff || 5000;
      const sync = () => Object.assign(inst, { wave: $("#bsWave").value, attack: +$("#bsA").value / 1000, decay: +$("#bsD").value / 1000,
        sustain: +$("#bsS").value / 100, release: +$("#bsR").value / 1000, cutoff: +$("#bsC").value, name: `Built-in ${$("#bsWave").value}` });
      ["#bsWave", "#bsA", "#bsD", "#bsS", "#bsR", "#bsC"].forEach((s) => ($(s).onchange = sync));
      sync();
    }
  };
  // ---- chop kits (mashlab/chops.py)
  const songs = [...S.songs.values()].filter((s) => s.status === "ready" && s.kind !== "sample").sort((a, b) => a.title.localeCompare(b.title));
  const PLAY_HELP = {
    keys: "Each key plays its own chop, counting up from the 'from key' note - play the chops like pads.",
    order: "Every note plays the next chop, whatever its pitch: the part's rhythm plays through the stem in order (a vocal follows the melody's rhythm).",
    shuffle: "Every note plays the next chop of a shuffled order - a mixed-up chopped version. ↻ gives another order.",
    drums: "The hits sorted into kick / snare / hat / open hat / crash / tom and played by the General MIDI drum notes, so a drum part plays these drum sounds. Change a chop's label if it's sorted wrong.",
    cue: "Like cue points: every note this part uses gets its own chop. Hitting that note plays the stem from that chop's start for as long as the note lasts (on past the chop's end), from the same start every time; two different notes at once play their two chops. Change which chop each note gets below.",
    flow: "Chops made to measure for this part: the chops are laid end to end (the silences between them left out) and every note plays the next stretch, cut to exactly its length - no stretching, no repeats. Rests don't use any up.",
  };
  // lyrics found from here (the song's Lyrics job): the dialog re-cuts when it's done
  if (!window.__retexLyricsHook) {
    window.__retexLyricsHook = true;
    on("event:job", (m) => {
      const j = m.job;
      if (j?.kind !== "lyrics" || (j.status !== "done" && j.status !== "error")) return;
      (window.__retexLyricsWait?.[j.ref] || (() => {}))(j);
    });
  }
  async function chopsBody(el, $) {
    if (inst.type !== "chops" || inst.chops === undefined) {      // new (the type menu sets only the type)
      const drum = !!part.drum;
      inst = { type: "chops", song: songs[0]?.id || "", stem: drum ? "drums" : "vocals", slice: drum ? "hits" : "words",
        play: drum ? "drums" : "order", base: drum ? 36 : 48, seed: 1, mono: !drum, fit: "whole", pitch: false, release: 0.03,
        from: null, to: null, chops: [], gain_db: inst.gain_db || 0 };
    }
    el.innerHTML = `
      <div class="row-gap" style="flex-wrap:wrap">
        <label>Song <select id="chSong" style="max-width:320px">${songs.map((s) => `<option value="${esc(s.id)}">${esc(s.title)}${s.kind === "midi" ? " (MIDI)" : ""}</option>`).join("")}</select></label>
        <label>Stem <select id="chStem"></select></label>
        <label>Cut into <select id="chSlice">
          <option value="hits">hits (drum hits, notes, syllables)</option><option value="words">words (from its lyrics)</option>
          <option value="beats">beats</option><option value="half">half beats</option><option value="bars">bars</option>
          <option value="phrases">phrases</option></select></label>
      </div>
      <div class="row-gap" style="margin-top:6px;flex-wrap:wrap">
        <label>From <input id="chFrom" type="number" step="0.1" min="0" style="width:72px"> s</label>
        <label>to <input id="chTo" type="number" step="0.1" min="0" style="width:72px"> s</label>
        <button id="chHl" class="small" title="Shift+drag across a clip in the timeline first: its song, stem and that stretch">Use the highlighted span</button>
        <button id="chCut" class="primary small">Cut it</button>
        <button id="chLyrics" class="small" hidden title="Runs Re-Tex's lyrics finder on this song (Whisper on the GPU, a minute or two): the words and when each is sung">Find its words now</button>
        <span id="chInfo" class="dim"></span>
      </div>
      <div class="row-gap" style="margin-top:8px;flex-wrap:wrap">
        <label>Each note plays <select id="chPlay">
          <option value="keys">its own chop (a chop per key)</option><option value="order">the next chop, in order</option>
          <option value="shuffle">the next chop, shuffled</option><option value="drums">a drum kit (kick, snare, hats… on the drum notes)</option>
          <option value="cue">its own chop's start, for as long as the note lasts (cue points)</option>
          <option value="flow">the next stretch, cut to the note's length</option></select></label>
        <label id="chBaseL">from key <select id="chBase">${Array.from({ length: 128 }, (_, i) => `<option value="${i}">${noteName(i)}</option>`).join("")}</select></label>
        <label id="chSeedL">order <input id="chSeed" type="number" min="1" style="width:56px"> <button id="chReseed" class="small" title="Another shuffle">↻</button></label>
      </div>
      <div class="row-gap" style="margin-top:6px;flex-wrap:wrap">
        <label title="A new chop cuts off the one still playing"><input id="chMono" type="checkbox"> one at a time</label>
        <label title="Off: every chop at full level, whatever the note's velocity"><input id="chVel" type="checkbox"> velocity sets the level</label>
        <label id="chSkipL" title="A held note plays on through the stem from its cue point - with this on, the stem's pauses are left out so it keeps sounding"><input id="chSkip" type="checkbox"> skip the stem's pauses</label>
        <label id="chFitL" title="How long each chop lasts against its note">Length <select id="chFit">
          <option value="whole">the whole chop</option><option value="cut">cut at the note's end</option>
          <option value="repeat">repeat until the note ends</option><option value="stretch">stretched to the note (pitch kept)</option></select></label>
        <label id="chPitchL" title="Tuned: each chop's own pitch is measured and moved to exactly the note's pitch, keeping its length and the voice's character. Follows the notes: higher notes play it faster and higher, like a record">Pitch <select id="chPitch">
          <option value="off">as sung</option><option value="follow">follows the notes (faster / slower)</option><option value="tune">tuned to each note</option></select></label>
        <label id="chOctL" title="Moves each chop no more than 6 semitones, to the nearest copy of the note - so a voice doesn't turn chipmunk or growl"><input id="chOct" type="checkbox"> keep the singer's octave</label>
        <span id="chPitchInfo" class="dim" style="font-size:12px"></span>
      </div>
      <p id="chHelp" class="dim" style="margin:6px 0 0"></p>
      <div id="chSayRow" style="margin-top:8px">
        <div class="row-gap" style="flex-wrap:wrap">
          <label style="flex:1;min-width:300px">Make it say <input id="chSay" type="text" style="width:100%" placeholder="type new lyrics - the singer's own words, in your order"></label>
          <button id="chSayGo" class="small primary">Find the words</button>
          <button id="chSayClear" class="small" title="Back to every word in the song's order">Clear</button>
          <label title="Stop after the last word (otherwise it starts the sentence again)"><input id="chOnce" type="checkbox"> once</label>
        </div>
        <div id="chSayInfo" class="dim" style="font-size:12px;margin-top:4px"></div>
      </div>
      <div id="chAssign"></div>
      <div id="chGrid" style="margin-top:8px;display:flex;flex-wrap:wrap;gap:3px;max-height:230px;overflow:auto"></div>`;
    const stemsFor = async () => {
      const info = await fullSong(inst.song).catch(() => S.songs.get(inst.song));
      const list = stemsOf(info);
      $("#chStem").innerHTML = list.map((st) => `<option value="${esc(st)}">${esc(stemLabel(info, st))}</option>`).join("");
      if (!list.includes(inst.stem)) inst.stem = list.includes("vocals") && !part.drum ? "vocals" : list.includes("drums") && part.drum ? "drums" : list[0];
      $("#chStem").value = inst.stem;
      return info;
    };
    const show = () => {
      const mode = inst.play;
      $("#chBaseL").hidden = mode === "drums";
      $("#chSeedL").hidden = mode !== "shuffle";
      $("#chPitchL").hidden = mode === "drums" || mode === "flow";
      $("#chOctL").hidden = $("#chPitchL").hidden || inst.pitch !== "tune";
      [...$("#chPitch").options].forEach((o) => (o.hidden = o.value === "follow" && !(mode === "order" || mode === "shuffle")));
      $("#chFitL").hidden = mode === "flow" || mode === "cue";
      $("#chSkipL").hidden = mode !== "cue";
      $("#chSayRow").hidden = !(inst.slice === "words" && mode === "order");
      sayInfo();
      $("#chBaseL").hidden = mode === "drums" || mode === "cue";
      assignPanel();
      $("#chHelp").textContent = PLAY_HELP[mode];
      const ch = inst.chops || [];
      const grid = $("#chGrid");
      if (!ch.length) { grid.innerHTML = `<span class="dim">No chops yet - press Cut it.</span>`; return; }
      const lo = Math.min(...ch.map((c) => c[3])), hi = Math.max(...ch.map((c) => c[3]));
      const usedBy = {};
      if (mode === "cue") Object.entries(inst.assign || {}).forEach(([p, i]) => (usedBy[i] ||= []).push(+p));
      const shown = ch.slice(0, 400);
      grid.innerHTML = shown.map((c, i) => {
        const lvl = hi > lo ? (c[3] - lo) / (hi - lo) : 1;
        const tag = mode === "keys" ? (inst.base + i <= 127 ? noteName(inst.base + i) : "-") : mode === "drums" || mode === "flow" ? ""
          : mode === "cue" ? (usedBy[i] ? usedBy[i].map(noteName).join(" ") + " ·" : "") : `${i + 1}`;
        const word = typeof c[4] === "string" ? ` ${esc(c[4].slice(0, 14))}` : "";
        const cls = mode === "drums" ? `<select data-cls="${i}" style="font-size:10px;padding:0 2px">${["kick", "snare", "hat", "open", "crash", "tom"].map((k) => `<option${k === c[2] ? " selected" : ""}>${k}</option>`).join("")}</select>` : "";
        return `<span style="display:inline-flex;align-items:center;gap:2px;border-radius:5px;padding:2px 4px;background:rgba(30,215,96,${(0.08 + 0.35 * lvl).toFixed(2)})">
          <button class="small" data-ch="${i}" title="${c[0].toFixed(2)}-${c[1].toFixed(2)} s · click to hear it" style="padding:1px 5px">${tag}${word || (mode === "drums" || mode === "flow" ? "▶" : "")}</button>${cls}</span>`;
      }).join("") + (ch.length > shown.length ? `<span class="dim">+${ch.length - shown.length} more</span>` : "");
      grid.querySelectorAll("button[data-ch]").forEach((b) => (b.onclick = () => {
        const c = ch[+b.dataset.ch];
        api.post("/api/midi/chop_audition", { song: inst.song, stem: inst.stem, a: c[0], b: c[1] }).catch((e) => toast(e.message, true));
      }));
      grid.querySelectorAll("select[data-cls]").forEach((sel) => (sel.onchange = () => { ch[+sel.dataset.cls][2] = sel.value; }));
    };
    // ---- cue points: which chop each of the part's notes plays
    const partPitches = (how) => {
      const ns = [...(part.notes || [])].sort((a, b) => a[0] - b[0]);
      const ps = [...new Set(ns.map((n) => n[2]))];
      return how === "low" ? ps.sort((a, b) => a - b) : ps;
    };
    const assignBy = (how) => {
      const ch = inst.chops || [], N = Math.max(1, ch.length), ps = partPitches(how);
      let order = ps.map((_, k) => k % N);
      if (how === "shuffle") {                       // a seeded shuffle, so ↻ gives a new one
        let x = (inst.seed || 1) * 9301 + 49297;
        const idx = Array.from({ length: N }, (_, k) => k);
        for (let k = N - 1; k > 0; k--) { x = (x * 9301 + 49297) % 233280; const r = Math.floor((x / 233280) * (k + 1)); [idx[k], idx[r]] = [idx[r], idx[k]]; }
        order = ps.map((_, k) => idx[k % N]);
      }
      inst.assign = Object.fromEntries(ps.map((p, k) => [String(p), order[k]]));
    };
    const chopLabel = (c, i) => `${i + 1} · ${typeof c[4] === "string" ? c[4] : c[2]} · ${c[0].toFixed(2)} s`;
    function assignPanel() {
      const box = $("#chAssign");
      if (inst.play !== "cue") { box.innerHTML = ""; return; }
      const ch = inst.chops || [];
      if (!ch.length) { box.innerHTML = ""; return; }
      const ps = partPitches("low");
      if (!ps.length) { box.innerHTML = `<p class="dim">This part has no notes yet - draw some, then the notes it uses show here.</p>`; return; }
      if (!inst.assign || ps.some((p) => inst.assign[String(p)] === undefined || inst.assign[String(p)] >= ch.length)) {
        const keep = inst.assign || {};
        assignBy("low");
        Object.entries(keep).forEach(([p, i]) => { if (i < ch.length) inst.assign[p] = i; });
      }
      const count = {};
      (part.notes || []).forEach((n) => (count[n[2]] = (count[n[2]] || 0) + 1));
      const opts = ch.map((c, i) => `<option value="${i}">${esc(chopLabel(c, i))}</option>`).join("");
      box.innerHTML = `
        <div class="row-gap" style="flex-wrap:wrap;margin-top:8px"><span class="dim">Give the notes chops:</span>
          <button class="small" data-how="low">lowest note → first chop</button>
          <button class="small" data-how="first">first played → first chop</button>
          <button class="small" data-how="shuffle" title="Random chops (↻ next to 'order' above changes the shuffle)">shuffled</button></div>
        <div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(270px,1fr));gap:4px 10px;margin-top:6px;max-height:210px;overflow:auto">
          ${ps.slice().reverse().map((p) => `<div class="row-gap" style="gap:4px">
            <b style="min-width:38px">${noteName(p)}</b><span class="dim" style="min-width:34px">${count[p]}×</span>
            <select data-p="${p}" style="flex:1;min-width:0">${opts}</select>
            <button class="small" data-hear="${p}" title="Hear it from its start">▶</button></div>`).join("")}
        </div>`;
      box.querySelectorAll("select[data-p]").forEach((sel) => {
        sel.value = String(inst.assign[sel.dataset.p]);
        sel.onchange = () => { inst.assign[sel.dataset.p] = +sel.value; show(); };
      });
      box.querySelectorAll("button[data-how]").forEach((b) => (b.onclick = () => { assignBy(b.dataset.how); show(); }));
      box.querySelectorAll("button[data-hear]").forEach((b) => (b.onclick = () => {
        const c = ch[inst.assign[b.dataset.hear]];
        const lens = (part.notes || []).filter((n) => n[2] === +b.dataset.hear).map((n) => n[1]).sort((x, y) => x - y);
        const sec = Math.min(4, Math.max(0.25, (lens[lens.length >> 1] || 1) * 60 / (d.bpm || 120)));    // a typical note of it
        api.post("/api/midi/chop_audition", { song: inst.song, stem: inst.stem, a: c[0], b: c[0] + sec }).catch((e) => toast(e.message, true));
      }));
    }
    // ---- "make it say": new lyrics from the song's own sung words
    function sayInfo() {
      const box = $("#chSayInfo");
      if (!inst.sequence?.length) { box.textContent = "Each note plays the next word of the song, in the song's order - or type new lyrics above."; return; }
      const ch = inst.chops || [];
      box.innerHTML = `Saying: <b>${inst.sequence.map((i) => esc(ch[i]?.[4] ?? "?")).join(" ")}</b>` +
        (inst.sayMissing?.length ? ` <span style="color:#e0a050">· not sung in this song: ${inst.sayMissing.map(esc).join(", ")}</span>` : "") +
        ` · ${inst.sequence.length} notes per pass`;
    }
    $("#chSay").value = inst.sayText || "";
    $("#chOnce").checked = !!inst.seq_once;
    $("#chOnce").onchange = (e) => (inst.seq_once = e.target.checked);
    $("#chSayClear").onclick = () => { delete inst.sequence; delete inst.sayText; delete inst.sayMissing; $("#chSay").value = ""; show(); };
    const doSay = async () => {
      const text = $("#chSay").value.trim();
      if (!text) return;
      try {
        const r = await api.post("/api/midi/say", { chops: inst.chops || [], text });
        if (!r.sequence.length) { $("#chSayInfo").textContent = `None of those words are sung in this song (${r.words} different words to pick from).`; return; }
        Object.assign(inst, { sequence: r.sequence, sayText: text, sayMissing: r.missing });
        show();
      } catch (e) { toast(e.message, true); }
    };
    $("#chSayGo").onclick = doSay;
    $("#chSay").onkeydown = (e) => { if (e.key === "Enter") { e.preventDefault(); doSay(); } };
    const lyricsButton = async () => {
      const b = $("#chLyrics");
      b.hidden = true;
      if (inst.slice !== "words" || !inst.song) return;
      const r = await api.get(`/api/midi/words_status?song=${encodeURIComponent(inst.song)}`).catch(() => ({ words: 0 }));
      b.hidden = r.words > 0;
    };
    $("#chLyrics").onclick = async () => {
      const song = inst.song;
      try {
        await api.post(`/api/songs/${encodeURIComponent(song)}/lyrics`);
      } catch (e) { toast(e.message, true); return; }
      $("#chLyrics").disabled = true;
      $("#chInfo").textContent = "finding its words (Whisper on the GPU)… the chops appear when it's done";
      (window.__retexLyricsWait ||= {})[song] = (j) => {
        delete window.__retexLyricsWait[song];
        if (!document.body.contains($("#chLyrics"))) return;       // the dialog was closed meanwhile
        $("#chLyrics").disabled = false;
        if (j.status === "error") { $("#chInfo").textContent = `couldn't find its words: ${j.error || "?"}`; return; }
        if (inst.song === song) cut();
      };
    };
    const cut = async () => {
      if (!inst.song) { toast("Pick a song first", true); return; }
      $("#chInfo").textContent = "cutting…";
      lyricsButton();
      try {
        const r = await api.post("/api/midi/chops", { song: inst.song, stem: inst.stem, slice: inst.slice, from: inst.from, to: inst.to });
        inst.chops = r.chops;
        if (inst.sayText && inst.slice === "words") {      // new chops: find the typed words again
          const sr = await api.post("/api/midi/say", { chops: r.chops, text: inst.sayText }).catch(() => null);
          if (sr) Object.assign(inst, { sequence: sr.sequence, sayMissing: sr.missing });
        } else delete inst.sequence;
        delete inst.pitches;
        if (inst.pitch === "tune") setTimeout(measure, 0);
        const info = S.songs.get(inst.song);
        inst.name = `chops: ${info?.title || inst.song} · ${stemLabel(await fullSong(inst.song).catch(() => info), inst.stem)}`;
        const counts = {};
        r.chops.forEach((c) => (counts[c[2]] = (counts[c[2]] || 0) + 1));
        $("#chInfo").textContent = `${r.chops.length} chops` + (inst.play === "drums" ? ` (${Object.entries(counts).map(([k, v]) => `${v} ${k}`).join(", ")})` : "")
          + (inst.play === "keys" && r.chops.length > 128 - inst.base ? ` - keys reach the first ${128 - inst.base}` : "");
      } catch (e) { inst.chops = []; $("#chInfo").textContent = e.message; }
      show();
    };
    $("#chSong").value = inst.song;
    await stemsFor();
    $("#chSlice").value = inst.slice; $("#chPlay").value = inst.play; $("#chBase").value = String(inst.base);
    $("#chSeed").value = inst.seed || 1;
    $("#chFrom").value = inst.from ?? ""; $("#chTo").value = inst.to ?? "";
    $("#chVel").checked = inst.velocity ?? inst.play === "drums";
    $("#chMono").checked = !!inst.mono; $("#chFit").value = inst.fit || (inst.gate ? "cut" : "whole"); $("#chPitch").value = inst.pitch === "tune" ? "tune" : inst.pitch === true ? "follow" : "off";
    $("#chOct").checked = inst.tune_octave !== false;
    $("#chSong").onchange = async (e) => { inst.song = e.target.value; inst.from = inst.to = null; $("#chFrom").value = $("#chTo").value = ""; await stemsFor(); cut(); };
    $("#chStem").onchange = (e) => {
      inst.stem = e.target.value;
      if (/drum/i.test(inst.stem) && inst.slice !== "hits") { inst.slice = "hits"; $("#chSlice").value = "hits"; }
      cut();
    };
    $("#chSlice").onchange = (e) => { inst.slice = e.target.value; cut(); };
    $("#chFrom").onchange = (e) => { inst.from = e.target.value === "" ? null : +e.target.value; cut(); };
    $("#chTo").onchange = (e) => { inst.to = e.target.value === "" ? null : +e.target.value; cut(); };
    $("#chCut").onclick = cut;
    $("#chPlay").onchange = (e) => {
      inst.play = e.target.value;
      inst.mono = inst.play !== "drums" && inst.play !== "cue"; $("#chMono").checked = inst.mono;
      if (inst.velocity === undefined) $("#chVel").checked = inst.play === "drums";
      if (inst.play === "drums" && inst.slice !== "hits") { inst.slice = "hits"; $("#chSlice").value = "hits"; cut(); } else show();
    };
    $("#chBase").onchange = (e) => { inst.base = +e.target.value; show(); };
    $("#chSeed").onchange = (e) => { inst.seed = Math.max(1, +e.target.value || 1); };
    $("#chReseed").onclick = () => { inst.seed = (inst.seed || 1) + 1; $("#chSeed").value = inst.seed; };
    $("#chMono").onchange = (e) => (inst.mono = e.target.checked);
    $("#chVel").onchange = (e) => (inst.velocity = e.target.checked);
    $("#chSkip").checked = inst.skip_gaps !== false;
    $("#chSkip").onchange = (e) => (inst.skip_gaps = e.target.checked);
    $("#chFit").onchange = (e) => { inst.fit = e.target.value; delete inst.gate; };
    const measure = async () => {                    // each chop's own pitch, for tuning
      if (inst.pitch !== "tune" || !(inst.chops || []).length) return;
      if ((inst.pitches || []).length === inst.chops.length) return;
      $("#chPitchInfo").textContent = "measuring each chop's pitch…";
      try {
        const r = await api.post("/api/midi/chop_pitches", { song: inst.song, stem: inst.stem, chops: inst.chops });
        inst.pitches = r.pitches;
        const n = r.pitches.filter((x) => x != null).length;
        $("#chPitchInfo").textContent = `${n} of ${r.pitches.length} chops have a clear pitch (the rest play as they are)`;
      } catch (e) { $("#chPitchInfo").textContent = e.message; }
    };
    $("#chPitch").onchange = (e) => {
      const v = e.target.value;
      inst.pitch = v === "tune" ? "tune" : v === "follow";
      show(); measure();
    };
    $("#chOct").onchange = (e) => (inst.tune_octave = e.target.checked);
    $("#chHl").onclick = async () => {
      const { currentHighlight } = await import("./timeline.js");
      const h = currentHighlight();
      if (!h || !h.clips.length || h.t1 - h.t0 < 1e-3) { toast("Shift+drag across a clip in the timeline first", true); return; }
      try {
        const r = await api.post("/api/midi/clip_span", { project: S.project, clip_id: h.clips[0], t0: h.t0, t1: h.t1 });
        Object.assign(inst, { song: r.song, stem: r.stem, from: r.from, to: r.to });
        $("#chSong").value = r.song; await stemsFor();
        $("#chFrom").value = r.from; $("#chTo").value = r.to;
        cut();
      } catch (e) { toast(e.message, true); }
    };
    if (!(inst.chops || []).length && inst.song) cut(); else show();
  }

  $("#inFolders").onclick = () => foldersDialog(() => { m.close(); instrumentDialog(sid, part.id); });
  $("#inType").onchange = (e) => { inst = { type: e.target.value, gain_db: inst.gain_db || 0 }; body(); };
  $("#inGain").onchange = (e) => { inst.gain_db = +e.target.value || 0; };
  $("#inPart").onchange = (e) => { part = d.parts.find((p) => p.id === e.target.value); inst = JSON.parse(JSON.stringify(part.instrument || {})); body(); };
  $("#inTry").onclick = async () => {
    const first = part.notes.length ? Math.floor(part.notes[0][0] / d.bpb) * d.bpb : 0;
    try {
      await api.post("/api/midi/audition", { sid, pid: part.id, instrument: inst, from: first, to: first + 4 * d.bpb, bpm: S.project?.bpm || d.bpm });
    } catch (e) { toast(e.message, true); }
  };
  $("#inStop").onclick = () => api.post("/api/midi/audition_stop");
  $("#inApply").onclick = async () => {
    if (inst.type === "chops" && !(inst.chops || []).length) { toast("The chop kit has no chops yet - press Cut it", true); return; }
    try {
      await api.put(`/api/midi/${sid}/parts/${part.id}`, { instrument: inst });
      toast(`${part.name}: playing it with ${inst.name || inst.type}…`);
      m.close();
    } catch (e) { toast(e.message, true); }
  };
  body();
}

export async function foldersDialog(after) {
  const [f, fonts, vst] = await Promise.all([api.get("/api/instruments/folders"), api.get("/api/instruments/soundfonts"), api.get("/api/instruments/vst3")]);
  const builtinSf = fonts.folders.filter((x) => !f.sf2.includes(x));
  const builtinVst = vst.folders.filter((x) => !f.vst3.includes(x));
  const m = modal("Instrument folders", `
    <p class="lead">Re-Tex looks in these folders and every folder inside them.</p>
    <h4 style="margin:10px 0 4px">SoundFonts (.sf2 / .sf3)</h4>
    <div class="dim" style="font-size:12px">${builtinSf.map(esc).join("<br>")}</div>
    <textarea id="fdSf" rows="3" style="width:100%;margin-top:4px" spellcheck="false" placeholder="one folder per line">${esc(f.sf2.join("\n"))}</textarea>
    <button id="fdSfAdd" class="small">Add a folder…</button> <span class="dim">${fonts.soundfonts.length} soundfonts found now</span>
    <h4 style="margin:14px 0 4px">VST3 plugins</h4>
    <div class="dim" style="font-size:12px">${builtinVst.map(esc).join("<br>")}</div>
    <textarea id="fdVst" rows="3" style="width:100%;margin-top:4px" spellcheck="false" placeholder="one folder per line">${esc(f.vst3.join("\n"))}</textarea>
    <button id="fdVstAdd" class="small">Add a folder…</button>
    <div class="row-gap" style="margin-top:12px"><span style="flex:1"></span><button id="fdSave" class="primary">Save</button></div>
    <p class="dim">After adding VST3 folders, press "Scan again" on the VST3 page so the new plugins are found.</p>`, { wide: true });
  const $ = (s) => m.body.querySelector(s);
  const add = (ta) => async () => {
    const r = await api.post("/api/settings/browse", { title: "Choose a folder", path: "" }).catch((e) => ({ error: e.message }));
    if (r.path) ta.value = (ta.value.trim() ? ta.value.trim() + "\n" : "") + r.path;
    else if (r.error) toast(r.error, true);
  };
  $("#fdSfAdd").onclick = add($("#fdSf"));
  $("#fdVstAdd").onclick = add($("#fdVst"));
  $("#fdSave").onclick = async () => {
    const lines = (ta) => ta.value.split(/\r?\n/).map((x) => x.trim()).filter(Boolean);
    try {
      await api.put("/api/instruments/folders", { sf2: lines($("#fdSf")), vst3: lines($("#fdVst")) });
      sf2Cache = new Map();
      toast("Folders saved");
      m.close();
      after?.();
    } catch (e) { toast(e.message, true); }
  };
}

// ---------------------------------------------------------------- MIDI from a stem, timing, export, new
export async function transcribeStem(sid, stem) {
  try {
    await api.post("/api/midi/transcribe", { song: sid, stem });
    toast(`MuScriptor is turning the ${stem} into MIDI (about a minute or two on the GPU) - it lands in the library`, false, 7000);
  } catch (e) { toast(e.message, true); }
}

export function transcribeMenu(sid) {
  const m = modal("Turn a stem into MIDI", `
    <p class="lead">MuScriptor listens to the stem and writes down its notes. The MIDI song uses this song's own beat grid, so its clips
      line up with the stem's. Then give it any instrument.</p>
    <div class="row-gap">${["bass", "other", "vocals", "drums"].map((s) => `<button data-st="${s}">${s}</button>`).join("")}</div>`);
  m.body.querySelectorAll("button[data-st]").forEach((b) => (b.onclick = () => { m.close(); transcribeStem(sid, b.dataset.st); }));
}

export function timingDialog(sid) {
  const songs = [...S.songs.values()].filter((s) => s.status === "ready" && s.kind !== "sample" && s.id !== sid).sort((a, b) => a.title.localeCompare(b.title));
  const m = modal("Play it on another song's timing", `
    <p class="lead">Its notes then sound on that song's beats - its drift, tempo changes and bar lines - so it sits with that song as a
      recording would. Beat 1 of the MIDI goes on that song's first beat.</p>
    <div class="row-gap"><select id="tmSong" style="min-width:420px">${songs.map((s) => `<option value="${esc(s.id)}">${esc(s.title)}${s.kind === "midi" ? " (MIDI)" : ""}</option>`).join("")}</select>
      <button id="tmGo" class="primary">Use its timing</button></div>`);
  m.body.querySelector("#tmGo").onclick = async () => {
    try { await api.post(`/api/midi/${sid}/timing`, { song: m.body.querySelector("#tmSong").value }); toast("Moving the notes onto its timing…"); m.close(); }
    catch (e) { toast(e.message, true); }
  };
}

export function exportMid(sid) {
  const a = document.createElement("a");
  a.href = `/api/midi/${encodeURIComponent(sid)}/export.mid`;
  a.download = "";
  document.body.appendChild(a); a.click(); a.remove();
}

export function newMidiDialog() {
  const bpm = S.project?.bpm || 120, bpb = S.project?.bpb || 4;
  const m = modal("New MIDI song", `
    <div class="row-gap" style="flex-wrap:wrap">
      <label>Name <input id="nmTitle" type="text" value="New MIDI" style="width:220px"></label>
      <label>Tempo <input id="nmBpm" type="number" step="0.5" value="${bpm}" style="width:70px"> BPM</label>
      <label>Bars <input id="nmBars" type="number" min="1" value="8" style="width:60px"></label>
      <label><input id="nmDrum" type="checkbox"> drums</label>
      <button id="nmGo" class="primary">Make it</button>
    </div>
    <p class="dim">An empty part to draw in: it opens in the piano roll. Add it to the timeline from the song view like any song.</p>`);
  m.body.querySelector("#nmGo").onclick = async () => {
    try {
      const r = await api.post("/api/midi/new", { title: m.body.querySelector("#nmTitle").value, bpm: +m.body.querySelector("#nmBpm").value,
        bpb, bars: +m.body.querySelector("#nmBars").value, drum: m.body.querySelector("#nmDrum").checked });
      m.close();
      S.selSong = r.id; emit("songSelected", r.id);
      openPianoRoll(r.id, "p1");
    } catch (e) { toast(e.message, true); }
  };
}
document.getElementById("btnNewMidi")?.addEventListener("click", newMidiDialog);
