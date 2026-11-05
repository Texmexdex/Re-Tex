// Play / stop / seek / loop / tempo controls and the clock.
import { ICONS } from "./icons.js";
import { api } from "./api.js";
import { S, on, emit, edit, transportPos, fmtTime, secPerBeat, toast } from "./state.js";
import { showMenu } from "./ui.js";

const btnPlay = document.getElementById("btnPlay");
const clockTime = document.getElementById("clockTime");
const clockBar = document.getElementById("clockBar");
const bpmInput = document.getElementById("bpmInput");
const btnLoop = document.getElementById("btnLoop");

export async function play(fromSec) {
  if (!S.project) return;
  try {
    const st = await api.post("/api/transport", fromSec != null ? { action: "play", pos: fromSec } : { action: "play" });
    applyState(st);
  } catch (e) { toast(e.message, true); }
}
export async function stop() {
  applyState(await api.post("/api/transport", { action: "stop" }));
}
export function togglePlay() { S.transport.playing ? stop() : play(); }
export async function seekTo(sec) {
  applyState(await api.post("/api/transport", { action: "seek", pos: sec }));
}

export function applyState(st) {
  if (!st) return;
  // server time -> client clock offset (both in seconds)
  S.clockOffset = st.t - performance.now() / 1000;
  S.transport = st;
  btnPlay.classList.toggle("on", st.playing);
  btnPlay.innerHTML = st.playing ? ICONS.stop : ICONS.play;
  if (st.error) toast(st.error, true);
  emit("transport");
}

function tick() {
  const pos = transportPos();
  clockTime.textContent = fmtTime(pos);
  if (S.project) {
    const beat = pos / secPerBeat(), bpb = S.project.bpb;
    clockBar.textContent = `${Math.floor(beat / bpb) + 1}.${Math.floor(beat % bpb) + 1}`;
  }
  requestAnimationFrame(tick);
}
requestAnimationFrame(tick);

btnPlay.onclick = togglePlay;
document.getElementById("btnStart").onclick = () => seekTo(S.project?.loop?.on ? S.project.loop.start * secPerBeat() : 0);
btnLoop.onclick = () => toggleLoop();
const btnFollow = document.getElementById("btnFollow");
export function toggleFollow() {
  S.follow = !S.follow;
  try { localStorage.setItem("follow", S.follow ? "1" : "0"); } catch {}
  btnFollow.classList.toggle("on", S.follow);
  toast(S.follow ? "Following the playhead" : "Not following the playhead - the view stays where you put it", false, 1800);
}
btnFollow.classList.toggle("on", S.follow);
btnFollow.onclick = () => { toggleFollow(); btnFollow.blur(); };
// metronome: clicks on the project's grid, made by the player next to the mix (sample-locked to what you hear)
const btnClick = document.getElementById("btnClick");
const clickPref = (() => { try { return JSON.parse(localStorage.getItem("metronome") || "{}"); } catch { return {}; } })();
const CLICK_LEVELS = [["Soft", 0.25], ["Medium", 0.5], ["Loud", 0.9]];
function sendClick() {
  try { localStorage.setItem("metronome", JSON.stringify(clickPref)); } catch {}
  btnClick.classList.toggle("on", !!clickPref.on);
  api.post("/api/transport", { action: "metronome", on: !!clickPref.on, gain: clickPref.gain ?? 0.5 }).catch(() => {});
}
export function toggleMetronome() {
  clickPref.on = !clickPref.on;
  sendClick();
  toast(clickPref.on ? "Metronome on - clicks on the mix's grid while it plays (K)" : "Metronome off", false, 1800);
}
btnClick.onclick = () => { toggleMetronome(); btnClick.blur(); };
btnClick.oncontextmenu = (e) => {
  e.preventDefault();
  showMenu(e, [["Metronome level"], ...CLICK_LEVELS.map(([name, g]) =>
    [name, () => { clickPref.gain = g; sendClick(); }, Math.abs((clickPref.gain ?? 0.5) - g) < 0.01 ? "now" : ""])]);
};
sendClick();                                        // the server starts with it off: tell it what you had

export function toggleLoop() {
  if (!S.project) return;
  edit((p) => {
    if (!p.loop || p.loop.end <= p.loop.start) {
      const b = Math.floor(transportPos() / secPerBeat() / p.bpb) * p.bpb;
      p.loop = { on: true, start: b, end: b + 4 * p.bpb };
    } else p.loop.on = !p.loop.on;
  });
}
bpmInput.onchange = () => {
  const v = Math.max(40, Math.min(250, +bpmInput.value || 120));
  if (!S.project || v === S.project.bpm) return;
  // keep the playhead on the same musical position
  const beat = transportPos() / secPerBeat();
  // a run of tempo clicks is one undo step; the server only renders the tempo you stop on
  edit((p) => {
    p.bpm = v;
    // songs already in the arrangement keep their ½× / 1× / 2× choice: flipping it here would change which
    // part of the song every clip plays (the ½× 1× 2× buttons rescale the clips; a tempo change can't)
    const used = new Set(p.clips.map((c) => c.song));
    for (const sid of Object.keys(p.songs)) {
      const info = S.songs.get(sid);
      if (!info || used.has(sid)) continue;
      p.songs[sid].mult = [0.5, 1, 2].reduce((b, m) => Math.abs(Math.log(v / (info.bpm * m))) < Math.abs(Math.log(v / (info.bpm * b))) ? m : b, 1);
    }
  }, { coalesce: "tempo", label: "Tempo" });
  seekTo(beat * 60 / v);
  toast(`Tempo ${v} BPM - quick drafts in a second or two, full quality in about a minute`);
};
document.getElementById("volume").oninput = (e) => api.post("/api/transport", { action: "volume", value: +e.target.value });

on("project", () => {
  if (!S.project) return;
  if (document.activeElement !== bpmInput) bpmInput.value = S.project.bpm;
  btnLoop.classList.toggle("on", !!S.project.loop?.on);
});
