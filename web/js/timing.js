// Every song plays as recorded: one fixed speed (project tempo / the song's tempo), its own timing kept -
// drifts, odd bars and tempo changes included. Nothing is stretched beat by beat.

const cache = new WeakMap();      // song info -> landing
const partsCache = new WeakMap(); // song info -> tempo parts
const marksCache = new WeakMap(); // song info -> { key, conf, marks }

// Where each beat of the song lands, counted in the song's own beats at its tempo: W[j] = fb + (t_j - t_fb) / p
// (p = the song's beat period, fb = its first beat). The exact twin of mashlab/project.py song_tspec +
// mashlab/timing.py positions() ("natural", no pins). at(beat index) -> W, inv(W) -> beat index.
export function landing(info) {
  if (!info?.beats || info.beats.length < 2 || !info.bpm) return null;
  if (cache.has(info)) return cache.get(info);
  const b = info.beats, n = b.length, p = 60 / info.bpm;
  const fb = Math.min(Math.max(0, Math.trunc(info.first_beat || 0)), n - 1);
  const W = new Float64Array(n);
  for (let j = 0; j < n; j++) W[j] = fb + (b[j] - b[fb]) / p;
  const at = (x) => { const i = Math.max(0, Math.min(n - 2, Math.floor(x))); return W[i] + (x - i) * (W[i + 1] - W[i]); };
  const inv = (w) => {
    if (w <= W[0]) return (w - W[0]) / (W[1] - W[0]);
    if (w >= W[n - 1]) return n - 1 + (w - W[n - 1]) / (W[n - 1] - W[n - 2]);
    let lo = 0, hi = n - 1;
    while (hi - lo > 1) { const m = (lo + hi) >> 1; if (W[m] <= w) lo = m; else hi = m; }
    return lo + (w - W[lo]) / (W[hi] - W[lo]);
  };
  const tw = (t) => fb + (t - b[fb]) / p;           // audio time (s) -> W, and back
  const wt = (w) => b[fb] + (w - fb) * p;
  const out = { W, at, inv, tw, wt };
  cache.set(info, out);
  return out;
}

function median(a) {
  const s = Float64Array.from(a).sort();
  const n = s.length;
  return n ? (n % 2 ? s[(n - 1) / 2] : (s[n / 2 - 1] + s[n / 2]) / 2) : 0;
}

// Parts of the song the beat detector found at a tempo more than 6 % away from the song's own:
// [{ a, b: beat indices, bpm }]. Shown on clips for information only - they play as recorded like the rest.
export function tempoParts(info) {
  if (!info?.beats || !info.segments || !info.bpm) return [];
  if (partsCache.has(info)) return partsCache.get(info);
  const b = info.beats, out = [];
  const lo = info.first_beat || 0, hi = info.last_beat ?? b.length - 1;
  for (const [s0, s1] of info.segments) {
    const a = Math.max(s0, lo), z = Math.min(s1, hi);
    if (z - a < 4) continue;
    const d = [];
    for (let j = a; j < z; j++) d.push(b[j + 1] - b[j]);
    const bpm = 60 / median(d);
    if (Math.abs(Math.log(bpm / info.bpm)) > Math.log(1.06)) out.push({ a, b: z, bpm });
  }
  partsCache.set(info, out);
  return out;
}

// How much the song's own tempo moves within its main-tempo parts: half the 5-95 % range of the tempo over
// 16-beat windows, in % (0.7 % for the median song; the detector's beat-to-beat jitter averages out).
export function tempoVariation(info) {
  if (!info?.beats || !info.segments || !info.bpm) return 0;
  const b = info.beats, p = 60 / info.bpm, loc = [];
  const lo = info.first_beat || 0, hi = info.last_beat ?? b.length - 1;
  for (const [s0, s1] of info.segments) {
    const a = Math.max(s0, lo), z = Math.min(s1, hi + 1);
    if (z - a < 24) continue;
    const d = [];
    for (let j = a; j < z - 1; j++) d.push(b[j + 1] - b[j]);
    if (Math.abs(Math.log(median(d) / p)) > Math.log(1.06)) continue;
    for (let s = a + 4; s + 16 < z - 4; s += 8) loc.push(60 / ((b[s + 16] - b[s]) / 16));
  }
  if (loc.length < 3) return 0;
  loc.sort((x, y) => x - y);
  const q = (f) => loc[Math.min(loc.length - 1, Math.max(0, Math.round(f * (loc.length - 1))))];
  return (q(0.95) - q(0.05)) / 2 / info.bpm * 100;
}

// ---------------------------------------------------------------- beat marks
// What's drawn on a song's clips. Marks only: they move nothing in the audio (songs play as recorded).
//  * the detector's beats + downbeats, each flagged sure / unsure: unsure = fewer than 40 % of the beats around it
//    (+-4) have a clear Beat This! beat probability (>= 0.5), or the mark is a guess outside the detected range.
//    (A beat between two sure ones isn't in doubt: where the model hears a half-time pulse, every other beat is
//    weak and the stretch is still sure.)
//  * or, once you've edited a song's marks, yours: info.user_marks = { t: [seconds], d: [0 beat | 1 downbeat |
//    2 "bars start here"], s: [1 sure | 0 unsure], bpb } - saved with the song, so every project shows them
export const CLEAR = 50, SURE_SHARE = 0.4;

function smoothConf(conf, lo, hi, w = 4) {
  const n = conf.length, out = new Uint8Array(n);
  for (let j = 0; j < n; j++) {
    let a = Math.max(0, j - w), b = Math.min(n, j + w + 1), clear = 0;
    for (let i = a; i < b; i++) if (conf[i] >= CLEAR) clear++;
    out[j] = j >= lo && j <= hi && clear / (b - a) >= SURE_SHARE ? 1 : 0;    // outside the detected range: a guess
  }
  for (let j = 1; j < n - 1; j++) if (out[j] && !out[j - 1] && !out[j + 1]) out[j] = 0;   // a lone sure beat in an unsure stretch
  return out;
}

// the detector's marks as an editable list (what you start from when you first edit a song's marks)
export function detectedMarks(info) {
  const b = info.beats, n = b.length, d = new Array(n).fill(0);
  for (const j of info.bars || []) if (j >= 0 && j < n) d[j] = 1;
  const sure = info.beat_conf ? smoothConf(info.beat_conf, info.first_beat || 0, info.last_beat ?? n - 1) : new Uint8Array(n).fill(1);
  // MuScriptor second opinion (mashlab/muscheck.py): where its notes clearly disagree with a mark, it's unsure
  const mv = info.mus_check?.v;
  if (mv && mv.length === n) for (let j = 0; j < n; j++) if (mv[j] === 2) sure[j] = 0;
  const end = (info.duration || b[n - 1]) + 1e-6, t = [], dd = [], ss = [];
  for (let j = 0; j < n; j++) if (b[j] >= -1e-6 && b[j] <= end) { t.push(b[j]); dd.push(d[j]); ss.push(sure[j]); }
  return { t, d: dd, s: ss, bpb: info.bpb || 4 };
}

// -> { w: positions (song beats, landing W), t: seconds, down, sure, user: are these your marks, bpb }
export function songMarks(info) {
  const L = landing(info);
  if (!L) return null;
  const hit = marksCache.get(info);
  if (hit && hit.key === info.user_marks && hit.conf === info.beat_conf && hit.mus === info.mus_check) return hit.marks;
  const m = info.user_marks || detectedMarks(info);
  const marks = {
    t: Float64Array.from(m.t), w: Float64Array.from(m.t, L.tw),
    down: Uint8Array.from(m.d, (x) => (x ? 1 : 0)), sure: Uint8Array.from(m.s || m.t.map(() => 1)),
    user: !!info.user_marks, bpb: m.bpb || info.bpb || 4,
  };
  marksCache.set(info, { key: info.user_marks, conf: info.beat_conf, mus: info.mus_check, marks });
  return marks;
}

// ---- editing (all on the explicit list m = { t, d, s, bpb }; each returns nothing and changes m) ----
function reflag(m) {
  const anchors = [];
  m.d.forEach((x, k) => x === 2 && anchors.push(k));
  if (!anchors.length) return;
  const bpb = m.bpb || 4;
  let ai = 0;
  for (let k = 0; k < m.t.length; k++) {
    while (ai + 1 < anchors.length && anchors[ai + 1] <= k) ai++;
    if (m.d[k] === 2) continue;
    m.d[k] = (((k - anchors[ai]) % bpb) + bpb) % bpb === 0 ? 1 : 0;
  }
}
function insertAt(m, t, d, s) {
  let k = 0;
  while (k < m.t.length && m.t[k] < t) k++;
  m.t.splice(k, 0, t); m.d.splice(k, 0, d); m.s.splice(k, 0, s);
  return k;
}
// drag mark k to time t (kept between its neighbours)
export function moveMark(m, k, t, gap = 0.02) {
  const lo = k > 0 ? m.t[k - 1] + gap : -Infinity, hi = k + 1 < m.t.length ? m.t[k + 1] - gap : Infinity;
  m.t[k] = Math.min(hi, Math.max(lo, t)); m.s[k] = 1;
}
export function addMark(m, t) { insertAt(m, t, 0, 1); reflag(m); }
export function deleteMark(m, k) { m.t.splice(k, 1); m.d.splice(k, 1); m.s.splice(k, 1); reflag(m); }
// shift the marks from time t0 on (or before it) by dt seconds; a shifted mark landing within 0.3 beat of another
// is dropped
export function shiftMarks(m, t0, dt, before, p) {
  for (let k = 0; k < m.t.length; k++) if (before ? m.t[k] < t0 : m.t[k] >= t0) { m.t[k] += dt; m.s[k] = 1; }
  const idx = m.t.map((_, k) => k).sort((a, b) => m.t[a] - m.t[b]), keep = [];
  for (const k of idx) if (!keep.length || m.t[k] - m.t[keep[keep.length - 1]] >= 0.3 * p) keep.push(k);
  m.t = keep.map((k) => m.t[k]); m.d = keep.map((k) => m.d[k]); m.s = keep.map((k) => m.s[k]);
  reflag(m);
}
// bars start at mark k: downbeat every bpb marks from there, until the next mark you set the same way (the first
// one also counts back to the start)
export function barStart(m, k) { m.d[k] = 2; m.s[k] = 1; reflag(m); }
export function setBpb(m, n) {
  m.bpb = n;
  if (!m.d.includes(2)) { const k = m.d.indexOf(1); m.d[Math.max(0, k)] = 2; }
  reflag(m);
}
// your taps replace the marks they cover (and any within 3/4 of a tap from either end - the same beats, placed
// differently); the bar count carries on from the downbeat before them
export function tapReplace(m, taps) {
  if (taps.length < 2) return;
  const half = 0.75 * (taps[taps.length - 1] - taps[0]) / (taps.length - 1);
  const a = taps[0] - half, b = taps[taps.length - 1] + half;
  const keep = m.t.map((t, k) => (t < a || t > b ? k : -1)).filter((k) => k >= 0);
  m.t = keep.map((k) => m.t[k]); m.d = keep.map((k) => m.d[k]); m.s = keep.map((k) => m.s[k]);
  let lastDown = -1;
  for (let k = 0; k < m.t.length && m.t[k] < a; k++) if (m.d[k]) lastDown = k;
  const ks = taps.map((t) => insertAt(m, t, 0, 1));
  if (m.d.includes(2)) reflag(m);
  else if (lastDown >= 0) {
    const bpb = m.bpb || 4;
    for (const k of ks) m.d[k] = (k - lastDown) % bpb === 0 ? 1 : 0;
  }
}
// "these marks are right": the marks in [t0, t1] count as sure
export function confirmMarks(m, t0, t1) { for (let k = 0; k < m.t.length; k++) if (m.t[k] >= t0 && m.t[k] <= t1) m.s[k] = 1; }

// ---------------------------------------------------------------- lock to marks (mashlab/lock.py)
// A song locked to its marks plays its locked stems (warped once so every mark sits on a steady grid). The lock
// belongs to the marks / stems / beat grid it was made from: lockSig is the twin of lock.py sig().
const lockCache = new WeakMap();
export function lockSig(info) {
  const b = info.beats, um = info.user_marks;
  const dur = (info.duration || b[b.length - 1]) + 1e-6;
  const m = um && um.t && um.t.length ? um.t : b.filter((x) => x >= -1e-6 && x <= dur);
  const ms = (x) => Math.floor(x * 1000 + 0.5);
  let s = 0;
  for (const x of m) s += ms(x);
  return `${m.length}-${s}-r${info.stems_rev || 0}-b${b.length}-${ms(b[0])}`;
}
// the lock made from these marks (info.user_marks: the song's grid, or a project's own timing laid over it -
// state.js projectInfo): the song's current lock, or one kept in info.locks (lock.py find is the twin)
export function lockFor(info) {
  if (!info?.beats) return null;
  const s = lockSig(info);
  for (const lk of [info.lock, ...Object.values(info.locks || {})]) if (lk?.rev && lk.sig === s) return lk;
  return null;
}
// the project has this song locked AND a locked version made from the grid it uses exists
export function lockActive(project, sid, info) {
  return !!(project?.songs?.[sid]?.lock && lockFor(info));
}
// the song as its locked version: beat times in the locked audio, its steady tempo; marks moved along with it
export function lockView(info) {
  const lk = lockFor(info);
  const hit = lockCache.get(info);
  if (hit && hit.lock === lk && hit.um === info.user_marks) return hit.view;
  const b = info.beats, lb = lk.beats, n = b.length;
  const toLocked = (t) => {                                   // original audio time -> locked audio time
    if (t <= b[0]) return lb[0] + (t - b[0]) * (lb[1] - lb[0]) / (b[1] - b[0]);
    if (t >= b[n - 1]) return lb[n - 1] + (t - b[n - 1]) * (lb[n - 1] - lb[n - 2]) / (b[n - 1] - b[n - 2]);
    let lo = 0, hi = n - 1;
    while (hi - lo > 1) { const k = (lo + hi) >> 1; if (b[k] <= t) lo = k; else hi = k; }
    return lb[lo] + (t - b[lo]) * (lb[hi] - lb[lo]) / (b[hi] - b[lo]);
  };
  const view = { ...info, beats: lb, bpm: lk.bpm, duration: toLocked(info.duration || b[n - 1]),
    user_marks: info.user_marks ? { ...info.user_marks, t: info.user_marks.t.map(toLocked) } : null };
  lockCache.set(info, { lock: lk, um: info.user_marks, view });
  return view;
}
