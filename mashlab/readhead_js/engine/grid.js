// ReadHead engine — musical grid.
// Everything in the engine is measured in BEATS (quarter notes) against one
// project grid, never in seconds. That is what keeps edits from drifting.
//
// A grid is either FIXED (bpm + t0) or a BEAT MAP (the real time of every beat,
// for songs whose tempo drifts). Both answer beatToSec / secToBeat the same way,
// so nothing else in the engine cares which one it has.

export function makeGrid(bpm, t0 = 0, beatsPerBar = 4) {
  return { bpm, t0, bpb: beatsPerBar };
}

/**
 * @param {number[]} times  time (s) of consecutive beats
 * @param {number} idx0     index in `times` of beat 0 (a downbeat)
 */
export function makeGridMap(times, idx0, beatsPerBar = 4) {
  const iv = [];
  for (let i = 1; i < times.length; i++) iv.push(times[i] - times[i - 1]);
  const sorted = iv.slice().sort((a, b) => a - b);
  const med = sorted[Math.floor(sorted.length / 2)] || 0.5;
  return {
    bpm: Math.round((60 / med) * 100) / 100,
    t0: times[idx0],
    bpb: beatsPerBar,
    map: { times: Float64Array.from(times), idx0 },
  };
}

export const spb = (g) => 60 / g.bpm; // nominal seconds per beat

export function beatToSec(g, b) {
  if (!g.map) return g.t0 + b * (60 / g.bpm);
  const { times, idx0 } = g.map;
  const n = times.length;
  const k = b + idx0;
  if (k <= 0) return times[0] + k * (times[1] - times[0]);
  if (k >= n - 1) return times[n - 1] + (k - (n - 1)) * (times[n - 1] - times[n - 2]);
  const i = Math.floor(k), f = k - i;
  return times[i] + f * (times[i + 1] - times[i]);
}

export function secToBeat(g, s) {
  if (!g.map) return (s - g.t0) / (60 / g.bpm);
  const { times, idx0 } = g.map;
  const n = times.length;
  if (s <= times[0]) return (s - times[0]) / (times[1] - times[0]) - idx0;
  if (s >= times[n - 1]) return n - 1 + (s - times[n - 1]) / (times[n - 1] - times[n - 2]) - idx0;
  let lo = 0, hi = n - 1;
  while (hi - lo > 1) { const m = (lo + hi) >> 1; if (times[m] <= s) lo = m; else hi = m; }
  return lo + (s - times[lo]) / (times[lo + 1] - times[lo]) - idx0;
}

/** Shift which beat counts as "1" (bar phase) by whole beats. */
export function shiftGrid(g, beats) {
  if (!g.map) return { ...g, t0: g.t0 + beats * (60 / g.bpm) };
  return { ...g, t0: beatToSec(g, beats), map: { ...g.map, idx0: g.map.idx0 + beats } };
}

/** Double (factor 2) or halve (factor 0.5) the tempo reading — fixes octave mistakes. */
export function scaleGrid(g, factor) {
  if (!g.map) return { ...g, bpm: g.bpm * factor };
  const { times, idx0 } = g.map;
  let nt = [], ni = idx0;
  if (factor === 2) {
    for (let i = 0; i < times.length; i++) {
      nt.push(times[i]);
      if (i + 1 < times.length) nt.push((times[i] + times[i + 1]) / 2);
    }
    ni = idx0 * 2;
  } else {
    for (let i = idx0 % 2; i < times.length; i += 2) nt.push(times[i]);
    ni = Math.floor(idx0 / 2);
  }
  return makeGridMap(nt, ni, g.bpb);
}

// Snap / note-value table. `beats` is the length in quarter-note beats
// (the bar-based entries scale with beatsPerBar).
export const DIVISIONS = [
  { id: '4bar',  label: '4 bars',              beats: (bpb) => 4 * bpb },
  { id: '2bar',  label: '2 bars',              beats: (bpb) => 2 * bpb },
  { id: 'bar',   label: 'bar',                 beats: (bpb) => bpb },
  { id: '1/2',   label: '1/2  (half bar)',     beats: () => 2 },
  { id: '1/4D',  label: '1/4 dotted',          beats: () => 1.5 },
  { id: '1/4',   label: '1/4  (beat)',         beats: () => 1 },
  { id: '1/4T',  label: '1/4T (⅔ beat)',       beats: () => 2 / 3 },
  { id: '1/8D',  label: '1/8 dotted',          beats: () => 0.75 },
  { id: '1/8',   label: '1/8  (½ beat)',       beats: () => 0.5 },
  { id: '1/8T',  label: '1/8T (⅓ beat)',       beats: () => 1 / 3 },
  { id: '1/16',  label: '1/16 (¼ beat)',       beats: () => 0.25 },
  { id: '1/16T', label: '1/16T (⅙ beat)',      beats: () => 1 / 6 },
  { id: '1/32',  label: '1/32 (⅛ beat)',       beats: () => 0.125 },
];

export function divBeats(id, bpb = 4) {
  const d = DIVISIONS.find((x) => x.id === id);
  return d ? d.beats(bpb) : 1;
}

const EPS = 1e-9;
export const snap = (b, div) => Math.round(b / div + EPS) * div;
export const snapFloor = (b, div) => Math.floor(b / div + EPS) * div;
export const snapCeil = (b, div) => Math.ceil(b / div - EPS) * div;

// Exact-ish beat equality for event bookkeeping (triplets are not binary
// fractions, so never compare beats with ===).
export const beq = (a, b) => Math.abs(a - b) < 1e-6;
