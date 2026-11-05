// ReadHead engine — the read-offset lane.
//
// A lane is a sorted list of events. Each event changes how a stem is READ from
// timeline beat `t` onward:
//   off  : read offset in beats. source beat = t - off. +off = later (repeat), -off = skip ahead
//   rate : 1 forward, -1 reverse (reads backwards from where it was)
//   gain : 1 on, 0 muted (gates)
// Before the first event the stem plays untouched (off 0, rate 1, gain 1).
// This is exactly what dragging a snapped clip left/right during playback does.
import { beq } from './grid.js';

export const DEFAULT_STATE = Object.freeze({ off: 0, rate: 1, gain: 1 });

export function makeEvent(t, off = 0, rate = 1, gain = 1, extra = {}) {
  return { t, off, rate, gain, ...extra };
}

export function sortLane(lane) {
  return lane.sort((a, b) => a.t - b.t);
}

export function stateAt(lane, t) {
  let s = DEFAULT_STATE;
  for (const e of lane) {
    if (e.t <= t + 1e-9) s = e; else break;
  }
  return s;
}

/** source beat being read at timeline beat t */
export function srcAt(lane, t) {
  let e = null;
  for (const x of lane) { if (x.t <= t + 1e-9) e = x; else break; }
  if (!e) return t;
  const start = e.t - e.off;
  return start + e.rate * (t - e.t);
}

/**
 * Piecewise segments over [tA, tB): {t0, t1, src0, rate, gain}
 * src0 = source beat at t0. src(t) = src0 + rate * (t - t0).
 */
export function segments(lane, tA, tB) {
  const segs = [];
  const evs = lane.filter((e) => e.t > tA + 1e-9 && e.t < tB - 1e-9);
  let cur = { t: tA, ...stateAtFull(lane, tA) };
  // tRef/srcRef = the event that started this read (anchor), so a renderer can map
  // samples identically no matter where its render window begins
  const push = (t0, t1, st) => {
    if (t1 - t0 <= 1e-9) return;
    segs.push({ t0, t1, src0: st.src0 + st.rate * (t0 - st.tRef), rate: st.rate, gain: st.gain, tRef: st.tRef, srcRef: st.src0 });
  };
  let st = cur;
  let t = tA;
  for (const e of evs) {
    push(t, e.t, st);
    st = { tRef: e.t, src0: e.t - e.off, rate: e.rate, gain: e.gain };
    t = e.t;
  }
  push(t, tB, st);
  return segs;
}

function stateAtFull(lane, t) {
  let e = null;
  for (const x of lane) { if (x.t <= t + 1e-9) e = x; else break; }
  if (!e) return { tRef: 0, src0: 0, rate: 1, gain: 1 };
  return { tRef: e.t, src0: e.t - e.off, rate: e.rate, gain: e.gain };
}

/** Drop events that change nothing (same read position/rate/gain as just before). */
export function normalizeLane(lane) {
  sortLane(lane);
  const out = [];
  for (const e of lane) {
    // later event at the same time wins
    if (out.length && beq(out[out.length - 1].t, e.t)) out.pop();
    const prevSrc = srcAtList(out, e.t);
    const prev = out.length ? out[out.length - 1] : DEFAULT_STATE;
    const newSrc = e.t - e.off;
    if (beq(prevSrc, newSrc) && prev.rate === e.rate && prev.gain === e.gain) continue;
    out.push(e);
  }
  return out;
}
function srcAtList(list, t) {
  if (!list.length) return t;
  const e = list[list.length - 1];
  return e.t - e.off + e.rate * (t - e.t);
}

/** Remove events in [t0, t1) (optionally only those matching `pred`). */
export function clearRange(lane, t0, t1, pred = () => true) {
  return lane.filter((e) => !(e.t >= t0 - 1e-9 && e.t < t1 - 1e-9 && pred(e)));
}

/** Insert/replace an event at t (keeps the lane sorted). */
export function upsert(lane, ev) {
  const out = lane.filter((e) => !beq(e.t, ev.t));
  out.push(ev);
  return sortLane(out);
}

/** Offset currently in effect at t, expressed as `t - src` (works after reverses too). */
export function effectiveOffset(lane, t) {
  return t - srcAt(lane, t);
}

export function cloneLane(lane) {
  return lane.map((e) => ({ ...e }));
}
