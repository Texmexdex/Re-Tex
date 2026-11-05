// ReadHead engine — tempo/grid estimation, per-cell feature views, song structure.
import { NBANDS } from './dsp.js';
import { makeGrid, makeGridMap, beatToSec, secToBeat } from './grid.js';

export const CELLS_PER_BEAT = 4;
export let LABEL_SIM = 0.9;
export function setLabelSim(v) { LABEL_SIM = v; }

// ---------------------------------------------------------------- helpers
function percentile(arr, p) {
  const a = Array.from(arr).filter((x) => Number.isFinite(x)).sort((x, y) => x - y);
  if (!a.length) return 1;
  return a[Math.min(a.length - 1, Math.max(0, Math.floor(p * (a.length - 1))))];
}
function interp(arr, x) {
  if (x < 0 || x >= arr.length - 1) return 0;
  const i = Math.floor(x), f = x - i;
  return arr[i] * (1 - f) + arr[i + 1] * f;
}
export function cos(a, b) {
  let d = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) { d += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
  return na > 0 && nb > 0 ? d / Math.sqrt(na * nb) : 0;
}

// Combined onset envelope from several stems' frame features.
export function onsetEnvelope(frameSets, weights) {
  const n = Math.max(...frameSets.map((f) => f.nFrames));
  const env = new Float32Array(n);
  frameSets.forEach((f, si) => {
    const w = weights ? weights[si] : 1;
    const norm = percentile(f.flux, 0.95) || 1;
    for (let i = 0; i < f.nFrames; i++) env[i] += (w * f.flux[i]) / norm;
  });
  // remove slow trend, half-wave rectify
  const win = 16;
  const out = new Float32Array(n);
  let acc = 0;
  const q = [];
  for (let i = 0; i < n + win; i++) {
    if (i < n) { acc += env[i]; q.push(env[i]); }
    if (q.length > 2 * win + 1) acc -= q.shift();
    const c = i - win;
    if (c >= 0 && c < n) out[c] = Math.max(0, env[c] - acc / q.length);
  }
  return out;
}

// ---------------------------------------------------------------- tempo
const combAt = (env, P, ph, a = 0, b = env.length) => {
  let s = 0, c = 0;
  let x = ph + Math.ceil((a - ph) / P) * P;
  for (; x < b - 1; x += P) { s += interp(env, x); c++; }
  return c ? s / c : 0;
};
function bestPhase(env, P, a = 0, b = env.length, step = 0.1) {
  let best = { ph: 0, s: -Infinity }, sum = 0, cnt = 0;
  for (let ph = 0; ph < P; ph += step) {
    const s = combAt(env, P, ph, a, b);
    sum += s; cnt++;
    if (s > best.s) best = { ph, s };
  }
  best.contrast = best.s - sum / cnt;
  return best;
}

/** Coarse BPM from the onset autocorrelation. */
export function detectTempo(env, hopSec, { minBpm = 55, maxBpm = 215, prior = 110 } = {}) {
  const n = env.length;
  const lagOf = (bpm) => 60 / (bpm * hopSec);
  const maxLag = Math.min(n - 1, Math.ceil(lagOf(minBpm / 2)) + 4);
  const acf = new Float64Array(maxLag + 1);
  let mean = 0;
  for (let i = 0; i < n; i++) mean += env[i];
  mean /= n;
  for (let L = 1; L <= maxLag; L++) {
    let s = 0;
    for (let i = 0; i + L < n; i++) s += (env[i] - mean) * (env[i + L] - mean);
    acf[L] = s / (n - L);
  }
  const acfAt = (x) => interp(acf, x);
  let best = null;
  for (let bpm = minBpm; bpm <= maxBpm; bpm += 0.25) {
    const L = lagOf(bpm);
    let s = acfAt(L) + 0.5 * acfAt(2 * L) + 0.25 * acfAt(4 * L) + 0.35 * acfAt(L / 2);
    const pw = Math.exp(-0.5 * (Math.log2(bpm / prior) / 1.0) ** 2);
    s *= 0.6 + 0.4 * pw;
    if (!best || s > best.s) best = { bpm, s };
  }
  return best.bpm;
}

/**
 * Track how the beat phase moves through the song at a given tempo: per-chunk
 * best phase, unwrapped, with a weighted straight-line fit. A flat line means
 * the tempo is right; a slope means it is slightly off; scatter means the
 * song's tempo genuinely moves.
 */
export function phaseDrift(env, hopSec, bpm, chunkSec = 20) {
  const P = 60 / (bpm * hopSec);
  const beat = P * hopSec;
  const g = bestPhase(env, P);
  const C = Math.max(Math.round(chunkSec / hopSec), Math.ceil(P * 8));
  // phase is circular, and a chunk locking to the off-beat (8th-note grooves) is not drift either:
  // judge steadiness modulo half a beat
  const wrap = (x) => x - Math.round(x / (beat / 2)) * (beat / 2);
  const pts = [];
  for (let a = 0; a + P * 4 < env.length; a += C) {
    const b = Math.min(env.length, a + C);
    const r = bestPhase(env, P, a, b, 0.2);
    pts.push({ t: ((a + b) / 2) * hopSec, o: wrap((r.ph - g.ph) * hopSec), w: Math.max(1e-6, r.contrast) });
  }
  const circMean = (use) => {
    let sx = 0, sy = 0;
    for (const p of use) { const ang = (2 * Math.PI * p.o) / (beat / 2); sx += p.w * Math.cos(ang); sy += p.w * Math.sin(ang); }
    return (Math.atan2(sy, sx) / (2 * Math.PI)) * (beat / 2);
  };
  let centre = circMean(pts);
  // chunks that lock to the off-beat (half-time bars, breakdowns) are outliers, not drift
  let keep = pts.filter((p) => Math.abs(wrap(p.o - centre)) < beat / 8);
  if (keep.length) centre = circMean(keep);
  keep = pts.filter((p) => Math.abs(wrap(p.o - centre)) < beat / 8);
  let r2 = 0, sw = 0;
  for (const p of keep) { const e = wrap(p.o - centre); r2 += p.w * e * e; sw += p.w; }
  // slow drift: compare the first and last third of the song
  const third = keep.length >= 6 ? Math.floor(keep.length / 3) : 0;
  const driftMs = third ? Math.abs(wrap(circMean(keep.slice(-third)) - circMean(keep.slice(0, third)))) * 1000 : 0;
  const keepFrac = pts.length ? keep.length / pts.length : 0;
  return { bpm, rmsMs: sw ? Math.sqrt(r2 / sw) * 1000 : 999, endDriftMs: driftMs, keepFrac, phaseFrames: g.ph, comb: g.s, contrast: g.contrast, pts };
}

/** Beat tracking by dynamic programming (Ellis 2007) for songs that drift. */
export function beatTrack(env, hopSec, bpm, tightness = 300) {
  const n = env.length;
  const P = 60 / (bpm * hopSec);
  let m = 0, sd = 0;
  for (let i = 0; i < n; i++) m += env[i];
  m /= n;
  for (let i = 0; i < n; i++) sd += (env[i] - m) ** 2;
  sd = Math.sqrt(sd / n) || 1;
  const o = Float64Array.from(env, (x) => x / sd);
  const score = new Float64Array(n), back = new Int32Array(n).fill(-1);
  const lo = Math.round(P * 0.6), hi = Math.round(P * 1.6);
  for (let t = 0; t < n; t++) {
    let best = -Infinity, bi = -1;
    for (let d = lo; d <= hi; d++) {
      const tau = t - d;
      if (tau < 0) break;
      const v = score[tau] - tightness * Math.log(d / P) ** 2;
      if (v > best) { best = v; bi = tau; }
    }
    score[t] = o[t] + (bi >= 0 && best > 0 ? best : 0);
    back[t] = bi >= 0 && best > 0 ? bi : -1;
  }
  let t = n - 1, bv = -Infinity;
  for (let i = Math.max(0, n - Math.ceil(P)); i < n; i++) if (score[i] > bv) { bv = score[i]; t = i; }
  const frames = [];
  while (t >= 0) { frames.push(t); t = back[t]; }
  frames.reverse();
  // sub-frame refinement + light smoothing (keeps real drift, removes jitter)
  const raw = frames.map((f) => {
    if (f > 0 && f < n - 1 && env[f] >= env[f - 1] && env[f] >= env[f + 1]) {
      const y0 = env[f - 1], y1 = env[f], y2 = env[f + 1], den = y0 - 2 * y1 + y2;
      return (f + (den !== 0 ? Math.max(-0.5, Math.min(0.5, (0.5 * (y0 - y2)) / den)) : 0)) * hopSec;
    }
    return f * hopSec;
  });
  const W = 6, sm = [];
  for (let i = 0; i < raw.length; i++) {
    let sx = 0, sy = 0, sxx = 0, sxy = 0, c = 0;
    for (let j = Math.max(0, i - W); j <= Math.min(raw.length - 1, i + W); j++) { sx += j; sy += raw[j]; sxx += j * j; sxy += j * raw[j]; c++; }
    const den = c * sxx - sx * sx;
    const sl = den ? (c * sxy - sx * sy) / den : 0;
    sm.push((sy - sl * sx) / c + sl * i);
  }
  return sm;
}

/** Which beat of the bar is "1": harmonic changes and kicks land on downbeats. */
export function detectBarPhase(frameSets, beatTimes, bpb = 4) {
  const hopSec = frameSets[0].hopSec;
  const nB = beatTimes.length - 1;
  if (nB < bpb * 2) return 0;
  const chromaB = [], lowB = new Float64Array(nB);
  for (let i = 0; i < nB; i++) {
    const a = Math.max(0, Math.floor(beatTimes[i] / hopSec)), b = Math.floor(beatTimes[i + 1] / hopSec);
    const c = new Float64Array(12);
    for (const f of frameSets) {
      for (let k = a; k < b && k < f.nFrames; k++) for (let p = 0; p < 12; p++) c[p] += f.chroma[k * 12 + p];
      for (let k = a - 2; k <= a + 2; k++) if (k >= 0 && k < f.nFrames) lowB[i] += f.low[k];
    }
    chromaB.push(c);
  }
  const lowN = percentile(lowB, 0.9) || 1;
  const scores = new Float64Array(bpb), counts = new Float64Array(bpb);
  for (let i = 1; i < nB; i++) {
    scores[i % bpb] += (1 - cos(chromaB[i], chromaB[i - 1])) * 2 + 0.5 * (lowB[i] / lowN);
    counts[i % bpb]++;
  }
  let q = 0;
  for (let k = 1; k < bpb; k++) if (scores[k] / counts[k] > scores[q] / counts[q]) q = k;
  return q;
}

/**
 * Full grid estimate: coarse tempo -> octave choice -> drift-corrected exact
 * tempo -> (if the song really drifts) a beat-by-beat map -> bar phase.
 */
export function estimateGrid(frameSets, roles = [], bpb = 4, { forceBpm = null } = {}) {
  const hopSec = frameSets[0].hopSec;
  const w = frameSets.map((_, i) => (roles[i] === 'drums' ? 2 : roles[i] === 'vocal' ? 0.5 : 1));
  const env = onsetEnvelope(frameSets, w);
  const durSec = env.length * hopSec;
  let bpm = forceBpm || detectTempo(env, hopSec);
  if (!forceBpm) {
    const prior = (b) => Math.exp(-0.5 * (Math.log2(b / 110) / 0.8) ** 2);
    let best = { b: bpm, s: -1 };
    for (const b of [bpm / 2, bpm, bpm * 2]) {
      if (b < 55 || b > 215) continue;
      const s = bestPhase(env, 60 / (b * hopSec)).s * prior(b);
      if (s > best.s) best = { b, s };
    }
    bpm = best.b;
  }
  // exact tempo: the whole-song comb only peaks sharply at the true tempo,
  // because any error accumulates over hundreds of beats
  const globalComb = (b) => bestPhase(env, 60 / (b * hopSec), 0, env.length, 0.25).s;
  let fine = { b: bpm, s: globalComb(bpm) };
  for (let b = bpm - 1; b <= bpm + 1 + 1e-9; b += 0.02) {
    const sc = globalComb(b);
    if (sc > fine.s) fine = { b, s: sc };
  }
  for (let b = fine.b - 0.02; b <= fine.b + 0.02 + 1e-9; b += 0.005) {
    const sc = globalComb(b);
    if (sc > fine.s) fine = { b, s: sc };
  }
  bpm = fine.b;
  // songs made in a DAW sit on a round tempo: prefer it when it fits as well
  for (const r of [Math.round(bpm), Math.round(bpm * 2) / 2, Math.round(bpm * 10) / 10]) {
    if (Math.abs(r - bpm) > 0.15) continue;
    if (globalComb(r) >= fine.s * 0.985) { bpm = r; break; }
  }
  const d = phaseDrift(env, hopSec, bpm);
  const fixedOK = d.rmsMs < 18 && d.keepFrac >= 0.6 && d.endDriftMs < 30;
  let times;
  if (fixedOK) {
    const P = 60 / bpm;
    const ph0 = d.phaseFrames * hopSec;
    times = [];
    for (let t = ph0 % P; t < durSec + P; t += P) times.push(t);
  } else {
    times = beatTrack(env, hopSec, bpm);
  }
  const q = detectBarPhase(frameSets, times, bpb);
  let grid;
  if (fixedOK) grid = makeGrid(Math.round(bpm * 1000) / 1000, times[q], bpb);
  else grid = makeGridMap(times, q, bpb);
  const conf = Math.max(0, Math.min(1, d.contrast / (d.comb || 1)));
  return { grid, confidence: conf, mode: fixedOK ? 'fixed' : 'map', driftMs: d.rmsMs, keepFrac: d.keepFrac };
}

// ---------------------------------------------------------------- cell view
/**
 * Aggregate frame features into grid cells (1/4 beat). The generator only ever
 * asks musical questions ("how loud is source beat 37.5?"), answered here.
 */
export function makeView(fr, grid) {
  const hopSec = fr.hopSec;
  const dur = fr.nFrames * hopSec;
  const cellBeats = 1 / CELLS_PER_BEAT;
  const bStart = Math.floor(secToBeat(grid, 0));
  const bEnd = Math.ceil(secToBeat(grid, dur));
  const nCells = Math.max(1, (bEnd - bStart) * CELLS_PER_BEAT);
  const rms = new Float32Array(nCells), onset = new Float32Array(nCells);
  const chroma = new Float32Array(nCells * 12), bands = new Float32Array(nCells * NBANDS);
  for (let c = 0; c < nCells; c++) {
    const b = bStart + c * cellBeats;
    const a = Math.max(0, Math.floor(beatToSec(grid, b) / hopSec));
    const e = Math.min(fr.nFrames, Math.max(a + 1, Math.floor(beatToSec(grid, b + cellBeats) / hopSec)));
    let r = 0, o = 0, cnt = 0;
    for (let k = a; k < e; k++) {
      r += fr.rms[k]; cnt++;
      // onset: strongest flux in the first part of the cell (attack at the cell start)
      if (k < a + Math.max(2, (e - a) / 2)) o = Math.max(o, fr.flux[k]);
      for (let p = 0; p < 12; p++) chroma[c * 12 + p] += fr.chroma[k * 12 + p];
      for (let q = 0; q < NBANDS; q++) bands[c * NBANDS + q] += fr.bands[k * NBANDS + q];
    }
    rms[c] = cnt ? r / cnt : 0;
    onset[c] = o;
    if (cnt) for (let q = 0; q < NBANDS; q++) bands[c * NBANDS + q] /= cnt;
  }
  const rN = percentile(rms, 0.95) || 1, oN = percentile(onset, 0.95) || 1;
  for (let c = 0; c < nCells; c++) { rms[c] = Math.min(2, rms[c] / rN); onset[c] = Math.min(2, onset[c] / oN); }

  const idx = (beat) => {
    const c = Math.floor((beat - bStart) * CELLS_PER_BEAT + 1e-6);
    return c >= 0 && c < nCells ? c : -1;
  };
  const view = {
    grid, bStart, bEnd, nCells,
    rmsCells: rms, onsetCells: onset,
    rms: (beat) => { const c = idx(beat); return c < 0 ? 0 : rms[c]; },
    onset: (beat) => { const c = idx(beat); return c < 0 ? 0 : onset[c]; },
    chroma: (beat) => { const c = idx(beat); return c < 0 ? null : chroma.subarray(c * 12, c * 12 + 12); },
    rmsSpan(b0, b1) {
      let s = 0, n = 0;
      for (let b = b0; b < b1 - 1e-9; b += cellBeats) { s += view.rms(b); n++; }
      return n ? s / n : 0;
    },
    // chroma summed over a span, weighted by loudness (silence carries no key)
    chromaSpan(b0, b1) {
      const out = new Float64Array(12);
      for (let b = b0; b < b1 - 1e-9; b += cellBeats) {
        const c = idx(b);
        if (c < 0) continue;
        for (let p = 0; p < 12; p++) out[p] += chroma[c * 12 + p];
      }
      return out;
    },
    // vector describing one beat: pitch content + timbre + rhythm, for similarity
    beatVec(beat) {
      const b = Math.floor(beat + 1e-6);
      if (bvCache.has(b)) return bvCache.get(b);
      const v = beatVecRaw(b);
      bvCache.set(b, v);
      return v;
    },
  };
  const bvCache = new Map();
  function beatVecRaw(b) {
    {
      const v = new Float64Array(12 + NBANDS + CELLS_PER_BEAT + 1);
      let cn = 0;
      for (let k = 0; k < CELLS_PER_BEAT; k++) {
        const c = idx(b + k * cellBeats);
        if (c < 0) continue;
        for (let p = 0; p < 12; p++) v[p] += chroma[c * 12 + p];
        for (let q = 0; q < NBANDS; q++) v[12 + q] += bands[c * NBANDS + q] / CELLS_PER_BEAT;
        v[12 + NBANDS + k] = onset[c];
        v[12 + NBANDS + CELLS_PER_BEAT] += rms[c] / CELLS_PER_BEAT;
        cn++;
      }
      // normalise the chroma block, centre the timbre block
      let s = 0;
      for (let p = 0; p < 12; p++) s += v[p] * v[p];
      s = Math.sqrt(s) || 1;
      for (let p = 0; p < 12; p++) v[p] /= s;
      let m = 0;
      for (let q = 0; q < NBANDS; q++) m += v[12 + q];
      m /= NBANDS;
      for (let q = 0; q < NBANDS; q++) v[12 + q] = (v[12 + q] - m) * 0.25;
      return v;
    }
  }
  return view;
}

// ---------------------------------------------------------------- structure
/**
 * Bars, phrase alignment and sections (A/B/C labels) from all stems together.
 */
export function analyzeStructure(views, grid, { phraseBars = 4, durationSec, forcePhraseOffset = null } = {}) {
  const bpb = grid.bpb;
  const totalBeats = secToBeat(grid, durationSec);
  const totalBars = Math.max(1, Math.ceil(totalBeats / bpb));
  const barVec = [], energy = new Float64Array(totalBars);
  for (let bar = 0; bar < totalBars; bar++) {
    const b0 = bar * bpb, b1 = b0 + bpb;
    const parts = [];
    for (const v of views) {
      const r = v.rmsSpan(b0, b1);
      energy[bar] += r;
      const ch = v.chromaSpan(b0, b1);
      let s = 0;
      for (let p = 0; p < 12; p++) s += ch[p] * ch[p];
      s = Math.sqrt(s) || 1;
      parts.push(r * 3);
      for (let p = 0; p < 12; p++) parts.push((ch[p] / s) * Math.min(1, r * 4));
    }
    barVec.push(parts);
  }
  const eMax = Math.max(...energy) || 1;
  for (let i = 0; i < totalBars; i++) energy[i] /= eMax;
  // Stack each bar with the bars that follow it (one phrase worth), so a whole
  // chord progression is compared with a whole progression, not bar against bar.
  const single = barVec.slice();
  for (let i = 0; i < totalBars; i++) {
    const st = [];
    for (let k = 0; k < phraseBars; k++) {
      const v = single[i + k] || single[i].map(() => 0);
      for (const x of v) st.push(x);
    }
    barVec[i] = st;
  }

  // novelty at bar i: how different the phrase after i is from the phrase before,
  // relative to how much the music was changing anyway on either side
  const P = phraseBars;
  const d = (a, b) => (a < 0 || b < 0 || a + P > totalBars || b + P > totalBars ? null : 1 - cos(barVec[a], barVec[b]));
  const novelty = new Float64Array(totalBars);
  for (let i = 1; i < totalBars; i++) {
    const cross = d(i - P, i);
    if (cross === null) continue;
    const cross2 = d(i - P, i + P);
    const within = [d(i - 2 * P, i - P), d(i, i + P)].filter((x) => x !== null);
    const w = within.length ? within.reduce((a, b) => a + b, 0) / within.length : 0;
    const c = cross2 === null ? cross : 0.5 * (cross + cross2);
    novelty[i] = Math.max(0, c - 0.5 * w);
  }
  // phrase alignment: which bar offset puts phrase starts on the biggest changes
  let phraseOffset = 0, bestS = -1;
  for (let o = 0; o < phraseBars; o++) {
    let s = 0;
    for (let i = o; i < totalBars; i += phraseBars) s += novelty[i] + 0.25 * Math.max(0, energy[i] - (energy[i - 1] ?? 0));
    if (s > bestS) { bestS = s; phraseOffset = o; }
  }
  if (forcePhraseOffset != null) phraseOffset = ((forcePhraseOffset % phraseBars) + phraseBars) % phraseBars;
  // section boundaries on phrase starts with strong novelty
  const cands = [];
  for (let i = phraseOffset; i < totalBars; i += phraseBars) if (i > 0) cands.push(i);
  const minGap = Math.max(8, phraseBars * 2);
  const nv = cands.map((i) => novelty[i]).sort((a, b) => a - b);
  const median = nv.length ? nv[Math.floor(nv.length / 2)] : 0;
  const thresh = Math.max(0.015, 1.5 * median);
  const chosen = [];
  cands
    .filter((i) => novelty[i] > thresh)
    .sort((a, b) => novelty[b] - novelty[a])
    .forEach((i) => { if (chosen.every((c) => Math.abs(c - i) >= minGap) && i >= minGap / 2 && totalBars - i >= minGap / 2) chosen.push(i); });
  chosen.sort((a, b) => a - b);
  const bounds = [0, ...chosen, totalBars];
  const sections = [];
  // two sections get the same label when their phrases line up as near-copies
  const secSim = (a0, a1, b0, b1) => {
    let s = 0, n = 0;
    for (let k = 0; a0 + k + P <= a1 && b0 + k + P <= b1; k += P) { s += cos(barVec[a0 + k], barVec[b0 + k]); n++; }
    return n ? s / n : 0;
  };
  const reps = [];
  for (let s = 0; s < bounds.length - 1; s++) {
    const a = bounds[s], b = bounds[s + 1];
    let en = 0;
    for (let i = a; i < b; i++) en += energy[i];
    let label = -1, best = 0;
    reps.forEach((r, l) => { const v = secSim(r.a, r.b, a, b); if (v > best) { best = v; if (v > LABEL_SIM) label = l; } });
    if (label < 0) { label = reps.length; reps.push({ a, b }); }
    sections.push({ startBar: a, endBar: b, label: String.fromCharCode(65 + (label % 26)), energy: en / Math.max(1, b - a), sim: best });
  }
  return { phraseBars, phraseOffset, totalBars, totalBeats, energy: Array.from(energy), novelty: Array.from(novelty), sections };
}
