// ReadHead engine — vocal analysis for Arrange mode.
// Finds the sung LINES (split at breaths/gaps), the SYLLABLES inside each line
// (onset times), each line's anchor bar (keeping pickups), and which lines repeat
// (hook detection) — so vocals can be placed on purpose, not by dice.
import { secToBeat } from './grid.js';
import { cos } from './analysis.js';

function pct(arr, p) {
  const a = Array.from(arr).filter(Number.isFinite).sort((x, y) => x - y);
  return a.length ? a[Math.min(a.length - 1, Math.floor(p * (a.length - 1)))] : 0;
}

/**
 * @param fr    frame features of the vocal stem (dsp.analyzeFrames)
 * @param grid  project grid
 * @param view  cell view of the vocal stem (analysis.makeView)
 * @returns {lines, syllables, voiced}
 */
export function analyzeVocal(fr, grid, view, opt = {}) {
  const n = fr.nFrames, hop = fr.hopSec, bpb = grid.bpb;
  const beatSec = 60 / grid.bpm;
  // ---- voice activity with hysteresis
  const loud = pct(fr.rms, 0.95) || 1e-6;
  const onT = (opt.onDb != null ? 10 ** (opt.onDb / 20) : 0.12) * loud;
  const offT = onT * 0.5;
  const voiced = new Uint8Array(n);
  let on = false;
  for (let i = 0; i < n; i++) {
    const r = fr.rms[i];
    if (!on && r > onT) on = true; else if (on && r < offT) on = false;
    voiced[i] = on ? 1 : 0;
  }
  // regions
  let regions = [];
  for (let i = 0; i < n; i++) {
    if (voiced[i] && (i === 0 || !voiced[i - 1])) regions.push({ a: i, b: i + 1 });
    else if (voiced[i]) regions[regions.length - 1].b = i + 1;
  }
  regions = regions.filter((r) => (r.b - r.a) * hop >= 0.06);
  // merge word gaps into lines; a breath / rest longer than this splits lines
  const gapSplit = Math.max(opt.gapSec ?? 0.3, 0.55 * beatSec);
  const merged = [];
  for (const r of regions) {
    const last = merged[merged.length - 1];
    if (last && (r.a - last.b) * hop < gapSplit) last.b = r.b; else merged.push({ ...r });
  }
  const minLine = Math.max(0.2, 0.5 * beatSec);

  // ---- syllable onset detection function
  const lr = new Float32Array(n);
  for (let i = 0; i < n; i++) lr[i] = Math.log(fr.rms[i] + 1e-5);
  const lrs = new Float32Array(n);
  for (let i = 0; i < n; i++) lrs[i] = (lr[Math.max(0, i - 1)] + lr[i] + lr[Math.min(n - 1, i + 1)]) / 3;
  const rise = new Float32Array(n);
  for (let i = 2; i < n; i++) rise[i] = Math.max(0, lrs[i] - lrs[i - 2]);
  const vFlux = [], vRise = [];
  for (let i = 0; i < n; i++) if (voiced[i]) { vFlux.push(fr.flux[i]); vRise.push(rise[i]); }
  const fN = pct(vFlux, 0.95) || 1, rN = pct(vRise, 0.95) || 1;
  const det = new Float32Array(n);
  for (let i = 0; i < n; i++) det[i] = fr.flux[i] / fN + 1.2 * (rise[i] / rN);
  const ds = new Float32Array(n);
  for (let i = 0; i < n; i++) ds[i] = (det[Math.max(0, i - 1)] + 2 * det[i] + det[Math.min(n - 1, i + 1)]) / 4;

  const minSp = Math.max(6, Math.round(0.085 / hop));
  const lines = [];
  const syllables = [];
  for (const m of merged) {
    if ((m.b - m.a) * hop < minLine) continue;
    const syl = [];
    // local adaptive threshold
    for (let i = m.a; i < m.b; i++) {
      let isMax = true;
      for (let k = -4; k <= 4; k++) if (k && ds[i + k] > ds[i]) { isMax = false; break; }
      if (!isMax) continue;
      let med = [];
      for (let k = -25; k <= 25; k++) if (i + k >= 0 && i + k < n) med.push(ds[i + k]);
      med.sort((x, y) => x - y);
      const thr = med[med.length >> 1] + 0.15;
      if (ds[i] < Math.max(0.3, thr)) continue;
      if (syl.length && i - syl[syl.length - 1].f < minSp) {
        if (ds[i] > syl[syl.length - 1].s) syl[syl.length - 1] = { f: i, s: ds[i] };
        continue;
      }
      syl.push({ f: i, s: ds[i] });
    }
    // the line's first syllable is at its onset (strongest rise in the first ~60 ms)
    let f0 = m.a, best = -1;
    for (let i = Math.max(0, m.a - 3); i < Math.min(m.b, m.a + Math.round(0.06 / hop)); i++) if (ds[i] > best) { best = ds[i]; f0 = i; }
    if (!syl.length || syl[0].f - f0 > minSp) syl.unshift({ f: f0, s: Math.max(best, 0.5) });
    else syl[0].f = Math.min(syl[0].f, f0);
    // the detector peaks a little after the real attack: walk back to where the
    // energy starts rising, so a chop lands the consonant ON the beat, not after it
    for (const s of syl) {
      let base = s.f, bv = Infinity;
      for (let i = Math.max(0, s.f - 6); i <= s.f; i++) if (lrs[i] < bv) { bv = lrs[i]; base = i; }
      const top = lrs[Math.min(n - 1, s.f + 1)];
      let on = s.f;
      for (let i = base; i <= s.f; i++) if (lrs[i] > bv + 0.2 * (top - bv)) { on = i; break; }
      // sub-frame: interpolate the crossing
      const y0 = lrs[Math.max(0, on - 1)], y1 = lrs[on], lvl = bv + 0.2 * (top - bv);
      const frac = y1 > y0 ? Math.max(0, Math.min(1, (lvl - y0) / (y1 - y0))) : 1;
      s.t = (on - 1 + frac) * hop;
    }
    // drop syllables too close to the line end (no vowel after them)
    const sy = syl.filter((s, k) => k === 0 || (m.b - s.f) * hop > 0.05);
    const startSec = sy[0].t, endSec = m.b * hop;
    const startBeat = secToBeat(grid, startSec), endBeat = secToBeat(grid, endSec);
    // anchor bar: a line starting up to 1.5 beats before a bar line is a pickup into it
    const anchorBar = Math.floor((startBeat + 1.5) / bpb + 1e-9);
    const line = {
      idx: lines.length, startSec, endSec, startBeat, endBeat, anchorBar,
      lenBars: Math.max(1, Math.ceil((endBeat - anchorBar * bpb) / bpb - 1e-6)),
      syllables: [], energy: 0,
    };
    let e = 0;
    for (let i = m.a; i < m.b; i++) e += fr.rms[i];
    line.energy = e / Math.max(1, m.b - m.a) / loud;
    sy.forEach((s, k) => {
      const next = sy[k + 1];
      const tSec = s.t;
      const syl1 = {
        line: line.idx, k, sec: tSec, beat: secToBeat(grid, tSec),
        endBeat: secToBeat(grid, next ? next.t : endSec), strength: s.s,
      };
      line.syllables.push(syl1);
      syllables.push(syl1);
    });
    lines.push(line);
  }

  // ---- the vocal's own bar phase: which beat of its source counts as "1".
  // Lines start on (or just before, as a pickup) the singer's downbeats.
  let phase = 0, bestPh = -1;
  for (let ph = 0; ph < bpb; ph++) {
    let sc = 0;
    for (const l of lines) {
      const x = l.startBeat - ph;
      let best = 0;
      for (let k = Math.floor(x / bpb) - 1; k <= Math.floor(x / bpb) + 1; k++) {
        const d = x - k * bpb;
        if (d >= -1.5 && d <= 0.75) { const z = (d < 0 ? d + 0.25 : d) / 0.5; best = Math.max(best, Math.exp(-(z * z))); }
      }
      sc += best * Math.sqrt(l.syllables.length);
    }
    for (const sy of syllables) {
      const x = sy.beat - ph;
      if (Math.abs(x - Math.round(x / bpb) * bpb) < 0.15) sc += 0.15 * sy.strength;
    }
    if (sc > bestPh) { bestPh = sc; phase = ph; }
  }
  if (opt.barPhase != null) phase = ((opt.barPhase % bpb) + bpb) % bpb;
  for (const l of lines) {
    l.anchorBar = Math.floor((l.startBeat - phase + 1.5) / bpb + 1e-9);
    l.anchorBeat = l.anchorBar * bpb + phase;
    l.lenBars = Math.max(1, Math.ceil((l.endBeat - l.anchorBeat) / bpb - 1e-6));
  }

  // ---- which lines are the same phrase (hook detection)
  const lineVec = (ln) => {
    const v = [];
    const a = ln.anchorBeat, nb = Math.max(1, Math.round(ln.endBeat - a));
    for (let k = 0; k < Math.min(nb, 16); k++) v.push(view.beatVec(a + k));
    return v;
  };
  const vecs = lines.map(lineVec);
  const sim = (i, j) => {
    const a = vecs[i], b = vecs[j];
    const m = Math.min(a.length, b.length);
    if (!m) return 0;
    let s = 0;
    for (let k = 0; k < m; k++) s += cos(a[k], b[k]);
    const sc = lines[i].syllables.length, sd = lines[j].syllables.length;
    const sylSim = 1 - Math.abs(sc - sd) / Math.max(sc, sd, 1);
    return (s / m) * Math.sqrt(m / Math.max(a.length, b.length)) * (0.75 + 0.25 * sylSim);
  };
  const thr = opt.hookSim ?? 0.86;
  const cluster = new Int32Array(lines.length).fill(-1);
  let nc = 0;
  for (let i = 0; i < lines.length; i++) {
    if (cluster[i] >= 0) continue;
    cluster[i] = nc;
    for (let j = i + 1; j < lines.length; j++) if (cluster[j] < 0 && sim(i, j) > thr) cluster[j] = nc;
    nc++;
  }
  const members = Array.from({ length: nc }, () => []);
  lines.forEach((l, i) => { l.cluster = cluster[i]; members[cluster[i]].push(i); });
  let hook = -1, bestScore = 0;
  members.forEach((ms, c) => {
    if (ms.length < 2) return;
    const en = ms.reduce((a, i) => a + lines[i].energy, 0) / ms.length;
    const sc = ms.length * (0.6 + en);
    if (sc > bestScore) { bestScore = sc; hook = c; }
  });
  lines.forEach((l) => (l.isHook = l.cluster === hook && hook >= 0));
  return { lines, syllables, hookCluster: hook, clusters: members, voicedThreshold: onT, barPhase: phase };
}
