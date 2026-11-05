// ReadHead engine — the chop writer.
//
// It writes read-offset events (the same thing you do by dragging a snapped clip)
// with three layers of judgement:
//   1. STRUCTURE  — phrases start back in sync ("gravity"), edits cluster at phrase
//                   tails, intensity follows section energy, each section gets an
//                   edit motif that comes back (with mutation) when the section does.
//   2. GESTURES   — repeat, stutter, roll (accelerating ratchet), skip, reverse,
//                   gate, displace, jump-to-similar. Every one is just offset events.
//   3. CONTENT    — candidates are scored on clean cut points (cut at onsets/gaps, not
//                   mid-vowel), keeping the stem audible, key fit against the backing
//                   as it has already been edited, and (for jumps) sounding alike.
import { makeRng, hashInts } from './rng.js';
import { cos } from './analysis.js';
import { makeEvent, normalizeLane } from './lane.js';

export const GESTURES = ['stutter', 'repeatBar', 'roll', 'skip', 'reverse', 'gate', 'displace', 'jump'];

export const GESTURE_INFO = {
  stutter:   { label: 'Stutter',     color: '#ff6b6b', desc: 'repeat a short chunk 2-4x' },
  repeatBar: { label: 'Repeat bar',  color: '#ffa94d', desc: 'replay the last bar' },
  roll:      { label: 'Roll',        color: '#ffd43b', desc: 'accelerating ratchet on one hit' },
  skip:      { label: 'Skip',        color: '#69db7c', desc: 'jump ahead' },
  reverse:   { label: 'Reverse',     color: '#4dabf7', desc: 'play a chunk backwards' },
  gate:      { label: 'Gate',        color: '#9775fa', desc: 'rhythmic mute pattern' },
  displace:  { label: 'Displace',    color: '#f783ac', desc: 'shift off the beat, then back' },
  jump:      { label: 'Jump',        color: '#3bc9db', desc: 'jump to a similar-sounding spot' },
  realign:   { label: 'Realign',     color: '#868e96', desc: 'back in sync' },
  line:      { label: 'Line',        color: '#e8e8e8', desc: 'a vocal line placed on a bar' },
  hook:      { label: 'Hook',        color: '#ff8787', desc: 'the most-repeated vocal line' },
  ratchet:   { label: 'Ratchet',     color: '#ffe066', desc: 'syllable repeats accelerating into a change' },
  sylStutter:{ label: 'Syllable stutter', color: '#ff922b', desc: 'one syllable replayed on the grid' },
  echo:      { label: 'Echo',        color: '#74c0fc', desc: 'last syllable repeating and fading' },
  gap:       { label: 'Gap',         color: '#495057', desc: 'vocal silent' },
  loop:      { label: 'Loop',        color: '#ffa94d', desc: 'a 1- or 2-bar vocal fragment repeated' },
  alternate: { label: 'Alternate',   color: '#f783ac', desc: 'two neighbouring bars swapped back and forth' },
  stammer:   { label: 'Stammer',     color: '#ff922b', desc: 'the phrase opening restarts once' },
  manual:    { label: 'Manual',      color: '#ffffff', desc: 'your edit' },
};

export const ROLE_WEIGHTS = {
  vocal: { stutter: 3, repeatBar: 2, jump: 2, roll: 1.2, skip: 1, reverse: 0.6, displace: 0.5, gate: 0.6 },
  drums: { roll: 3, stutter: 3, gate: 1.5, reverse: 1, repeatBar: 1, skip: 1, displace: 0.8, jump: 0.3 },
  bass:  { stutter: 2, repeatBar: 2, skip: 1, gate: 1, roll: 0.6, jump: 0.5, reverse: 0.3 },
  other: { repeatBar: 2, stutter: 2, jump: 1.5, reverse: 1, gate: 1, roll: 1, displace: 0.4, skip: 0.6 },
  mix:   { stutter: 3, roll: 2, repeatBar: 2, skip: 1, reverse: 0.8, gate: 1, jump: 1, displace: 0.5 },
};
const ROLE_DENSITY = { drums: 1.2, vocal: 0.8, bass: 0.6, other: 0.9, mix: 1.0 };
const ROLE_SEAM = { vocal: 1.0, other: 0.7, bass: 0.8, drums: 0.3, mix: 0.6 };
const ORDER = { drums: 0, bass: 1, other: 2, mix: 3, vocal: 4 };

export const DEFAULT_OPTS = {
  seed: 1,
  salt: 0,
  phraseBars: 4,
  minDiv: 0.25,          // smallest chop in beats (0.25 = 1/16 note)
  triplets: true,
  gravity: true,         // every phrase starts back in sync
  linkedFills: true,     // all stems do the same fill into a new section
  harmony: true,         // reject moves that put pitched material out of key with the backing
  motifReuse: 0.7,       // chance a repeated section reuses (and mutates) its edit motif
  pushChance: 0.35,      // chance a repeat/skip leaves the stem displaced instead of catching up
  maxPushBars: 2,
  gestures: Object.fromEntries(GESTURES.map((g) => [g, true])),
};

// ------------------------------------------------------------------ helpers
const clamp = (x, a, b) => Math.max(a, Math.min(b, x));

function buildPhrases(structure, P, bpb) {
  const { totalBars, phraseOffset, sections } = structure;
  const secAt = (bar) => sections.findIndex((s) => bar >= s.startBar && bar < s.endBar);
  const out = [];
  const starts = [];
  if (phraseOffset > 0) starts.push(0);
  for (let b = phraseOffset; b < totalBars; b += P) starts.push(b);
  // section boundaries are always phrase starts
  for (const s of sections) if (!starts.includes(s.startBar)) starts.push(s.startBar);
  starts.sort((a, b) => a - b);
  for (let i = 0; i < starts.length; i++) {
    const a = starts[i], b = i + 1 < starts.length ? starts[i + 1] : totalBars;
    if (b <= a) continue;
    const si = Math.max(0, secAt(a));
    out.push({ idx: out.length, startBar: a, endBar: b, bars: b - a, sectionIdx: si, section: sections[si] });
  }
  out.forEach((p, i) => {
    const nx = out[i + 1];
    p.sectionEnd = !nx || nx.sectionIdx !== p.sectionIdx;
    p.startBeat = p.startBar * bpb;
    p.endBeat = p.endBar * bpb;
  });
  return out;
}

function euclid(k, n, rot = 0) {
  const out = [];
  for (let i = 0; i < n; i++) out.push(((i + rot) * k) % n < k ? 1 : 0);
  return out;
}

// ------------------------------------------------------------------ planning
function lengthsFor(type, intensity, opts, bpb, rng) {
  const md = opts.minDiv;
  const T = opts.triplets;
  const pickL = (opts_) => rng.weighted(opts_.filter((o) => o.L >= md - 1e-9));
  switch (type) {
    case 'stutter': {
      const o = pickL([
        { L: 1, w: 2 }, { L: 0.5, w: 2 }, { L: 2, w: 0.8 }, { L: 0.25, w: 0.4 + intensity * 1.5 },
        ...(T ? [{ L: 1 / 3, w: 0.9 }, { L: 2 / 3, w: 0.6 }, { L: 1 / 6, w: intensity * 0.6 }] : []),
        { L: 0.125, w: intensity * 0.5 },
      ]) || { L: 1 };
      const maxReps = Math.max(2, Math.floor((o.L >= 2 ? 2 * bpb : bpb) / o.L));
      const reps = clamp(2 + rng.int(3), 2, Math.min(4, maxReps));
      return { L: o.L, reps, span: o.L * reps };
    }
    case 'repeatBar': return { L: bpb, span: bpb };
    case 'roll': {
      const span = rng.weighted([{ s: 1, w: 0.8 }, { s: 2, w: 1.5 }, { s: bpb, w: 0.3 + intensity }]).s;
      return { span, accel: rng.chance(0.8), triplet: T && rng.chance(0.3) };
    }
    case 'skip': {
      const L = rng.weighted([{ L: 1, w: 1 }, { L: 2, w: 1 }, { L: bpb, w: 1 }, { L: 0.5, w: intensity }].filter((x) => x.L >= md));
      return { L: L.L, span: L.L };
    }
    case 'reverse': {
      const s = rng.weighted([{ s: 0.5, w: 1 }, { s: 1, w: 1.5 }, { s: 2, w: 0.7 }].filter((x) => x.s >= md));
      return { span: s.s };
    }
    case 'gate': {
      const span = rng.weighted([{ s: 1, w: 1 }, { s: 2, w: 1.2 }, { s: bpb, w: 0.5 + intensity }]).s;
      const step = rng.weighted([{ s: 0.25, w: 2 }, { s: 0.5, w: 1 }, ...(T ? [{ s: 1 / 3, w: 0.8 }, { s: 1 / 6, w: 0.4 }] : [])].filter((x) => x.s >= md)).s;
      return { span, step };
    }
    case 'displace': {
      const d = rng.weighted([{ d: 0.5, w: 2 }, { d: 0.25, w: 1 }, { d: 0.75, w: 0.6 }, ...(T ? [{ d: 1 / 3, w: 1 }, { d: 2 / 3, w: 0.5 }] : [])].filter((x) => x.d >= md - 1e-9)).d;
      return { span: rng.chance(0.6) ? bpb : 2 * bpb, d };
    }
    case 'jump': {
      const span = rng.weighted([{ s: 2, w: 1 }, { s: bpb, w: 1.5 }, { s: 2 * bpb, w: 0.8 }]).s;
      return { span };
    }
  }
  return { span: 1 };
}

function placeGesture(g, phraseBars, bpb, rng) {
  // where in the phrase: tail-weighted, like a producer placing fills
  const bars = [];
  const ws = [];
  const spanBars = Math.ceil(g.span / bpb - 1e-9);
  for (let bar = 0; bar <= phraseBars - spanBars; bar++) {
    if (bar === 0 && g.span >= bpb) continue; // bar-long moves never sit on the phrase downbeat
    const last = bar + spanBars - 1;
    let w = 0.25;
    if (last === phraseBars - 1) w += 1.3;
    else if ((last + 1) % 4 === 0) w += 0.6;
    else if ((last + 1) % 2 === 0) w += 0.25;
    bars.push(bar); ws.push(w);
  }
  if (!bars.length) return null;
  const bar = rng.weighted(bars, ws);
  let at = bar * bpb;
  if (g.span < bpb) {
    // position inside the bar, on the gesture's own grid, later beats favoured
    const unit = g.span >= 2 ? 2 : g.span >= 1 ? 1 : Math.max(g.span, 0.5);
    const slots = [];
    const sw = [];
    for (let x = 0; x + g.span <= bpb + 1e-9; x += unit) {
      if (bar === 0 && x === 0) continue; // the phrase downbeat stays clean
      slots.push(x); sw.push(0.6 + (x / bpb) * 2);
    }
    if (!slots.length) return null;
    at += rng.weighted(slots, sw);
  } else if (bar === 0) {
    return null;
  }
  return at;
}

function newGesture(type, intensity, phraseBars, bpb, opts, rng) {
  const p = lengthsFor(type, intensity, opts, bpb, rng);
  const g = { type, ...p };
  if (type === 'repeatBar' || type === 'skip' || type === 'stutter') g.push = rng.chance(opts.pushChance * (type === 'stutter' ? 0.4 : 1));
  g.at = placeGesture(g, phraseBars, bpb, rng);
  return g.at == null ? null : g;
}

function overlaps(list, g) {
  return list.some((h) => g.at < h.at + h.span - 1e-9 && h.at < g.at + g.span - 1e-9);
}

function newPlan(role, intensity, phraseBars, bpb, opts, rng) {
  const weights = ROLE_WEIGHTS[role] || ROLE_WEIGHTS.mix;
  const types = GESTURES.filter((t) => opts.gestures[t] && weights[t]);
  if (!types.length) return [];
  const n = Math.min(phraseBars * 2, Math.floor(intensity * phraseBars * 0.9 + rng()));
  const plan = [];
  for (let tries = 0; plan.length < n && tries < n * 6; tries++) {
    const type = rng.weighted(types, types.map((t) => weights[t] * (t === 'roll' || t === 'reverse' ? 0.5 + intensity : 1)));
    const g = newGesture(type, intensity, phraseBars, bpb, opts, rng);
    if (g && !overlaps(plan, g)) plan.push(g);
  }
  return plan.sort((a, b) => a.at - b.at);
}

function mutatePlan(plan, role, intensity, phraseBars, bpb, opts, rng) {
  const out = plan.map((g) => ({ ...g }));
  if (!out.length) return newPlan(role, intensity, phraseBars, bpb, opts, rng);
  const r = rng();
  const i = rng.int(out.length);
  if (r < 0.45) {
    // same move, new length / repeat count
    const g = out[i];
    const p = lengthsFor(g.type, intensity, opts, bpb, rng);
    const ng = { ...g, ...p };
    const rest = out.filter((_, k) => k !== i);
    if (!overlaps(rest, ng) && ng.at + ng.span <= phraseBars * bpb + 1e-9) out[i] = ng;
  } else if (r < 0.65 && out.length > 1) {
    out.splice(i, 1);
  } else if (r < 0.85) {
    const weights = ROLE_WEIGHTS[role] || ROLE_WEIGHTS.mix;
    const types = GESTURES.filter((t) => opts.gestures[t] && weights[t]);
    const g = newGesture(rng.weighted(types, types.map((t) => weights[t])), intensity, phraseBars, bpb, opts, rng);
    if (g && !overlaps(out, g)) out.push(g);
  }
  return out.sort((a, b) => a.at - b.at);
}

// ------------------------------------------------------------------ realisation
// Turn one planned gesture into concrete events at absolute beat t, given the
// offset in effect (base). Returns candidate variants; scoring picks one.
function variants(g, t, base, ctx) {
  const { bpb, minDiv } = ctx;
  const E = (tt, off, rate = 1, gain = 1) => makeEvent(tt, off, rate, gain, { g: g.type, gen: true });
  const out = [];
  switch (g.type) {
    case 'stutter': {
      // A: repeat the chunk starting here.  B: repeat the chunk that just played.
      for (const prevChunk of [false, true]) {
        const ev = [];
        const k0 = prevChunk ? 0 : 1;
        for (let i = k0; i < g.reps; i++) ev.push(E(t + i * g.L, base + (i + (prevChunk ? 1 : 0)) * g.L));
        const pushOff = base + (g.reps - (prevChunk ? 0 : 1)) * g.L;
        const end = g.push ? pushOff : base;
        ev.push(E(t + g.span, end));
        out.push({ ev, newOff: end, v: prevChunk ? 'prev' : 'here' });
      }
      break;
    }
    case 'repeatBar': {
      const ev = [E(t, base + bpb)];
      if (!g.push) ev.push(E(t + bpb, base));
      out.push({ ev, newOff: g.push ? base + bpb : base });
      break;
    }
    case 'roll': {
      const S = g.span;
      const div = g.triplet ? 3 : 2;
      const segs = [];
      let rem = S, seg = S / 2;
      while (rem > 1e-9) {
        let L = seg / div;
        if (L < minDiv - 1e-9 || seg > rem + 1e-9) {
          // fill whatever is left with the finest allowed hits
          L = Math.max(minDiv, g.triplet ? minDiv * 2 / 3 : minDiv);
          const n = Math.max(1, Math.round(rem / L));
          segs.push({ len: rem, L: rem / n });
          break;
        }
        segs.push({ len: seg, L });
        rem -= seg;
        seg /= 2;
      }
      if (!g.accel) segs.reverse();
      // two anchors: ratchet the hit on this beat, or the hit half a beat earlier
      for (const back of [0, Math.max(minDiv, 0.5)]) {
        const a = t - base - back; // source beat being ratcheted
        const ev = [];
        let tau = t;
        for (const s of segs) {
          for (let x = 0; x < s.len - 1e-9; x += s.L) ev.push(E(tau + x, tau + x - a));
          tau += s.len;
        }
        ev.push(E(t + S, base));
        out.push({ ev, newOff: base, v: back ? 'early' : 'on' });
      }
      break;
    }
    case 'skip': {
      const ev = [E(t, base - g.L)];
      if (!g.push) ev.push(E(t + g.L, base));
      out.push({ ev, newOff: g.push ? base - g.L : base });
      break;
    }
    case 'reverse': {
      // A: rewind what just played.  B: play the upcoming chunk backwards.
      out.push({ ev: [E(t, base, -1), E(t + g.span, base, 1)], newOff: base, v: 'rewind' });
      out.push({ ev: [E(t, base - g.span, -1), E(t + g.span, base, 1)], newOff: base, v: 'ahead' });
      break;
    }
    case 'gate': {
      const n = Math.round(g.span / g.step);
      const pats = [];
      for (const k of [Math.ceil(n * 0.5), Math.ceil(n * 0.625), Math.ceil(n * 0.75)]) pats.push(euclid(k, n), euclid(k, n, 1));
      pats.push(Array.from({ length: n }, (_, i) => (i % 2 === 0 ? 1 : 0)));
      for (const pat of pats) {
        const ev = [];
        pat.forEach((on, i) => ev.push(E(t + i * g.step, base, 1, on)));
        ev.push(E(t + g.span, base, 1, 1));
        out.push({ ev, newOff: base, v: pat.join('') });
      }
      break;
    }
    case 'displace': {
      out.push({ ev: [E(t, base + g.d), E(t + g.span, base)], newOff: base, v: 'late' });
      out.push({ ev: [E(t, base - g.d), E(t + g.span, base)], newOff: base, v: 'early' });
      break;
    }
    case 'jump': {
      const nat = t - base;
      const { view } = ctx.stem;
      const lo = Math.max(view.bStart, 0), hi = view.bEnd - g.span;
      for (let p = nat - Math.floor((nat - lo) / bpb) * bpb; p <= hi; p += bpb) {
        if (Math.abs(p - nat) < 2 * bpb) continue;
        out.push({ ev: [E(t, t - p), E(t + g.span, base)], newOff: base, v: `@${p}`, jumpTo: p });
      }
      break;
    }
  }
  return out;
}

function srcFn(ev, base) {
  // events sorted, all inside the gesture; before the first event the base offset applies
  return (tau) => {
    let e = null;
    for (const x of ev) { if (x.t <= tau + 1e-9) e = x; else break; }
    if (!e) return { s: tau - base, gain: 1 };
    return { s: e.t - e.off + e.rate * (tau - e.t), gain: e.gain };
  };
}

function scoreVariant(v, g, t, base, ctx) {
  const { stem, role, backing, opts, chaos } = ctx;
  const view = stem.view;
  const span = g.span;
  const f = srcFn(v.ev, base);
  const step = 0.25;
  let eNat = 0, eNew = 0, n = 0;
  for (let tau = t; tau < t + span - 1e-9; tau += step) {
    eNat += view.rms(tau - base);
    const r = f(tau);
    eNew += r.gain * view.rms(r.s);
    n++;
  }
  eNat /= n || 1; eNew /= n || 1;
  let score = 0;
  const pitched = role !== 'drums';
  // pointless: stuttering / gating silence
  if (eNew < 0.08 && role !== 'drums' && g.type !== 'gate') return -10;
  if (pitched && eNat > 0.25 && eNew < 0.35 * eNat && g.type !== 'gate') score -= 2;
  score += 0.5 * Math.min(1, eNew);

  // seams: cutting INTO a sustained sound is worst, cutting one OFF is next
  const sustain = (s) => view.rms(s) * (1 - Math.min(1, view.onset(s)));
  let seam = 0;
  const pts = v.ev.map((e) => e.t);
  for (const tau of pts) {
    const before = f(tau - 1e-6), after = f(tau);
    if (Math.abs(before.s - after.s) < 1e-6 && before.gain === after.gain) continue;
    seam += 0.5 * sustain(before.s) * before.gain + 1.0 * sustain(after.s) * after.gain;
  }
  seam /= Math.max(1, Math.sqrt(pts.length));
  score -= (ROLE_SEAM[role] ?? 0.6) * seam;

  // key fit against the backing as it has already been edited
  if (opts.harmony && pitched && role !== 'mix' && backing) {
    let fitN = 0, fitE = 0, w = 0;
    for (let tau = t; tau < t + span - 1e-9; tau += 0.5) {
      const bk = backing(tau);
      if (!bk) continue;
      const cn = view.chroma(tau - base), r = f(tau), ce = view.chroma(r.s);
      const wn = view.rms(tau - base), we = view.rms(r.s) * r.gain;
      if (cn) fitN += wn * cos(cn, bk);
      if (ce) fitE += we * cos(ce, bk);
      w += Math.max(wn, we);
    }
    if (w > 0) {
      const d = (fitE - fitN) / w;
      if (d < -(0.12 + 0.25 * chaos) && !g.linked) return -10;
      score += 1.2 * d;
    }
  }
  // jumps should land somewhere that sounds alike
  if (g.type === 'jump') {
    let sim = 0, k = 0;
    for (let b = 0; b < span; b += 1) {
      sim += cos(view.beatVec(t - base + b), view.beatVec(v.jumpTo + b));
      k++;
    }
    sim /= k || 1;
    if (sim < 0.55) return -10;
    score += 2 * sim;
  }
  return score;
}

// ------------------------------------------------------------------ main
/**
 * @param ctx  { grid, structure, stems: [{id, role, view, chaos (0..1), enabled}] }
 * @param opts DEFAULT_OPTS overrides
 * @param prev previous lanes {id: events} (kept for disabled stems / outside range)
 * @param range optional {startBar, endBar} to reroll only those phrases
 * @returns {lanes, report}
 */
export function generate(ctx, optsIn = {}, prev = {}, range = null) {
  const opts = { ...DEFAULT_OPTS, ...optsIn, gestures: { ...DEFAULT_OPTS.gestures, ...(optsIn.gestures || {}) } };
  const { grid, structure } = ctx;
  const bpb = grid.bpb;
  const P = opts.phraseBars || structure.phraseBars;
  const phrases = buildPhrases(structure, P, bpb);
  const maxE = Math.max(1e-6, ...structure.sections.map((s) => s.energy));
  const inRange = (ph) => !range || (ph.endBar > range.startBar && ph.startBar < range.endBar);

  const stems = ctx.stems.slice().sort((a, b) => (ORDER[a.role] ?? 3) - (ORDER[b.role] ?? 3));
  const meanChaos = stems.filter((s) => s.enabled).reduce((a, s) => a + s.chaos, 0) / Math.max(1, stems.filter((s) => s.enabled).length);

  // shared fills into new sections: every stem does the same rhythm at once
  const fills = {};
  if (opts.linkedFills) {
    for (const ph of phrases) {
      if (!ph.sectionEnd || ph.bars < 2) continue;
      const rng = makeRng(hashInts(opts.seed, 'fill', ph.idx, inRange(ph) && range ? opts.salt : 0));
      if (!rng.chance(0.35 + 0.6 * meanChaos)) continue;
      const type = rng.weighted(['roll', 'stutter', 'gate', 'reverse'].filter((t) => opts.gestures[t]), [3, 2, 1, 1].filter((_, i) => opts.gestures[['roll', 'stutter', 'gate', 'reverse'][i]]));
      if (!type) continue;
      const p = lengthsFor(type, meanChaos, opts, bpb, rng);
      const span = type === 'stutter' ? p.span : Math.min(p.span, rng.chance(0.5) ? 2 : 1 + (meanChaos > 0.6 ? 1 : 0));
      fills[ph.idx] = { type, ...p, span, push: false, linked: true, at: ph.bars * bpb - span };
    }
  }

  const lanes = {};
  const report = { phrases: phrases.length, fills: Object.keys(fills).length, counts: {} };
  const done = [];
  for (const stem of stems) {
    const role = stem.role || 'mix';
    const prevLane = prev[stem.id] || [];
    if (!stem.enabled) { lanes[stem.id] = prevLane.map((e) => ({ ...e })); done.push(stem); continue; }
    // the backing this stem must stay in key with = pitched stems already written
    const harmonicStems = done.filter((s) => s.role === 'bass' || s.role === 'other');
    const backing = harmonicStems.length
      ? (tau) => {
        const acc = new Float64Array(12);
        let any = false;
        for (const h of harmonicStems) {
          const s = srcAtLane(lanes[h.id], tau);
          const c = h.view.chroma(s.s);
          if (!c || !s.gain) continue;
          const w = h.view.rms(s.s);
          for (let p = 0; p < 12; p++) acc[p] += c[p] * w;
          any = true;
        }
        return any ? acc : null;
      }
      : null;
    const events = [];
    const motifs = new Map();
    let carryOff = 0;
    const counts = {};
    for (const ph of phrases) {
      if (!inRange(ph)) {
        if (opts.gravity) events.push(makeEvent(ph.startBeat, 0, 1, 1, { g: 'realign', gen: true }));
        for (const e of prevLane) if (e.t >= ph.startBeat - 1e-9 && e.t < ph.endBeat - 1e-9) events.push({ ...e });
        carryOff = srcAtLane(prevLane, ph.endBeat - 1e-6).s;
        carryOff = ph.endBeat - 1e-6 - carryOff;
        continue;
      }
      const rng = makeRng(hashInts(opts.seed, stem.id, ph.idx, range ? opts.salt : 0));
      const energy = (ph.section?.energy ?? 1) / maxE;
      const intensity = clamp(stem.chaos * (ROLE_DENSITY[role] ?? 1) * (0.3 + 0.7 * energy), 0, 1.5);
      const key = ph.section?.label ?? 'A';
      let plan;
      if (stem.chaos <= 0) plan = [];
      else if (!range && motifs.has(key) && motifs.get(key).bars === ph.bars && rng.chance(opts.motifReuse)) {
        // the motif evolves phrase by phrase (A, A', A'', A''') and the original
        // comes back every 4th time, like a hook returning
        const m = motifs.get(key);
        m.count++;
        plan = m.count % 4 === 0 ? m.orig.map((g) => ({ ...g })) : mutatePlan(m.last, role, intensity, ph.bars, bpb, opts, rng);
        m.last = plan;
      } else {
        plan = newPlan(role, intensity, ph.bars, bpb, opts, rng);
        if (!motifs.has(key)) motifs.set(key, { orig: plan, last: plan, bars: ph.bars, count: 0 });
      }
      if (fills[ph.idx] && stem.chaos > 0) {
        const fl = fills[ph.idx];
        plan = plan.filter((g) => !overlaps([fl], g)).concat([{ ...fl }]).sort((a, b) => a.at - b.at);
      }
      let base = opts.gravity ? 0 : carryOff;
      if (opts.gravity) events.push(makeEvent(ph.startBeat, 0, 1, 1, { g: 'realign', gen: true }));
      const gctx = { bpb, minDiv: opts.minDiv, stem, role, backing, opts, chaos: stem.chaos };
      const placed = [];
      for (let gi = 0; gi < plan.length; gi++) {
        const g = plan[gi];
        const nextAt = gi + 1 < plan.length ? plan[gi + 1].at : ph.bars * bpb;
        const prevEnd = placed.length ? placed[placed.length - 1].at + placed[placed.length - 1].span : 0;
        // nudge candidates: the planned spot first, then neighbours on the gesture's own grid
        const unit = g.linked ? 0 : g.span >= bpb ? bpb : g.span >= 1 ? 1 : 0.5;
        const spots = unit ? [g.at, g.at - unit, g.at + unit, g.at - 2 * unit, g.at + 2 * unit] : [g.at];
        let best = null, bestAt = g.at;
        for (const at of spots) {
          if (at < (unit ? 0.5 : 0) || at < prevEnd - 1e-9 || at + g.span > nextAt + 1e-9 || at + g.span > ph.bars * bpb + 1e-9) continue;
          if (at < 1e-9 && !g.linked) continue;
          const t = ph.startBeat + at;
          const movePenalty = Math.abs(at - g.at) / Math.max(1, unit) * 0.12;
          for (const v of variants(g, t, base, gctx)) {
            v.score = scoreVariant(v, g, t, base, gctx) - movePenalty + rng() * 0.05;
            if (!best || v.score > best.score) { best = v; bestAt = at; }
          }
        }
        if (!best || best.score <= -5) continue;
        const t = ph.startBeat + bestAt;
        placed.push({ at: bestAt, span: g.span });
        let newOff = best.newOff;
        if (Math.abs(newOff) > opts.maxPushBars * bpb) {
          // too far out: catch up at the end of the gesture instead
          best.ev.push(makeEvent(t + g.span, base, 1, 1, { g: g.type, gen: true }));
          newOff = base;
        }
        for (const e of best.ev) events.push(e);
        base = newOff;
        counts[g.type] = (counts[g.type] || 0) + 1;
      }
      carryOff = base;
    }
    lanes[stem.id] = normalizeLane(events);
    report.counts[stem.id] = counts;
    done.push(stem);
  }
  return { lanes, report, phrases };
}

function srcAtLane(lane, tau) {
  let e = null;
  for (const x of lane || []) { if (x.t <= tau + 1e-9) e = x; else break; }
  if (!e) return { s: tau, gain: 1 };
  return { s: e.t - e.off + e.rate * (tau - e.t), gain: e.gain };
}

export { buildPhrases };
