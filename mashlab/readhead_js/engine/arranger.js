// ReadHead engine — Arrange mode.
//
// Places vocal LINES and SYLLABLE chops into the instrumental's structure on
// purpose, instead of rolling dice per phrase:
//   breakdown (drums out)  -> vocal drops out too, or one sparse fragment
//   build (into a change)  -> a line, then a syllable ratchet accelerating into the change
//   drop / chorus          -> the hook, repeated to fill, with a syllable stutter lead-in
//   verse                  -> the next lines IN ORDER (the words keep making sense)
//   intro / outro          -> tease / hook + echo tail
// Lines keep their own rhythm and pickups (shifted by whole bars); a line that
// overruns its slot is cut at a syllable boundary, never mid-word; every syllable
// chop starts exactly on the syllable's onset so it lands on the beat.
import { makeEvent, normalizeLane, srcAt } from './lane.js';
import { makeRng, hashInts } from './rng.js';
import { snapFloor, snapCeil } from './grid.js';
import { arrangeFlow } from './flow.js';

const median = (a) => { const s = a.slice().sort((x, y) => x - y); return s.length ? s[s.length >> 1] : 0; };
const quant = (a, p) => { const s = a.slice().sort((x, y) => x - y); return s.length ? s[Math.min(s.length - 1, Math.floor(p * (s.length - 1)))] : 0; };

export const ROLE_INFO = {
  intro:     { label: 'intro',     color: '#7d8590' },
  verse:     { label: 'verse',     color: '#6fbf73' },
  build:     { label: 'build',     color: '#ffd43b' },
  drop:      { label: 'drop',      color: '#ff6b6b' },
  chorus:    { label: 'chorus',    color: '#ffa94d' },
  breakdown: { label: 'breakdown', color: '#4dabf7' },
  outro:     { label: 'outro',     color: '#7d8590' },
};

/**
 * What each phrase of the INSTRUMENTAL is doing, from the backing stems only.
 * @param backing [{role, view}]
 */
export function phraseRoles(phrases, backing, bpb) {
  const drums = backing.filter((s) => s.role === 'drums');
  const perPhrase = phrases.map((ph) => {
    let e = 0, d = 0, rise = 0;
    const bars = [];
    for (let bar = ph.startBar; bar < ph.endBar; bar++) {
      let eb = 0;
      for (const s of backing) eb += s.view.rmsSpan(bar * bpb, (bar + 1) * bpb);
      bars.push(eb);
      e += eb;
      if (drums.length) for (const s of drums) d += s.view.rmsSpan(bar * bpb, (bar + 1) * bpb);
      else {
        // no drum stem: rhythmic density of the backing stands in for "drums present"
        for (const s of backing) { let o = 0; for (let b = bar * bpb; b < (bar + 1) * bpb; b += 0.25) o += s.view.onset(b); d += o / (bpb * 4); }
      }
    }
    const nb = Math.max(1, ph.endBar - ph.startBar);
    if (bars.length >= 2) rise = (bars[bars.length - 1] - bars[0]) / (Math.max(...bars) + 1e-9);
    return { e: e / nb, d: d / nb, rise };
  });
  const es = perPhrase.map((p) => p.e), ds = perPhrase.map((p) => p.d);
  const eMax = Math.max(1e-9, ...es), dMax = Math.max(1e-9, ...ds);
  perPhrase.forEach((p) => { p.e /= eMax; p.d /= dMax; });
  const active = perPhrase.filter((p) => p.e > 0.08);
  const medE = median(active.map((p) => p.e)) || 0.5, medD = median(active.map((p) => p.d)) || 0.5;
  const hiE = quant(active.map((p) => p.e), 0.75);
  const firstFull = perPhrase.findIndex((p) => p.e >= 0.6 * medE && p.d >= 0.5 * medD);
  let lastFull = -1;
  perPhrase.forEach((p, i) => { if (p.e >= 0.6 * medE && p.d >= 0.5 * medD) lastFull = i; });
  const roles = perPhrase.map((p, i) => {
    if (firstFull > 0 && i < firstFull) return 'intro';
    if (lastFull >= 0 && i > lastFull) return 'outro';
    if (p.d < 0.45 * medD || p.e < 0.5 * medE) return 'breakdown';
    const prev = perPhrase[i - 1];
    const newSection = i > 0 && phrases[i].sectionIdx !== phrases[i - 1].sectionIdx;
    const prevBreak = prev && (prev.d < 0.45 * medD || prev.e < 0.5 * medE);
    // drums coming back after a breakdown is a drop; so is a clearly louder new section
    if (prevBreak && p.e >= 0.8 * medE) return 'drop';
    if (prev && newSection && p.e >= Math.max(medE * 1.05, prev.e + 0.1)) return 'drop';
    return p.e >= hiE ? 'chorus' : 'verse';
  });
  // a build is the phrase leading INTO a drop (or into a clearly louder section)
  for (let i = 0; i < roles.length - 1; i++) {
    if ((roles[i] === 'verse' || roles[i] === 'chorus') && (roles[i + 1] === 'drop' || (phrases[i + 1].sectionIdx !== phrases[i].sectionIdx && perPhrase[i + 1].e > perPhrase[i].e + 0.15) || perPhrase[i].rise > 0.35)) roles[i] = 'build';
  }
  return roles.map((r, i) => ({ role: r, ...perPhrase[i] }));
}

export const ARRANGE_DEFAULTS = {
  seed: 1,
  salt: 0,
  minDiv: 0.25,
  triplets: true,
  space: 0.35,          // how much room to leave (verse slots ending early, sparser breakdowns)
  hookOnDrops: true,
  keepOrder: true,
  breakdown: 'auto',    // 'silent' | 'sparse' | 'auto'   (lines style)
  style: 'flow',        // 'flow' = your Re-Tex way (read forward, loops, ratchets) | 'lines' = place whole lines per section
  tailBeats: 0.25,      // let a line's last syllable ring this long before the gap
};

/**
 * @param ctx {grid, phrases, roles, vocal: {lines, syllables}, chop (0..1)}
 * @returns {events, plan}
 */
export function arrangeVocal(ctx, optsIn = {}) {
  const o = { ...ARRANGE_DEFAULTS, ...optsIn };
  const { grid, phrases, roles, vocal } = ctx;
  const chop = ctx.chop ?? 0.5;
  const bpb = grid.bpb, md = o.minDiv;
  const lines = vocal.lines.filter((l) => l.syllables.length);
  const ev = [];
  const plan = [];
  if (!lines.length) return { events: [], plan };
  const hooks = lines.filter((l) => l.isHook);
  const hookSet = hooks.length ? hooks : [lines.reduce((a, b) => (b.energy * b.syllables.length > a.energy * a.syllables.length ? b : a))];
  const verses = o.keepOrder ? lines.filter((l) => !l.isHook || !hooks.length) : lines.slice();
  if (!verses.length) verses.push(...lines);
  let vCur = 0, hCur = 0;
  let contentEnd = -Infinity; // timeline beat where the last placed audio ends
  let lastLine = null;

  const E = (t, off, gain, g, rate = 1) => ev.push(makeEvent(t, off, rate, gain, { g, gen: true }));
  const silence = (t, g = 'gap') => E(t, 0, 0, g);

  /** place a line so its anchor bar lands on targetBar; cut at a syllable boundary before `limit` */
  function placeLine(line, targetBar, limit, tag, { allowPickup = true } = {}) {
    const off = targetBar * bpb - line.anchorBeat;
    let t0 = snapFloor(line.startBeat + off - 0.06, md);
    if (!allowPickup || t0 < contentEnd - 1e-9) t0 = Math.max(targetBar * bpb, snapCeil(contentEnd, md));
    if (t0 >= limit - md) return null;
    // the line plays syllables until one would start too late to finish in the slot
    const syl = line.syllables;
    let endT = snapCeil(line.endBeat + off + o.tailBeats, md);
    let cutAt = null;
    for (let k = 0; k < syl.length; k++) {
      const st = syl[k].beat + off;
      if (st < t0 - 1e-6) continue;
      if (st > limit - 0.5) { cutAt = k; break; }
    }
    if (cutAt != null) endT = Math.min(endT, snapFloor(syl[cutAt].beat + off - 0.02, md));
    endT = Math.min(endT, limit);
    if (endT <= t0 + md) return null;
    // a pickup reaches back over the slot's opening silence: remove that silence
    for (let i = ev.length - 1; i >= 0; i--) if (ev[i].gain === 0 && ev[i].t > t0 + 1e-9 && ev[i].t <= targetBar * bpb + 1e-9 && ev[i].g === 'gap') ev.splice(i, 1);
    E(t0, off, 1, tag);
    silence(endT);
    contentEnd = endT;
    lastLine = { line, off, endT, cutAt };
    return { t0, endT, off };
  }

  /** a syllable replayed at each time in `times`; each hit is the syllable's first L beats */
  function sylHits(sy, times, L, tag, gains) {
    const audible = Math.min(L, snapCeil(Math.max(md, sy.endBeat - sy.beat), md));
    times.forEach((t, i) => {
      E(t, t - sy.beat, gains ? gains[i] : 1, tag);
      if (audible < L - 1e-9) silence(t + audible, tag);
    });
    const end = times[times.length - 1] + L;
    silence(end, tag);
    contentEnd = Math.max(contentEnd, end);
    return end;
  }

  /** accelerating ratchet on one syllable over [a, b): 1 → 1/2 → 1/4 … beats, into the change */
  function ratchet(sy, a, b, rng) {
    const R = b - a;
    if (R < md * 2) return;
    const trip = o.triplets && rng.chance(0.3);
    const floorL = trip ? Math.max(1 / 6, md * 2 / 3) : md;
    const times = [], lens = [];
    let t = a, seg = R / 2, L = (R >= 4 ? 1 : 0.5) * (trip ? 2 / 3 : 1);
    while (t < b - 1e-9) {
      if (L < floorL - 1e-9) { L = floorL; seg = b - t; }
      const end = Math.min(b, t + Math.max(seg, L));
      for (let x = t; x < end - 1e-9; x += L) { times.push(x); lens.push(Math.min(L, b - x)); }
      t = end; seg /= 2; L /= 2;
    }
    const breath = rng.chance(0.6) && R >= 2; // the last half beat before the change goes silent
    const sylLen = Math.max(md, sy.endBeat - sy.beat);
    times.forEach((x, i) => {
      if (breath && x >= b - 0.5 - 1e-9) return;
      E(x, x - sy.beat, 1, 'ratchet');
      if (sylLen < lens[i] - 1e-9) silence(x + Math.min(lens[i], snapCeil(sylLen, md)), 'ratchet');
    });
    if (breath) silence(b - 0.5, 'ratchet');
    silence(b, 'ratchet');
    contentEnd = b;
  }

  const firstStrong = (line) => line.syllables.reduce((a, s) => (s.strength > a.strength ? s : a), line.syllables[0]);
  const lastSyl = (line, cutAt) => line.syllables[Math.max(0, (cutAt ?? line.syllables.length) - 1)];

  phrases.forEach((ph, pi) => {
    const rng = makeRng(hashInts(o.seed, 'arr', pi, ctx.range && pi >= ctx.range.a && pi < ctx.range.b ? o.salt : 0));
    const role = roles[pi]?.role ?? 'verse';
    const S = ph.startBeat, Eb = ph.endBeat, bars = ph.endBar - ph.startBar;
    if (ctx.range && (pi < ctx.range.a || pi >= ctx.range.b)) {
      // keep what was there (caller merges)
      plan.push({ pi, role, kept: true });
      return;
    }
    silence(S, 'gap');
    if (contentEnd > S) contentEnd = S; // a slot owns its own time
    if (ctx.range && pi === ctx.range.a) contentEnd = S; // a reroll never reaches back before its range
    const actions = [];
    const nextVerse = () => { const l = verses[vCur % verses.length]; vCur++; return l; };
    const nextHook = () => { const l = hookSet[hCur % hookSet.length]; hCur++; return l; };
    const fillLines = (pick, fromBar, toBeat, tag, leadIn) => {
      let bar = fromBar, guard = 0, first = true;
      while (bar * bpb < toBeat - bpb / 2 && guard++ < 16) {
        const line = pick();
        // stutter lead-in on the first syllable ("f-f-fragmented"), in the beat before the line
        if (leadIn && !first && chop > 0.25 && rng.chance(0.25 + 0.5 * chop)) {
          const sy = line.syllables[0];
          const L = rng.chance(0.5) ? 0.25 : o.triplets && rng.chance(0.4) ? 1 / 3 : 0.5;
          const n = Math.round(1 / L);
          const t0 = bar * bpb - 1;
          if (t0 >= contentEnd - 1e-9) { sylHits(sy, Array.from({ length: n }, (_, i) => t0 + i * L), L, 'sylStutter'); actions.push('stutter lead-in'); }
        }
        const r = placeLine(line, bar, toBeat, tag);
        if (!r) break;
        actions.push(`${tag === 'hook' ? 'hook' : 'line'} ${line.idx + 1}`);
        first = false;
        // echo the last syllable into the gap after the line
        const room = (Math.ceil((r.endT + 0.01) / bpb) * bpb) - r.endT;
        if (chop > 0.2 && room >= 1 && rng.chance(0.2 + 0.5 * chop)) {
          const sy = lastSyl(line, lastLine.cutAt);
          const L = rng.chance(0.5) ? 0.5 : 0.25;
          const n = Math.min(3, Math.floor(room / L));
          if (n >= 2) {
            const ts = Array.from({ length: n }, (_, i) => snapCeil(r.endT, L) + i * L).filter((t) => t + L <= toBeat + 1e-9);
            if (ts.length >= 2) { sylHits(sy, ts, L, 'echo', ts.map((_, i) => Math.pow(0.6, i + 1))); actions.push('echo'); }
          }
        }
        bar = Math.ceil((contentEnd + 0.01) / bpb);
        // breathing room between lines in sparse settings
        if (rng.chance(o.space * 0.5)) bar++;
      }
    };

    switch (role) {
      case 'intro': {
        if (chop > 0.3 && bars >= 2 && rng.chance(0.5)) {
          const sy = firstStrong(hookSet[0]);
          ratchet(sy, Eb - Math.min(bpb, 4), Eb, rng);
          actions.push('hook tease');
        } else actions.push('instrumental');
        break;
      }
      case 'breakdown': {
        const sparse = o.breakdown === 'sparse' || (o.breakdown === 'auto' && rng.chance(0.15 + 0.4 * chop * (1 - o.space)));
        if (sparse && lastLine) {
          // one dry fragment: the last word we heard, repeated, then silence
          const sy = lastSyl(lastLine.line, lastLine.cutAt);
          const at = S + Math.min(bars - 1, 1 + (bars > 2 ? rng.int(2) : 0)) * bpb;
          const L = rng.chance(0.5) ? 1 : 0.5;
          sylHits(sy, [at, at + L, at + 2 * L], L, 'echo', [1, 0.6, 0.35]);
          actions.push('sparse fragment');
        } else actions.push('vocal out');
        break;
      }
      case 'build': {
        const half = S + Math.max(1, Math.floor(bars / 2)) * bpb;
        fillLines(nextVerse, ph.startBar, half, 'line', false);
        const sy = firstStrong(rng.chance(0.6) ? hookSet[0] : (lastLine ? lastLine.line : hookSet[0]));
        ratchet(sy, Math.max(half, snapCeil(contentEnd, md)), Eb, rng);
        actions.push('ratchet into change');
        break;
      }
      case 'drop':
      case 'chorus': {
        const useHook = o.hookOnDrops && (role === 'drop' || rng.chance(0.5));
        fillLines(useHook ? nextHook : nextVerse, ph.startBar, Eb, useHook ? 'hook' : 'line', true);
        break;
      }
      case 'outro': {
        if (rng.chance(0.6)) {
          fillLines(nextHook, ph.startBar, Math.min(Eb, S + 2 * bpb), 'hook', false);
          if (lastLine) {
            const sy = lastSyl(lastLine.line, lastLine.cutAt);
            const ts = [0, 1, 2, 3].map((i) => snapCeil(contentEnd, 1) + i).filter((t) => t + 1 <= Eb);
            if (ts.length >= 2) sylHits(sy, ts, 1, 'echo', ts.map((_, i) => Math.pow(0.55, i + 1)));
          }
          actions.push('fade-out echoes');
        } else actions.push('instrumental');
        break;
      }
      default: {
        // verse: lines in order; with space, let the last bar breathe
        const until = rng.chance(o.space) && bars >= 4 ? Eb - bpb : Eb;
        fillLines(nextVerse, ph.startBar, until, 'line', true);
        // occasional syllable stutter inside a verse line, on a strong syllable
        if (chop > 0.35 && lastLine && rng.chance(chop * 0.6)) {
          const { line, off } = lastLine;
          const cands = line.syllables.filter((s) => s.beat + off >= S + 1 && s.beat + off < lastLine.endT - 1 && s.endBeat - s.beat >= 0.2);
          if (cands.length) {
            const sy = cands[rng.int(cands.length)];
            const at = snapFloor(sy.beat + off + 0.03, md); // grid slot where this syllable falls
            const L = sy.endBeat - sy.beat >= 0.5 ? 0.5 : 0.25;
            // replay the syllable twice in place, then resume the line exactly where it would be
            E(at, at - sy.beat, 1, 'sylStutter');
            E(at + L, at + L - sy.beat, 1, 'sylStutter');
            E(at + 2 * L, off, 1, 'line');
            actions.push('syllable stutter');
          }
        }
      }
    }
    plan.push({ pi, role, startBar: ph.startBar, endBar: ph.endBar, actions });
  });
  return { events: ev, plan };
}

/**
 * Arrange every vocal stem. Returns lanes for vocal stems only.
 * @param ctx {grid, phrases, stems:[{id, role, view, vocal, chaos, enabled}]}
 */
export function arrange(ctx, opts = {}, prev = {}, rangePhrases = null) {
  const bpb = ctx.grid.bpb;
  const backing = ctx.stems.filter((s) => s.role !== 'vocal');
  const roles = phraseRoles(ctx.phrases, backing.length ? backing : ctx.stems, bpb);
  const lanes = {};
  const plans = {};
  for (const s of ctx.stems) {
    if (s.role !== 'vocal' || !s.enabled || !s.vocal) continue;
    const range = rangePhrases;
    const o2 = { ...ARRANGE_DEFAULTS, ...opts, seed: hashInts(opts.seed ?? 1, s.id) };
    const fn = o2.style === 'lines' ? arrangeVocal : arrangeFlow;
    const { events, plan } = fn({ grid: ctx.grid, phrases: ctx.phrases, roles, vocal: s.vocal, chop: s.chaos, range }, o2);
    let all = events;
    if (range) {
      // keep the previous lane outside the rerolled phrases
      const a = ctx.phrases[range.a].startBeat, b = ctx.phrases[range.b - 1].endBeat;
      const keep = (prev[s.id] || []).filter((e) => e.t < a - 1e-9 || e.t >= b - 1e-9);
      const inside = events.filter((e) => e.t >= a - 1e-9 && e.t < b - 1e-9);
      const hasB = keep.some((e) => Math.abs(e.t - b) < 1e-6);
      // whatever the old lane was playing at b (e.g. the next line, entered via a pickup) resumes at b
      const old = prev[s.id] || [];
      const stB = old.filter((e) => e.t <= b + 1e-9).pop();
      const restore = stB ? makeEvent(b, b - srcAt(old, b), stB.rate, stB.gain, { g: stB.g, gen: true }) : makeEvent(b, 0, 1, 1, { g: 'realign', gen: true });
      all = keep.filter((e) => e.t < a - 1e-9).concat(inside, hasB ? [] : [restore], keep.filter((e) => e.t >= b - 1e-9));
    }
    lanes[s.id] = normalizeLane(all);
    plans[s.id] = plan;
  }
  return { lanes, roles, plans };
}
