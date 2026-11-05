// ReadHead engine — "Flow" arrangement: the author's own way of arranging a vocal in Re-Tex,
// learned from one of his hand-made projects (62 hand-placed clips), not invented.
//
// What his arrangement does, and what this reproduces:
//   - the vocal mostly READS FORWARD through the song (a cursor), keeping its own bar phase
//     (every move is a whole number of the SINGER's bars, so downbeats stay downbeats)
//   - every cut starts on a syllable onset that sits on a beat
//   - devices, each a few bars long:
//       play        the vocal carries on
//       loop        a 1- or 2-bar fragment repeated 2-4x (his 244/252<-194, 311..335<-249)
//       alternate   two neighbouring bars swapped back and forth (his 220..236<-174/170)
//       stammer     the phrase opening restarts once, then carries on (his 132/136<-66)
//       ratchet     a 1-bar motif repeated each bar: syllable x4 in 16ths, the syllable for a
//                   beat, the 2 beats before it — then release into the line (his 167..180<-142)
//       hook loop   the most-repeated line, looped on drops / choruses
//       sparse      one bar, then room (breakdowns)
//   - a loop's audible length stops before a syllable it would chop (his "len 6.75 of 8")
//   - the vocal enters over the end of a long instrumental intro, not at bar 1
import { makeEvent } from './lane.js';
import { makeRng, hashInts } from './rng.js';
import { snapFloor } from './grid.js';

export const DEVICE_INFO = {
  play:      { label: 'play',      color: '#e8e8e8' },
  loop:      { label: 'loop',      color: '#ffa94d' },
  alternate: { label: 'alternate', color: '#f783ac' },
  stammer:   { label: 'stammer',   color: '#ff922b' },
  ratchet:   { label: 'ratchet',   color: '#ffe066' },
  hook:      { label: 'hook loop', color: '#ff8787' },
  sparse:    { label: 'sparse',    color: '#74c0fc' },
};

export function arrangeFlow(ctx, o) {
  const { grid, phrases, roles, vocal } = ctx;
  const chop = ctx.chop ?? 0.5;
  const bpb = grid.bpb, md = o.minDiv ?? 0.25;
  const tail = o.tailBeats ?? 0.25;
  const ph0 = vocal.barPhase || 0;
  const lines = vocal.lines.filter((l) => l.syllables.length);
  const ev = [], plan = [];
  if (!lines.length) return { events: ev, plan };
  const syl = vocal.syllables.slice().sort((a, b) => a.beat - b.beat);
  const srcFirst = Math.min(...lines.map((l) => l.anchorBeat));
  const srcEnd = Math.max(...lines.map((l) => l.endBeat));
  const E = (t, off, gain, g) => ev.push(makeEvent(t, off, 1, gain, { g, gen: true }));
  const gap = (t) => E(t, 0, 0, 'gap');

  // ---------------------------------------------------------- source helpers
  const voicedFrac = (a, b) => {
    let v = 0;
    for (const l of lines) v += Math.max(0, Math.min(b, l.endBeat) - Math.max(a, l.startBeat));
    return v / Math.max(1e-9, b - a);
  };
  const lowerIdx = (x) => { let lo = 0, hi = syl.length; while (lo < hi) { const m = (lo + hi) >> 1; if (syl[m].beat < x) lo = m + 1; else hi = m; } return lo; };
  const onsetNear = (x, tol = 0.2) => {
    const i = lowerIdx(x);
    let best = null;
    for (const j of [i - 1, i]) if (syl[j] && Math.abs(syl[j].beat - x) <= tol && (!best || Math.abs(syl[j].beat - x) < Math.abs(best.beat - x))) best = syl[j];
    return best;
  };
  const srcBar = (x) => ph0 + Math.round((x - ph0) / bpb) * bpb;      // nearest singer's bar line
  const lineAt = (x) => lines.find((l) => x >= l.startBeat - 1.6 && x < l.endBeat + 0.01);
  /** audible length of a fragment [a, a+L): end at the line's end, or before a syllable the cut would chop */
  function audible(a, L, toLineEnd = true) {
    let end = a + L;
    const ln = toLineEnd ? lineAt(a + 0.05) : null;
    if (ln && ln.endBeat + tail < end) end = ln.endBeat + tail;
    const i = lowerIdx(a + 0.1);
    for (let j = i; j < syl.length && syl[j].beat < a + L; j++) {
      const s = syl[j];
      // a syllable that starts in the last 3/4 beat and runs past the cut would be chopped: stop before it
      if (s.beat > a + L - 0.75 && s.endBeat > a + L + 0.05) { end = Math.min(end, s.beat - 0.01); break; }
    }
    // the silence starts exactly at the syllable boundary (not rounded to the grid, which
    // would clip the end of the previous syllable)
    return Math.max(md, end - a);
  }
  /** place source [a, a+L) at timeline t (syllables land where they were sung, relative to the bar) */
  function place(t, a, L, tag, toLineEnd = tag !== 'line') {
    E(t, t - a, 1, tag);
    const aud = audible(a, L, toLineEnd);
    if (aud < L - 1e-9) gap(t + aud);
  }
  // the next singer's bar at/after x that has real singing in it (skip dead air)
  function nextVoiced(x, maxBars = 12) {
    let b = srcBar(x);
    if (b < x - 0.01) b += bpb;
    for (let k = 0; k < maxBars; k++, b += bpb) if (voicedFrac(b, b + bpb) > 0.3) return b;
    return null;
  }
  // a bar to loop / ratchet: starts on a syllable, well filled
  function goodBar(x, rng) {
    const cands = [];
    for (let k = 0; k < 6; k++) {
      const b = srcBar(x) + k * bpb;
      if (b >= srcEnd) break;
      const on = onsetNear(b, 0.2);
      const v = voicedFrac(b, b + bpb);
      if (v < 0.4) continue;
      cands.push({ b, w: v + (on ? 0.8 : 0) - k * 0.08 });
    }
    if (!cands.length) return null;
    cands.sort((p, q) => q.w - p.w);
    return cands[0].b; // always the best-filled bar starting on a syllable: same material, same choice
  }

  // ---------------------------------------------------------- the hook fragment
  const hooks = lines.filter((l) => l.isHook);
  const hookLine = hooks.length ? hooks[0] : lines.reduce((a, b) => (b.energy * b.syllables.length > a.energy * a.syllables.length ? b : a));
  const hookStart = (() => { const b = srcBar(hookLine.anchorBeat); return onsetNear(b, 0.3) || voicedFrac(b, b + bpb) > 0.4 ? b : hookLine.anchorBeat; })();

  // ---------------------------------------------------------- devices
  let cur = srcBar(srcFirst);
  const advance = (n) => {
    cur += n;
    if (voicedFrac(cur, cur + bpb) < 0.3) { const nv = nextVoiced(cur); cur = nv ?? srcBar(srcFirst); }
    if (cur >= srcEnd - bpb) cur = srcBar(srcFirst); // ran out: come back round (his 395.5<-33.5)
  };
  const D = {
    play(t0, t1) {
      // carry on — but not into dead air: skip ahead (whole singer's bars) to where it's sung
      const len = t1 - t0;
      if (voicedFrac(cur, cur + len) < 0.45) {
        for (let k = 1; k <= 24; k++) {
          const b = cur + k * bpb;
          if (b + len > srcEnd + bpb) break;
          if (voicedFrac(b, b + len) >= 0.55) { cur = b; break; }
        }
      }
      place(t0, cur, t1 - t0, 'line');
      advance(t1 - t0);
      return 'play';
    },
    loop(t0, t1, rng) {
      const Lb = (t1 - t0) >= 4 * bpb && rng.chance(0.5) ? 2 : 1;
      const a = goodBar(cur, rng) ?? cur;
      const L = Lb * bpb;
      for (let t = t0; t < t1 - 1e-9; t += L) place(t, a, Math.min(L, t1 - t), 'loop');
      cur = a; advance(L);
      return `loop ${Lb} bar${Lb > 1 ? 's' : ''}`;
    },
    alternate(t0, t1, rng) {
      const x = goodBar(cur, rng) ?? cur, y = x + bpb;
      let k = 0;
      for (let t = t0; t < t1 - 1e-9; t += bpb, k++) place(t, k % 2 === 0 ? y : x, Math.min(bpb, t1 - t), 'alternate');
      cur = y; advance(bpb);
      return 'alternate bars';
    },
    stammer(t0, t1, rng) {
      const a = goodBar(cur, rng) ?? cur;
      const Ls = rng.chance(0.5) ? bpb : bpb / 2;
      place(t0, a, Ls, 'stammer');
      place(t0 + Ls, a, t1 - t0 - Ls, 'line');
      cur = a; advance(t1 - t0 - Ls);
      return 'stammer';
    },
    ratchet(t0, t1, rng) {
      const b = goodBar(cur, rng) ?? cur;
      // the ratcheted syllable: on a beat of this bar, short enough that 16th retriggers don't chop a vowel
      let s = null;
      for (let k = 0; k < bpb && !s; k++) { const c = onsetNear(b + k, 0.12); if (c && c.endBeat - c.beat <= 0.4) s = c; }
      s = s || onsetNear(b, 0.3) || syl[Math.min(syl.length - 1, lowerIdx(b))];
      const nBars = Math.max(1, Math.round((t1 - t0) / bpb) - (rng.chance(0.5) ? 1 : 0));
      const trip = o.triplets && rng.chance(0.25);
      for (let k = 0; k < nBars; k++) {
        const B = t0 + k * bpb;
        const last = k === nBars - 1 && nBars * bpb >= t1 - t0;
        // his motif: syllable x4 in 16ths, the syllable for a beat, the 2 beats before it
        const hitL = trip ? 1 / 3 : Math.max(md, 0.25);
        const nHits = Math.round(1 / hitL);
        for (let i = 0; i < nHits; i++) E(B + i * hitL, B + i * hitL - s.beat, 1, 'ratchet');
        E(B + 1, B + 1 - s.beat, 1, 'ratchet');
        if (last && rng.chance(0.6)) {
          // into the change: faster hits, then half a beat of silence
          for (let i = 0; i < 6; i++) E(B + 2 + i * 0.25, B + 2 + i * 0.25 - s.beat, 1, 'ratchet');
          gap(B + bpb - 0.5);
        } else place(B + 2, s.beat - 2, 2, 'ratchet');
      }
      const rel = t0 + nBars * bpb;
      if (rel < t1 - 1e-9) {
        E(rel, rel - s.beat, 1, 'line'); // release: the line carries on from the ratcheted syllable
        const a = s.beat;
        const aud = audible(a, t1 - rel, false);
        if (aud < t1 - rel - 1e-9) gap(rel + aud);
        cur = srcBar(a); advance(t1 - rel);
      } else { cur = srcBar(s.beat); advance(bpb); }
      return `ratchet x${nBars}`;
    },
    hook(t0, t1, rng) {
      const Lb = rng.chance(0.65) ? 2 : 1;
      const L = Lb * bpb;
      for (let t = t0; t < t1 - 1e-9; t += L) place(t, hookStart, Math.min(L, t1 - t), 'hook');
      return `hook loop ${Lb} bar${Lb > 1 ? 's' : ''}`;
    },
    sparse(t0, t1) {
      if (voicedFrac(cur, cur + bpb) < 0.3) { const nv = nextVoiced(cur); if (nv != null) cur = nv; }
      place(t0, cur, bpb, 'line');
      gap(t0 + bpb);
      advance(bpb);
      return 'one bar, then room';
    },
  };

  // ---------------------------------------------------------- the vocal's entry
  const nonIntro = roles.findIndex((r) => r.role !== 'intro');
  const introLen = nonIntro > 0 ? phrases[nonIntro].startBar - phrases[0].startBar : 0;
  const leadBars = introLen >= 12 ? 8 : 0; // enter 8 bars before the drums arrive (his bar 19 of 26)
  const entryBar = nonIntro > 0 ? phrases[nonIntro].startBar - leadBars : 0;

  const W = (role, rng) => {
    const c = chop;
    const table = {
      verse:     [['play', 1.4 - c], ['loop', 0.25 + 0.4 * c], ['stammer', 0.15 + 0.3 * c], ['alternate', 0.1 + 0.4 * c]],
      intro:     [['play', 1]],
      chorus:    [['hook', 0.5], ['play', 0.6 - 0.3 * c], ['loop', 0.3 + 0.3 * c], ['alternate', 0.2 * c]],
      drop:      [['hook', 0.7], ['loop', 0.3 + 0.3 * c], ['play', 0.3]],
      build:     [['ratchet', 0.5 + 0.6 * c], ['loop', 0.3], ['stammer', 0.2]],
      breakdown: [['sparse', 0.6 + (o.space ?? 0.35)], ['play', 0.6], ['loop', 0.3 * c]],
      outro:     [['hook', 0.6], ['sparse', 0.4]],
    }[role] || [['play', 1]];
    return rng.weighted(table.map((x) => x[0]), table.map((x) => Math.max(0.01, x[1])));
  };

  // a phrase leading into a breakdown or a drop builds tension (his ratchets at bars 43-46)
  const nextRole = (pi) => roles[pi + 1]?.role;
  // STRUCTURE, not dice: every section kind (its role + the music's section label) gets ONE treatment and ONE
  // variation, decided once. Its phrases then follow a fixed symmetric pattern:
  //   'same'   A A A A      'alt'  A B A B      'four'  A A A B   (B = the variation)
  // and every decision inside a treatment (loop length, which syllable, rests) is seeded by the section kind,
  // not the phrase, so a repeat of the same material gets the same chop every time.
  const pattern = o.symmetry || 'four';
  const kinds = new Map();
  const kindOf = (role, label) => {
    const key = `${role}|${label ?? ''}`;
    if (!kinds.has(key)) {
      const r = makeRng(hashInts(o.seed ?? 1, 'kind', key, ctx.range ? o.salt ?? 0 : 0));
      const base = W(role, r);
      let vari = W(role, r);
      for (let i = 0; i < 6 && vari === base; i++) vari = W(role, r);
      if (vari === base) vari = base === 'play' ? 'loop' : 'play';
      kinds.set(key, { key, base, vari, n: 0, rest: role === 'verse' && r.chance((o.space ?? 0.35) * 0.5),
        split: r.chance(0.5), preChange: r.chance(0.35 + 0.5 * chop) });
    }
    return kinds.get(key);
  };
  phrases.forEach((ph, pi) => {
    const role = roles[pi]?.role ?? 'verse';
    const S = ph.startBeat, Eb = ph.endBeat;
    const acts = [];
    gap(S); // a phrase owns its own time; devices below fill it
    if (ph.endBar <= entryBar) { plan.push({ pi, role, startBar: ph.startBar, endBar: ph.endBar, actions: ['instrumental'] }); return; }
    const t0 = Math.max(S, entryBar * bpb);
    const K = kindOf(role, ph.section?.label);
    const k = K.n++;
    const isVar = pattern === 'alt' ? k % 2 === 1 : pattern === 'four' ? k % 4 === 3 : false;
    let dev = isVar ? K.vari : K.base;
    // the phrase leading into a breakdown or drop always builds the same way for this section kind
    const intoChange = (role === 'verse' || role === 'chorus') && ['breakdown', 'drop'].includes(nextRole(pi));
    if (intoChange && K.preChange) dev = 'ratchet';
    const rng = makeRng(hashInts(o.seed ?? 1, 'dev', K.key, dev, intoChange ? 'pre' : isVar ? 'var' : 'base'));
    // some devices take the phrase's back half only, with the line playing first
    if (dev === 'ratchet' && Eb - t0 >= 3 * bpb) {
      const half = t0 + Math.floor((Eb - t0) / bpb / 2) * bpb;
      acts.push(D.play(t0, half));
      acts.push(D.ratchet(half, Eb, rng));
    } else if ((dev === 'loop' || dev === 'stammer') && role === 'verse' && Eb - t0 >= 4 * bpb && K.split) {
      const half = t0 + 2 * bpb;
      acts.push(D[dev](t0, half, rng));
      acts.push(D.play(half, Eb));
    } else acts.push(D[dev](t0, Eb, rng));
    // room to breathe: a verse kind that rests, rests its last bar every time
    if (K.rest && !isVar && Eb - t0 >= 4 * bpb && dev !== 'ratchet') { gap(Eb - bpb); acts.push('last bar rests'); }
    plan.push({ pi, role, startBar: ph.startBar, endBar: ph.endBar, actions: acts, kind: K.key, variation: isVar });
  });
  return { events: ev, plan };
}
