// ReadHead engine, run by Re-Tex (mashlab/readhead.py) as:  node cli.mjs <request.json>
// Reads mono float32 inputs the Python side wrote, analyses them on the PROJECT's grid
// (Re-Tex's tempo, bar 1 = beat 0), and prints one JSON object: the edit lane + the plan.
// Pure computation: reads its inputs, writes only feature caches next to them, prints JSON.
import { readFileSync, writeFileSync, existsSync } from 'node:fs';
import { analyzeFrames, NBANDS } from './engine/dsp.js';
import { makeView, analyzeStructure } from './engine/analysis.js';
import { makeGrid } from './engine/grid.js';
import { analyzeVocal } from './engine/vocal.js';
import { generate, buildPhrases } from './engine/generator.js';
import { phraseRoles, arrangeVocal, ARRANGE_DEFAULTS } from './engine/arranger.js';
import { arrangeFlow } from './engine/flow.js';
import { normalizeLane } from './engine/lane.js';
import { hashInts } from './engine/rng.js';

const req = JSON.parse(readFileSync(process.argv[2], 'utf8'));
const loadF32 = (p) => { const b = readFileSync(p); return new Float32Array(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength)); };

// frame features, cached next to the input (inputs are named by content hash, so a cache hit is exact)
function frames(inp) {
  const cp = inp.path + '.fr';
  if (existsSync(cp) && existsSync(cp + '.json')) {
    const m = JSON.parse(readFileSync(cp + '.json', 'utf8'));
    const all = loadF32(cp);
    let o = 0;
    const take = (n) => { const a = all.subarray(o, o + n); o += n; return a; };
    return { ...m, rms: take(m.nFrames), flux: take(m.nFrames), low: take(m.nFrames), chroma: take(m.nFrames * 12), bands: take(m.nFrames * NBANDS) };
  }
  const fr = analyzeFrames(loadF32(inp.path), inp.sr);
  const parts = [fr.rms, fr.flux, fr.low, fr.chroma, fr.bands];
  const total = parts.reduce((a, p) => a + p.length, 0);
  const all = new Float32Array(total);
  let o = 0;
  for (const p of parts) { all.set(p, o); o += p.length; }
  try {
    writeFileSync(cp, Buffer.from(all.buffer));
    writeFileSync(cp + '.json', JSON.stringify({ sr: fr.sr, hop: fr.hop, hopSec: fr.hopSec, nFrames: fr.nFrames, nbands: fr.nbands }));
  } catch { /* cache is optional */ }
  return fr;
}

const SECTION_ROLE = { intro: 'intro', verse: 'verse', 'pre-chorus': 'build', prechorus: 'build', build: 'build', buildup: 'build',
  chorus: 'chorus', hook: 'chorus', drop: 'drop', breakdown: 'breakdown', bridge: 'breakdown', break: 'breakdown', outro: 'outro', solo: 'verse' };

const t0 = Date.now();
const grid = makeGrid(req.bpm, 0, req.bpb || 4);
const bpb = grid.bpb;
const P = req.opts?.phraseBars || 4;
const backing = req.backing.map((b) => ({ ...b, fr: frames(b) }));
for (const b of backing) b.view = makeView(b.fr, grid);
const durationSec = (req.lengthBeats * 60) / req.bpm;
let structure;
const marks = (req.sections || []).filter((m) => Number.isFinite(m.beat)).sort((a, b) => a.beat - b.beat);
if (marks.length) {
  // YOUR section markers decide the structure (and the roles, where the names say it)
  structure = analyzeStructure(backing.map((b) => b.view), grid, { phraseBars: P, durationSec, forcePhraseOffset: Math.round(marks[0].beat / bpb) % P });
  const totalBars = structure.totalBars;
  const bounds = marks.map((m) => Math.round(m.beat / bpb)).filter((b, i, a) => b < totalBars && (i === 0 || b > a[i - 1]));
  if (bounds[0] > 0) bounds.unshift(0);
  structure.sections = bounds.map((a, i) => {
    const b = i + 1 < bounds.length ? bounds[i + 1] : totalBars;
    const mk = marks.find((m) => Math.round(m.beat / bpb) === a);
    let e = 0;
    for (let k = a; k < b; k++) e += structure.energy[k] || 0;
    return { startBar: a, endBar: b, label: mk ? String(mk.name || 'section') : 'start', energy: e / Math.max(1, b - a), marked: !!mk };
  });
} else {
  structure = analyzeStructure(backing.map((b) => b.view), grid, { phraseBars: P, durationSec });
}
const phrases = buildPhrases(structure, P, bpb);
const roles = phraseRoles(phrases, backing.map((b) => ({ role: b.role, view: b.view })), bpb);
if (marks.length) phrases.forEach((ph, i) => {
  const sec = ph.section;
  const r = sec && sec.marked ? SECTION_ROLE[String(sec.label).toLowerCase().replace(/\s*\d+$/, '').trim()] : null;
  if (r) roles[i].role = r;
});

const out = { ok: true, grid: { bpm: req.bpm, bpb }, phrases: phrases.map((p, i) => ({ startBar: p.startBar, endBar: p.endBar, role: roles[i].role, section: p.section?.label })) };
const o = req.opts || {};
if (req.mode === 'arrange') {
  const src = { ...req.source, fr: frames(req.source) };
  src.view = makeView(src.fr, grid);
  const vocal = analyzeVocal(src.fr, grid, src.view, { barPhase: Number.isFinite(o.vocalPhase) ? o.vocalPhase : null });
  const opts = { ...ARRANGE_DEFAULTS, ...o, seed: hashInts(o.seed ?? 1, 'vocal') };
  const fn = opts.style === 'lines' ? arrangeVocal : arrangeFlow;
  const res = fn({ grid, phrases, roles, vocal, chop: o.chop ?? 0.5 }, opts);
  out.lane = normalizeLane(res.events);
  out.plan = res.plan;
  out.vocal = { lines: vocal.lines.length, syllables: vocal.syllables.length, barPhase: vocal.barPhase, hooks: vocal.lines.filter((l) => l.isHook).map((l) => l.idx + 1) };
} else {
  const src = { ...req.source, fr: frames(req.source) };
  src.view = makeView(src.fr, grid);
  const res = generate({ grid, structure, stems: [{ id: 'lane', role: req.source.role || 'mix', view: src.view, chaos: o.chop ?? 0.5, enabled: true }] },
    { ...o, gestures: { ...(o.gestures || {}), reverse: false } }); // Re-Tex clips can't play backwards
  out.lane = res.lanes.lane;
  out.plan = res.phrases.map((p) => ({ startBar: p.startBar, endBar: p.endBar }));
  out.counts = res.report.counts.lane || {};
}
out.ms = Date.now() - t0;
process.stdout.write(JSON.stringify(out));
