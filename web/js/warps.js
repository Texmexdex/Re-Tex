// Stretching / contracting part of a clip without slicing it - the browser twin of mashlab/warps.py.
// clip.warps = [{ a, b, r, kp }]: the song's source beats a..b play at r x the clip's own speed (r < 1 = that part
// takes longer); kp = keep pitch (false: pitch follows the speed; null = whatever the clip does). Everything in
// the clip after it moves with it, and the clip's end moves too.

// -> [{ u0, u1, w0, w1, r, kp }]: clip offsets u (project beats from the clip start) and landing positions w
// (song beats at the song's own tempo: where the audio really is, timing.js landing) of each constant-speed piece
export function pieces(c, mult, Lm) {
  const k = (c.rate || 1) / mult;
  const wf = (b) => (Lm ? Lm.at(b) : b);
  const L = c.len;
  const regs = (c.warps || []).filter((g) => g.b > g.a && g.r > 0)
    .map((g) => ({ wa: wf(g.a), wb: wf(g.b), r: g.r, kp: g.kp ?? null })).sort((x, y) => x.wa - y.wa);
  const out = [];
  let u = 0, w = wf(c.src);
  const push = (w1, r, kp) => {
    const du = (w1 - w) / (k * r);
    if (du <= 1e-9) return;
    const u1 = Math.min(L, u + du);
    const we = w + (u1 - u) * k * r;
    out.push({ u0: u, u1, w0: w, w1: we, r, kp });
    u = u1; w = we;
  };
  for (const g of regs) {
    if (u >= L) break;
    if (g.wb <= w) continue;
    if (g.wa > w) push(g.wa, 1, null);
    if (u < L) push(g.wb, g.r, g.kp);
  }
  if (u < L) out.push({ u0: u, u1: L, w0: w, w1: w + (L - u) * k, r: 1, kp: null });
  return out;
}

// landing position at clip offset u (project beats), and the reverse
export function wAtU(ps, u) {
  if (!ps.length) return 0;
  let p = ps[ps.length - 1];
  for (const q of ps) if (u <= q.u1) { p = q; break; }
  if (u < ps[0].u0) p = ps[0];
  const du = p.u1 - p.u0;
  return du > 1e-12 ? p.w0 + (u - p.u0) * (p.w1 - p.w0) / du : p.w0;
}
export function uAtW(ps, w) {
  if (!ps.length) return 0;
  let p = ps[ps.length - 1];
  for (const q of ps) if (w <= q.w1) { p = q; break; }
  if (w < ps[0].w0) p = ps[0];
  const dw = p.w1 - p.w0;
  return dw > 1e-12 ? p.u0 + (w - p.w0) * (p.u1 - p.u0) / dw : p.u0;
}

// the source-beat span [a, b] that clip offsets u0..u1 cover
export function spanAt(c, mult, Lm, u0, u1) {
  const ps = pieces(c, mult, Lm);
  const inv = (w) => (Lm ? Lm.inv(w) : w);
  return [inv(wAtU(ps, u0)), inv(wAtU(ps, u1))];
}

// warps rewritten so source beats a..b get fn({ r, kp }) -> { r, kp }
function rewrite(warps, a, b, fn) {
  const cuts = new Set([a, b]);
  for (const g of warps || []) { cuts.add(g.a); cuts.add(g.b); }
  const xs = [...cuts].sort((x, y) => x - y);
  const out = [];
  for (let i = 0; i + 1 < xs.length; i++) {
    const x0 = xs[i], x1 = xs[i + 1];
    if (x1 - x0 < 1e-9) continue;
    const mid = (x0 + x1) / 2;
    const g = (warps || []).find((q) => q.a <= mid && mid < q.b);
    let v = { r: g ? g.r : 1, kp: g ? g.kp ?? null : null };
    if (mid > a && mid < b) v = fn(v);
    if (Math.abs(v.r - 1) < 1e-9) continue;                 // plain speed: no warp needed
    const prev = out[out.length - 1];
    if (prev && Math.abs(prev.b - x0) < 1e-9 && Math.abs(prev.r - v.r) < 1e-9 && prev.kp === v.kp) prev.b = x1;
    else out.push({ a: x0, b: x1, r: v.r, kp: v.kp });
  }
  return out;
}

// clip offsets u0..u1 take newLen project beats instead: stretch (>) or contract (<) that part only; what
// follows in the clip moves, and the clip's end moves with it. kp: keep pitch in that part (true/false).
export function stretchSpan(c, mult, Lm, u0, u1, newLen, kp) {
  const oldLen = u1 - u0;
  if (oldLen < 1e-6 || newLen < 1e-6) return;
  const [a, b] = spanAt(c, mult, Lm, u0, u1);
  const f = newLen / oldLen;
  c.warps = rewrite(c.warps, a, b, (v) => ({ r: v.r / f, kp }));
  if (!c.warps.length) delete c.warps;
  c.len = Math.max(0.05, c.len + (newLen - oldLen));
}

// back to the clip's own speed in u0..u1
export function resetSpan(c, mult, Lm, u0, u1) {
  const ps0 = pieces(c, mult, Lm);
  const w0 = wAtU(ps0, u0), w1 = wAtU(ps0, u1);
  const [a, b] = spanAt(c, mult, Lm, u0, u1);
  c.warps = rewrite(c.warps, a, b, () => ({ r: 1, kp: null }));
  if (!c.warps.length) delete c.warps;
  const k = (c.rate || 1) / mult;
  c.len = Math.max(0.05, c.len + ((w1 - w0) / k - (u1 - u0)));
}

// pitch mode for whatever is stretched in u0..u1
export function setSpanPitch(c, mult, Lm, u0, u1, kp) {
  const [a, b] = spanAt(c, mult, Lm, u0, u1);
  c.warps = rewrite(c.warps, a, b, (v) => ({ r: v.r, kp: Math.abs(v.r - 1) < 1e-9 ? null : kp }));
  if (!c.warps.length) delete c.warps;
}
