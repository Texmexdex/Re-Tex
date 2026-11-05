// Deterministic seeded randomness so a seed always gives the same edit,
// and a single phrase can be rerolled without touching the others.

export function hashInts(...xs) {
  let h = 2166136261 >>> 0;
  for (const x of xs) {
    const s = String(x);
    for (let i = 0; i < s.length; i++) {
      h ^= s.charCodeAt(i);
      h = Math.imul(h, 16777619) >>> 0;
    }
    h ^= 0x9e3779b9;
    h = Math.imul(h, 16777619) >>> 0;
  }
  return h >>> 0;
}

export function makeRng(seed) {
  let a = seed >>> 0;
  const next = () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  next.int = (n) => Math.floor(next() * n);
  next.pick = (arr) => arr[Math.floor(next() * arr.length)];
  next.chance = (p) => next() < p;
  // weighted pick: items [{w, ...}] or parallel weights
  next.weighted = (items, weights) => {
    const ws = weights || items.map((x) => x.w);
    let tot = 0;
    for (const w of ws) tot += Math.max(0, w);
    if (tot <= 0) return items[0];
    let r = next() * tot;
    for (let i = 0; i < items.length; i++) {
      r -= Math.max(0, ws[i]);
      if (r <= 0) return items[i];
    }
    return items[items.length - 1];
  };
  return next;
}
