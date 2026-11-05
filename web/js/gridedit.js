// The beat grid window's editing rules (gridtool.js), with no page attached so tests run them in node.
// M = { t: [s], d: [0|1] a bar's 1, s: [0|1] sure, c: [0|1] placed by you, p0 } - edited in place.
//
// A bar is its 1 (a bar line) and the beats up to the next 1. A bar's beats are always spread evenly between its two
// bar lines - unless you placed one yourself (Ctrl+drag), and then that bar keeps its shape in proportion. Nothing
// before the bar you're working on ever moves (the user's rules, Oct 19):
//   drag a bar line          the bar before it squeezes / stretches evenly; everything after moves along with it
//   Ctrl+drag a bar line     only the two bars either side of it change
//   drag a beat              its bar's tempo changes so the beat lands there (still even); everything after moves along
//   Ctrl+drag a beat         only that beat moves
//   Shift+drag anything      every mark moves
//   delete a mark            the bar's beats spread evenly again (4 beats -> 3); a deleted bar line joins its bars
//   add a beat               the bar's beats spread evenly again (3 -> 4); exactly where you put it with Ctrl

export const MIN = 0.03;            // s: marks never get closer than this
export const UNEVEN = 1.25;         // a bar whose longest beat is this much longer than its shortest is uneven

const clamp = (x, lo, hi) => Math.min(Math.max(x, lo), hi);
export function prevDown(M, i) { for (let j = i - 1; j >= 0; j--) if (M.d[j]) return j; return -1; }
export function nextDown(M, i) { for (let j = i + 1; j < M.t.length; j++) if (M.d[j]) return j; return -1; }
export function placed(M, a, b) { for (let j = a + 1; j < b; j++) if (M.c[j]) return true; return false; }

/** The beats strictly between marks a and b, fitted to where a and b are now: evenly - or, in a bar with a beat you
 * placed yourself, keeping every beat's place in proportion (B = the marks before this edit, same indices). */
export function fitBetween(M, a, b, B = M) {
  if (a < 0 || b < 0 || b - a < 2) return;
  const ta = M.t[a], tb = M.t[b];
  if (placed(M, a, b)) {
    const ba = B.t[a], bb = B.t[b];
    for (let j = a + 1; j < b; j++) M.t[j] = ta + ((B.t[j] - ba) / (bb - ba)) * (tb - ta);
  } else for (let j = a + 1; j < b; j++) M.t[j] = ta + ((j - a) / (b - a)) * (tb - ta);
}

/** Move mark k to time T, starting from the marks B ({t, c}) as they were when the drag began.
 * how: "plain" | "one" (Ctrl) | "all" (Shift). */
export function dragMark(M, k, T, how, B) {
  M.t = B.t.slice(); M.c = B.c.slice();
  const n = M.t.length, last = n - 1;
  if (how === "all") {
    const dt = Math.max(T - B.t[k], -B.t[0]);
    for (let i = 0; i < n; i++) M.t[i] = B.t[i] + dt;
    return;
  }
  const a = prevDown(M, k);
  if (M.d[k]) {
    const lo = a >= 0 ? M.t[a] + (k - a) * MIN : k > 0 ? M.t[k - 1] + MIN : 0;
    if (how === "one") {                              // only the bars on either side of this bar line
      const nb = nextDown(M, k), b = nb >= 0 ? nb : k < last ? last : -1;
      M.t[k] = clamp(T, lo, b >= 0 ? M.t[b] - (b - k) * MIN : Infinity);
      fitBetween(M, a, k, B); fitBetween(M, k, b, B);
      return;
    }
    const dt = Math.max(T, lo) - B.t[k];              // the bar before squeezes / stretches, the rest moves along
    for (let i = k; i < n; i++) M.t[i] = B.t[i] + dt;
    fitBetween(M, a, k, B);
    return;
  }
  if (how === "one" || a < 0) {                       // just this beat (a beat before the first bar has no bar)
    M.t[k] = clamp(T, k > 0 ? M.t[k - 1] + MIN : 0, k < last ? M.t[k + 1] - MIN : Infinity);
    if (how === "one" && a >= 0) M.c[k] = 1;
    return;
  }
  // a beat inside a bar: the bar's tempo changes so it lands at T (its beats stay even), the rest moves along
  const nb = nextDown(M, k), end = nb >= 0 ? nb : last;
  const T2 = Math.max(T, M.t[a] + (k - a) * MIN);
  const even = !placed(M, a, nb >= 0 ? nb : last + 1);
  const per = (T2 - M.t[a]) / (k - a), f = (T2 - M.t[a]) / (B.t[k] - B.t[a]);
  for (let i = a + 1; i <= end; i++) M.t[i] = even ? M.t[a] + (i - a) * per : M.t[a] + (B.t[i] - B.t[a]) * f;
  if (nb >= 0) { const dt = M.t[nb] - B.t[nb]; for (let i = nb + 1; i < n; i++) M.t[i] = B.t[i] + dt; }
}

/** Delete mark i; its bar (or the two bars a deleted bar line joins) spreads its beats evenly again. -> done? */
export function deleteMark(M, i) {
  if (i < 0 || i >= M.t.length || M.t.length <= 2) return false;
  for (const key of ["t", "d", "s", "c"]) M[key].splice(i, 1);
  if (M.p0 != null) M.p0 = M.p0 === i ? null : M.p0 > i ? M.p0 - 1 : M.p0;
  const a = prevDown(M, i), b = i < M.t.length && M.d[i] ? i : nextDown(M, i);
  if (a >= 0 && b >= 0) fitBetween(M, a, b);
  return true;
}

/** Add a beat at T: its bar spreads its beats evenly again - or, exact, it stays at T (a beat you placed). -> its
 * index, or -1 (too close to a mark). */
export function addMark(M, T, exact = false) {
  let i = M.t.findIndex((t) => t > T);
  if (i < 0) i = M.t.length;
  if ((i > 0 && T - M.t[i - 1] < MIN) || (i < M.t.length && M.t[i] - T < MIN)) return -1;
  M.t.splice(i, 0, T); M.d.splice(i, 0, 0); M.s.splice(i, 0, 1); M.c.splice(i, 0, exact ? 1 : 0);
  if (M.p0 != null && M.p0 >= i) M.p0++;
  const a = prevDown(M, i), b = nextDown(M, i);
  if (!exact && a >= 0 && b >= 0) fitBetween(M, a, b);
  return i;
}

/** Mark i becomes a bar's 1, or a plain beat (nothing moves). */
export function toggleDown(M, i) {
  M.d[i] = M.d[i] ? 0 : 1;
  if (!M.d[i] && M.p0 === i) M.p0 = null;
}

/** Bars whose beats aren't even (and that hold no beat you placed): [[a, b]] (their bar lines). */
export function unevenBars(M) {
  const out = [];
  let a = M.d.indexOf(1);
  while (a >= 0) {
    const b = nextDown(M, a);
    if (b < 0) break;
    if (b - a >= 2 && !placed(M, a, b)) {
      let lo = Infinity, hi = 0;
      for (let j = a; j < b; j++) { const iv = M.t[j + 1] - M.t[j]; lo = Math.min(lo, iv); hi = Math.max(hi, iv); }
      if (hi > lo * UNEVEN) out.push([a, b]);
    }
    a = b;
  }
  return out;
}
