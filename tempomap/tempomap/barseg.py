"""Bars by cutting the whole song into bars at once (a semi-Markov dynamic programme over the subdivisions).

Every way of cutting the subdivisions into bars of 3 to 16 subdivisions is scored, and the best is kept:
- each bar should sound like the bar before it (the same rhythm and notes, position by position) - or, in a riff
  that alternates two bar lengths, like the bar two back;
- each bar should start on an accent (kick, crash, bass, a chord change);
- usual lengths cost nothing, unusual ones a little (8 = 4/4 first);
- changing the length costs something, so a meter holds through a section and changes where the music changes
  (where the bar stops sounding like the one before) - not here and there.
Because the whole song is decided at once, bar 1 and the bar lengths come out together, and a section's meter is
decided by all of its bars, not a few at a time.
"""
import numpy as np

LENGTHS = tuple(range(3, 17))
# cost of each length in subdivisions, for halves (sub 2) and thirds (sub 3) of the beat
PRIOR = {2: {8: 0.0, 6: 0.35, 12: 0.45, 4: 0.6, 7: 0.5, 5: 0.6, 16: 0.7, 9: 0.6, 10: 0.6, 14: 0.6, 11: 0.7,
             13: 0.75, 15: 0.75, 3: 0.9},
         3: {12: 0.0, 6: 0.25, 9: 0.5, 15: 0.7, 3: 0.9}}
DEFAULT_COST = 1.0
W = dict(rep=1.0, down=0.6, switch=5.0, chroma=0.25, split=0.0, alt=1.0, alt_on=False)


def _lagged(X, maxlag):
    """C[lag][t] = sum over k < t of <x_k, x_k-lag>; N[t] = sum over k < t of |x_k|^2."""
    n = len(X)
    N = np.r_[0.0, np.cumsum(np.sum(X * X, axis=1))]
    C = {}
    for lag in range(1, maxlag + 1):
        d = np.zeros(n)
        if lag < n:
            d[lag:] = np.sum(X[lag:] * X[:-lag], axis=1)
        C[lag] = np.r_[0.0, np.cumsum(d)]
    return C, N


def _sims(C, N, n, L, lag):
    """For every bar start s: cosine similarity of [s, s+L) with the stretch `lag` earlier (0 off the start)."""
    out = np.zeros(n + 1)
    s = np.arange(lag, n - L + 1)
    if not len(s):
        return out
    num = C[lag][s + L] - C[lag][s]
    a = N[s + L] - N[s]
    b = N[s - lag + L] - N[s - lag]
    ok = (a > 1e-9) & (b > 1e-9)
    out[s[ok]] = num[ok] / np.sqrt(a[ok] * b[ok])
    return out


def segment(on, ch, acc, subs, w=None):
    """on: onsets per subdivision [n, F]; ch: chroma [n, 12]; acc: accent per subdivision; subs: 2/3 per
    subdivision. -> list of bar starts (subdivision indices)."""
    w = dict(W, **(w or {}))
    n = len(on)
    if n < 2 * max(LENGTHS):
        return [0]
    chn = ch - ch.mean(0)
    chn = chn / (np.linalg.norm(chn, axis=1, keepdims=True) + 1e-9)
    X = np.hstack([on, w["chroma"] * chn])
    maxlag = 2 * max(LENGTHS)
    C, N = _lagged(X, maxlag)
    a = (acc - np.mean(acc)) / (np.std(acc) + 1e-9)
    Ls = list(LENGTHS)
    K = len(Ls)
    cost = np.array([[PRIOR.get(int(subs[min(s, n - 1)]), PRIOR[2]).get(L, DEFAULT_COST) for L in Ls] for s in range(n)])
    NEG = -1e18
    R1 = {L: _sims(C, N, n, L, L) for L in Ls}
    R2 = {}
    for L in Ls:                                   # [kj, s]: one riff-cycle back, for each previous length
        R2[L] = np.stack([R1[L] if L2 == L else np.maximum(
            _sims(C, N, n, L, L + L2) if L + L2 <= maxlag else np.zeros(n + 1), 0.5 * R1[L]) for L2 in Ls])
    sw = w["switch"] * (1.0 - np.eye(K))           # [ki, kj]
    # a long bar whose second half plays its first half is two bars (a 16 that is 8 + 8)
    half_pen = {}
    for L in Ls:
        if L >= 10 and L % 2 == 0:
            h = L // 2
            sh = _sims(C, N, n, h, h)                 # [s + h, s + L) against [s, s + h): index by s + h
            v = np.zeros(n + 1)
            v[:n + 1 - h] = sh[h:]
            half_pen[L] = w["split"] * np.clip(v - 0.5, 0, None) * (L / 8.0)
        else:
            half_pen[L] = np.zeros(n + 1)
    if not w.get("alt_on", True):
        return _dp1(n, K, Ls, a, cost, half_pen, R1, R2, sw, w, NEG)
    # state = (length of the bar before, length of this bar): a riff that alternates two lengths (5 + 7) pays the
    # change once, then only `alt` per bar as long as each bar repeats the one two back
    eye = np.eye(K, dtype=bool)
    # T[ki][k2, k1]: cost of a new bar ki after bars k2 (two back) then k1 (the last)
    T = np.empty((K, K, K))
    for ki in range(K):
        same = eye[ki][None, :].repeat(K, 0)          # ki == k1
        alt = eye[ki][:, None].repeat(K, 1)           # ki == k2
        T[ki] = np.where(same, 0.0, np.where(alt, w["alt"], w["switch"]))
    best = np.full((n + 1, K, K), NEG)               # [e, k1 (bar before), ki (this bar)]
    back = np.full((n + 1, K, K), -1, dtype=np.int16)
    start_free = max(Ls)
    for e in range(1, n + 1):
        for ki, L in enumerate(Ls):
            s = e - L
            if s < 0:
                continue
            base = w["down"] * a[s] - cost[s, ki] - half_pen[L][s]
            rep = w["rep"] * R2[L][:, s] * (L / 8.0)  # [k1]
            if s >= Ls[0]:
                M = best[s] - T[ki]                    # [k2, k1]
                k2 = np.argmax(M, axis=0)
                v = M[k2, np.arange(K)] + base + rep
                ok = v > NEG / 2
                best[e, ok, ki] = v[ok]
                back[e, ok, ki] = k2[ok]
            if s < start_free:                         # a first bar: no bar before it
                c0 = base + w["rep"] * R1[L][s] * (L / 8.0)
                if c0 > best[e, ki, ki]:
                    best[e, ki, ki] = c0
                    back[e, ki, ki] = -2
    ends = range(max(1, n - max(Ls)), n + 1)
    e, k1, ki = max(((e, k1, k) for e in ends for k1 in range(K) for k in range(K)), key=lambda x: best[x])
    starts = []
    while e > 0:
        L = Ls[ki]
        s = e - L
        starts.append(s)
        k2 = int(back[e, k1, ki])
        if k2 < 0:
            break
        e, k1, ki = s, k2, k1
    return sorted(starts)


def _dp1(n, K, Ls, a, cost, half_pen, R1, R2, sw, w, NEG):
    """The simpler programme: state = this bar's length only."""
    best = np.full((n + 1, K), NEG)
    back = np.full((n + 1, K), -1, dtype=int)
    start_free = max(Ls)
    for e in range(1, n + 1):
        for ki, L in enumerate(Ls):
            s = e - L
            if s < 0:
                continue
            base = w["down"] * a[s] - cost[s, ki] - half_pen[L][s]
            cand0 = base + w["rep"] * R1[L][s] * (L / 8.0) if s < start_free else NEG
            if s >= Ls[0]:
                tot = best[s] + base + w["rep"] * R2[L][:, s] * (L / 8.0) - sw[ki]
                kj = int(np.argmax(tot))
                if tot[kj] > cand0 and tot[kj] > NEG / 2:
                    best[e, ki], back[e, ki] = tot[kj], kj
                    continue
            if cand0 > NEG / 2:
                best[e, ki], back[e, ki] = cand0, -2
    ends = range(max(1, n - max(Ls)), n + 1)
    e, ki = max(((e, k) for e in ends for k in range(K)), key=lambda x: best[x[0], x[1]])
    starts = []
    while e > 0 and ki >= 0:
        L = Ls[ki]
        s = e - L
        starts.append(s)
        kj = back[e, ki]
        if kj == -2:
            break
        e, ki = s, kj
    return sorted(starts)
