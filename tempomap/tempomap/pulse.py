"""Tempo sections: where the song keeps one steady pulse, at what tempo, and exactly where its beats fall.

1. The detector's beats are cut into stretches of steady spacing (where the detector thinks the tempo stays put).
   The detector is right about WHERE things change, and about how fast it counted inside a stretch, far more often
   than about WHAT it counted: it counts a triplet feel as a faster tempo, a 7/8 riff as another one, and guesses
   in a quiet intro.
2. The longest stretch is taken to be counted at the right LEVEL (the detector's beat, not its half or double).
   Its exact tempo is the one that holds the whole stretch together best: the onsets summed at that rate and its
   multiples over the entire stretch at once (whole-stretch coherence, scanned in steps of 0.005 BPM). A steady
   studio take has one tempo that keeps every bar of the stretch in phase; a tempo even 0.3 % off loses the phase
   within a minute, so the peak is sharp. (The detector's own count is NOT precise enough: on Fear Inoculum it
   says 89.84-89.96 where the music holds together twice as well at 90.87.)
3. Outward from it, each stretch is first asked: does the grid next to it, carried on in phase, still land on the
   music here (the onsets keep falling at the same places in its beat)? Then it's the same pulse - studio songs
   change the grouping far more often than the tempo. Otherwise it gets a tempo of its own (the detector's count,
   at the counting level nearest its neighbour's) fitted the same way; with no clear pulse at all, its neighbour's
   grid carries on. Neighbours with the same tempo and phase become one section; the same tempo with a jump in
   phase stays two (an edit).
4. Where the tempo changes, the change goes on the beat where one grid stops fitting the hits and the next starts.
"""
import numpy as np

MIN_BEATS = 12            # the shortest stretch of steady spacing in the detector's beats
MIN_SEC = 8.0             # stretches shorter than this are folded into a neighbour
HARM = ((1, 1.0), (2, 0.7), (3, 0.5), (4, 0.5))
SAME = 0.015              # a pulse within 1.5 % of the neighbour's counts as the neighbour's
KEEP_SAME = 0.5           # ... and is kept when it's at least this strong against the strongest pulse there (a
                          # regrouping - 4 against 3, a triplet feel - makes another pulse the strongest on top;
                          # the grid underneath is still the old one)
KEEP_RATIO = 0.3          # ... or this strong when the strongest one is a simple ratio of it (3:2, 4:3 ...)
RATIOS_ = (1.5, 2 / 3, 4 / 3, 0.75, 1.25, 0.8, 5 / 3, 0.6, 2.0, 0.5)
FEELS = {1.0: "1:1", 0.5: "1:2", 2.0: "2:1", 1.5: "3:2", 2 / 3: "2:3", 4 / 3: "4:3", 0.75: "3:4", 1.25: "5:4",
         0.8: "4:5", 5 / 3: "5:3", 0.6: "3:5", 3.0: "3:1", 1 / 3: "1:3"}
REGROUP = 0.004           # a stretch whose own count is within 0.4 % of one of these ratios of the main pulse is the
                          # main grid regrouped (a triplet feel, 4 over 3, half time) - not a new tempo
TACTUS = (60.0, 200.0)    # the range a counted pulse is kept in (a 390 "BPM" is a 16th note, a 45 a half note)
CLEAR = 3.0               # a pulse must stand this far above the typical salience to count at all
MERGE = 0.004             # neighbouring sections whose fitted tempos are within 0.4 % ...
PHASE_OK = 0.08           # ... and whose grids agree to within this share of a beat become one


# ---------------------------------------------------------------- 1. the detector's stretches

def _segments(x, min_len=MIN_BEATS, pen=0.05):
    """Change points of a series (least squares, a penalty per segment): [(a, b)] index ranges."""
    n = len(x)
    if n < 2 * min_len:
        return [(0, n)]
    cs = np.concatenate([[0.0], np.cumsum(x)])
    cs2 = np.concatenate([[0.0], np.cumsum(x * x)])
    best = np.full(n + 1, np.inf)
    prev = np.zeros(n + 1, int)
    best[0] = 0.0
    for b in range(min_len, n + 1):
        a = np.arange(0, b - min_len + 1)
        ok = np.isfinite(best[a])
        if not ok.any():
            continue
        a = a[ok]
        m = b - a
        cost = (cs2[b] - cs2[a]) - (cs[b] - cs[a]) ** 2 / m
        v = best[a] + cost + pen * min_len
        k = int(np.argmin(v))
        best[b], prev[b] = v[k], a[k]
    if not np.isfinite(best[n]):
        return [(0, n)]
    out, b = [], n
    while b > 0:
        out.append((prev[b], b))
        b = prev[b]
    return out[::-1]


def stretches(beats, dur):
    """The detector's beats as stretches of steady spacing, covering the whole song: [(t0, t1, detector BPM)]."""
    b = np.asarray(beats, float)
    if len(b) < 2 * MIN_BEATS + 2:
        bpm = 60.0 / float(np.median(np.diff(b))) if len(b) > 2 else 120.0
        return [(0.0, dur, bpm)]
    iv = np.diff(b)
    segs = _segments(np.log(np.clip(iv, 1e-3, None)))
    out = [[float(b[a]), float(b[c]), 60.0 / float(np.median(iv[a:c]))] for a, c in segs]
    out[0][0], out[-1][1] = 0.0, dur
    for i in range(len(out) - 1):                      # (they touch: one ends where the next starts)
        out[i + 1][0] = out[i][1]
    changed = True
    while changed and len(out) > 1:                    # fold the short ones into the closer-tempo neighbour
        changed = False
        for i, s in enumerate(out):
            if s[1] - s[0] >= MIN_SEC:
                continue
            nb = [j for j in (i - 1, i + 1) if 0 <= j < len(out)]
            j = min(nb, key=lambda j: abs(np.log(out[j][2] / s[2])))
            out[j][0], out[j][1] = min(out[j][0], s[0]), max(out[j][1], s[1])
            del out[i]
            changed = True
            break
    return [tuple(s) for s in out]


# ---------------------------------------------------------------- 2. the pulse in the audio

class Salience:
    """How strongly the onsets between t0 and t1 repeat at a beat rate: S(bpm) = the weighted size of their Fourier
    components at the rate and its multiples, over their total. One zero-padded FFT; evaluated anywhere."""

    def __init__(self, env, fps, t0, t1):
        i0, i1 = int(max(0, t0 * fps)), int(min(len(env), t1 * fps))
        x = np.asarray(env[i0:i1], float)
        self.total = float(x.sum()) + 1e-12
        n = max(16, len(x))
        npad = 1 << int(np.ceil(np.log2(8 * n)))
        self.mag = np.abs(np.fft.rfft(x - x.mean(), npad))
        self.df = fps / npad

    def __call__(self, bpm):
        bpm = np.atleast_1d(np.asarray(bpm, float))
        s = np.zeros(len(bpm))
        for h, w in HARM:
            k = np.clip(np.round(bpm * h / 60.0 / self.df).astype(int), 0, len(self.mag) - 1)
            s += w * self.mag[k]
        return s / self.total


def _grid(lo, hi, step=0.0005):
    return np.exp(np.arange(np.log(lo), np.log(hi), step))


LEVELS = (1.0, 0.5, 2.0, 2 / 3, 1.5, 1 / 3, 3.0)
LEVEL_KEEP = 0.5          # another counting level of the strongest pulse is taken if it is this strong and nearer


def _peak(sal, bpm, span=0.015):
    g = _grid(bpm * (1 - span), bpm * (1 + span))
    s = sal(g)
    k = int(np.argmax(s))
    return float(g[k]), float(s[k])


def choose_pulse(sal, ref, rng=1.6):
    """The pulse of a stretch next to one whose pulse is `ref`: (bpm, how it was chosen, clarity). "same": the
    neighbour's pulse is still there; "own": a clearly different pulse, counted at the level nearest the
    neighbour's (the strongest peak is often the eighths: dense cymbals put more harmonics there); "none": no
    clear pulse."""
    g = _grid(ref / rng, ref * rng)
    s = sal(g)
    clarity = float(s.max() / (np.median(s) + 1e-12))
    near = np.abs(g / ref - 1) < SAME
    i_same = np.flatnonzero(near)[int(np.argmax(s[near]))]
    i_best = int(np.argmax(s))
    if clarity < CLEAR:
        return ref, "none", clarity
    if s[i_same] >= KEEP_SAME * s[i_best]:
        return float(g[i_same]), "same", clarity
    best, sb = float(g[i_best]), float(s[i_best])
    cands = [_peak(sal, best * k) for k in LEVELS]
    ok = [(b, v) for b, v in cands if v >= LEVEL_KEEP * sb and ref / 2.2 < b < ref * 2.2]
    b = min(ok, key=lambda c: abs(np.log(c[0] / ref)))[0] if ok else best
    return b, "own", clarity


def first_pulse(sal, det_bpm):
    """The longest stretch's pulse: the salience peak within 4 % of what the detector counted there."""
    g = _grid(det_bpm * 0.96, det_bpm * 1.04)
    s = sal(g)
    gg = _grid(det_bpm / 1.6, det_bpm * 1.6)
    return float(g[int(np.argmax(s))]), float(s.max() / (np.median(sal(gg)) + 1e-12))


# ---------------------------------------------------------------- 3. fit to the hits

def _peak_near(curve, fps, t, reach):
    """(time, height) of the strongest point of curve within reach s of t."""
    i0, i1 = int(max(0, (t - reach) * fps)), int(min(len(curve), (t + reach) * fps + 1))
    if i1 - i0 < 2:
        return t, 0.0
    k = int(np.argmax(curve[i0:i1]))
    return (i0 + k) / fps, float(curve[i0 + k])


def onset_peaks(curve, fps, t0, t1, q=85):
    """The clear hits between t0 and t1: [(time, height)] of the curve's local peaks above its q-th percentile."""
    i0, i1 = int(max(1, t0 * fps)), int(min(len(curve) - 1, t1 * fps))
    x = curve[i0 - 1:i1 + 1]
    if len(x) < 3:
        return np.zeros(0), np.zeros(0)
    thr = max(np.percentile(x, q), 1e-6)
    pk = np.flatnonzero((x[1:-1] >= x[:-2]) & (x[1:-1] > x[2:]) & (x[1:-1] >= thr)) + 1
    from scipy.ndimage import maximum_filter1d       # one peak per 40 ms: the strongest within 40 ms either side
    r = max(1, int(0.04 * fps))
    mx = maximum_filter1d(x, size=2 * r + 1, mode="nearest")
    keep = pk[x[pk] >= mx[pk]]
    if len(keep) > 1:                                # (equal heights closer than 40 ms: the first)
        keep = keep[np.r_[True, np.diff(keep) > r]]
    return (i0 - 1 + keep) / fps, x[keep]


def fit_hits(feat, t0, t1, bpm, phase=None):
    """Exact tempo and phase over [t0, t1] from the hits: -> dict(period, anchor (a beat time), rms (s, of the hits
    that count), used (hits), share (of beats with a hit on them), sub (2 or 3: the subdivision the hits sit on)).

    Every clear hit is put on the nearest subdivision of the beat (halves, or thirds - whichever the hits sit on
    better) and hit time is fitted against beat number with a robust straight line. Hits on subdivisions count, so
    a riff grouped 3+3+3+2 - hits that land on the eighths but rarely on the beat - fits as well as a backbeat.
    phase: a beat time to start from (else the phase of the onsets' beat-rate component)."""
    p = 60.0 / bpm
    if phase is None:
        i0, i1 = int(t0 * feat.fps), int(t1 * feat.fps)
        x = feat.env[i0:i1]
        tt = np.arange(i0, i1) / feat.fps
        c = np.sum(x * np.exp(-2j * np.pi * tt / p))
        phase = (-np.angle(c) / (2 * np.pi)) * p
    k0 = int(np.ceil((t0 - phase) / p))
    k1 = int(np.floor((t1 - phase) / p))
    none = dict(period=p, anchor=phase + k0 * p, rms=None, used=0, share=0.0, sub=2)
    if k1 - k0 < 4:
        return none
    ht, hh = onset_peaks(feat.hits, feat.fps, t0, t1)
    et, eh = onset_peaks(feat.env, feat.fps, t0, t1, q=90)
    if len(ht) < 8:
        ht, hh = et, eh
    if len(ht) < 8:
        return none
    best = None
    for sub in (2, 3):
        a, per = phase, p
        for tol in (0.2, 0.1):                           # (share of a subdivision a hit may sit off it)
            k = (ht - a) / per
            q = np.round(k * sub) / sub
            r = ht - (a + q * per)
            w = hh * (np.abs(r) < tol * per / sub)
            if (w > 0).sum() < 8:
                break
            for _ in range(5):                           # robust straight line: hit time = a + beat number * per
                A = np.stack([np.ones(len(q)), q], 1)
                W = np.sqrt(w)
                sol, *_ = np.linalg.lstsq(A * W[:, None], ht * W, rcond=None)
                r = ht - A @ sol
                w = hh * (np.abs(r) < tol * per / sub) / np.maximum(1.0, np.abs(r) / 0.012)
            a, per = float(sol[0]), float(sol[1])
        k = (ht - a) / per
        q = np.round(k * sub) / sub
        r = ht - (a + q * per)
        inl = np.abs(r) < min(0.03, 0.12 * per)
        if inl.sum() < 6:
            continue
        on_beat = np.abs(q - np.round(q)) < 1e-9
        score = float(hh[inl].sum()) - 0.5 * float(hh[~inl].sum())
        cand = dict(period=per, anchor=a + int(np.ceil((t0 - a) / per)) * per,
                    rms=float(np.sqrt(np.average(r[inl] ** 2, weights=hh[inl]))), used=int(inl.sum()),
                    share=float(len(np.unique(np.round(k[inl & on_beat]))) / max(1, k1 - k0 + 1)), sub=sub, _score=score)
        if best is None or score > best["_score"] * (1.15 if sub == 3 else 1.0):
            best = cand
    if best is None:
        return none
    best.pop("_score")
    return best


# ---------------------------------------------------------------- 3a. the tempo that holds a stretch together

def coherent_tempo(feat, t0, t1, bpm0, span=0.04, step=0.005):
    """The tempo near bpm0 (within span) at which the onsets between t0 and t1 add up best over the whole stretch at
    once: the weighted size of their components at the beat rate and its multiples. Each harmonic is moved down to
    0 Hz and summed in 0.25 s blocks first, so a fine scan is cheap. -> (bpm, coherence)."""
    i0, i1 = int(max(0, t0 * feat.fps)), int(min(feat.n, t1 * feat.fps))
    x = np.asarray(feat.env[i0:i1], float)
    if len(x) < feat.fps * 4 or x.sum() <= 0:
        return bpm0, 0.0
    x = x - x.mean()
    blk = max(1, int(round(0.25 * feat.fps)))
    nb = len(x) // blk
    t = (i0 + np.arange(nb * blk)) / feat.fps
    tb = (i0 + (np.arange(nb) + 0.5) * blk) / feat.fps
    z = [(x[:nb * blk] * np.exp(-2j * np.pi * bpm0 * h / 60.0 * t)).reshape(nb, blk).sum(axis=1) for h, _ in HARM]
    norm = np.abs(x).sum() + 1e-12

    def size(bpms):
        m = np.zeros(len(bpms))
        for (h, w), zh in zip(HARM, z):
            df = (np.asarray(bpms) - bpm0) * h / 60.0
            for a in range(0, len(df), 128):
                m[a:a + 128] += w * np.abs(np.exp(-2j * np.pi * np.outer(df[a:a + 128], tb)) @ zh)
        return m / norm
    g = np.arange(bpm0 * (1 - span), bpm0 * (1 + span), step)
    s = size(g)
    k = int(np.argmax(s))
    fine = g[k] + np.arange(-step, step + 1e-9, step / 10)
    sf = size(fine)
    j = int(np.argmax(sf))
    return float(fine[j]), float(sf[j])


def phase_of(feat, t0, t1, period):
    """A beat time for a grid of this period over [t0, t1]: the slot comb (16ths, or triplets) slid across one slot
    in 2 ms steps to where it catches the most hit weight, then the slot the kick and snare land on, then nudged
    onto the hits (the median of how far they sit off their slots). -> (beat time, sub)."""
    ht, hh = onset_peaks(feat.hits, feat.fps, t0, t1)
    if len(ht) < 6:
        i0, i1 = int(max(0, t0 * feat.fps)), int(min(feat.n, t1 * feat.fps))
        x = feat.env[i0:i1]
        tt = np.arange(i0, i1) / feat.fps
        c = np.sum((x - x.mean()) * np.exp(-2j * np.pi * tt / period))
        return (-np.angle(c) / (2 * np.pi)) * period, 2
    best = None
    for sub in (2, 3):
        slot = period / (4 if sub == 2 else 3)
        for sh in np.arange(0.0, slot, 0.002):
            _, r = _snap(ht, t0 + sh, period, sub)
            sc = float(hh[np.abs(r) < 0.012].sum())
            if best is None or sc > best[0] * (1.15 if sub == 3 and best[2] == 2 else 1.0):
                if best is None or sub == best[2] or sc > best[0] * 1.15:
                    best = (sc, t0 + sh, sub)
    _, a, sub = best
    a = _beat_phase(feat, t0, t1, a, period, sub)
    if len(ht) >= 6:
        for _ in range(2):
            _, r = _snap(ht, a, period, sub)
            inl = np.abs(r) < 0.03
            if inl.sum() >= 6:
                a += float(np.median(r[inl]))
    return a, sub


def grid_fit(feat, t0, t1, bpm0, span=0.04):
    """Tempo (coherent_tempo) and phase (phase_of) over [t0, t1], and how well the hits sit on that grid.
    -> dict(period, anchor, rms, used, share, sub, coherence)."""
    bpm, coh = coherent_tempo(feat, t0, t1, bpm0, span)
    p = 60.0 / bpm
    a, sub = phase_of(feat, t0, t1, p)
    ht, hh = onset_peaks(feat.hits, feat.fps, t0, t1)
    if len(ht) < 6:
        return dict(period=p, anchor=a, rms=None, used=0, share=0.0, sub=sub, coherence=coh)
    q, r = _snap(ht, a, p, sub)
    inl = np.abs(r) < TRACK_TOL
    k0, k1 = np.ceil((t0 - a) / p), np.floor((t1 - a) / p)
    on_beat = inl & (np.abs(q - np.round(q)) < 1e-6)
    return dict(period=p, anchor=a + k0 * p, rms=float(np.sqrt(np.average(r[inl] ** 2, weights=hh[inl]))) if inl.sum() >= 4 else None,
                used=int(inl.sum()), share=float(len(np.unique(np.round(q[on_beat]))) / max(1, k1 - k0 + 1)), sub=sub, coherence=coh)


# ---------------------------------------------------------------- 3b. the detector's count, and a fit that can't slip

SLOTS = {2: (0.0, 0.25, 0.5, 0.75), 3: (0.0, 1 / 3, 2 / 3)}   # where in a beat a hit may sit: 16ths, or triplets
TRACK_TOL = 0.018                                  # s a hit may sit off its slot and still count
STEP = 5.0                                         # s the fitted window grows by at a time


def det_fit(beats, t0, t1):
    """The detector's count over [t0, t1]: a robust straight line through its beats -> (period, anchor) or None.
    Each gap is counted in half beats of the typical spacing, so a beat it dropped, or a half-beat jump where it
    started counting again on an off-beat (after an odd bar), doesn't tilt the line."""
    b = np.asarray(beats, float)
    b = b[(b >= t0) & (b < t1)]
    if len(b) < 8:
        return None
    iv = np.diff(b)
    p0 = float(np.median(iv))
    steps = np.round(iv / p0 * 2) / 2
    keep = steps >= 0.5
    b = np.r_[b[0], b[1:][keep]]
    k = np.r_[0.0, np.cumsum(steps[keep])]
    w = np.ones(len(b))
    for _ in range(5):
        A = np.stack([np.ones(len(k)), k], 1)
        sol, *_ = np.linalg.lstsq(A * np.sqrt(w)[:, None], b * np.sqrt(w), rcond=None)
        r = b - A @ sol
        w = 1.0 / np.maximum(1.0, np.abs(r) / 0.015)
    return float(sol[1]), float(sol[0])


def _snap(ht, a, per, sub=2):
    """Each hit's beat number on the slot grid (16ths, or triplets) and how far off its slot it is (s)."""
    k = (ht - a) / per
    base = np.floor(k)
    frac = k - base
    slots = np.array(SLOTS[sub] + (1.0,))
    j = np.argmin(np.abs(frac[:, None] - slots[None, :]), axis=1)
    q = base + slots[j]
    return q, ht - (a + q * per)


def track_fit(feat, t0, t1, period, anchor):
    """Exact tempo and phase over [t0, t1], fitted to the hits outward from the densest 20 s, the window growing
    STEP s at a time - each step's grid is only carried a few seconds before the hits correct it, so it can't drift
    onto the next subdivision. Straight (16ths) or triplet slots, whichever the hits in the first window sit on.
    -> dict like fit_hits."""
    ht, hh = onset_peaks(feat.hits, feat.fps, t0, t1)
    none = dict(period=period, anchor=anchor, rms=None, used=0, share=0.0, sub=2)
    if len(ht) < 10:
        return none
    span = 20.0
    if t1 - t0 > span:                                     # where the drumming is densest
        c = np.arange(t0, t1 - span + 1e-9, 2.0)
        dens = [hh[(ht >= x) & (ht < x + span)].sum() for x in c]
        lo = float(c[int(np.argmax(dens))])
    else:
        lo = t0
    hi = min(t1, lo + span)

    def fit(lo, hi, a, per, sub):
        m = (ht >= lo) & (ht < hi)
        if m.sum() < 6:
            return a, per, 0.0
        x, h = ht[m], hh[m]
        got = 0.0
        for _ in range(4):
            q, r = _snap(x, a, per, sub)
            w = h * (np.abs(r) < TRACK_TOL) / np.maximum(1.0, np.abs(r) / 0.01)
            if (w > 0).sum() < 6:
                break
            A = np.stack([np.ones(len(q)), q], 1)
            sol, *_ = np.linalg.lstsq(A * np.sqrt(w)[:, None], x * np.sqrt(w), rcond=None)
            a, per = float(sol[0]), float(sol[1])
            got = float(w.sum() / h.sum())
        return a, per, got
    tries = [fit(lo, hi, anchor, period, s) + (s,) for s in (2, 3)]
    a, per, _, sub = max(tries, key=lambda x: x[2] * (1.0 if x[3] == 2 else 0.87))
    while not (lo <= t0 and hi >= t1):
        lo, hi = max(t0, lo - STEP), min(t1, hi + STEP)
        a, per, _ = fit(lo, hi, a, per, sub)
    a = _beat_phase(feat, t0, t1, a, per, sub)
    q, r = _snap(ht, a, per, sub)
    inl = np.abs(r) < TRACK_TOL
    if inl.sum() < 6:
        return none
    k0, k1 = np.ceil((t0 - a) / per), np.floor((t1 - a) / per)
    beats_hit = len(np.unique(np.round(q[inl & (np.abs(q - np.round(q)) < 1e-6)])))
    return dict(period=per, anchor=a + k0 * per, rms=float(np.sqrt(np.average(r[inl] ** 2, weights=hh[inl]))),
                used=int(inl.sum()), share=float(beats_hit / max(1, k1 - k0 + 1)), sub=sub)


def _beat_phase(feat, t0, t1, a, per, sub):
    """Which of the grid's slots is the beat. Hits on 16ths fit a grid shifted by any number of 16ths just as well,
    so the beat is put where the kick and snare land (summed over the stretch) - the backbeat, not the hi-hat."""
    shifts = SLOTS[sub]
    kick = feat.parts.get("kick")
    body = feat.hits if kick is None else kick + 0.8 * feat.parts.get("snare", 0 * kick)
    best, bs = a, -1.0
    for sh in shifts:
        aa = a + sh * per
        k0, k1 = int(np.ceil((t0 - aa) / per)), int(np.floor((t1 - aa) / per))
        if k1 < k0:
            continue
        idx = feat.idx(aa + np.arange(k0, k1 + 1) * per)
        r = max(1, int(0.015 * feat.fps))
        v = float(np.mean([body[max(0, i - r):i + r + 1].max() for i in idx]))
        if v > bs:
            best, bs = aa, v
    return best


def lock_strength(feat, period, anchor, t0, t1):
    """How much of the onsets between t0 and t1 keep landing at the same places in this grid's beat: the phasors
    at the beat and its 2, 3 and 4 x, weighted like the salience (needs the grid's PHASE right, not just its tempo)."""
    i0, i1 = int(max(0, t0 * feat.fps)), int(min(feat.n, t1 * feat.fps))
    x = feat.env[i0:i1]
    if len(x) < 10 or x.sum() <= 1e-9:
        return 0.0
    pos = (np.arange(i0, i1) / feat.fps - anchor) / period
    return float(sum(w * abs(np.sum(x * np.exp(-2j * np.pi * h * pos))) for h, w in HARM) / x.sum())


CONTINUE = 0.5            # the neighbour's grid carried on keeps at least this share of its own lock strength


# ---------------------------------------------------------------- the whole song

def sections(feat, det_beats, dur):
    """-> [dict(start, end, bpm, anchor, period, how, clarity, rms, share)] covering [0, dur], in order. `how`:
    "main" (the longest stretch), "same" (kept its neighbour's pulse), "own" (a tempo of its own), "none" (no clear
    pulse: its neighbour's grid carried on)."""
    st = stretches(det_beats, dur)
    main = max(range(len(st)), key=lambda i: st[i][1] - st[i][0])
    out = [None] * len(st)
    t0, t1, dbpm = st[main]
    sal = Salience(feat.env, feat.fps, t0, t1)
    df = det_fit(det_beats, t0, t1)
    f = grid_fit(feat, t0, t1, 60.0 / df[0] if df else dbpm)
    gg = _grid(dbpm / 1.6, dbpm * 1.6)
    clar = float(sal(60.0 / f["period"]).max() / (np.median(sal(gg)) + 1e-12))
    for _ in range(3):                           # a pulse a musician would count: 60 to 200 BPM
        b = 60.0 / f["period"]
        if b > TACTUS[1]:
            f = grid_fit(feat, t0, t1, b / 2, span=0.01)
        elif b < TACTUS[0]:
            f = grid_fit(feat, t0, t1, b * 2, span=0.01)
        else:
            break
    f["anchor"] = _det_phase(det_beats, t0, t1, f["period"], f["anchor"])
    out[main] = dict(start=t0, end=t1, bpm=60.0 / f["period"], how="main", clarity=clar, det_bpm=dbpm, **f)
    order = list(range(main - 1, -1, -1)) + list(range(main + 1, len(st)))
    for i in order:
        nb = out[i + 1] if i < main else out[i - 1]
        t0, t1, dbpm = st[i]
        base = nb if nb["how"] != "none" else next((x for x in out if x and x["how"] != "none"), out[main])
        # 1. the neighbour's pulse: is it still the one that holds this stretch together? (its tempo within 0.3 %
        # against the strongest pulse at any counting level near it)
        sal = Salience(feat.env, feat.fps, t0, t1)
        g = _grid(base["bpm"] / 1.6, base["bpm"] * 1.6)
        s = sal(g)
        clar = float(s.max() / (np.median(s) + 1e-12))
        same_bpm, c_same = coherent_tempo(feat, t0, t1, base["bpm"], span=0.003, step=0.002)
        top_bpm, c_best = coherent_tempo(feat, t0, t1, float(g[int(np.argmax(s))]), span=0.01, step=0.005)
        ratio = top_bpm / base["bpm"]
        simple = any(abs(np.log(ratio / q)) < 0.015 for q in RATIOS_)
        if clar >= CLEAR and c_same >= (KEEP_RATIO if simple else KEEP_SAME) * c_best:
            out[i] = dict(start=t0, end=t1, bpm=base["bpm"], how="same", clarity=clar, det_bpm=dbpm,
                          period=base["period"], anchor=base["anchor"], rms=None, used=0, share=0.0, sub=base.get("sub", 2))
            continue
        if clar < CLEAR:
            out[i] = dict(start=t0, end=t1, bpm=base["bpm"], how="none", clarity=clar, det_bpm=dbpm)
            continue
        # 2. a pulse of its own: the detector's count here, at the level nearest the song's main pulse (not the
        # neighbour's - one wrong section mustn't pull the next one's counting level along with it)
        base = out[main]
        df = det_fit(det_beats, t0, t1)
        q = _regroup(60.0 / df[0] if df else dbpm, base["bpm"])
        if q:
            out[i] = _carry(base, t0, t1, clar, dbpm, q)
            continue
        if df:
            dp, da = df
            cands = [(dp / k, k) for k in LEVELS]
            cands = [(p, k) for p, k in cands if 60.0 / p < 2.2 * base["bpm"] and 60.0 / p > base["bpm"] / 2.2
                     and TACTUS[0] <= 60.0 / p <= TACTUS[1]] or cands
            sv = [(p, k, float(sal(60.0 / p)[0])) for p, k in cands]
            top = max(v for _, _, v in sv) if sv else 0.0
            ok = [(p, k) for p, k, v in sv if v >= LEVEL_KEEP * top]
            p, k = min(ok, key=lambda c: abs(np.log((60.0 / c[0]) / base["bpm"]))) if ok else (dp, 1.0)
            f = grid_fit(feat, t0, t1, 60.0 / p)
        else:
            bpm, _, _ = choose_pulse(sal, base["bpm"])
            f = grid_fit(feat, t0, t1, bpm)
        if f["used"] < 8:
            out[i] = dict(start=t0, end=t1, bpm=base["bpm"], how="none", clarity=clar, det_bpm=dbpm)
            continue
        q = _regroup(60.0 / f["period"], base["bpm"])
        if q:
            out[i] = _carry(base, t0, t1, clar, dbpm, q)
            continue
        for _ in range(3):                       # (the same 60-200 BPM as the main pulse)
            b = 60.0 / f["period"]
            if b > TACTUS[1]:
                f = grid_fit(feat, t0, t1, b / 2, span=0.01)
            elif b < TACTUS[0]:
                f = grid_fit(feat, t0, t1, b * 2, span=0.01)
            else:
                break
        f["anchor"] = _det_phase(det_beats, t0, t1, f["period"], f["anchor"])
        out[i] = dict(start=t0, end=t1, bpm=60.0 / f["period"], how="own", clarity=clar, det_bpm=dbpm, **f)
    merged = []
    for s in out:                                        # neighbours that are one grid become one section
        if merged and _same_grid(merged[-1], s):
            m = merged[-1]
            m["end"] = s["end"]
            m["parts"] = m.get("parts", 1) + 1
            if s.get("feel"):
                m.setdefault("feels", []).append(dict(start=s["start"], end=s["end"], feel=s["feel"]))
            if s["how"] != "none" and m["how"] != "none":
                f = grid_fit(feat, m["start"], m["end"], m["bpm"], span=0.006)
                if f["used"] >= 8:
                    m.update(f)
                    m["bpm"] = 60.0 / f["period"]
            elif m["how"] == "none" and s["how"] != "none":
                for k in ("bpm", "period", "anchor", "rms", "used", "share", "how", "clarity"):
                    if k in s:
                        m[k] = s[k]
            continue
        merged.append(dict(s))
    _fill_free(merged)
    _place_changes(feat, merged)
    for s in merged:
        _maybe_follow(feat, s, det_beats)
    return merged


# ---------------------------------------------------------------- 3. takes that drift

FOLLOW_WIN = 3.0          # s of hits either side of a beat that steer the next one
FOLLOW_PHASE = 0.12       # how much of the measured offset each beat takes up (phase)
FOLLOW_RATE = 0.012       # ... and how much goes into the tempo
FOLLOW_MIN = 0.15         # hits weaker than this (circular strength) don't steer
DRIFT_STEADY = 0.6        # a straight grid that stays within 15 ms of its typical offset less than this share ...
DRIFT_GAIN = 0.2          # ... and a following grid that does at least this much better -> the section follows


def follow(feat, t0, t1, period, anchor):
    """A beat grid that follows a take's drift, from a beat `anchor` (where the grid is known to be right) forward to
    t1 and back to t0: beat by beat it moves toward where the drum hits sit in the 16ths (or triplets) around it - a
    slow tracker, so single hits, fills and regroupings don't pull it, but a tempo that wanders over bars does.
    -> beat times over [t0, t1]."""
    ht, hh = onset_peaks(feat.hits, feat.fps, t0 - FOLLOW_WIN, t1 + FOLLOW_WIN, q=80)

    def run(direction):
        b, per, out = anchor, period, []
        while t0 - 1e-9 <= b <= t1 + 1e-9:
            out.append(b)
            i0, i1 = np.searchsorted(ht, [b - FOLLOW_WIN, b + FOLLOW_WIN])
            err = 0.0
            if i1 - i0 >= 6:
                h = ht[i0:i1]
                w = hh[i0:i1] * (0.5 + 0.5 * np.cos(np.pi * (h - b) / FOLLOW_WIN))
                w = w / (w.sum() + 1e-12)
                z, k = max(((np.sum(w * np.exp(2j * np.pi * k * (h - b) / per)), k) for k in (4, 3)), key=lambda x: abs(x[0]))
                if abs(z) >= FOLLOW_MIN:
                    err = np.angle(z) / (2 * np.pi * k) * per        # s, + = the hits are later than this beat
            per = float(np.clip(per + direction * FOLLOW_RATE * err, period * 0.85, period * 1.15))
            b = b + direction * per + FOLLOW_PHASE * err
        return out

    def offsets(bs):
        per = np.gradient(bs) if len(bs) > 1 else np.full(len(bs), period)
        e, ok = np.zeros(len(bs)), np.zeros(len(bs), bool)
        for j, b in enumerate(bs):
            i0, i1 = np.searchsorted(ht, [b - FOLLOW_WIN, b + FOLLOW_WIN])
            if i1 - i0 < 6:
                continue
            h = ht[i0:i1]
            w = hh[i0:i1] * (0.5 + 0.5 * np.cos(np.pi * (h - b) / FOLLOW_WIN))
            w = w / (w.sum() + 1e-12)
            z, k = max(((np.sum(w * np.exp(2j * np.pi * k * (h - b) / per[j])), k) for k in (4, 3)), key=lambda x: abs(x[0]))
            if abs(z) >= FOLLOW_MIN:
                e[j], ok[j] = np.angle(z) / (2 * np.pi * k) * per[j], True
        return e, ok

    bs = np.array(run(-1.0)[1:][::-1] + run(1.0))
    for _ in range(2):                                 # take up what the tracker lagged behind (a steady ramp)
        e, ok = offsets(bs)
        if ok.sum() < 8:
            break
        j = np.arange(len(bs))
        e = np.interp(j, j[ok], e[ok])
        ker = np.hanning(17)
        e = np.convolve(e, ker / ker.sum(), mode="same")
        nb = bs + e
        if np.all(np.diff(nb) > 0.25 * period):
            bs = nb
    # k=1 beat-phase guard: k in (4, 3) subdivision tracking can lock 1/4 or 1/3 beat off the real kick+snare
    # hits while reporting low subdivision RMS (e.g. Reyin: +105.6 ms = exactly 0.25 beat off). Check the actual
    # beat-level onset curve across [-0.5, +0.5] beats and recenter onto the true k=1 beat phase when off.
    if len(bs) >= 8:
        kick = feat.parts.get("kick")
        body = feat.hits if kick is None else kick + 0.85 * feat.parts.get("snare", 0 * kick)
        per_m = float(np.median(np.diff(bs)))
        if per_m > 0.1 and len(body) > 16 and float(body.max()) > 1e-6:
            ker = np.array([0.25, 0.5, 1.0, 0.5, 0.25], float)
            sm = np.convolve(body, ker / ker.sum(), mode="same")
            deltas = np.linspace(-0.5, 0.5, 101)
            sc = np.array([float(sm[feat.idx(bs + d * per_m)].mean()) for d in deltas])
            z0 = float(sc[50])
            loc = np.abs(deltas) <= 0.10
            loc_i = int(np.where(loc)[0][np.argmax(sc[loc])])
            glob_i = int(np.argmax(sc))
            if float(sc[glob_i]) >= 1.8 * max(float(sc[loc_i]), z0, 1e-9):
                bs = bs + float(deltas[glob_i]) * per_m
            elif float(sc[loc_i]) >= 1.15 * max(z0, 1e-9):
                bs = bs + float(deltas[loc_i]) * per_m
    return bs


DET_PHASE_R = 0.7         # the detector's beats set the beat's phase when they sit this steadily on the grid


def _det_phase(det_beats, t0, t1, period, anchor):
    """The grid's phase from the detector's beats, when they count at this tempo and sit steadily on it (a beat
    tracker places the beat better than "where the kick and snare land", which in an odd meter or a syncopated
    riff can be an eighth or a sixteenth off). Otherwise the anchor as it was."""
    b = np.asarray(det_beats, float)
    b = b[(b >= t0) & (b < t1)]
    if len(b) < 8:
        return anchor
    iv = np.median(np.diff(b))
    if abs(iv / period - 1) > 0.03:              # the detector counts at another level here
        return anchor
    z = np.mean(np.exp(2j * np.pi * (b - anchor) / period))
    if abs(z) < DET_PHASE_R:
        return anchor
    return anchor + np.angle(z) / (2 * np.pi) * period


def _maybe_follow(feat, s, det_beats=()):
    """A section whose straight grid slides against the hits (a take with no click, or one that speeds up) gets a
    grid that follows it - only where that measurably sits better."""
    from . import check
    if s["how"] == "none" or s["end"] - s["start"] < 30:
        return
    straight = _beats_between(s, s["start"], s["end"])
    a = check.summary(check.lock(feat, straight))
    if a.get("steady") is None or a["steady"] >= DRIFT_STEADY:
        return
    lk = check.lock(feat, straight)                  # start where the straight grid sits best: fitted there afresh
    c = lk["t"][int(np.argmax(lk["strength"]))] if lk["t"] else 0.5 * (s["start"] + s["end"])
    w0 = max(s["start"], min(c - 12.0, s["end"] - 25.0))
    f = grid_fit(feat, w0, min(s["end"], w0 + 25.0), s["bpm"], span=0.06)
    f["anchor"] = _det_phase(det_beats, w0, min(s["end"], w0 + 25.0), f["period"], f["anchor"])
    a0 = f["anchor"] + np.round((c - f["anchor"]) / f["period"]) * f["period"]
    fb = follow(feat, s["start"], s["end"], f["period"], a0)
    b = check.summary(check.lock(feat, fb))
    if b.get("steady") is None or b["steady"] < a["steady"] + DRIFT_GAIN:
        return
    iv = np.diff(fb)
    s.update(beats=fb, how="follows", straight_bpm=s["bpm"], bpm=60.0 / float(np.median(iv)),
             bpm_lo=60.0 / float(np.percentile(iv, 95)), bpm_hi=60.0 / float(np.percentile(iv, 5)),
             steady_straight=a["steady"], steady_follow=b["steady"])


def _regroup(bpm, main_bpm):
    """The ratio name if `bpm` is the main pulse counted another way (within REGROUP), else None."""
    r = bpm / main_bpm
    q = min(FEELS, key=lambda x: abs(np.log(r / x)))
    return FEELS[q] if abs(np.log(r / q)) < REGROUP else None


def _carry(base, t0, t1, clar, dbpm, feel):
    """The main grid carried on through [t0, t1] (regrouped there, `feel` = what the detector counted)."""
    return dict(start=t0, end=t1, bpm=base["bpm"], how="same", clarity=clar, det_bpm=dbpm, period=base["period"],
                anchor=base["anchor"], rms=None, used=0, share=0.0, sub=base.get("sub", 2), feel=feel)


def _same_grid(a, b):
    if a["how"] == "none" or b["how"] == "none":
        return a["how"] == "none" and b["how"] == "none" or abs(np.log(a["bpm"] / b["bpm"])) < SAME
    if abs(a["period"] / b["period"] - 1) > MERGE:
        return False
    # does a's grid, carried on, land on b's beats?
    k = (b["anchor"] - a["anchor"]) / a["period"]
    return abs(k - round(k)) < PHASE_OK


def _fill_free(secs):
    """A section with no clear pulse takes its neighbour's grid, carried on (the next one's if it has one, so an
    intro lines up with where the song comes in)."""
    for i, s in enumerate(secs):
        if s["how"] != "none":
            continue
        src = next((secs[j] for j in list(range(i + 1, len(secs))) + list(range(i - 1, -1, -1)) if secs[j]["how"] != "none"), None)
        if src is None:
            p = 60.0 / s["bpm"]
            s.update(period=p, anchor=s["start"], rms=None, used=0, share=0.0)
            continue
        s.update(period=src["period"], anchor=src["anchor"], bpm=src["bpm"], rms=None, used=0, share=0.0)


def _beats_between(s, t0, t1):
    if s.get("beats") is not None:                     # a section that follows the drift: its own beat list
        b = np.asarray(s["beats"], float)
        return b[(b >= t0 - 1e-9) & (b <= t1 + 1e-9)]
    p, a = s["period"], s["anchor"]
    k0, k1 = int(np.ceil((t0 - a) / p - 1e-9)), int(np.floor((t1 - a) / p - 1e-9))
    return a + np.arange(k0, k1 + 1) * p


def _place_changes(feat, secs):
    """Move each change of tempo onto the beat where the hits stop following one grid and start following the next
    (searched within 8 beats of where the detector saw it)."""
    def fit_score(times):
        if not len(times):
            return 0.0
        return sum(_peak_near(feat.hits, feat.fps, t, 0.025)[1] for t in times)
    for i in range(len(secs) - 1):
        a, b = secs[i], secs[i + 1]
        if a["how"] == "none" or b["how"] == "none" or abs(a["period"] / b["period"] - 1) <= MERGE:
            b0 = _beats_between(b, b["start"] - b["period"] * 0.5, b["start"] + b["period"] * 0.5)
            if len(b0):
                a["end"] = b["start"] = float(b0[0])
            continue
        lo = max(a["start"] + 4 * a["period"], b["start"] - 8 * b["period"])
        hi = min(b["end"] - 4 * b["period"], b["start"] + 8 * b["period"])
        cands = _beats_between(b, lo, hi)
        if not len(cands):
            continue
        win0, win1 = lo - 8 * a["period"], hi + 8 * b["period"]
        best = max(cands, key=lambda c: fit_score(_beats_between(a, win0, c - 0.5 * a["period"]))
                   + fit_score(_beats_between(b, c, win1)))
        a["end"] = b["start"] = float(best)
