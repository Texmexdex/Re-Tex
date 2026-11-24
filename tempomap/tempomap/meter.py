"""Bars: how many subdivisions each bar has, and where bar 1 falls - from the music, not assumed to be 4/4.

1. The grid's subdivisions (halves of the beat, or thirds where the feel is triplet - judged a few beats at a time:
   a song can go triplet at the same tempo).
2. Each subdivision gets a short description of what happens on it: kick, snare, cymbal, bass and other onsets, and
   the chord sounding (chroma).
3. Bar length = how often the RHYTHM repeats (the drums, bass and other onsets), measured a few bars at a time: the
   similarity of every subdivision to the one L later. Chords don't vote on the length (a progression repeats
   every 2-4 bars, which would make every bar look 4 bars long); where there is no rhythm to go by (an intro with
   no drums), how regularly the chord CHANGES does. In 4/4 the pattern also repeats every half bar (kick-snare-kick-snare), and a two-bar riff
   repeats every two bars, so the length is chosen as the most usual one that the music supports (8 subdivisions
   = 4/4 first, then 3/4, ...) - but only when the pattern really repeats at that length; a riff in 7 repeats at
   7, 14, 21 and never at 8.
4. A long repeating unit is split where the rhythm starts over inside it (the subdivisions after that point play
   what the bar's first ones play) on a clear accent: a 12 that goes 5 + 7 becomes a bar of 5/8 and a bar of 7/8;
   a 5/4 whose kicks and snares just alternate stays one bar.
5. Bar 1 = where the accents are strongest along the bar: kick, crash, chord change, bass.
"""
import numpy as np

PRIOR = {2: [8, 6, 12, 7, 5, 9, 10, 11, 13, 14, 15, 16, 4], 3: [12, 6, 9, 15, 18, 24]}
LMAX = {2: 32, 3: 36}
SUPPORT = 0.92            # the length's multiples must reach this share of the best repetition score ...
OWN = 0.6                 # ... and the length itself this share (a progression or a crash every 4 bars makes the
                          # 4-bar repeat the strongest; the bar itself still repeats clearly)
WIN = 48                  # subdivisions the repetition is measured over (centred)
HOP = 4
SPLIT_ACCENT = 0.5        # a long unit splits where its rhythm starts over (this alike to the bar's start) ...
SPLIT_SAME = 0.8          # ... on an accent at least this strong against bar 1's


def sub_runs(feat, beat_t, sub_hint):
    """Subdivision per beat (2 or 3): the onsets locked to the grid at 2 and 4 x the beat against 3 and 6 x, a few
    beats at a time. -> list of 2 / 3, one per beat interval."""
    n = len(beat_t) - 1
    if n <= 0:
        return []
    pos_t = np.asarray(beat_t)
    t = np.arange(feat.n) / feat.fps
    inside = (t >= pos_t[0]) & (t < pos_t[-1])
    pos = np.full(feat.n, np.nan)
    pos[inside] = np.interp(t[inside], pos_t, np.arange(len(pos_t)))
    e = feat.env
    raw = []
    for j in range(0, n, 2):
        a, b = max(0, j - 4), min(n, j + 4)
        m = inside & (pos >= a) & (pos < b)
        if m.sum() < 10 or e[m].sum() <= 0:
            raw.append(sub_hint)
            continue
        x, p = e[m], pos[m]
        mag = {h: abs(np.sum(x * np.exp(-2j * np.pi * h * p))) / x.sum() for h in (2, 3, 4, 6)}
        raw.append(3 if mag[3] + mag[6] > 1.2 * (mag[2] + mag[4]) else 2)
    raw = np.repeat(raw, 2)[:n]
    k = 5                                          # smooth: a feel lasts more than a couple of beats
    out = np.array([int(np.round(np.median(raw[max(0, i - k):i + k + 1]))) for i in range(n)])
    return [3 if x >= 3 else 2 for x in out]


def tatums(beat_t, subs):
    """Subdivision times and, for each, (its beat's subdivision count, whether it starts a beat)."""
    t, s, on = [], [], []
    for j in range(len(beat_t) - 1):
        a, b = beat_t[j], beat_t[j + 1]
        for q in range(subs[j]):
            t.append(a + (b - a) * q / subs[j])
            s.append(subs[j])
            on.append(q == 0)
    if len(beat_t):
        t.append(beat_t[-1]); s.append(subs[-1] if subs else 2); on.append(True)
    return np.array(t), np.array(s), np.array(on)


def describe(feat, tt):
    """What happens on each subdivision: (onsets [n, 5]: kick, snare, cymbal, bass, other near it - each column
    scaled to mean 0, spread 1; chroma [n, 12]: the chord over it, as measured)."""
    parts = [feat.parts.get(k) for k in ("kick", "snare", "cymbal", "bass", "other")]
    reach = int(0.02 * feat.fps)
    rows = []
    idx = feat.idx(tt)
    for p in parts:
        if p is None:
            rows.append(np.zeros(len(tt)))
            continue
        rows.append(np.array([p[max(0, i - reach):i + reach + 1].max() for i in idx]))
    on = np.stack(rows, 1)
    ci = np.clip(np.round(np.asarray(tt) * feat.cfps).astype(int), 0, len(feat.chroma) - 1)
    ce = np.clip(np.r_[ci[1:], ci[-1] + 1], ci + 1, None)
    ch = np.stack([feat.chroma[a:b].mean(0) if b > a else feat.chroma[min(a, len(feat.chroma) - 1)] for a, b in zip(ci, ce)])
    on = np.log1p(5.0 * on)                          # (a crash shouldn't count ten times a kick)
    on = (on - on.mean(0)) / (on.std(0) + 1e-9)
    on[:, 2] *= 0.5                                  # cymbals mark phrases as much as bars
    return on, ch


def accents(feat, tt, on, ch, subs):
    """How much each subdivision sounds like the start of a bar: kick, crash, bass, and the chord changing there
    (the chord over the beat after it against the beat before it)."""
    n = len(tt)
    chg = np.zeros(n)
    raw_ch = ch
    for k in range(n):
        s = subs[k]
        a, b = raw_ch[max(0, k - s):k], raw_ch[k:k + s]
        if len(a) and len(b):
            u, v = a.mean(0), b.mean(0)
            chg[k] = 1.0 - float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-9))
    chg = (chg - chg.mean()) / (chg.std() + 1e-9)
    return on[:, 0] + 0.7 * on[:, 2] + 0.5 * on[:, 3] + 1.0 * chg, chg


def _change_scores(chg, lags):
    """S[L][k] from the chord changes alone: how often a change L subdivisions later matches one here (a local
    autocorrelation of the change curve, scaled like a similarity)."""
    n = len(chg)
    x = np.clip(chg, 0, None)
    out = {}
    for L in lags:
        s = np.full(n, np.nan)
        if L < n:
            s[:n - L] = x[:n - L] * x[L:]
        v = np.nan_to_num(s)
        c = np.concatenate([[0.0], np.cumsum(v)])
        e = np.concatenate([[0.0], np.cumsum(x * x)])
        h = WIN // 2
        lo = np.clip(np.arange(n) - h, 0, n)
        hi = np.clip(np.arange(n) + h, 0, n)
        out[L] = (c[hi] - c[lo]) / np.maximum(1e-9, e[hi] - e[lo])
    return out


def _repeat_scores(F, lags):
    """S[L][k]: how alike the pattern is to itself L subdivisions later, averaged over a window centred on k."""
    n = len(F)
    Fn = F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-9)
    out = {}
    for L in lags:
        s = np.full(n, np.nan)
        if L < n:
            s[:n - L] = np.sum(Fn[:n - L] * Fn[L:], axis=1)
        v = np.nan_to_num(s)
        c = np.concatenate([[0.0], np.cumsum(v)])
        cnt = np.concatenate([[0], np.cumsum(~np.isnan(s))])
        h = WIN // 2
        lo = np.clip(np.arange(n) - h, 0, n)
        hi = np.clip(np.arange(n) + h, 0, n)
        m = np.maximum(1, cnt[hi] - cnt[lo])
        out[L] = (c[hi] - c[lo]) / m
    return out


def choose_length(S, k, sub):
    """The bar length at subdivision k (see the module text). -> (length, confidence 0..1)."""
    lags = [L for L in S if not np.isnan(S[L][k])]
    if not lags:
        return PRIOR[sub][0], 0.0
    best = max(S[L][k] for L in lags)
    if best <= 0.05:
        return PRIOR[sub][0], 0.0
    for B in PRIOR[sub]:
        if B not in S:
            continue
        mult = [m * B for m in (1, 2, 3, 4) if m * B in S]
        sup = max(S[L][k] for L in mult)
        if sup >= SUPPORT * best and S[B][k] >= OWN * best:
            return B, float(S[B][k])
    return max(lags, key=lambda L: S[L][k]), float(best)


SWITCH = 2.5              # points (of HOP subdivisions) a new bar length must win for before the path changes


def _segment(Sr, Sh, loud, pts, sub, first):
    """Bar length at every point as one path with as few changes as the music asks for (dynamic programming over
    the point-by-point choices): a change of length costs SWITCH, so a length has to keep winning for about two
    bars before the path takes it - a bar or two of a fill doesn't become a meter. -> (lengths, confidence)."""
    cands = sorted(set(first))
    n = len(pts)
    if not n:
        return first, []
    sc = np.array([[1.0 if c == f else 0.0 for c in cands] for f in first])
    best = sc[0].copy()
    back = np.zeros((n, len(cands)), int)
    for q in range(1, n):
        stay = best
        move = best.max() - SWITCH
        back[q] = np.where(stay >= move, np.arange(len(cands)), int(np.argmax(best)))
        best = np.maximum(stay, move) + sc[q]
    path = [int(np.argmax(best))]
    for q in range(n - 1, 0, -1):
        path.append(int(back[q][path[-1]]))
    path = path[::-1]
    lens = [cands[j] for j in path]
    conf = []
    for q, k in enumerate(pts):
        S = Sr
        conf.append(float(np.nan_to_num(S[lens[q]][k])))
    return lens, conf


def bars(feat, tt, subs, det_downbeats=(), det_beats=()):
    """Bar lines over the subdivisions: -> (bar starts [subdivision indices], meter runs [dict(i0, i1, length,
    sub, confidence)])."""
    n = len(tt)
    if n < 16:
        return [0], [dict(i0=0, i1=n, length=8, sub=2, confidence=0.0)]
    on, ch = describe(feat, tt)
    acc, chg = accents(feat, tt, on, ch, subs)
    rhythm = np.abs(on).sum(1)
    out_runs = []
    # runs of one subdivision kind first (a triplet-feel part is measured on its own)
    kinds, i = [], 0
    while i < n:
        j = i
        while j < n and subs[j] == subs[i]:
            j += 1
        kinds.append((i, j, int(subs[i])))
        i = j
    for i0, i1, sub in kinds:
        lags = list(range(4 if sub == 2 else 6, LMAX[sub] + 1))
        Sr = _repeat_scores(on[i0:i1], lags)
        Sh = _change_scores(chg[i0:i1], lags)
        loud = feat.hits[feat.idx(tt[i0:i1])]
        pts = list(range(0, i1 - i0, HOP))
        ch_ = []
        for k in pts:                                  # rhythm where there is some, else the chord changes
            w = loud[max(0, k - WIN // 2):k + WIN // 2]
            has_rhythm = float(np.mean(w > 0.15)) > 0.2 if len(w) else False
            ch_.append(choose_length(Sr if has_rhythm else Sh, k, sub))
        sm, conf = _segment(Sr, Sh, loud, pts, sub, [c[0] for c in ch_])
        q = 0
        while q < len(sm):
            r = q
            while r < len(sm) and sm[r] == sm[q]:
                r += 1
            a = i0 + pts[q]
            b = i0 + (pts[r] if r < len(sm) else i1 - i0)
            out_runs.append(dict(i0=a, i1=b, length=int(sm[q]), sub=sub, confidence=float(np.mean(conf[q:r]))))
            q = r
    # runs shorter than two of their bars join the neighbour that is more sure
    changed = True
    while changed and len(out_runs) > 1:
        changed = False
        for i, r in enumerate(out_runs):
            if r["i1"] - r["i0"] >= 2 * r["length"]:
                continue
            nb = [j for j in (i - 1, i + 1) if 0 <= j < len(out_runs) and out_runs[j]["sub"] == r["sub"]]
            if not nb:
                continue
            j = max(nb, key=lambda j: out_runs[j]["confidence"])
            out_runs[j]["i0"], out_runs[j]["i1"] = min(out_runs[j]["i0"], r["i0"]), max(out_runs[j]["i1"], r["i1"])
            del out_runs[i]
            changed = True
            break
    merged = []
    for r in out_runs:
        if merged and merged[-1]["length"] == r["length"] and merged[-1]["sub"] == r["sub"]:
            merged[-1]["i1"] = r["i1"]
            continue
        merged.append(r)
    # a run that is two (or three) of its neighbour's bars is that meter (two 7/8 bars read as one 14); a scrap of
    # 2/4 at a change belongs to the run beside it
    small = {2: 5, 3: 6}
    changed = True
    while changed and len(merged) > 1:
        changed = False
        for i in range(len(merged) - 1):
            A, B = merged[i], merged[i + 1]
            if A["sub"] != B["sub"]:
                continue
            la, lb = A["length"], B["length"]
            keep = None
            if la < small[A["sub"]] and lb >= small[A["sub"]]:
                keep = B
            elif lb < small[A["sub"]] and la >= small[A["sub"]]:
                keep = A
            elif la != lb and max(la, lb) % min(la, lb) == 0 and min(la, lb) >= small[A["sub"]]:
                keep = A if la < lb else B
            if keep is None:
                continue
            keep["i0"], keep["i1"] = A["i0"], B["i1"]
            merged[i:i + 2] = [keep]
            changed = True
            break
    det = np.asarray(det_downbeats, float)
    starts = []
    for r in merged:
        L, i0, i1 = r["length"], r["i0"], r["i1"]
        prof = np.zeros(L)
        for ph in range(L):
            ks = np.arange(i0 + ph, i1, L)
            prof[ph] = acc[ks].mean() if len(ks) else -9
        # (the detector's bar lines get no vote: after any odd bar they count on from the wrong beat)
        ph = int(np.argmax(prof))
        r["phase"] = ph
        r["profile"] = [round(float(x), 3) for x in prof]
        # one strong second accent inside a long unit: split it there (5 + 7, 3 + 3 + 2 ...)
        r["split"] = None
        if L >= 10:
            rel = np.roll(prof, -ph)
            starts_ = np.arange(i0 + ph, i1 - L + 1, L)
            if len(starts_) >= 2:
                pat = np.stack([on[k:k + L] for k in starts_]).mean(0)      # the bar's rhythm, position by position
                best = None
                for j in range(4, L - 3):
                    mlen = min(j, L - j)
                    a, b = pat[:mlen].ravel(), pat[j:j + mlen].ravel()
                    sim = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
                    if sim >= SPLIT_SAME and rel[j] >= SPLIT_ACCENT * rel[0] and (best is None or sim > best[1]):
                        best = (j, sim)
                if best:
                    r["split"] = int(best[0])
        r["first"] = i0 + ph
    _refine_changes(on, merged)
    for r in merged:
        k = r["first"]
        while k < r["i1"]:
            starts.append(k)
            if r["split"] and k + r["split"] < r["i1"]:
                starts.append(k + r["split"])
            k += r["length"]
    starts = sorted(set(s for s in starts if 0 <= s < n))
    return starts, merged


def _bar_pattern(on, first, L, a, b):
    ks = [k for k in range(first, b - L + 1, L) if k >= a]
    return np.stack([on[k:k + L] for k in ks]).mean(0) if len(ks) >= 2 else None


def _like(x, p):
    if p is None or len(x) != len(p):
        return 0.0
    u, v = x.ravel(), p.ravel()
    return float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-9))


def _refine_changes(on, runs):
    """Put each change of bar length on the bar line where the bars before it play like the first run's bars and
    the bars after it like the second's (searched within 3 bars of where the window first saw it). The second run's
    bars then count from that bar line."""
    for i in range(len(runs) - 1):
        A, B = runs[i], runs[i + 1]
        if A["sub"] != B["sub"]:
            continue
        LA, LB = A["length"], B["length"]
        pa = _bar_pattern(on, A["first"], LA, A["i0"], A["i1"] - 2 * LA)
        pb = _bar_pattern(on, B["first"] + 2 * LB, LB, B["i0"], B["i1"])
        if pa is None or pb is None:
            continue
        lo, hi = B["i0"] - 3 * LA, B["i0"] + 3 * LB
        cands = [k for k in range(A["first"], hi + 1, LA) if k >= max(lo, A["i0"] + LA)]
        best, bk = None, None
        for k in cands:
            sa = sum(_like(on[k - m * LA:k - (m - 1) * LA], pa) for m in (1, 2, 3) if k - m * LA >= A["i0"])
            sb = sum(_like(on[k + m * LB:k + (m + 1) * LB], pb) for m in (0, 1, 2) if k + (m + 1) * LB <= B["i1"])
            if best is None or sa + sb > best:
                best, bk = sa + sb, k
        if bk is not None and A["i0"] < bk < B["i1"] - LB:
            A["i1"] = B["i0"] = bk
            B["first"] = bk


def _detector_agrees(tt, subs, i0, i1, det_beats):
    """Did the detector count the same beats here (nearly every one of its beats on one of ours, about as many)?"""
    b = np.asarray(det_beats, float)
    on = [tt[k] for k in range(i0, min(i1, len(tt))) if k == 0 or subs[k] != subs[k - 1] or (k - i0) % subs[k] == 0]
    if not len(b) or len(on) < 8:
        return False
    d = b[(b >= tt[i0]) & (b < tt[min(i1, len(tt) - 1)])]
    if len(d) < 8:
        return False
    on = np.asarray(on)
    j = np.clip(np.searchsorted(on, d), 1, len(on) - 1)
    hit = np.minimum(np.abs(on[j] - d), np.abs(on[j - 1] - d)) < 0.03
    return hit.mean() > 0.85 and 0.85 < len(d) / len(on) < 1.15


def label(length, sub):
    """The usual name of a bar of `length` subdivisions."""
    if sub == 3:
        return f"{length}/8"
    return f"{length // 2}/4" if length % 2 == 0 else f"{length}/8"


METHOD = "seg"            # "seg": the whole song cut into bars at once (barseg); "local": the older window method


def bars_seg(feat, tt, subs, w=None):
    """Bar lines by cutting the whole song into bars at once (see barseg). Same return as bars()."""
    from . import barseg
    n = len(tt)
    if n < 40:
        return [0], [dict(i0=0, i1=n, length=8, sub=2, confidence=0.0)]
    on, ch = describe(feat, tt)
    acc, chg = accents(feat, tt, on, ch, subs)
    if ACCENT == "learned":
        acc = downbeat_features(on, chg, subs) @ np.asarray(DOWNBEAT_COEF)
    acc = (acc - acc.mean()) / (acc.std() + 1e-9)
    dl = getattr(feat, "down_logits", None)
    if dl is not None and len(dl) and BT_WEIGHT > 0:   # Beat This!'s downbeat logits, at each subdivision
        fr = np.clip(np.round(np.asarray(tt) * 50.0).astype(int), 0, len(dl) - 1)
        x = np.maximum.reduce([dl[np.clip(fr + k, 0, len(dl) - 1)] for k in (-1, 0, 1)])
        x = np.clip(x, -8, 8)
        acc = acc * LOCAL_WEIGHT + BT_WEIGHT * (x - x.mean()) / (x.std() + 1e-9)
    starts = barseg.segment(on, ch, acc, np.asarray(subs), w)
    # how sure: where Beat This!'s downbeat probability is, how much of it the run's bar lines carry
    prob = None
    if dl is not None and len(dl):
        fr = np.clip(np.round(np.asarray(tt) * 50.0).astype(int), 0, len(dl) - 1)
        lg = np.maximum.reduce([dl[np.clip(fr + k, 0, len(dl) - 1)] for k in (-1, 0, 1)])
        prob = 1.0 / (1.0 + np.exp(-np.clip(lg, -20, 20)))
    runs = []
    for k, s in enumerate(starts):
        e = starts[k + 1] if k + 1 < len(starts) else n
        L = e - s
        if k + 1 == len(starts):
            L = runs[-1]["length"] if runs else L
        if runs and runs[-1]["length"] == L and runs[-1]["i1"] == s:
            runs[-1]["i1"] = e
        else:
            runs.append(dict(i0=s, i1=e, length=int(L), sub=int(subs[min(s, n - 1)]), confidence=0.5))
    if prob is not None:
        st = np.asarray(starts)
        for r in runs:
            b = st[(st >= r["i0"]) & (st < r["i1"])]
            r["confidence"] = float(np.mean(prob[b])) if len(b) else 0.0
    return list(starts), runs


WIDE = True
# learned on the MIDI answer key (tests/downbeat_train.py): bar-1-ness = DOWNBEAT_COEF . downbeat_features
DOWNBEAT_COEF = [0.2579, 0.6431, -0.2998, 0.0792, 0.3674, -0.1178, -0.1008, -0.0633, -0.0049, -0.0076, 0.217, -0.0285, 0.1567, -0.0477, 0.0016, -0.016, -0.0091, 0.0997, -0.1967, 0.5274, -0.1006, -0.1854, -0.0887, -0.0499, -0.2221, 0.4978, 0.0769, -0.0688, 0.0251, -0.0139, -0.0063, -0.0125, -0.0093, -0.1457, -0.0859, -0.0285, 0.1567, -0.0477, 0.0016, -0.016, -0.1006, -0.0014, 0.1146, -0.1967, -0.1288, 0.1116, -0.066, 0.0822, -0.4057, -0.2991, -0.0612, -0.2244, 0.031, -0.1178, -0.1008, 0.41, -0.0295, -0.0433, -0.1854, 0.0228, -0.0958, -0.1631, 0.1271, 0.0515, -0.0535, -0.0633, -0.0049, 0.3852, 0.0539, 0.0061, 0.0416, -0.1, -0.0872, -0.1327, -0.1017, -0.1585, 0.0932, -0.0076, 0.217, 0.2229, 0.0841, 0.0238, 0.0121, -0.0778, -0.0102, -0.3302, -0.2288, -0.0775, -0.1489, 0.2371, 0.0819, -0.3812, -0.0069, 0.1131, 0.3492, 0.1473, -0.0997, -0.0325, 0.1403, 0.0034, -0.0873, -0.1392, 0.1134, 0.3603, -0.0224, 0.1157, -0.1081, -0.1777, -0.0427, -0.282, -0.0332, -0.0829, -0.0521]
ACCENT = "learned"
BT_WEIGHT = 1.0           # Beat This!'s downbeat logits in the accent (when the features carry them)
LOCAL_WEIGHT = 0.5        # ... and the onsets' own accent        # "learned" or "hand" (kick + crash + bass + chord change)
DB_CHG = 19               # column of the chord change in downbeat_features


def downbeat_features(on, chg, subs):
    """What a subdivision and the beats around it sound like, for telling bar 1 from the rest: onsets here (5),
    one, two and three beats later (kick and snare: 6), a beat before (5), how busy the drums are just before
    against just after (2), the cymbal a subdivision before (1: the end of a fill), the chord change here and a
    beat either side (3), whether the feel is triplet (1). -> [n, 23]."""
    n = len(on)
    sub = np.asarray(subs, int)

    def shift(x, k):                                   # x[i + k], zero off the ends
        out = np.zeros_like(x)
        if k > 0:
            out[:n - k] = x[k:]
        elif k < 0:
            out[-k:] = x[:n + k]
        else:
            out[:] = x
        return out
    beat = int(np.round(np.median(sub))) if n else 2
    cols = [on[:, c] for c in range(5)]
    for k in (1, 2, 3):
        cols += [shift(on[:, 0], k * beat), shift(on[:, 1], k * beat)]
    cols += [shift(on[:, c], -beat) for c in range(5)]
    ks = np.abs(on[:, 0]) + np.abs(on[:, 1])
    c = np.r_[0.0, np.cumsum(ks)]
    i = np.arange(n)
    w = 4 * beat
    before = (c[i] - c[np.maximum(0, i - w)]) / w
    after = (c[np.minimum(n, i + w)] - c[i]) / w
    cols += [before, after, shift(on[:, 2], -1)]
    cols += [chg, shift(chg, -beat), shift(chg, beat)]
    cols += [(sub == 3).astype(float)]
    if WIDE:                                           # the whole neighbourhood, position by position
        for k in list(range(-4, 0)) + list(range(1, 12)):
            cols += [shift(on[:, c], k) for c in range(5)] + [shift(chg, k)]
    return np.stack(cols, 1)
