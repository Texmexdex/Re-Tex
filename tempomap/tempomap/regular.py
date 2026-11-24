"""The grid where the music doesn't say: the user's rules (Nov 16).

1. Where there's no beat to hear (an ambient intro or outro, a break with no drums and no pulse), the grid carries
   on at the tempo of the music next to it: evenly spaced beats, so a lock plays that stretch at one steady speed -
   no warping where there's nothing to line up. Marked "no beat".
2. Where the bars come out in another meter than the song mostly is in, but that stretch is short (under 4 bars),
   unsure, or has no beat, it is gridded in the song's main meter, carried on from the bar line before it. If the
   count lands on the bar line after it, nothing else changes; if it doesn't, the one bar that takes up the
   difference is marked so you can check it. Marked "assumed".
3. A long stretch (4 bars or more) in another meter that the bar finder isn't sure of, where there are beats: gridded
   the same way (plays as if nothing changes) and marked "unknown" - yours to correct.
A long stretch in another meter that the bar finder IS sure of (confidence >= KEEP) stays as it is: a real 7/8.

Every span this touches is listed in map["spans"] ({start, end, kind, note}) and its marks are unsure (amber), so
nothing here is quietly decided for you.
"""
import numpy as np

GAPS, BARS, STRAIGHT = True, True, True     # each rule can be switched off (tests/key_run.py measures them)
KEEP = 0.45           # a detected odd meter this sure (and 4+ bars long, with beats) is real and stays
MIN_BARS = 4          # shorter than this many bars of the main meter: assumed to be the main meter
HITS_SHARE = 0.25     # a window with this share of the song's usual drum level or more has a beat ...
PULSE = 0.15          # ... and so does one whose onsets keep to the grid this well (any of 1-8 per beat)
MIN_GAP = 4           # beats: a shorter dip (a fill, a one-beat stop) isn't a "no beat" stretch


def presence(feat, beats, half=4):
    """-> per beat, True where there's a beat to hear around it (drums at a usual level, or onsets keeping to the
    grid). Windows of 2 x half beats."""
    b = np.asarray(beats, float)
    n = len(b)
    if n < 2 * half + 2:
        return np.ones(n, bool)
    fps = feat.fps
    idx = np.arange(n, dtype=float)
    level = np.zeros(n)
    pulse = np.zeros(n)
    for i in range(n):
        a, z = b[max(0, i - half)], b[min(n - 1, i + half)]
        i0, i1 = int(a * fps), max(int(a * fps) + 2, int(z * fps))
        h = feat.hits[i0:i1]
        e = feat.env[i0:i1]
        level[i] = float(h.mean()) if len(h) else 0.0
        if len(e) and e.sum() > 1e-9:
            pos = np.interp(np.arange(i0, i0 + len(e)) / fps, b, idx)
            pulse[i] = max(abs(np.sum(e * np.exp(2j * np.pi * k * pos))) / e.sum() for k in (1, 2, 3, 4, 6, 8))
    usual = float(np.median(level[level > 0.25 * level.max()])) if level.max() > 0 else 0.0
    return (level >= HITS_SHARE * usual) | (pulse >= PULSE)


def no_beat_spans(feat, beats):
    """-> [(i0, i1)]: runs of beats (i0 <= i < i1) with nothing to hear, at least MIN_GAP long."""
    p = presence(feat, beats)
    out, i, n = [], 0, len(p)
    while i < n:
        if p[i]:
            i += 1
            continue
        j = i
        while j < n and not p[j]:
            j += 1
        if j - i >= MIN_GAP and j - i < n:              # (a song with no beat anywhere: leave it alone)
            out.append((i, j))
        i = j
    return out


STRAIGHT_SLACK = 0.003  # s: a steady line is taken when it sits on the hits within this of the followed beats
STRAIGHT_NEAR = (0.1, 0.2)   # beats: the line's median / 90th-percentile distance from the followed beats
STRAIGHT_MIN = 32       # beats: shorter runs keep the follower's beats


def _hit_error(ht, grid, sub):
    """Median distance (s) of the hits from the nearest subdivision of a beat grid."""
    g = np.asarray(grid, float)
    if len(ht) < 8 or len(g) < 4:
        return None
    pos = np.interp(ht, g, np.arange(len(g), dtype=float))
    per = np.interp(ht, g[:-1], np.diff(g))
    q = pos * sub
    return float(np.median(np.abs(q - np.round(q)) * per / sub))


def straighten(feat, beats, present):
    """Where the beats follow small wobbles the drums don't have (a tracker settling after an intro, a fill pulling
    it), one steady tempo: for each run of beats with something to hear, a straight line fitted to its hits replaces
    the followed beats when it sits on the hits as well (within STRAIGHT_SLACK). Real drift (a live take) fits worse
    straight and keeps its following. -> (new beat times, [(i0, i1, bpm)] runs made steady)."""
    from .pulse import fit_hits, onset_peaks
    b = np.asarray(beats, float).copy()
    p_ok = np.asarray(present, bool)
    out, done = b.copy(), []
    i, n = 0, len(b)
    while i < n:
        if not p_ok[i]:
            i += 1
            continue
        j = i
        while j < n and p_ok[j]:
            j += 1
        if j - i >= STRAIGHT_MIN:
            seg = b[i:j]
            p0 = float(np.median(np.diff(seg)))
            r = fit_hits(feat, seg[0], seg[-1], 60.0 / p0, phase=float(seg[0]))
            if r.get("rms") is not None and r["used"] >= 16 and abs(r["period"] / p0 - 1) < 0.03:
                ia = int(np.argmin(np.abs(seg - r["anchor"])))
                line = r["anchor"] + (np.arange(len(seg)) - ia) * r["period"]
                ht, _ = onset_peaks(feat.hits, feat.fps, seg[0], seg[-1], q=80)
                # measured on the finest subdivision the hits keep to (16ths in a blast beat): against chance
                # (a quarter of a subdivision), the subdivision the steady line fits best
                subs = [k for k in (2, 3, 4, 6) if _hit_error(ht, line, k) is not None]
                if not subs:
                    i = j
                    continue
                k = min(subs, key=lambda q: _hit_error(ht, line, q) / (r["period"] / q / 4))
                e_line, e_now = _hit_error(ht, line, k), _hit_error(ht, seg, k)
                dev = np.abs(line - seg) / p0                # it smooths the beats; it doesn't move the grid:
                near = np.median(dev) <= STRAIGHT_NEAR[0] and np.percentile(dev, 90) <= STRAIGHT_NEAR[1]
                if near and e_line is not None and e_now is not None and e_line <= e_now + STRAIGHT_SLACK:
                    out[i:j] = line
                    done.append((i, j, round(60.0 / r["period"], 3), round(e_line * 1000, 1), round(e_now * 1000, 1)))
        i = j
    return out, done


def remap(beats, spans):
    """-> f(times): the time warp that makes the beats of every no-beat span even. A span at the start or end goes
    on at the tempo of the 8 beats next to it; one in the middle spreads its beats evenly between the beats either
    side (same count, one steady speed). Outside the spans nothing moves."""
    b = np.asarray(beats, float)
    return warp_fn(b, even_gaps(b, spans))


def even_gaps(beats, spans):
    """The beats with every no-beat span made even (see remap)."""
    b = np.asarray(beats, float)
    n = len(b)
    new = b.copy()
    for i0, i1 in spans:
        if i0 == 0:
            p = float(np.median(np.diff(b[i1:i1 + 25]))) if n - i1 >= 3 else float(np.median(np.diff(b)))
            new[:i1] = b[i1] - (i1 - np.arange(i1)) * p
        elif i1 >= n:
            p = float(np.median(np.diff(b[max(0, i0 - 25):i0]))) if i0 >= 3 else float(np.median(np.diff(b)))
            new[i0:] = b[i0 - 1] + (np.arange(i0, n) - (i0 - 1)) * p
        else:
            a, z = i0 - 1, i1
            new[a:z + 1] = np.linspace(b[a], b[z], z - a + 1)
    return new


def warp_fn(b, new):
    """-> f(times): moves every time the way the beats b moved to `new` (through beat number, so the subdivisions
    and bar lines between two beats keep their place between them)."""
    b, new = np.asarray(b, float), np.asarray(new, float)
    n = len(b)
    idx = np.arange(n, dtype=float)

    def f(t):
        t = np.asarray(t, float)
        # through beat index: inside the beats, interpolate; outside, carry on at the end spacing
        k = np.interp(t, b, idx)
        lo, hi = t < b[0], t > b[-1]
        k = np.where(lo, (t - b[0]) / max(1e-6, b[1] - b[0]), k)
        k = np.where(hi, n - 1 + (t - b[-1]) / max(1e-6, b[-1] - b[-2]), k)
        out = np.interp(k, idx, new)
        out = np.where(k < 0, new[0] + k * (new[1] - new[0]), out)
        out = np.where(k > n - 1, new[-1] + (k - (n - 1)) * (new[-1] - new[-2]), out)
        return out
    return f


def main_meter(runs, tt):
    """(length, sub) of the meter the song is in for the most time."""
    tot = {}
    for r in runs:
        key = (r["length"], r["sub"])
        tot[key] = tot.get(key, 0.0) + float(tt[min(r["i1"], len(tt) - 1)] - tt[r["i0"]])
    return max(tot, key=tot.get) if tot else (8, 2)


def bars(tt, beat_t, bar_starts, runs, gap_spans, label):
    """The bar rules (2 and 3 above). tt: subdivision times; beat_t: the beats; bar_starts: subdivision indices of
    the bar lines; runs: the bar finder's meter runs; gap_spans: no-beat spans as (start, end) times; label(length,
    sub) names a meter. -> (bar_starts, runs, spans) with the runs it replaced merged into the main meter."""
    tt, beat_t = np.asarray(tt, float), np.asarray(beat_t, float)
    if len(runs) < 2 or len(beat_t) < 8:
        return list(bar_starts), runs, []
    L, sub = main_meter(runs, tt)
    bpb = max(1, L // sub)
    main_lab = label(L, sub)
    bt = np.asarray(tt[list(bar_starts)], float)
    spans = []

    def beat_of(t):
        return int(np.argmin(np.abs(beat_t - t)))

    def nobeat_share(a, z):
        tot = sum(max(0.0, min(z, e) - max(a, s)) for s, e in gap_spans)
        return tot / max(1e-6, z - a)

    new_bt = list(bt)
    for r in runs:
        if (r["length"], r["sub"]) == (L, sub):
            continue
        a, z = float(tt[r["i0"]]), float(tt[min(r["i1"], len(tt) - 1)])
        j0, j1 = beat_of(a), beat_of(z)
        nbars = (j1 - j0) / bpb
        quiet = nobeat_share(a, z)
        if r.get("confidence", 0.0) >= KEEP and nbars >= MIN_BARS and quiet < 0.5:
            continue                                    # a real odd meter: it stays
        kind = "no beat" if quiet >= 0.5 else ("unknown" if nbars >= MIN_BARS else "assumed")
        # the main meter carried on from the bar line where the run starts, up to the next bar line after it
        keep = [x for x in new_bt if not (a - 1e-6 <= x < z - 1e-6)]
        nxt = [x for x in keep if x >= z - 1e-6]
        jn = beat_of(nxt[0]) if nxt else len(beat_t) - 1
        fill = [float(beat_t[j]) for j in range(j0, jn, bpb)]
        left = (jn - j0) % bpb
        note = f"heard as {label(r['length'], r['sub'])} (sure {r.get('confidence', 0.0):.2f}); gridded as {main_lab}"
        if left and nxt:                                # it doesn't come out even: one odd bar, flagged
            if left < bpb / 2 and len(fill) > 1:
                fill = fill[:-1]                         # a long bar (bpb + left) rather than a stub
                odd = bpb + left
            else:
                odd = left
            note += f" - the count doesn't come out even: the bar before {round(float(beat_t[jn]), 2)} s is {odd} beats, check it"
        new_bt = sorted(keep + fill)
        spans.append(dict(start=round(a, 3), end=round(z, 3), kind=kind, note=note))
        r["replaced"] = kind
    starts = sorted(set(int(np.argmin(np.abs(tt - x))) for x in new_bt))
    out = []
    for r in runs:
        rr = dict(r)
        rep = rr.pop("replaced", None)
        r.pop("replaced", None)
        w = 0.0 if rep else float(tt[min(rr["i1"], len(tt) - 1)] - tt[rr["i0"]])   # (a replaced run's own
        if rep:                                                                    # sureness doesn't count)
            rr["length"], rr["sub"] = L, sub
        if out and (out[-1]["length"], out[-1]["sub"]) == (rr["length"], rr["sub"]) and out[-1]["i1"] == rr["i0"]:
            o = out[-1]
            tw = o["_w"] + w
            if tw > 0:
                o["confidence"] = (o["confidence"] * o["_w"] + rr["confidence"] * w) / tw
            o["i1"], o["_w"] = rr["i1"], tw
        else:
            rr["_w"] = w
            out.append(rr)
    for o in out:
        o.pop("_w", None)
    return starts, out, spans


def merge(spans):
    """Overlapping spans of the same kind as one, their notes joined."""
    out = []
    for sp in sorted(spans, key=lambda x: x["start"]):
        if out and sp["kind"] == out[-1]["kind"] and sp["start"] <= out[-1]["end"] + 1e-3:
            o = out[-1]
            o["end"] = max(o["end"], sp["end"])
            if sp["note"] not in o["note"]:
                o["note"] += "; " + sp["note"]
        else:
            out.append(dict(sp))
    return out
