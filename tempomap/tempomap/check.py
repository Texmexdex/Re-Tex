"""How well a grid sits on the music, a few seconds at a time.

In every window the clear drum hits are placed on the grid as a phase within its 16ths (or its triplets, whichever
they line up on better) and averaged as a circle: strength = how much of the hits keep landing at one place in the
16th (0 = anywhere, 1 = all at one spot; chance is about 0.1); offset = where that place is (ms, + = the music is
later than the grid). Over a song, a grid at the right tempo gives a FLAT offset (the same few ms everywhere); at
the wrong tempo the offset slides and wraps; with the wrong pulse the strength drops toward chance.
"""
import numpy as np

WIN, HOP = 8.0, 2.0


def lock(feat, beats, win=WIN, hop=HOP):
    """-> dict(t: window centres, strength, offset_ms) for the beat times `beats` (any grid)."""
    from .pulse import onset_peaks
    b = np.asarray(beats, float)
    if len(b) < 4:
        return dict(t=[], strength=[], offset_ms=[])
    ht, hh = onset_peaks(feat.hits, feat.fps, b[0], b[-1], q=80)
    if len(ht) < 8:
        return dict(t=[], strength=[], offset_ms=[])
    pos = np.interp(ht, b, np.arange(len(b)))
    per = np.interp(ht, b[:-1], np.diff(b))
    out_t, st, off = [], [], []
    for c in np.arange(b[0] + win / 2, b[-1] - win / 2 + 1e-9, hop):
        m = (ht >= c - win / 2) & (ht < c + win / 2)
        if m.sum() < 6:
            continue
        w = hh[m] / hh[m].sum()
        best = None
        for k in (4, 3):                                   # 16ths, triplets
            z = np.sum(w * np.exp(2j * np.pi * k * pos[m]))
            if best is None or abs(z) > abs(best[0]):
                best = (z, k)
        z, k = best
        o = np.angle(z) / (2 * np.pi * k) * float(np.median(per[m]))
        out_t.append(round(float(c), 2))
        st.append(round(float(abs(z)), 3))
        off.append(round(o * 1000.0, 1))
    return dict(t=out_t, strength=st, offset_ms=off)


def summary(lk):
    """Median strength; the middle half of the offsets (ms) over the windows with at least half the median strength
    (where the music has a pulse at all); and the share of those within 15 ms of their median (steady)."""
    s, o = np.asarray(lk["strength"]), np.asarray(lk["offset_ms"])
    if len(s) < 3:
        return dict(strength=0.0, spread_ms=None, steady=None, windows=0)
    ok = s >= 0.5 * np.median(s)
    q1, q3 = np.percentile(o[ok], [25, 75])
    med = float(np.median(o[ok]))
    return dict(strength=round(float(np.median(s)), 3), spread_ms=round(float(q3 - q1), 1), median_ms=round(med, 1),
                steady=round(float(np.mean(np.abs(o[ok] - med) <= 15)), 3), windows=int(ok.sum()))
