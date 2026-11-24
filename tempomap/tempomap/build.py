"""A song's tempo map, start to finish: sections of steady tempo, the beats, their subdivisions, the bars."""
import numpy as np

from . import features, meter, pulse, regular


def beats_of(sections, dur):
    """Every beat of the map, in order: each section's own steady grid, from its start to the next one's."""
    out = []
    for s in sections:
        out.extend(pulse._beats_between(s, s["start"], s["end"] - 1e-6))
    b = np.array(sorted(out))
    if len(b) > 1:                                       # (no two closer than half a beat, where sections meet)
        keep = [b[0]]
        for x in b[1:]:
            if x - keep[-1] > 0.5 * np.median(np.diff(b)) * 0.6:
                keep.append(x)
        b = np.array(keep)
    b = even_joins(b) if EVEN_JOINS else b
    return b[(b >= 0) & (b <= dur)]


EVEN_JOINS = False
REGULAR = True            # regular.py: the user's rules for no-beat stretches and short / unsure odd meters


def even_joins(b, lo=0.75, hi=1.5):
    """Where two sections meet, the beat before the join is moved (or one put in) so no beat there is shorter than
    3/4 or longer than 1 1/2 of the beats around it - a lock pulls every beat onto one beat of the master, and a
    half-beat there would play at double speed for a moment."""
    b = list(b)
    k = 2
    while k < len(b) - 2:
        g = b[k] - b[k - 1]
        ref = 0.5 * ((b[k - 1] - b[k - 2]) + (b[k + 1] - b[k]))
        if ref <= 0:
            k += 1
            continue
        if g < lo * ref:                                 # a short beat: drop the beat before it
            del b[k - 1]
            continue
        if g > hi * ref:                                 # a long gap: put a beat in its middle
            b.insert(k, b[k - 1] + 0.5 * g)
            continue
        k += 1
    return np.array(b)


def subdivisions(secs, beat_t, subs, dur):
    """The subdivision grid, section by section: each beat split into its 2 or 3, and a section's last stretch
    (from its last beat to where the next section starts) filled at the section's own spacing - never squeezed, so
    a section that ends half a beat after a beat (an odd bar) ends on its own grid. -> (times, sub of each, starts a
    beat?)"""
    t, s, on = [], [], []
    bsub = dict(zip(np.round(beat_t[:-1], 5), subs))
    for sec in secs:
        bs = [b for b in beat_t if sec["start"] - 1e-6 <= b < sec["end"] - 1e-6]
        if not bs:
            continue
        for j, b in enumerate(bs):
            q = bsub.get(round(b, 5), 2)
            nxt = bs[j + 1] if j + 1 < len(bs) else None
            if nxt is None:                               # to the section's end at its own spacing
                step = sec["period"] / q
                n = int(round((min(sec["end"], dur) - b) / step))
                n = max(1, n)
                for k in range(n):
                    t.append(b + k * step); s.append(q); on.append(k % q == 0)
                continue
            for k in range(q):
                t.append(b + (nxt - b) * k / q); s.append(q); on.append(k == 0)
    tt = np.array(t)
    keep = np.r_[True, np.diff(tt) > 0.03] if len(tt) else np.zeros(0, bool)
    return tt[keep], np.array(s)[keep], np.array(on)[keep]


def bar_beats(tt, ts, bar_starts):
    """The beats as a musician counts them: from each bar line, every 2 (or 3) subdivisions - a 7/8 bar counts
    2 + 2 + 2 + 1, so the beat starts again on every bar 1."""
    out = []
    edges = list(bar_starts) + [len(tt)]
    if edges and edges[0] > 0:                           # (a pickup before the first bar line)
        k = edges[0]
        while k > 0:
            k -= int(ts[max(0, k - 1)])
            out.append(max(0, k))
    for a, b in zip(edges[:-1], edges[1:]):
        k = a
        while k < b:
            out.append(k)
            k += int(ts[k])
    return sorted(set(out))


def tempo_map(info, stems, sr, det_beats, dur, down_logits=None):
    """-> map dict (see the README): sections, beats, subdivisions, bars, meters, plus what it was measured from.
    down_logits: Beat This!'s downbeat logits per 20 ms frame, if available (a strong cue for bar 1)."""
    feat = features.Features(stems, sr)
    if down_logits is not None:
        feat.down_logits = np.asarray(down_logits, float)
    return from_features(info, feat, det_beats, dur), feat


def rephase(secs, tt, on, bar_starts, beat_t, share=0.6):
    """Where the bar lines of a tempo section mostly fall the same way off the beat (bar 1 found an eighth after
    every counted beat: the pulse was put on the off-beat), the section's beats move onto the bar lines - the bar
    finder hears bar 1 from the downbeats and the rhythm, a stronger cue than where the kick and snare land.
    -> the beat times (the subdivisions stay as they are; only which of them are beats changes)."""
    if not len(tt) or not len(bar_starts):
        return beat_t
    pos = np.zeros(len(tt), int)                       # subdivisions since the last beat
    for k in range(1, len(tt)):
        pos[k] = 0 if on[k] else pos[k - 1] + 1
    out = []
    bs = np.asarray(bar_starts)
    for sec in secs:
        inside = (tt >= sec["start"] - 1e-6) & (tt < sec["end"] - 1e-6)
        bars_in = bs[(tt[bs] >= sec["start"] - 1e-6) & (tt[bs] < sec["end"] - 1e-6)]
        bt = beat_t[(beat_t >= sec["start"] - 1e-6) & (beat_t < sec["end"] - 1e-6)]
        f = 0.0
        if len(bars_in) >= 4 and len(bt) >= 2:
            # where in its beat each bar line falls (0 = on the beat, 0.5 = halfway to the next)
            j = np.clip(np.searchsorted(bt, tt[bars_in], side="right") - 1, 0, len(bt) - 2)
            frac = (tt[bars_in] - bt[j]) / (bt[j + 1] - bt[j])
            frac = np.round(np.clip(frac, 0, 0.999) * 12) / 12          # in twelfths: halves, thirds, quarters
            vals, cnt = np.unique(frac, return_counts=True)
            k = int(np.argmax(cnt))
            if vals[k] > 0.05 and cnt[k] >= share * len(bars_in):
                f = float(vals[k])
        if f:                                            # every beat moves the same share of its own length
            nxt = np.r_[bt[1:], bt[-1] + (bt[-1] - bt[-2])]
            out.extend(bt + f * (nxt - bt))
        else:
            out.extend(bt)
    out = np.array(sorted(out))
    return out if len(out) else beat_t


def from_features(info, feat, det_beats, dur):
    """The map from already measured features (see tempo_map)."""
    secs = pulse.sections(feat, det_beats, dur)
    beat_t = beats_of(secs, dur)
    subs = []
    for j in range(len(beat_t) - 1):
        s = next((x for x in secs if x["start"] - 1e-6 <= beat_t[j] < x["end"]), secs[-1])
        subs.append(s.get("sub", 2))
    subs = meter.sub_runs(feat, beat_t, subs[0] if subs else 2) if len(beat_t) > 8 else subs
    tt, ts, on = subdivisions(secs, beat_t, subs, dur)
    downs = [info["beats"][j] for j in info.get("bars") or [] if 0 <= j < len(info["beats"])]
    if meter.METHOD == "seg":
        bar_starts, runs = meter.bars_seg(feat, tt, ts)
    else:
        bar_starts, runs = meter.bars(feat, tt, ts, downs, det_beats)
    # the user's rules for where the music doesn't say (regular.py): no-beat stretches carry the grid on evenly;
    # short / unsure / beatless odd-meter stretches take the main meter - and every such span is listed
    spans = []
    gaps = regular.no_beat_spans(feat, beat_t) if REGULAR and regular.GAPS else []
    nb = len(beat_t)
    gap_t = [(float(beat_t[i0]), float(beat_t[min(i1, nb - 1)])) for i0, i1 in gaps]
    if REGULAR and regular.BARS:
        bar_starts, runs, rspans = regular.bars(tt, beat_t, bar_starts, runs, gap_t, meter.label)
        spans += rspans
    beat_t = rephase(secs, tt, on, bar_starts, beat_t)
    steady = []
    if REGULAR:
        present = np.ones(len(beat_t), bool)
        for i0, i1 in gaps:
            present[i0:i1] = False
        new, steady = regular.straighten(feat, beat_t, present) if regular.STRAIGHT else (beat_t.copy(), [])
        new = regular.even_gaps(new, gaps)
        f = regular.warp_fn(beat_t, new)
        tt, beat_t = f(tt), f(beat_t)
        for (i0, i1) in gaps:
            where = "start" if i0 == 0 else "end" if i1 >= nb else "middle"
            spans.append(dict(start=round(float(beat_t[i0]), 3), end=round(float(beat_t[min(i1, nb - 1)]), 3),
                              kind="no beat", note=f"nothing to keep time to ({where}): the grid carries on evenly at the "
                                                   f"tempo around it - a lock plays it at one steady speed"))
    spans = regular.merge(spans)
    if steady:                                          # a "follows" section that came out one steady tempo says so
        cover = np.zeros(nb, bool)
        for i0, i1, *_ in steady:
            cover[i0:i1] = True
        for i0, i1 in gaps:
            cover[i0:i1] = True
        for sec in secs:
            ins = np.flatnonzero((beat_t >= sec["start"] - 1e-6) & (beat_t < sec["end"] - 1e-6))
            if sec.get("how") == "follows" and len(ins) > 8 and cover[ins].all():
                sec["how"], sec["bpm"] = "main", 60.0 / float(np.median(np.diff(beat_t[ins])))
                sec.pop("bpm_lo", None), sec.pop("bpm_hi", None)
    bar_t = tt[bar_starts] if len(tt) else np.zeros(0)
    counted = bar_beats(tt, ts, bar_starts)
    meters = []
    for r in runs:
        lab = meter.label(r["length"], r["sub"])
        if r.get("split"):
            lab = f"{meter.label(r['split'], r['sub'])} + {meter.label(r['length'] - r['split'], r['sub'])}"
        meters.append(dict(start=float(tt[r["i0"]]), end=float(tt[min(r["i1"], len(tt) - 1)]), length=r["length"],
                           sub=r["sub"], label=lab, split=r.get("split"), confidence=round(r["confidence"], 3)))
    clean = []
    for s in secs:
        clean.append(dict(start=round(s["start"], 4), end=round(s["end"], 4), bpm=round(s["bpm"], 4),
                          how=s["how"], clarity=round(float(s.get("clarity", 0.0)), 2),
                          rms_ms=None if s.get("rms") is None else round(s["rms"] * 1000, 2),
                          share=round(float(s.get("share", 0.0)), 3), detector_bpm=round(float(s.get("det_bpm", 0)), 2),
                          parts=s.get("parts", 1),
                          bpm_range=[round(s["bpm_lo"], 2), round(s["bpm_hi"], 2)] if s.get("bpm_lo") else None,
                          feels=[dict(start=round(f["start"], 3), end=round(f["end"], 3), feel=f["feel"])
                                 for f in s.get("feels", [])] + ([dict(start=round(s["start"], 3), end=round(s["end"], 3),
                                                                      feel=s["feel"])] if s.get("feel") else [])))
    return dict(
        id=info.get("id"), title=info.get("title"), duration=round(dur, 3),
        sections=clean, meters=meters,
        beats=[round(float(tt[k]), 5) for k in counted], pulse_beats=[round(float(x), 5) for x in beat_t],
        tatums=[round(float(x), 5) for x in tt], tatum_sub=[int(x) for x in ts],
        bars=[int(x) for x in bar_starts], bar_times=[round(float(x), 5) for x in bar_t], spans=spans,
        steady=[dict(start=round(float(beat_t[i0]), 3), end=round(float(beat_t[i1 - 1]), 3), bpm=bpm, hit_ms=e1,
                     followed_ms=e0) for i0, i1, bpm, e1, e0 in steady],
        flux_lag_ms=round(feat.lag * 1000, 2),
    )


def retex_marks(m, unsure_below=0.3):
    """The map as Re-Tex's beat marks (its song grid: {t, d, s, bpb}): one mark per beat of the pulse; d = 1 where a
    bar line falls on that beat; s = 0 (unsure, amber in the Beat grid window) where the bars are unsure or there was
    no pulse to measure; bpb = the usual number of beats between bar lines. A bar line that falls between two beats
    (every other one in 7/8) can't be a mark: those bars show as one longer bar."""
    t = np.asarray(m["pulse_beats"], float)
    bars = np.asarray(m.get("bar_times") or [], float)
    if not len(t):
        return dict(t=[], d=[], s=[], bpb=4, source="tempomap")
    tol = 0.2 * float(np.median(np.diff(t))) if len(t) > 1 else 0.05
    if len(bars):
        j = np.clip(np.searchsorted(bars, t), 1, max(1, len(bars) - 1))
        near = np.minimum(np.abs(bars[np.minimum(j, len(bars) - 1)] - t), np.abs(bars[j - 1] - t))
        d = (near <= tol).astype(int)
    else:
        d = np.zeros(len(t), int)
    s = np.ones(len(t), int)
    runs = m.get("meters") or []
    if runs:                                         # unsure = clearly less sure than this song's bars usually are
        conf = np.array([r.get("confidence", 1.0) for r in runs])
        dur = np.array([max(1e-3, r["end"] - r["start"]) for r in runs])
        typical = float(np.average(conf, weights=dur))
        unsure_below = min(unsure_below, 0.6 * typical)
    for r in runs:
        if r.get("confidence", 1.0) < unsure_below:
            s[(t >= r["start"] - 1e-6) & (t < r["end"])] = 0
    for sec in m.get("sections") or []:
        if sec.get("how") == "none":
            s[(t >= sec["start"] - 1e-6) & (t < sec["end"])] = 0
    for sp in m.get("spans") or []:                  # regular.py's spans: gridded by rule, not heard - unsure
        s[(t >= sp["start"] - 1e-6) & (t <= sp["end"] + 1e-6)] = 0
    t, d, s = _even_marks(list(t), list(d), list(s))
    keep = [i for i, x in enumerate(t) if x >= 0]      # (a grid carried back before 0 s: those beats aren't marks)
    t, d, s = [t[i] for i in keep], [d[i] for i in keep], [s[i] for i in keep]
    d = np.asarray(d)
    k = np.flatnonzero(d)
    gaps = np.diff(k)
    bpb = int(np.bincount(gaps).argmax()) if len(gaps) else 4
    return dict(t=[round(float(x), 5) for x in t], d=[int(x) for x in d], s=[int(x) for x in s],
                bpb=max(1, min(16, bpb)), source="tempomap", spans=list(m.get("spans") or []))


def _even_marks(t, d, s, lo=0.75, hi=1.5):
    """Re-Tex's lock pulls every mark onto one beat of the master, so a beat much shorter or longer than the beats
    around it (where two tempo sections meet) would play at double / half speed for a moment: drop the mark before
    a short beat (its bar line, if any, moves to the next mark) or put one in the middle of a long gap (unsure)."""
    k = 1
    while k < len(t):
        g = t[k] - t[k - 1]
        left = [t[j] - t[j - 1] for j in range(max(1, k - 4), k)]
        right = [t[j] - t[j - 1] for j in range(k + 1, min(len(t), k + 5))]
        if not left or not right:
            k += 1
            continue
        ml, mr = float(np.median(left)), float(np.median(right))
        ref = min(ml, mr)                              # a tempo change between them is fine; an odd beat isn't
        if ref > 0 and g < lo * ref:
            if d[k - 1]:
                d[k] = 1
            del t[k - 1], d[k - 1], s[k - 1]
            continue
        if g > hi * max(ml, mr):
            t.insert(k, t[k - 1] + 0.5 * g)
            d.insert(k, 0)
            s.insert(k, 0)
            continue
        k += 1
    return t, d, s
