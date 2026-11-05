"""Repair a beat grid whose detector switched pulse between sections (the HORSE the Band case: 85 -> 221 -> 138 BPM
within one song - half time in one section, double time in the next). Inside a section the detector's beats are
consistent; across sections it counts at different levels, and locking that grid to one tempo slows a double-time
section to half speed. (Measured Oct 15: the lock itself is clean on right marks - only the marks were wrong.)

repair(beats, drums) -> beats on one pulse:
  1. cut the beats into stretches of steady tempo (change points on the log beat spacing),
  2. pick the song's pulse (the tempo that covers the most time, folded into 85-185 BPM),
  3. re-count each stretch at that pulse when it's a simple multiple of it: x2 / x1/2 (the classic detector octave
     error) always; x3/2, x2/3, x3, x1/3 only where the drum hits confirm it (the new beats land on hits at least as
     well) - else the stretch stays as it is and is reported as unsure; a genuine tempo change (any other ratio) stays,
  4. smooth the tempo curve and put each beat on the drum hit within 25 ms of it (a hand-fixed grid
     was made this way and locked without an audible artifact).
"""
import numpy as np

RATIOS = (1.0, 2.0, 0.5, 1.5, 2 / 3, 3.0, 1 / 3, 4.0, 0.25)
OCTAVE = (2.0, 0.5, 4.0, 0.25)  # always safe to re-count
HIT_KEEP = 0.95                # a 3:2-type re-count must keep at least this share of the stretch's drum-hit score
PULSE = (85.0, 185.0)          # BPM range the song's pulse is counted in
MIN_SEC = 12                   # beats: the shortest stretch of steady tempo
TOL = 0.09                     # a ratio this close to a simple multiple counts as one


def _segments(iv, min_len=MIN_SEC, pen=0.05):
    """Change points on log beat spacing: [(a, b)] index ranges (into iv) of steady tempo (dynamic programming)."""
    x = np.log(iv)
    n = len(x)
    cs, cs2 = np.concatenate([[0], np.cumsum(x)]), np.concatenate([[0], np.cumsum(x * x)])
    cost = lambda a, b: (cs2[b] - cs2[a]) - (cs[b] - cs[a]) ** 2 / (b - a)
    best = np.full(n + 1, np.inf)
    prev = np.zeros(n + 1, int)
    best[0] = 0.0
    for b in range(min_len, n + 1):
        for a in range(0, b - min_len + 1):
            if best[a] < np.inf:
                v = best[a] + cost(a, b) + pen * min_len
                if v < best[b]:
                    best[b], prev[b] = v, a
    if not np.isfinite(best[n]):
        return [(0, n)]
    out, b = [], n
    while b > 0:
        a = prev[b]
        out.append((a, b))
        b = a
    return out[::-1]


def _fold(bpm):
    while bpm > PULSE[1]:
        bpm /= 2
    while bpm < PULSE[0]:
        bpm *= 2
    return bpm


def plan(beats):
    """-> (pulse BPM, [(a, b, section BPM, ratio to the pulse or None for a genuine change)])"""
    b = np.asarray(beats, float)
    iv = np.diff(b)
    segs = _segments(iv)
    bpm = [60 / np.median(iv[a:c]) for a, c in segs]
    dur = [b[c] - b[a] for a, c in segs]
    # the pulse: the folded tempo that covers the most time (within 4 %)
    folded = [_fold(x) for x in bpm]
    cover = [sum(d for f2, d in zip(folded, dur) if abs(f2 / f - 1) < 0.04) for f in folded]
    pulse = folded[int(np.argmax(cover))]
    out = []
    for (a, c), t in zip(segs, bpm):
        r = t / pulse
        k = min(RATIOS, key=lambda q: abs(np.log(r / q)))
        out.append((a, c, t, k if abs(np.log(r / k)) < TOL else None))
    return pulse, out


def _recount(b, segs):
    """Each detector beat's position counted in pulse beats, and the beats at whole counts."""
    pos = np.zeros(len(b))
    for a, c, t, k in segs:
        pos[a] = round(pos[a])                  # every stretch starts on a whole pulse beat: a 3:2 stretch's leftover
        step = 1.0 / (k if k else 1.0)          # fraction mustn't shift everything after it (a double-time stretch:
        for i in range(a + 1, c + 1):           # each detector beat is half a pulse beat)
            pos[i] = pos[i - 1] + step
    q = np.arange(0, int(np.floor(pos[-1])) + 1, dtype=float)
    return np.interp(q, pos, b)


def _hit_env(drums, sr, hop=128):
    import librosa
    env = librosa.onset.onset_strength(y=np.asarray(drums, np.float32), sr=sr, hop_length=hop)
    return env / (np.percentile(env, 99) + 1e-9), hop


def _hits(env, hop, sr, times):
    if len(times) == 0:
        return 0.0
    f = np.clip(np.round(np.asarray(times) * sr / hop).astype(int), 0, len(env) - 1)
    return float(np.mean([env[max(0, x - 3):x + 4].max() for x in f]))


def repair(beats, drums=None, sr=44100):
    """The detector's beats re-counted on one pulse (see the module doc). -> (new beat times, report dict)"""
    b = np.asarray(beats, float)
    pulse, segs = plan(b)
    env = hop = None
    if drums is not None:
        env, hop = _hit_env(drums, sr)
    final, unsure = [], []
    for a, c, t, k in segs:
        if k in (None, 1.0) or k in OCTAVE or env is None:
            final.append((a, c, t, k if k in OCTAVE or k == 1.0 else None))
            continue
        # a 3:2-type stretch: keep the re-count only if its beats land on the drums as well as the detector's did
        alone = _recount(b[a:c + 1], [(0, c - a, t, k)])
        if _hits(env, hop, sr, _on_hits(alone, None, sr, env, hop) if len(alone) > 20 else alone) >= HIT_KEEP * _hits(env, hop, sr, b[a:c + 1]):
            final.append((a, c, t, k))
        else:
            final.append((a, c, t, None))
            unsure.append((round(float(b[a]), 2), round(float(b[c]), 2)))
    new = _recount(b, final)
    if env is not None and len(new) > 20:
        new = _on_hits(new, None, sr, env, hop)
    changed = sum(c - a for a, c, t, k in final if k not in (None, 1.0))
    rep = dict(pulse=round(pulse, 2), sections=[dict(start=round(float(b[a]), 2), end=round(float(b[c]), 2), bpm=round(t, 1),
                                                      ratio=k) for a, c, t, k in final],
               recounted_beats=int(changed), unsure=unsure, before=len(b), after=len(new))
    return new, rep


def _on_hits(t, drums, sr, env=None, hop=128):
    """Smooth the tempo curve, then each beat onto the drum hit within 25 ms (when there's a clear one)."""
    from scipy.signal import savgol_filter
    if env is None:
        env, hop = _hit_env(drums, sr)
    sm = savgol_filter(t, 17, 2) if len(t) > 17 else t
    w = int(0.025 * sr / hop)
    out = []
    for x in sm:
        f = int(round(x * sr / hop))
        seg = env[max(0, f - w):f + w + 1]
        out.append((max(0, f - w) + int(np.argmax(seg))) * hop / sr if len(seg) and seg.max() > 0.25 else x)
    out = np.array(out)
    return savgol_filter(out, 9, 2) if len(out) > 9 else out


LOCK_SPAN = 1.3               # a lock pulls everything to the song's main tempo: a stretch further off than this would
                               # play at a clearly wrong speed (Between the Trees' 139-166 BPM, +-9 %, locked cleanly)


def slips(beats):
    """Stretches a lock would play at the wrong speed: more than 30 % off the song's main tempo (time-weighted median
    of its stretches), or a switch to double / half time from the stretch before. -> [(start s, end s, ratio)], empty
    when the grid is fine to lock."""
    b = np.asarray(beats, float)
    if len(b) < 2 * MIN_SEC:
        return []
    iv = np.diff(b)
    segs = _segments(iv)
    bpm = np.array([60 / np.median(iv[a:c]) for a, c in segs])
    dur = np.array([b[c] - b[a] for a, c in segs])
    order = np.argsort(bpm)
    main = bpm[order][np.searchsorted(np.cumsum(dur[order]), dur.sum() / 2)]
    out = []
    for i, (a, c) in enumerate(segs):
        r = bpm[i] / main
        prev = bpm[i] / bpm[i - 1] if i else 1.0
        octave = i and min(abs(np.log(prev / k)) for k in OCTAVE) < TOL
        if abs(np.log(r)) > np.log(LOCK_SPAN) or octave:
            out.append((round(float(b[a]), 2), round(float(b[c]), 2), round(float(r), 3)))
    return out


def as_marks(info, drums=None, sr=44100):
    """repair() on the song's detector beats, as an editable marks list {t, d, s, bpb} (+ the report): the bar 1s
    where most of the detector's downbeats fall; stretches the drums couldn't confirm marked unsure."""
    dur = float(info.get("duration") or info["beats"][-1])
    det = np.array([x for x in info["beats"] if -1e-6 <= x <= dur + 1e-6])
    new, rep = repair(det, drums, sr)
    bpb = int(info.get("bpb") or 4)
    downs = [info["beats"][j] for j in info.get("bars") or [] if 0 <= j < len(info["beats"])]
    phase = 0
    if downs and len(new):
        idx = [int(np.argmin(np.abs(new - x))) for x in downs]
        phase = int(np.bincount(np.array(idx) % bpb, minlength=bpb).argmax())
    d = [1 if (i - phase) % bpb == 0 else 0 for i in range(len(new))]
    s = [0 if any(a <= x <= b for a, b in rep["unsure"]) else 1 for x in new]
    return dict(t=[round(float(x), 5) for x in new], d=d, s=s, bpb=bpb), rep
