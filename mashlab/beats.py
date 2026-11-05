"""Beat grid: Beat This! detection, then refined into a precise, musically consistent grid.

Beat This! frames are 20 ms, so raw beat times jitter by +-10 ms - too coarse for stacking songs.
We repair missed beats, split at edits (tempo/phase jumps), fit each segment with a straight line
(steady tempo) or a local regression (drifting tempo), then phase-lock the grid to drum transients.

Clips address source audio by *beat index* in the extended beat list, so edits and odd bars are fine.
"""
import threading
import numpy as np
import librosa

_model = None
_lock = threading.Lock()
GRID_VERSION = 5       # 5: a song one steady tempo fits all the way through gets that steady grid (steady_grid)
# 4: a triplet-feel passage (tracked at 4:3, 3:2 ... of the beat) between two main-tempo
#    sections stays on the main beat
# 3: same-tempo regimes that meet off the beat are re-phased on the evidence (join_choice)
# 2: main tempo by total coverage, 2/3/4x folding, meter from downbeat probabilities


def detect_full(y_mono, sr):
    """Raw beats and downbeats (seconds) from Beat This!, plus its beat and downbeat logits per 20 ms frame."""
    import torch
    from beat_this.inference import Audio2Frames, Postprocessor
    global _model
    with _lock:
        if _model is None:
            _model = (Audio2Frames(checkpoint_path="final0", device="cuda" if torch.cuda.is_available() else "cpu"),
                      Postprocessor(type="minimal"))
        a2f, post = _model
        with torch.no_grad():
            bl, dl = a2f(y_mono.astype(np.float32), sr)
        b, d = post(bl, dl)
        dl = dl.float().cpu().numpy()
        bl = bl.float().cpu().numpy()
    return np.asarray(b, float), np.asarray(d, float), dl, bl


def detect(y_mono, sr, logits=False):
    """Raw (beats, downbeats) in seconds from Beat This! (+ the downbeat logits per 20 ms frame)."""
    b, d, dl, _ = detect_full(y_mono, sr)
    return (b, d, dl) if logits else (b, d)


def body_env(y_drums_mono, sr):
    """Onset strength of the drums' snare/clap band (150-1500 Hz). Hi-hat eighths make full-band onsets fire
    as hard half a beat off as on the beat, and rock kicks are often syncopated; the snare (or clap) keeps
    to the beats."""
    import scipy.signal as sg
    y = sg.sosfilt(sg.butter(4, [150, 1500], "band", fs=sr, output="sos"), y_drums_mono).astype(np.float32)
    return _onset_env(y, sr)


def evidence(beat_logits, env, hop, sr, fps=50.0):
    """-> score(times): how beat-like each moment is. Beat This!'s beat probability (max within +-1 frame)
    and the drums' kick/snare onset strength (body_env, max within +-15 ms), each scaled by its 95th
    percentile, averaged. Either source alone can be fooled (the model by a busy passage, the drums by a
    syncopated fill)."""
    srcs = []
    if beat_logits is not None and len(beat_logits):
        act = _sig(beat_logits)
        srcs.append((act / (np.quantile(act, 0.95) + 1e-9), fps, 1))
    if env is not None and len(env):
        e = np.asarray(env, float)
        srcs.append((e / (np.quantile(e, 0.95) + 1e-9), sr / hop, max(1, int(0.015 * sr / hop))))

    def score(t):
        t = np.asarray(t, float)
        out = np.zeros(len(t))
        for a, rate, w in srcs:
            fr = np.round(t * rate).astype(int)
            v = np.zeros(len(t))
            for k in range(-w, w + 1):
                i = np.clip(fr + k, 0, len(a) - 1)
                v = np.maximum(v, np.where((fr + k >= 0) & (fr + k < len(a)), a[i], 0.0))
            out += np.minimum(v, 1.5)
        return out / max(1, len(srcs))
    return score


JOIN_WINDOW = 24          # beats either side of a join that decide it
JOIN_MOVE = 0.95          # a side is moved onto its neighbour's phase if it scores at least 95 % as well there
JOIN_STRONG = 0.5         # ... and at least half as well as the neighbour does on its own beats


def join_choice(A, B, score):
    """A, B: beat times of two consecutive regimes at the same tempo whose beats don't meet on a whole beat
    (a short or long beat at the join). Either the tracker lost the phase on one side - common: it locks
    onto the off-beat in a busy intro - or the music really has an odd-length bar / edit there.
    Scores the beats near the join as they are, with A moved onto B's phase, and with B moved onto A's.
    A side is moved when it fits the music at least about as well there (JOIN_MOVE); if both sides fit
    clearly better where they are, the break is real. Measured on this library: tracker slips gain when
    moved (A Perfect Circle's intro: 0.21 -> 0.83), real odd bars lose half or more (1.08 -> 0.23).
    -> (shift for A, shift for B, detail): one shift is 0; both 0 and detail set = a real break, keep it;
    detail None = nothing to decide (the beats already meet on a whole beat)."""
    A, B = np.asarray(A, float), np.asarray(B, float)
    if len(A) < 4 or len(B) < 4:
        return 0.0, 0.0, None
    p = 0.5 * (float(np.median(np.diff(A[-9:]))) + float(np.median(np.diff(B[:9]))))
    gap = B[0] - A[-1]
    m = gap / p
    r = max(1, int(np.floor(m + 0.5 - 1e-6)))            # whole beats the gap should be (1.5 -> 1)
    d = gap - r * p                                     # A later by d, or B earlier by d, closes it
    if abs(d) < 0.08 * p:
        return 0.0, 0.0, None
    a, b = A[-JOIN_WINDOW:], B[:JOIN_WINDOW]
    sa, sa2, sb, sb2 = (float(np.mean(score(x))) for x in (a, a + d, b, b - d))
    detail = dict(gap_beats=round(m, 2), a=round(sa, 3), a_moved=round(sa2, 3), b=round(sb, 3), b_moved=round(sb2, 3))
    # a side may move only if it fits at least about as well there AND lands on beats that are clearly
    # beat-like next to its neighbour's (with weak evidence on both phases the move is a coin toss)
    ok_a = sa2 >= JOIN_MOVE * sa and sa2 >= JOIN_STRONG * sb
    ok_b = sb2 >= JOIN_MOVE * sb and sb2 >= JOIN_STRONG * sa
    if not (ok_a or ok_b):
        return 0.0, 0.0, detail
    if ok_a and ok_b:
        ok_a = sa2 / max(sa, 1e-6) >= sb2 / max(sb, 1e-6)
    return (d, 0.0, detail) if ok_a else (0.0, -d, detail)


DP_TIGHT = 100.0          # tempo steadiness (Ellis-style: penalty = TIGHT * log(interval / period) ** 2)
DP_ODD = 1.5              # extra cost of an odd-length beat (> 12 % off): only where the music really has one
DP_FPS = 100


def _fill_gap(t0, t1, p, score):
    """Beats strictly between two beats t0 and t1 (the end of one regime, the start of the next) that follow
    the evidence at a steady period p. The tracker leaves gaps where it lost the plot - busy passages it
    heard in double time, pauses - and extrapolating the previous regime across them drifts off the music
    (A Perfect Circle: 62 s extrapolated, up to 0.45 beat off). A fractional number of beats in the gap is
    one odd-length beat, placed where the music has it."""
    n = int(round((t1 - t0) * DP_FPS))
    P = p * DP_FPS
    if n < 1.5 * P:
        return np.array([])
    t = t0 + np.arange(n + 1) / DP_FPS
    e = score(t)
    d = np.arange(max(1, int(0.35 * P)), int(1.75 * P) + 1)
    lr = np.log(d / P)
    # steady beats pay for any tempo wobble; an odd-length beat (> 12 % off) pays a flat price instead, so
    # a real half-beat bar costs the same wherever it is and lands where the music has it
    edge = np.log(1.12)
    trans = np.where(np.abs(lr) <= edge, -DP_TIGHT * lr ** 2, -DP_TIGHT * edge ** 2 - DP_ODD)
    best = np.full(n + 1, -np.inf)
    back = np.full(n + 1, -1, np.int64)
    best[0] = 0.0
    for f in range(int(d[0]), n + 1):
        prev = f - d
        ok = prev >= 0
        cand = best[prev[ok]] + trans[ok]
        k = int(np.argmax(cand))
        if np.isfinite(cand[k]):
            best[f] = cand[k] + (e[f] if f < n else 0.0)
            back[f] = prev[ok][k]
    out, f = [], n
    while f > 0 and back[f] >= 0:
        f = int(back[f])
        if f > 0:
            out.append(t[f])
    return np.array(out[::-1])


def _fill_gaps(parts, score, env, hop, sr):
    """Fill every inner gap of more than ~2 beats between same-tempo regimes with _fill_gap; the filled beats
    become regimes of their own (split at any odd-length beat), smoothed and phase-refined like the rest."""
    out = [parts[0]]
    for nxt in parts[1:]:
        A = out[-1]
        pa, pb = float(np.median(np.diff(A[-9:]))), float(np.median(np.diff(nxt[:9])))
        gap = nxt[0] - A[-1]
        m = gap / (0.5 * (pa + pb))
        # evenly spacing a short whole-beat gap is safe; a long one (extrapolation drifts) or one that isn't
        # a whole number of beats (the odd beat has to go somewhere) is filled from the evidence
        if abs(pa - pb) / pa < 0.05 and gap > 2.5 * pa and (m > 16 or abs(m - round(m)) > 0.15):
            fill = _fill_gap(A[-1], nxt[0], 0.5 * (pa + pb), score)
            if len(fill):
                seq = np.concatenate([[A[-1]], fill, [nxt[0]]])
                iv = np.diff(seq) / (0.5 * (pa + pb))
                cut = np.nonzero(np.abs(np.log(iv)) > np.log(1.12))[0]      # odd beats split the fill
                runs = np.split(fill, [c for c in cut if 0 < c < len(fill)])
                for r in runs:
                    if len(r) >= 8:
                        f, _ = _fit_segment(np.arange(len(r)), r)
                        r = _phase_refine(f, env, hop, sr)
                    if len(r):
                        out.append(np.asarray(r, float))
        out.append(nxt)
    return out


def _same_tempo(A, B):
    pa, pb = float(np.median(np.diff(A[-9:]))), float(np.median(np.diff(B[:9])))
    return abs(pa - pb) / pa < 0.03


def _phase_joins(parts, score):
    """Regimes (in order) -> the same, with every same-tempo join the evidence says the tracker slipped at
    moved onto one phase (join_choice). A move shifts the whole continuous run before the join, so joins
    already made continuous stay that way."""
    out, run = [np.asarray(parts[0], float)], 0
    for nxt in parts[1:]:
        nxt = np.asarray(nxt, float)
        A = np.concatenate(out[run:])
        if len(A) >= 4 and len(nxt) >= 4 and _same_tempo(A, nxt):
            da, db, det = join_choice(A, nxt, score)
            if da:
                for i in range(run, len(out)):
                    out[i] = out[i] + da
            nxt = nxt + db
            if det is not None and not (da or db):
                run = len(out)                      # a real break: a new run starts here
        else:
            run = len(out)
        out.append(nxt)
    return out


def _track(b, tol=0.15, min_beats=12):
    """Split detected beats into tempo regimes.

    Follows a local tempo line (last 16 accepted beats); each detected beat is snapped to a beat *index*
    on that line, so missed beats just skip an index and extra beats are dropped. A regime ends when a
    beat doesn't fit AND the next few don't either (a real tempo/phase change, not one bad detection).
    Returns [(k_int_array, t_array)] for regimes with >= min_beats beats.
    """
    segs, i, n = [], 0, len(b)
    while i < n - 3:
        g = np.diff(b[i:i + 4])
        if np.ptp(g) > 0.12 * np.median(g):
            i += 1
            continue
        ks, ts, j = [0, 1, 2, 3], list(b[i:i + 4]), i + 4
        while j < n:
            kk, tt = np.array(ks[-16:], float), np.array(ts[-16:])
            per, a0 = np.polyfit(kk, tt, 1)
            k = int(round((b[j] - a0) / per))
            if k <= ks[-1]:
                j += 1                                   # extra beat between two grid beats
                continue
            if abs(b[j] - (a0 + k * per)) < tol * per and k - ks[-1] <= 8:
                ks.append(k)
                ts.append(b[j])
                j += 1
                continue
            ahead = [abs(b[m] - (a0 + round((b[m] - a0) / per) * per)) < tol * per
                     for m in range(j + 1, min(n, j + 4))]
            if len(ahead) >= 2 and sum(ahead) >= len(ahead) - 1:
                j += 1                                   # lone outlier
                continue
            break
        if ks[-1] - ks[0] + 1 >= min_beats:
            segs.append((np.array(ks), np.array(ts)))
        i = j if j > i + 4 else i + 1
    return segs


def _loess(k, t, k_eval, W=8):
    """Local-linear smoothing of beat times against beat index (tricube weights), evaluated at k_eval."""
    out = np.empty(len(k_eval))
    for i, ke in enumerate(k_eval):
        d = np.abs(k - ke)
        sel = d <= W + 0.5
        if sel.sum() < 3:
            sel = np.argsort(d)[:4]
        kk, tt = k[sel].astype(float), t[sel]
        w = (1 - (np.abs(kk - ke) / (W + 1)) ** 3) ** 3
        A = np.vstack([np.ones_like(kk), kk]).T * w[:, None]
        coef, *_ = np.linalg.lstsq(A, tt * w, rcond=None)
        out[i] = coef[0] + coef[1] * ke
    return out


def _fit_segment(k, t):
    """Beat times at every integer index ks[0]..ks[-1]: a straight line if the tempo is steady."""
    k_all = np.arange(k[0], k[-1] + 1)
    p = np.polyfit(k, t, 1)
    r = np.abs(t - np.polyval(p, k))
    keep = r <= np.quantile(r, 0.9) + 1e-9
    p = np.polyfit(k[keep], t[keep], 1)
    r = np.abs(t - np.polyval(p, k))
    if np.quantile(r, 0.95) < 0.016:            # steady tempo (quantized/electronic): exact line
        return np.polyval(p, k_all), True
    return _loess(k, t, k_all), False


def _join(parts):
    """Chain fitted regimes into one beat list. Same tempo + whole-beat gap -> bridge seamlessly;
    otherwise keep going at the old tempo up to the change and start a new segment there."""
    beats, segs = list(parts[0]), [[0, 0]]
    for nxt in parts[1:]:
        pa = beats[-1] - beats[-2]
        pb = nxt[1] - nxt[0] if len(nxt) > 1 else pa          # a lone beat (between two odd ones)
        gap = nxt[0] - beats[-1]
        m = gap / pa
        if abs(pa - pb) / pa < 0.03 and abs(m - round(m)) < 0.2 and round(m) >= 1:
            r = int(round(m))
            beats.extend(beats[-1] + gap * np.arange(1, r) / r)
        else:
            while nxt[0] - beats[-1] > 1.5 * pa:
                beats.append(beats[-1] + pa)
            segs[-1][1] = len(beats)
            segs.append([len(beats), 0])
        beats.extend(nxt)
    segs[-1][1] = len(beats)
    return np.array(beats), segs


def _onset_env(y, sr, hop=64):
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=hop, n_mels=64, fmin=30, fmax=16000)
    e = librosa.onset.onset_strength(S=librosa.power_to_db(S), sr=sr, hop_length=hop, lag=1, max_size=3)
    return e / (e.max() + 1e-9), hop


def _phase_refine(t, env, hop, sr, max_ms=35):
    """Shift a grid segment by the constant offset that best lines beats up with transients."""
    if len(t) < 8:
        return t
    fr = t * sr / hop
    offs = np.arange(-int(max_ms / 1000 * sr / hop), int(max_ms / 1000 * sr / hop) + 1)
    x = np.arange(len(env))
    scores = np.array([np.interp(fr + o, x, env).sum() for o in offs])
    best = offs[np.argmax(scores)]
    base = scores[offs == 0][0]
    if scores.max() > base * 1.03:
        # parabolic sub-frame interpolation
        i = int(np.argmax(scores))
        if 0 < i < len(scores) - 1:
            a, b, c = scores[i - 1], scores[i], scores[i + 1]
            den = a - 2 * b + c
            frac = 0.5 * (a - c) / den if den != 0 else 0.0
        else:
            frac = 0.0
        return t + (best + frac) * hop / sr
    return t


FOLDS = (2, 3, 4)          # a regime tracked at n x (or 1/2 x) the main beat -> fold onto the main level
FOLD_TOL = 0.06


def _tempo(p):
    return 60.0 / float(np.median(np.diff(p)))


def _fold_ratio(r):
    """r = main ibi / regime ibi (>1: regime faster). -> n (decimate by n), -2 (subdivide), or 1."""
    if abs(r - 1) < FOLD_TOL:
        return 1
    for n in FOLDS:
        if abs(r / n - 1) < FOLD_TOL:
            return n
    if abs(r * 2 - 1) < FOLD_TOL:
        return -2
    return 0


def _main_tempo(parts):
    """The beat level to build the grid on: the tempo that covers the most time, counting regimes
    tracked at 2/3/4x or half of it too. (The longest single regime is not enough: a steady 30 s intro
    would beat a 150 s body that the tracker split into several shorter regimes.) When the winner is
    just another octave of the longest regime, the longest regime's octave is kept - the tracker's own
    choice of beat level for the bulk of the song."""
    tempos = [_tempo(p) for p in parts]
    durs = [p[-1] - p[0] for p in parts]

    def coverage(T):
        cov = 0.0
        for t, d in zip(tempos, durs):
            if _fold_ratio(t / T):           # t / T == main ibi / regime ibi
                cov += d * (1.0 if abs(t / T - 1) < FOLD_TOL else 0.8)
        return cov
    best = max(tempos, key=coverage)
    longest = tempos[int(np.argmax(durs))]
    if _fold_ratio(longest / best):          # same family: keep the longest regime's octave
        best = longest
    same = [(t, d) for t, d in zip(tempos, durs) if abs(t / best - 1) < FOLD_TOL]
    ts = np.repeat([t for t, _ in same], [max(1, int(d)) for _, d in same])
    return float(np.median(ts))


def _fold(p, main_ibi, env, hop, sr):
    """A regime tracked at 2/3/4x (or half) the main tempo -> the main beat level. When decimating, keep
    the phase whose beats carry the strongest drum transients."""
    if len(p) < 4:
        return p
    r = main_ibi / float(np.mean(np.diff(p)))           # >1: this regime's beats are faster
    n = _fold_ratio(r)
    if n >= 2 and len(p) >= 4 * n:
        x = np.arange(len(env))
        scores = [np.interp(p[k::n] * sr / hop, x, env).mean() for k in range(n)]
        return p[int(np.argmax(scores))::n]
    if n == -2:
        mid = (p[:-1] + p[1:]) / 2
        out = np.empty(len(p) + len(mid))
        out[0::2], out[1::2] = p, mid
        return out
    return p


TRIPLETS = (4 / 3, 3 / 4, 3 / 2, 2 / 3)   # main ibi / regime ibi of a passage tracked in a triplet feel


TRIPLET_PHASE = 0.1      # the main beat must carry straight through: the gap is a whole number of beats (+- this)
TRIPLET_EVIDENCE = 0.7   # ... and those beats are on the music: >= 70 % of the evidence either side has


def _drop_triplet_feel(parts, main_ibi, score):
    """Drop a run of regimes tracked at a triplet relation to the main beat (4:3, 3:2 ...) between two
    main-level regimes when the main beat carries straight through it: the gap is a whole number of main
    beats and evenly spaced beats across it sit on the music about as well as the beats either side. There
    the song keeps its beat and only the feel changes. (By and Down The River's bridge: drums in quarter-note
    triplets, tracked as pairs of them = 4/3 of a beat, squeezed 57 beats into 43 and moved everything after
    it by 3.5 bars.) A real tempo change at such a ratio doesn't come out a whole number of beats (measured on
    this library: 0.12 - 0.49 beat off) and stays as it is. The gap is filled at the main beat from the
    evidence (_fill_gaps), like any other passage the tracker lost."""
    level = lambda p: main_ibi / float(np.median(np.diff(p)))
    main = [len(p) > 1 and abs(level(p) - 1) < FOLD_TOL for p in parts]
    trip = [len(p) > 1 and any(abs(level(p) / q - 1) < FOLD_TOL for q in TRIPLETS) for p in parts]
    keep = [True] * len(parts)
    i = 0
    while i < len(parts):
        j = i
        while j < len(parts) and trip[j]:
            j += 1
        if j > i and i > 0 and j < len(parts) and main[i - 1] and main[j]:
            A, B = np.asarray(parts[i - 1], float), np.asarray(parts[j], float)
            pa, pb = float(np.median(np.diff(A[-9:]))), float(np.median(np.diff(B[:9])))
            gap = B[0] - A[-1]
            m = gap / (0.5 * (pa + pb))
            r = int(round(m))
            if abs(pa - pb) / pa < 0.05 and r >= 2 and abs(m - r) <= TRIPLET_PHASE:
                through = A[-1] + gap * np.arange(1, r) / r
                ref = min(float(np.mean(score(A[-16:]))), float(np.mean(score(B[:16]))))
                if float(np.mean(score(through))) >= TRIPLET_EVIDENCE * ref:
                    keep[i:j] = [False] * (j - i)
        i = max(j, i + 1)
    return [p for p, k in zip(parts, keep) if k]


def beat_confidence(beats, beat_logits, first_beat, last_beat, fps=50.0):
    """How sure Beat This! is of each beat: its beat probability at the beat (max within +-1 frame), 0..100.
    Beats outside the detected range (extended over silent intros / outros) are guesses: 0."""
    b = np.asarray(beats, float)
    out = np.zeros(len(b))
    if beat_logits is not None and len(beat_logits):
        a = _sig(beat_logits)
        fr = np.round(b * fps).astype(int)
        for k in (-1, 0, 1):
            i = fr + k
            ok = (i >= 0) & (i < len(a))
            out = np.maximum(out, np.where(ok, a[np.clip(i, 0, len(a) - 1)], 0.0))
    out[:first_beat] = 0
    out[last_beat + 1:] = 0
    return [int(round(v * 100)) for v in out]


CONTENT_WINDOW = 0.5     # fraction of a beat either side searched for a transient
CONTENT_FLOOR = 0.10     # a beat with less onset energy than this (of the song's typical beat) carries no pulse


def beat_content(beats, env, hop, sr, first_beat, last_beat):
    """Per-beat 0..1: how much rhythmic (onset) energy sits on each beat, relative to the song's own typical
    beat. Ambient pads, held chords, rubato intros / outros and near-silence score ~0: there is genuinely no
    pulse to detect there, as opposed to a pulse the tracker was merely unsure of. The UI uses this to tell
    'correctly beatless - nothing to check' from 'rhythmic, but the tracker wavered' so it only flags the
    latter. Reuses the drum/onset envelope the grid is already built from; no extra detection, no new model.

    env: onset-strength envelope (body_env / _onset_env, normalized), hop/sr its frame rate."""
    b = np.asarray(beats, float)
    out = np.zeros(len(b))
    if env is None or not len(env):
        out[first_beat:last_beat + 1] = 1.0        # no envelope to judge by: don't suppress anything
        return [int(round(v * 100)) for v in out]
    e = np.asarray(env, float)
    rate = sr / hop
    n = len(b)
    ibi = np.r_[np.diff(b), np.diff(b)[-1:] if n > 1 else [0.5]]
    peak = np.zeros(n)
    x = np.arange(len(e))
    for j in range(n):
        w = CONTENT_WINDOW * ibi[j]
        t = b[j] + np.linspace(-w, w, 9)
        peak[j] = float(np.interp(t * rate, x, e, left=0.0, right=0.0).max())
    inside = peak[first_beat:last_beat + 1]
    typical = float(np.quantile(inside, 0.75)) if len(inside) else 0.0
    if typical < 1e-6:
        out[first_beat:last_beat + 1] = 1.0
        return [int(round(v * 100)) for v in out]
    rel = np.clip(peak / typical, 0.0, 1.0)
    rel[rel < CONTENT_FLOOR] = 0.0                 # below the floor: treat as truly beatless
    out[first_beat:last_beat + 1] = rel[first_beat:last_beat + 1]
    out[:first_beat] = 0.0                         # extended guesses over silent intros / outros: no content
    out[last_beat + 1:] = 0.0
    return [int(round(v * 100)) for v in out]


def _sig(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, float)))


def _downbeat_act(beats, down_logits, fps=50.0):
    """Beat This! downbeat probability at each beat (max over +-1 frame)."""
    d = _sig(down_logits)
    fr = np.round(np.asarray(beats) * fps).astype(int)
    out = np.zeros(len(fr))
    for i, f in enumerate(fr):
        a, b = max(0, f - 1), min(len(d), f + 2)
        if b > a:
            out[i] = d[a:b].max()
    return out


def _contrast(act, m):
    """How much more downbeat-like every m-th beat is than the rest (best phase). -> (contrast, phase)."""
    best, ph_best = -1.0, 0
    for ph in range(m):
        on = act[ph::m]
        mask = np.ones(len(act), bool)
        mask[ph::m] = False
        if len(on) < 2 or mask.sum() < 2:
            continue
        c = on.mean() - act[mask].mean()
        if c > best:
            best, ph_best = c, ph
    return best, ph_best


def meter(act):
    """Beats per bar from downbeat probabilities: 4 unless 3-beat bars are clearly stronger; 6 when every
    other 3-beat group is the real downbeat (compound time counted in eighths, e.g. 6/8 at ~180 'BPM').
    Measured on this library: clear 4/4 songs give contrast ~0.75 at 4 and ~0.01 at 3."""
    c3, _ = _contrast(act, 3)
    c4, _ = _contrast(act, 4)
    c6, _ = _contrast(act, 6)
    if max(c3, c6) > c4 + 0.1 and max(c3, c6) >= 0.15:
        return 6 if c6 > 1.3 * c3 else 3
    return 4


def rebar(beats, segments, first_beat, last_beat, down_logits=None, didx=None, bpb=None):
    """Meter + bar lines for a beat list: from Beat This!'s downbeat probabilities where available,
    else from the detected downbeats (didx: their beat indices). Per segment, the bar phase with the
    strongest downbeat evidence. bpb: keep this meter instead of measuring it. -> (beats per bar, bar starts)."""
    beats = np.asarray(beats)
    act = _downbeat_act(beats, down_logits) if down_logits is not None and len(down_logits) else None
    didx = np.asarray(didx if didx is not None else [], int)
    if bpb:
        pass
    elif act is not None:
        bpb = meter(act[first_beat:last_beat + 1])
    else:
        gaps = np.diff(didx) if len(didx) > 1 else np.array([4])
        gaps = gaps[(gaps >= 2) & (gaps <= 8)]
        mode = int(np.bincount(gaps).argmax()) if len(gaps) else 4
        bpb = 3 if mode in (3, 6) else 4
    bars = []
    for a, b in segments:
        if act is not None and b - a >= 2 * bpb:
            _, phase = _contrast(act[a:b], bpb)
        else:
            d = didx[(didx >= a) & (didx < b)]
            phase = int(np.bincount((d - a) % bpb, minlength=bpb).argmax()) if len(d) else 0
        bars.extend(range(a + phase, b, bpb))     # beats before a+phase are a pickup, not a bar
    return int(bpb), [int(x) for x in bars]


STEADY_RANGE = 0.03       # the steady tempo is searched within +-3 % of the tracked main tempo
STEADY_WINDOW = 16        # beats per local check
STEADY_SHIFT = 0.07       # a stretch agrees with the steady grid if its best local shift is within 7 % of a beat ...
STEADY_SHARE = 0.85       # ... in at least 85 % of the stretches that have clear beats
STEADY_CLEAR = 1.2        # (a stretch has clear beats if its best shift scores 20 % above its average shift)
STEADY_SCORE = 0.75       # and the steady beats are at least 75 % as beat-like as the tracked ones (the tracked beats
#                           sit on each local peak, so they always score a bit higher, even on a click-tracked song)
last_steady = {}          # what the last steady_grid() call measured (for inspection / tuning)


def steady_grid(t_a, t_z, score, main_ibi, tracked):
    """A song recorded to a click is steady: one tempo and phase fit all of it. The tracker follows the evidence
    beat by beat, so where it wavers or drops a beat its grid wanders (It'll Be OK: tracked at 97.94 BPM, marks up
    to 190 ms off, 5 beats short - the song is a steady 98.97). Find the best steady pulse over the detected range
    [t_a, t_z] (+-STEADY_RANGE around the tracked tempo; fold the evidence by the period, best phase), then check it
    stretch by stretch: if it holds everywhere the beats are clear, and fits the music about as well as the tracked
    beats, the song is steady. -> steady beat times over [t_a, t_z], or None (tempo changes, a live take, rubato)."""
    global last_steady
    fps = 200.0
    tt = np.arange(t_a, t_z, 1.0 / fps)
    if len(tt) < fps * 20:
        last_steady = dict(reason="too short")
        return None
    cur = score(tt)
    nb = 96
    best = (-1.0, None, None)
    bpm0 = 60.0 / main_ibi
    for bpm in np.arange(bpm0 * (1 - STEADY_RANGE), bpm0 * (1 + STEADY_RANGE), 0.01):
        per = 60.0 / bpm
        k = np.minimum((((tt - t_a) % per) / per * nb).astype(int), nb - 1)
        m = np.bincount(k, weights=cur, minlength=nb) / np.maximum(np.bincount(k, minlength=nb), 1)
        m = (m + np.roll(m, 1) + np.roll(m, -1)) / 3              # a little smoothing across neighbouring phases
        i = int(np.argmax(m))
        if m[i] > best[0]:
            best = (float(m[i]), per, t_a + (i + 0.5) / nb * per)
    _, per, ph = best
    # refine tempo and phase on the actual beat positions
    grid = lambda per_, ph_: ph_ + np.arange(int(np.ceil((t_a - ph_) / per_)), int(np.floor((t_z - ph_) / per_)) + 1) * per_
    cand = [(float(score(grid(p_, ph + d)).mean()), p_, ph + d)
            for p_ in per * (1 + np.arange(-0.0004, 0.00041, 0.00005))
            for d in np.arange(-0.02, 0.0201, 0.002)]
    s_steady, per, ph = max(cand)
    g = grid(per, ph)
    tr = np.asarray(tracked, float)
    s_tracked = float(score(tr[(tr >= t_a) & (tr <= t_z)]).mean())
    shifts = np.arange(-0.25, 0.2501, 0.01)
    agree = clear = 0
    for w in range(0, len(g) - STEADY_WINDOW + 1, STEADY_WINDOW // 2):
        seg = g[w:w + STEADY_WINDOW]
        sc = np.array([score(seg + s * per).mean() for s in shifts])
        if sc.max() < STEADY_CLEAR * sc.mean():
            continue
        clear += 1
        agree += abs(shifts[int(np.argmax(sc))]) <= STEADY_SHIFT
    share = agree / clear if clear else 0.0
    ok = clear >= 4 and share >= STEADY_SHARE and s_steady >= STEADY_SCORE * s_tracked
    last_steady = dict(bpm=round(60 / per, 4), share=round(share, 3), clear=clear, s_steady=round(s_steady, 3),
                       s_tracked=round(s_tracked, 3), steady=bool(ok))
    return g if ok else None


def build_grid(y_mix_mono, y_drums_mono, sr, dur, raw=None):
    """raw: optional dict(beats, downbeats, down_logits, beat_logits, env, hop, body_env) - reuse a detection."""
    if raw is None:
        raw_b, raw_d, down_logits, beat_logits = detect_full(y_mix_mono, sr)
        env, hop = _onset_env(y_drums_mono if y_drums_mono is not None else y_mix_mono, sr)
        benv, _ = body_env(y_drums_mono if y_drums_mono is not None else y_mix_mono, sr)
    else:
        raw_b, raw_d, down_logits = raw["beats"], raw["downbeats"], raw.get("down_logits")
        beat_logits = raw.get("beat_logits")
        env, hop = raw["env"], int(raw["hop"])
        benv = raw.get("body_env", env)
    if len(raw_b) < 8:
        raise RuntimeError(f"only {len(raw_b)} beats found - is this a rhythmic song?")

    regimes = _track(raw_b) or _track(raw_b, tol=0.25, min_beats=6)
    if not regimes:
        raise RuntimeError("couldn't find a steady beat anywhere in this song")
    parts, stable = [], True
    for k, t in regimes:
        f, st = _fit_segment(k, t)
        stable &= st
        parts.append(_phase_refine(f, env, hop, sr))
    main_bpm = _main_tempo(parts)
    if main_bpm > 200 and down_logits is not None and len(down_logits):
        # > 200 BPM is an eighth-note track; fold to quarter notes - unless the bars are 6 of those
        # eighths (compound time, e.g. 6/8), where the eighth IS the natural counting level
        lp = max(parts, key=lambda p: p[-1] - p[0])
        if meter(_downbeat_act(lp, down_logits)) == 4:
            main_bpm /= 2
    main_ibi = 60.0 / main_bpm
    folded = []
    for p in parts:
        n = _fold_ratio(main_ibi / float(np.mean(np.diff(p))))
        if n == 1:
            q = p                                          # already at the main level
        elif n:
            q = _fold(p, main_ibi, env, hop, sr)
        elif folded:                                       # a real tempo change: match its neighbour's level
            q = _fold(p, float(np.mean(np.diff(folded[-1]))), env, hop, sr)
        else:
            q = p
        folded.append(q)
    score = evidence(beat_logits, benv, hop, sr)
    folded = _drop_triplet_feel(folded, main_ibi, score)
    parts = _fill_gaps(_phase_joins(folded, score), score, env, hop, sr)
    beats, segments = _join(parts)
    steady = steady_grid(float(beats[0]), float(beats[-1]), score, main_ibi, beats)
    if steady is not None:                                 # one steady tempo fits the whole song: use it
        beats, segments = np.asarray(steady, float), [[0, len(steady)]]
        parts, stable = [beats], True

    # extend the grid to cover the whole file (silent intros/outros get beats too)
    ibi0 = beats[1] - beats[0]
    ibi1 = beats[-1] - beats[-2]
    n_pre = int(np.ceil(beats[0] / ibi0)) + 1
    pre = beats[0] - ibi0 * np.arange(n_pre, 0, -1)
    n_post = int(np.ceil((dur - beats[-1]) / ibi1)) + 1
    post = beats[-1] + ibi1 * np.arange(1, n_post + 1)
    ext = np.concatenate([pre, beats, post])
    segments = [(a + n_pre, b + n_pre) for a, b in segments]
    segments[0] = (0, segments[0][1])
    segments[-1] = (segments[-1][0], len(ext))
    didx = np.array([int(np.argmin(np.abs(beats - d))) for d in raw_d if np.min(np.abs(beats - d)) < 0.07], int)
    didx = didx + n_pre

    bpb, bars = rebar(ext, segments, n_pre, n_pre + len(beats) - 1, down_logits, didx)

    # tempo of the main level: the AVERAGE beat over the regimes at that level, so a steady song played as recorded
    # stays on the grid over a whole clip (the median beat was up to 0.5 % off: marks slid ~3 beats over 600).
    # Where the two differ by more, the tempo really moves and the median is the more typical one.
    main = [p for p in parts if abs(float(np.mean(np.diff(p))) / main_ibi - 1) < FOLD_TOL]
    ib = np.concatenate([np.diff(p) for p in main]) if main else np.diff(beats)
    bpm = float(60.0 / np.median(ib))
    if steady is not None:
        bpm = 60.0 / float(np.mean(np.diff(beats)))            # exact: the steady grid's own tempo
    elif main:
        avg = 60.0 * sum(len(p) - 1 for p in main) / sum(float(p[-1] - p[0]) for p in main)
        if abs(avg / bpm - 1) < 0.005:
            bpm = avg
    conf = beat_confidence(ext, beat_logits, n_pre, n_pre + len(beats) - 1)
    content = beat_content(ext, benv, hop, sr, n_pre, n_pre + len(beats) - 1)
    return dict(beats=ext.tolist(), bars=bars, bpb=bpb, bpm=round(bpm, 4), beat_conf=conf, beat_content=content,
                segments=[list(s) for s in segments], stable=bool(stable),
                first_beat=int(n_pre), last_beat=int(n_pre + len(beats) - 1), grid_version=GRID_VERSION)
