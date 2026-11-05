"""Line a YuE2 take up with the mix it re-imagines: which moment of the take is each beat of the mix.

YuE2 follows the mix's melody and chords but not its clock: its tempo wanders and it drifts by bars over a song.
So the take is matched to the mix by what they share - harmony and rhythm - with a time-warping match (DTW over
chroma + onset strength, ~46 ms frames, banded), each mix beat's matching moment is smoothed over its neighbours
and snapped to the nearest beat the detector hears in the take. The result is a beat grid for the take on which
beat k IS mix beat k, so the engine's beat-by-beat warp plays bar N of the take on bar N of the mix.
"""
import numpy as np

FS = 22050             # analysis rate
HOP = 1024             # ~46 ms frames
BAND = 0.1             # DTW band: the take may wander up to 10 % of the song from the diagonal
ONSET_W = 0.35         # weight of the rhythm row next to the 12 chroma rows (tuned on two full-mix takes)
SMOOTH = 4             # beats each side the matched positions are averaged over
SNAP = 0.15            # snap to a heard beat within this share of a beat


def _features(y, sr):
    import librosa
    import soxr
    y = soxr.resample(np.asarray(y, np.float32), sr, FS)
    C = librosa.feature.chroma_cqt(y=y, sr=FS, hop_length=HOP)
    C = C / (np.linalg.norm(C, axis=0, keepdims=True) + 1e-9)
    O = librosa.onset.onset_strength(y=y, sr=FS, hop_length=HOP)
    O = (O - O.mean()) / (O.std() + 1e-9)
    n = min(C.shape[1], len(O))
    return np.vstack([C[:, :n], ONSET_W * O[None, :n]])


def align(mix, take, sr, bpm, n_beats, take_beats=None):
    """mix, take: mono audio at sr. Mix beat k is at k * 60 / bpm (the project grid). take_beats: beats heard in the
    take (s), to snap to. -> take times (s) of mix beats 0..n_beats-1, strictly increasing."""
    import librosa
    Fm, Ft = _features(mix, sr), _features(take, sr)
    _, wp = librosa.sequence.dtw(X=Fm, Y=Ft, metric="cosine", global_constraints=True, band_rad=BAND)
    wp = wp[::-1]
    fr = HOP / FS
    tm, tt = wp[:, 0] * fr, wp[:, 1] * fr
    um = np.unique(tm)
    ut = np.array([tt[tm == u].mean() for u in um])        # one take time per mix frame (the path's average)
    period = 60.0 / bpm
    tau = np.interp(np.arange(n_beats) * period, um, ut)
    # smooth: the warp path moves in frame steps; a tempo can't jump from beat to beat
    k = np.arange(n_beats)
    sm = np.array([tau[max(0, i - SMOOTH):i + SMOOTH + 1].mean() for i in k])
    if take_beats is not None and len(take_beats):
        tb = np.asarray(take_beats, float)
        j = np.clip(np.searchsorted(tb, sm), 1, len(tb) - 1)
        near = np.where(np.abs(tb[j - 1] - sm) < np.abs(tb[j] - sm), tb[j - 1], tb[j])
        sm = np.where(np.abs(near - sm) < SNAP * period, near, sm)
    for i in range(1, n_beats):                            # strictly forward, at least 40 % of a beat apart
        sm[i] = max(sm[i], sm[i - 1] + 0.4 * period)
    return sm


SKIP = 0.35            # cost of the take having one beat more / fewer than the mix at a point (vs a 1:1 step)


def _beat_features(y, sr, beats):
    """Per beat: chroma over the beat (L2-normalised) + the onset strength right at it."""
    import librosa
    import soxr
    y = soxr.resample(np.asarray(y, np.float32), sr, FS)
    hop = 512
    C = librosa.feature.chroma_cqt(y=y, sr=FS, hop_length=hop)
    O = librosa.onset.onset_strength(y=y, sr=FS, hop_length=hop)
    O = (O - O.mean()) / (O.std() + 1e-9)
    f = np.clip(np.round(np.asarray(beats) * FS / hop).astype(int), 0, C.shape[1] - 1)
    out = []
    for i in range(len(f)):
        a, b = f[i], f[i + 1] if i + 1 < len(f) else min(C.shape[1], f[i] + 8)
        c = C[:, a:max(b, a + 1)].mean(1)
        c = c / (np.linalg.norm(c) + 1e-9)
        o = O[max(0, a - 2):a + 3].max() if a < len(O) else 0.0
        out.append(np.r_[c, ONSET_W * o])
    return np.array(out)


def beat_level(take_beats, period):
    """The take's beats at the mix's beat level: split intervals of a half-time track, drop every other of a
    double-time one. None when its level doesn't relate to the mix's by 1, 2 or 1/2."""
    tb = np.asarray(take_beats, float)
    r = float(np.median(np.diff(tb))) / period
    if 0.8 < r < 1.25:
        return tb
    if 1.7 < r < 2.4:
        return np.sort(np.r_[tb, 0.5 * (tb[1:] + tb[:-1])])
    if 0.42 < r < 0.6:
        return tb[::2]
    return None


def align_beats(mix, take, sr, bpm, n_beats, take_beats):
    """Beat-to-beat match: mix beat k (at k * 60 / bpm) -> one of the take's own beats, with 1:1 steps preferred and
    a beat more or fewer in the take allowed at a cost (DTW over beat-synchronous chroma + onset). The take's beats
    are what its detector heard, so the result follows its real pulse, not frame jitter. Falls back to align() when
    the take's beat level can't be related to the mix's. -> take times (s) of mix beats 0..n_beats-1."""
    import librosa
    period = 60.0 / bpm
    dur = len(take) / sr
    tb = None
    if take_beats is not None:
        heard = np.asarray(take_beats, float)
        heard = heard[(heard >= 0) & (heard < dur)]             # the grid's extension past the audio isn't the take
        tb = beat_level(heard, period) if len(heard) > 8 else None
    if tb is None:
        return align(mix, take, sr, bpm, n_beats, take_beats)
    Fm = _beat_features(mix, sr, np.arange(n_beats) * period)
    Ft = _beat_features(take, sr, tb)
    cost = 1.0 - np.clip(Fm @ Ft.T / (np.linalg.norm(Fm, axis=1)[:, None] * np.linalg.norm(Ft, axis=1)[None, :] + 1e-9), -1, 1)
    steps = np.array([[1, 1], [1, 2], [2, 1]])
    # a skip step covers two beats, so it pays twice (else skipping would be the cheap way through) plus SKIP
    _, wp = librosa.sequence.dtw(C=cost, step_sizes_sigma=steps, weights_add=np.array([0.0, SKIP, SKIP]),
                                 weights_mul=np.array([1.0, 2.0, 2.0]), global_constraints=True, band_rad=BAND,
                                 subseq=False)
    wp = wp[::-1]
    tau = np.full(n_beats, np.nan)
    for k, j in wp:
        if np.isnan(tau[k]):
            tau[k] = tb[j]
    ok = ~np.isnan(tau)
    tau = np.interp(np.arange(n_beats), np.flatnonzero(ok), tau[ok])        # a mix beat the take skipped
    for i in range(1, n_beats):
        tau[i] = max(tau[i], tau[i - 1] + 0.4 * period)
    return tau


SKIP_BAR = 0.5         # cost of the take having a bar more / fewer than the mix at a point
DOWNBEAT_W = 0.3       # cost of matching a mix bar to a take bar that doesn't start on a heard downbeat


def _take_downbeats(take, sr, tb):
    """Per take beat: how downbeat-like the detector hears it (0..1, Beat This!'s downbeat activation)."""
    from . import beats as B
    _, _, dl, _ = B.detect_full(np.asarray(take, np.float32), sr)
    return np.clip(B._downbeat_act(tb, dl), 0.0, 1.0)


def _to_mix_level(take_grid, period, dur):
    """The take's beats and bar starts at the mix's beat level (a half-time take's beats split, each of its bars
    two of the mix's), only inside the audio. -> (beats, bar start indices) or None."""
    tb = np.asarray(take_grid["beats"], float)
    bars = list(take_grid["bars"])
    r = float(np.median(np.diff(tb))) / period
    if 1.7 < r < 2.4:                                     # half-time: double the beats, split every bar in two
        mid = 0.5 * (tb[1:] + tb[:-1])
        tb = np.sort(np.r_[tb, mid])
        half = int(take_grid.get("bpb", 4))
        bars = sorted({2 * b for b in bars} | {2 * b + half for b in bars})
    elif not 0.8 < r < 1.25:
        return None
    bars = [b for b in bars if b >= 0 and b + 4 <= len(tb) and tb[b] >= 0 and tb[b + 3] < dur]
    return tb, bars


def align_bars(mix, take, sr, bpm, n_beats, take_grid):
    """Bar-to-bar match: the mix's bars (every 4 beats from its beat 0) -> the take's bars (from its own detected
    downbeats), 1:1 steps preferred, a bar more or fewer in the take allowed at a cost (DTW over per-bar features:
    chroma of each of its 4 beats + their onsets). Beats 1-4 of a matched bar then map onto beats 1-4, so a take
    can never end up a beat out of phase - the failure of a beat-by-beat match, where the beats of one bar look
    alike. Falls back to align_beats when the take isn't in 4 at the mix's beat level.
    -> take times (s) of mix beats 0..n_beats-1."""
    import librosa
    period = 60.0 / bpm
    lv = _to_mix_level(take_grid, period, len(take) / sr) if take_grid.get("bpb", 4) == 4 else None
    if lv is None or len(lv[1]) < 4:
        return align_beats(mix, take, sr, bpm, n_beats, take_grid.get("beats"))
    tb, tbars0 = lv
    Fm = _beat_features(mix, sr, np.arange(n_beats + 1) * period)
    Ft = _beat_features(take, sr, tb)
    n_mbar = n_beats // 4
    fm = np.array([Fm[4 * m:4 * m + 4].ravel() for m in range(n_mbar)])
    # the take's own bar lines may sit on the wrong beat (its analysis guessed the phase): try all four, with a
    # preference for bars that start where the detector hears a downbeat, and keep the cheapest match
    down = _take_downbeats(take, sr, tb)
    steps = np.array([[1, 1], [1, 2], [2, 1]])
    best = None
    for ph in range(4):
        tbars = [b + ph for b in tbars0 if b + ph + 4 <= len(tb)]
        if len(tbars) < 4:
            continue
        ft = np.array([Ft[b:b + 4].ravel() for b in tbars])
        cost = 1.0 - np.clip(fm @ ft.T / (np.linalg.norm(fm, axis=1)[:, None] * np.linalg.norm(ft, axis=1)[None, :] + 1e-9), -1, 1)
        cost = cost + DOWNBEAT_W * (1.0 - down[tbars])[None, :]
        D, wp = librosa.sequence.dtw(C=cost, step_sizes_sigma=steps, weights_add=np.array([0.0, SKIP_BAR, SKIP_BAR]),
                                     weights_mul=np.array([1.0, 2.0, 2.0]), global_constraints=True, band_rad=BAND,
                                     subseq=False)
        total = float(D[-1, -1]) / len(wp)
        if best is None or total < best[0]:
            best = (total, wp, tbars)
    _, wp, tbars = best
    tau = np.full(n_beats, np.nan)
    for m, n in wp[::-1]:
        if np.isnan(tau[4 * m]):
            tau[4 * m:4 * m + 4] = tb[tbars[n]:tbars[n] + 4]
    ok = ~np.isnan(tau)
    tau = np.interp(np.arange(n_beats), np.flatnonzero(ok), tau[ok])        # mix bars the take skipped
    for i in range(1, n_beats):
        tau[i] = max(tau[i], tau[i - 1] + 0.4 * period)
    return tau


def grid(tau, take_dur, bpm, bpb=4):
    """A beat grid (analyze.analyze `grid`) for the take from its aligned beats: extended over the take's own
    lead-in and tail at the local beat length; bar lines every bpb beats in phase with the mix (mix beat 0 is a
    downbeat)."""
    tau = np.asarray(tau, float)
    p0 = float(np.median(np.diff(tau[:16]))) if len(tau) > 2 else 60.0 / bpm
    p1 = float(np.median(np.diff(tau[-16:]))) if len(tau) > 2 else 60.0 / bpm
    pre = max(0, int(np.floor(tau[0] / p0)))
    lead = tau[0] - p0 * np.arange(pre, 0, -1)
    post = int(np.floor((take_dur - tau[-1]) / p1))
    tail = tau[-1] + p1 * np.arange(1, post + 1)
    beats = np.concatenate([lead, tau, tail])
    first, last = pre, pre + len(tau) - 1
    bars = list(range(pre % bpb, len(beats), bpb))
    return dict(beats=[round(float(t), 5) for t in beats], bars=bars, bpb=bpb, bpm=round(float(bpm), 4),
                stable=True, segments=[[0, len(beats)]], first_beat=int(first), last_beat=int(last), grid_version=5)
