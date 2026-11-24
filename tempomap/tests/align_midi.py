"""Line a .mid up with a real recording of the same song, and carry the file's bar lines and beats onto the
recording - an answer key on real audio.

1. Both sides become the same kind of description, 10 times a second: the chord sounding (chroma, from the notes on
   the file's side, from the audio on the recording's side) and where notes start (bass, the rest, kick, snare).
2. Dynamic time warping finds the best monotone path between them (steps that keep the speed between half and
   double, so a repeat left out or an intro of a different length is skipped rather than smeared).
3. Every beat of the file is carried across the path, then each is nudged (within 60 ms) onto where the
   recording's onsets around it line up best with the file's notes around it.
4. How good the line-up is gets measured where the file has notes: the share of its note starts that land on an
   onset in the recording (within 50 ms). A key is only used where that is high.
"""
import numpy as np

FPS = 20.0
SLACK = 60.0              # s either side may start or end before the other (intros, outros, a fade)


def midi_notes(path):
    """-> list of (start s, end s, pitch, velocity, is_drum), with the file's tempo map (Re-Tex's reading: its
    first tempo from tick 0)."""
    import mido
    mf = mido.MidiFile(str(path), clip=True)
    ppq = mf.ticks_per_beat
    tempos = {}
    for tr in mf.tracks:
        t = 0
        for msg in tr:
            t += msg.time
            if msg.type == "set_tempo":
                tempos[t] = msg.tempo
    if not tempos:
        tempos = {0: 500000}
    tempos.setdefault(0, tempos[min(tempos)])
    marks = sorted(tempos.items())
    ticks = np.array([m[0] for m in marks], float)
    secs = np.zeros(len(marks))
    for j in range(1, len(marks)):
        secs[j] = secs[j - 1] + (ticks[j] - ticks[j - 1]) / ppq * marks[j - 1][1] / 1e6

    def sec(T):
        j = int(np.searchsorted(ticks, T, side="right") - 1)
        return secs[j] + (T - ticks[j]) / ppq * marks[j][1] / 1e6

    notes = []
    for tr in mf.tracks:
        t, open_ = 0, {}
        for msg in tr:
            t += msg.time
            if msg.type == "note_on" and msg.velocity:
                open_.setdefault((msg.channel, msg.note), []).append((t, msg.velocity))
            elif msg.type in ("note_off", "note_on"):
                st = open_.get((msg.channel, msg.note))
                if st:
                    t0, v = st.pop(0)
                    notes.append((sec(t0), sec(max(t, t0 + 1)), msg.note, v, msg.channel == 9))
    return notes, sec, ppq


def midi_desc(notes, dur):
    n = int(dur * FPS) + 2
    ch = np.zeros((n, 12))
    on = np.zeros((n, 4))                              # bass, other, kick, snare
    for s, e, p, v, drum in notes:
        a, b = int(s * FPS), max(int(s * FPS) + 1, int(e * FPS))
        if drum:
            if p in (35, 36):
                on[min(a, n - 1), 2] += v / 127
            elif p in (38, 40, 37, 39):
                on[min(a, n - 1), 3] += v / 127
            continue
        ch[a:min(b, n), p % 12] += v / 127
        on[min(a, n - 1), 0 if p < 52 else 1] += v / 127
    return ch, on


def audio_desc(feat, dur):
    n = int(dur * FPS) + 2
    t = np.arange(n) / FPS
    ci = np.clip((t * feat.cfps).astype(int), 0, len(feat.chroma) - 1)
    ch = feat.chroma[ci].astype(float)
    on = np.zeros((n, 4))
    for k, name in enumerate(("bass", "other", "kick", "snare")):
        p = feat.parts.get(name)
        if p is None:
            continue
        x = np.clip(p, 0, None)
        hop = feat.fps / FPS
        for i in range(n):
            a, b = int(i * hop), int((i + 1) * hop)
            if a < len(x):
                on[i, k] = x[a:b].max() if b > a else x[a]
    return ch, on


def _norm_rows(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)


def _smooth(on, k=3):
    ker = np.exp(-np.arange(k) / 1.5)
    out = np.zeros_like(on)
    for j in range(on.shape[1]):
        out[:, j] = np.convolve(on[:, j], ker)[:len(on)]
    return out


def dtw(cost):
    """Monotone path through cost [n, m] with steps (1,1), (1,2), (2,1): -> list of (i, j)."""
    n, m = cost.shape
    INF = 1e18
    D = np.full((n, m), INF)
    P = np.zeros((n, m), np.int8)
    lim = int(SLACK * FPS)                            # either side may start up to SLACK s late (a longer intro)
    D[0, :lim] = cost[0, :lim] * 3
    D[:lim, 0] = np.minimum(D[:lim, 0], cost[:lim, 0] * 3)
    for i in range(1, n):
        c = cost[i]
        a = np.full(m, INF)
        a[1:] = D[i - 1, :-1] + 2 * c[1:]             # diagonal
        b = np.full(m, INF)
        b[2:] = D[i - 1, :-2] + 2 * cost[i, 1:-1] + c[2:]    # (1, 2)
        cc = np.full(m, INF)
        if i >= 2:
            cc[1:] = D[i - 2, :-1] + 2 * cost[i - 1, 1:] + c[1:]  # (2, 1)
        best = np.minimum(np.minimum(a, b), cc)
        P[i] = np.where(best == a, 0, np.where(best == b, 1, 2))
        D[i] = np.minimum(D[i], best)
    # both may end early too: best end on the last row or column, per step taken (and not a stub)
    cands = [(n - 1, j) for j in range(m)] + [(i, m - 1) for i in range(n)]
    floor = 0.5 * min(n, m)
    lim = int(SLACK * FPS)
    i, j = min((c for c in cands if c[0] >= n - 1 - lim and c[1] >= m - 1 - lim and D[c] < 1e17),
               key=lambda c: D[c] / (c[0] + c[1] + 1))
    path = [(i, j)]
    while i > 0 and j > 0:
        p = P[i, j]
        if p == 0:
            i, j = i - 1, j - 1
        elif p == 1:
            i, j = i - 1, j - 2
        else:
            i, j = i - 2, j - 1
        if D[i, j] >= 1e17:
            break
        path.append((i, j))
    return path[::-1]


def _prep(x, k):
    x = x - x.mean(0)
    if k > 1:
        ker = np.ones(k) / k
        x = np.stack([np.convolve(x[:, j], ker, mode="same") for j in range(x.shape[1])], 1)
    return _norm_rows(x)


def align(notes, midi_dur, feat, real_dur, w=(1.0, 0.5, 0.5)):
    """-> function mapping file seconds to recording seconds (piecewise linear along the path), and the path.
    Each description is centred (what changes counts, not what is always there) and lightly smoothed; the cost
    is the drums' (kick, snare) dissimilarity, the notes' (bass, rest) and the chords', weighted by w."""
    mc, mo = midi_desc(notes, midi_dur)
    ac, ao = audio_desc(feat, real_dur)
    has_drums = mo[:, 2:].sum() > 0 and ao[:, 2:].sum() > 0
    cost = w[1] * (1 - _prep(mo[:, :2], 3) @ _prep(ao[:, :2], 3).T)
    cost += w[2] * (1 - _prep(mc, int(FPS)) @ _prep(ac, int(FPS)).T)
    if has_drums:
        cost += w[0] * (1 - _prep(mo[:, 2:], 3) @ _prep(ao[:, 2:], 3).T)
    path = np.array(dtw(cost), float) / FPS
    xs, ys = path[:, 0], path[:, 1]
    _, idx = np.unique(xs, return_index=True)
    xs, ys = xs[idx], ys[idx]

    def warp(t):
        return np.interp(t, xs, ys)
    return warp, (xs, ys)


def refine(times, notes_on, feat, warp, reach=0.06):
    """Each carried time nudged within `reach` s onto where the recording's onsets line up with the file's note
    starts within half a second around it. -> new times."""
    env = np.clip(feat.env, 0, None)
    fps = feat.fps
    no = np.asarray(sorted(notes_on))
    out = []
    for t in times:
        near = no[(no > t - 0.5) & (no < t + 0.5)]
        tw = warp(t)
        if len(near) < 2:
            out.append(tw)
            continue
        rel = warp(near) - tw
        best, bs = -1, 0.0
        for sh in np.arange(-reach, reach + 1e-9, 0.005):
            idx = np.round((tw + sh + rel) * fps).astype(int)
            idx = idx[(idx >= 0) & (idx < len(env))]
            sc = env[idx].sum() if len(idx) else 0
            if sc > best:
                best, bs = sc, sh
        out.append(tw + bs)
    return np.asarray(out)


def quality(drum_on, feat, warp, tol=0.03):
    """Share of the file's kick and snare notes (carried across) within tol of a hit in the recording, against the
    same notes moved an eighth of a second (chance on that recording)."""
    from tempomap.pulse import onset_peaks
    pk, _ = onset_peaks(feat.hits, feat.fps, 0, feat.n / feat.fps, q=80)
    t = warp(np.asarray(drum_on))

    def share(x):
        j = np.clip(np.searchsorted(pk, x), 1, len(pk) - 1)
        d = np.minimum(np.abs(pk[j] - x), np.abs(pk[j - 1] - x))
        return float(np.mean(d <= tol))
    return share(t), 0.5 * (share(t + 0.125) + share(t - 0.125))


# ---------------------------------------------------------------- beat by beat (more robust)

def _beat_feats_audio(feat, beats):
    """Per beat: kick, snare, bass, other onset strength at its four sixteenths (16) + the chord over it (12)."""
    parts = [np.clip(feat.parts.get(k, np.zeros(feat.n)), 0, None) for k in ("kick", "snare", "bass", "other")]
    reach = int(0.025 * feat.fps)
    rows = []
    for a, b in zip(beats[:-1], beats[1:]):
        r = []
        for p in parts:
            for q in range(4):
                i = int((a + (b - a) * q / 4) * feat.fps)
                r.append(p[max(0, i - reach):i + reach + 1].max() if 0 <= i < len(p) else 0.0)
        ci0, ci1 = int(a * feat.cfps), max(int(a * feat.cfps) + 1, int(b * feat.cfps))
        r += list(feat.chroma[min(ci0, len(feat.chroma) - 1):min(ci1, len(feat.chroma))].mean(0)) \
            if ci0 < len(feat.chroma) else [0.0] * 12
        rows.append(r)
    return np.asarray(rows, float)


def _beat_feats_midi(notes, beats):
    rows = np.zeros((len(beats) - 1, 28))
    b = np.asarray(beats)
    for s, e, p, v, drum in notes:
        k = int(np.searchsorted(b, s, side="right") - 1)
        if k < 0 or k >= len(b) - 1:
            continue
        q = int(np.round((s - b[k]) / (b[k + 1] - b[k]) * 4))
        if q == 4:
            k, q = k + 1, 0
            if k >= len(b) - 1:
                continue
        if drum:
            col = 0 if p in (35, 36) else (1 if p in (37, 38, 39, 40) else None)
        else:
            col = 2 if p < 52 else 3
        if col is not None:
            rows[k, col * 4 + q] += v / 127
        if not drum:                                   # chroma over the beats the note sounds through
            k1 = int(np.searchsorted(b, e, side="left") - 1)
            rows[k:max(k, k1) + 1, 16 + p % 12] += v / 127
    return rows


def _center(X, groups):
    out = []
    for g, wt in groups:
        x = X[:, g] - X[:, g].mean(0)
        x = x / (x.std(0) + 1e-9)
        out.append(wt * _norm_rows(x))
    return np.hstack(out)


def align_beats(notes, midi_beats, feat, det):
    """DTW over beats: the file's beats against the recording's (the detector's, tried as they are and at double /
    half the rate). -> (warp function file s -> recording s, chosen level, mean match of matched beats)."""
    mb = np.asarray(midi_beats, float)
    Fm_raw = _beat_feats_midi(notes, mb)
    groups = [(slice(0, 8), 1.0), (slice(8, 16), 0.6), (slice(16, 28), 0.6)]
    has_drums = Fm_raw[:, :8].sum() > 0
    if not has_drums:
        groups = groups[1:]
    Fm = _center(Fm_raw, groups)
    best = None
    d = np.asarray(det, float)
    for level in ("same", "double", "half"):
        if level == "double":
            rb = np.sort(np.r_[d, 0.5 * (d[:-1] + d[1:])])
        elif level == "half":
            rb = d[::2]
        else:
            rb = d
        Fa = _center(_beat_feats_audio(feat, rb), groups)
        wsum = sum(wt for _, wt in groups)
        cost = 1 - (Fm @ Fa.T) / wsum
        global FPS
        keep = FPS
        FPS = 1.0                                     # (dtw's slack is in steps here: 60 beats)
        try:
            path = dtw(cost)
        except ValueError:                            # the two can't be lined up end to end at this level
            continue
        finally:
            FPS = keep
        pi = np.array(path)
        score = float(np.mean(1 - cost[pi[:, 0], pi[:, 1]]))
        if best is None or score > best[2]:
            best = (pi, rb, score, level)
    pi, rb, score, level = best
    _, idx = np.unique(pi[:, 0], return_index=True)
    pi = pi[idx]
    xs, ys = mb[pi[:, 0]], rb[pi[:, 1]]

    def warp(t):
        return np.interp(t, xs, ys)
    return warp, level, score
