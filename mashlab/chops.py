"""Chop kits (instrument type "chops", instruments.py): a stem of a song in the library cut into chops - its hits,
its words (the lyrics' word timing), its beats / half beats / bars, or its phrases - leaving out the silent parts.
MIDI notes then pick chops:
  "keys"    each key its own chop (from the base note up), so you play the chops like pads
  "order"   every note plays the next chop, whatever its pitch: the part's rhythm plays through the stem in order
  "shuffle" every note plays the next chop of a shuffled order (a seed, so it's the same every time)
  "drums"   the hits sorted into kick / snare / hat / open hat / crash / tom, played by the General MIDI drum notes,
            so a song's drum part can play that song's own drum sounds
The chops are worked out once (make()) and kept in the instrument, so rendering is repeatable and they can be
auditioned and re-labelled in the instrument dialog:
  {"type": "chops", "song", "stem", "slice", "from": s | None, "to": s | None, "play": "keys", "base": 36,
   "seed": 1, "mono": true, "fit": "whole", "pitch": false, "release": 0.03, "chops": [[t0, t1, class, peak_db], ...]}
"cue": every note the part uses has its own chop (inst "assign": {pitch: chop index}); hitting that note plays
the stem from that chop's start for as long as the note lasts (on past the chop's end), from the same start every
time - like cue points. Different notes at once play their chops together.
"flow": every note plays the next stretch of the chops laid end to end (silences between them left out), cut
to exactly that note's length - chops made to measure for the part, no stretching, no repeats.
Velocity changes the level only with "velocity": true (the default for "drums" only).
How long a chop lasts ("fit"): "whole" plays it to its end; "cut" stops it when the note ends; "repeat" plays it
again and again until the note ends; "stretch" time-stretches it (pitch kept) to the note's length. Older kits
have "gate": true instead of "fit": "cut".
Times are seconds of the song as recorded (its stem files)."""
import numpy as np

from .config import SR

SLICES = ("hits", "words", "beats", "half", "bars", "phrases")
CLASSES = ("kick", "snare", "hat", "open", "crash", "tom")
DRUM_OF = {35: "kick", 36: "kick", 37: "snare", 38: "snare", 39: "snare", 40: "snare", 41: "tom", 42: "hat",
           43: "tom", 44: "hat", 45: "tom", 46: "open", 47: "tom", 48: "tom", 49: "crash", 50: "tom", 51: "open",
           52: "crash", 53: "open", 54: "hat", 55: "crash", 56: "snare", 57: "crash", 59: "open", 69: "hat", 70: "hat"}
FALLBACK = {"kick": ["tom", "snare"], "snare": ["tom", "kick"], "hat": ["open", "snare"], "open": ["hat", "crash"],
            "crash": ["open", "hat"], "tom": ["snare", "kick"]}
CLASS_NOTE = {"kick": 36, "snare": 38, "hat": 42, "open": 46, "crash": 49, "tom": 45}
HOP = 256


# -------------------------------------------------------------------- analysis
def _mono(y):
    return np.asarray(y, np.float32).mean(axis=0) if np.ndim(y) == 2 else np.asarray(y, np.float32)


def _env_db(m, hop=HOP, win=1024):
    """Frame loudness (dB) every `hop` samples, from a running sum (no big frame matrix)."""
    cs = np.concatenate([[0.0], np.cumsum(m.astype(np.float64) ** 2)])
    n = max(1, (len(m) - win) // hop + 1)
    i = np.arange(n) * hop
    e = (cs[np.minimum(i + win, len(m))] - cs[i]) / win
    return 10 * np.log10(e + 1e-12)


def _flux(m, hop=HOP, nfft=1024):
    """Spectral flux (how much new energy each frame brings), worked out in blocks to keep memory small."""
    from numpy.lib.stride_tricks import sliding_window_view
    if len(m) < nfft:
        return np.zeros(1)
    win = np.hanning(nfft).astype(np.float32)
    frames = sliding_window_view(m, nfft)[::hop]
    out = np.zeros(len(frames), np.float32)
    prev = None
    for s in range(0, len(frames), 2048):
        blk = np.log1p(np.abs(np.fft.rfft(frames[s:s + 2048] * win, axis=1)) * 10).astype(np.float32)
        d = np.diff(blk, axis=0, prepend=blk[:1] if prev is None else prev[None])
        out[s:s + len(blk)] = np.maximum(d, 0).sum(axis=1)
        prev = blk[-1]
    return out


def _onsets(m, act, nfft=1024):
    """Onset times (s). The signal gets nfft of silence in front, so a hit right at the start is found too."""
    f = _flux(np.concatenate([np.zeros(nfft, np.float32), m]), nfft=nfft)
    if len(f) < 3:
        return []
    from scipy.ndimage import maximum_filter1d, median_filter
    med = median_filter(f, size=int(0.4 * SR / HOP) | 1, mode="nearest")
    delta = 0.06 * float(np.percentile(f, 99) or 1.0)
    shift = nfft // HOP
    ok = np.zeros(len(f), bool)                          # loud enough where the frame's window sits
    n2 = max(0, min(len(act), len(f) - shift))
    ok[shift:shift + n2] = act[:n2]
    ok[:shift] = act[0] if len(act) else False
    peak = (f == maximum_filter1d(f, size=7)) & (f > med * 1.4 + delta) & ok
    out, last = [], -1e9
    gap = int(0.06 * SR / HOP)
    for k in np.flatnonzero(peak):
        if k - last >= gap:
            out.append(max(0.0, (k * HOP - nfft + nfft / 2) / SR))   # a frame's time = its window's middle
            last = k
    return out


def _lyrics_words(song):
    import json
    from . import library
    p = library.song_dir(song) / "lyrics.json"
    if not p.exists():
        return None
    out = []
    for seg in json.loads(p.read_text(encoding="utf-8")).get("segments") or []:
        ws = seg.get("words") or []
        for i, w in enumerate(ws):
            t0 = float(w["t"])
            t1 = float(ws[i + 1]["t"]) if i + 1 < len(ws) else float(seg.get("t1") or t0 + 0.6)
            out.append((t0, min(t1, t0 + 1.2), (w.get("w") or "").strip()))
    return out


def _features(m, a, b, db):
    """(log2 brightness, low share, low-mid share, high share, ring seconds) of the first 60 ms of a hit."""
    from scipy.signal.windows import tukey
    s0 = int(a * SR)
    x = m[s0:s0 + int(0.06 * SR)]
    if len(x) < 256:
        return None
    spec = np.abs(np.fft.rfft(x * tukey(len(x), 0.1))) ** 2     # flat-topped: the hit sits at the very start
    fr = np.fft.rfftfreq(len(x), 1 / SR)
    tot = float(spec.sum()) + 1e-12
    cen = float((spec * fr).sum() / tot)
    i0, i1 = int(a * SR / HOP), max(int(a * SR / HOP) + 1, int(b * SR / HOP))
    seg = db[i0:i1]
    pk = int(np.argmax(seg)) if len(seg) else 0
    below = np.flatnonzero(seg[pk:] < seg[pk] - 20) if len(seg) else []
    ring = (below[0] if len(below) else len(seg) - pk) * HOP / SR
    return (np.log2(max(cen, 20.0)), spec[fr < 150].sum() / tot, spec[(fr >= 150) & (fr < 600)].sum() / tot,
            spec[fr >= 5000].sum() / tot, ring)


def _rule(f):
    """One hit on its own: rough thresholds."""
    lc, low, lowmid, high, ring = f
    if low > 0.45 and high < 0.15:
        return "kick"
    if high > 0.4 and low < 0.15:
        return "hat" if ring < 0.12 else "open" if ring < 0.45 else "crash"
    if lowmid > 0.45 and low < 0.35 and high < 0.15:
        return "tom"
    return "snare"


def _sort_hits(feats, peaks):
    """Labels and a 'how good an example' score for a song's hits, relative to each other: three groups by sound
    (brightness, low share, high share), named kick < snare < hat by brightness; hats that ring are open hats /
    crashes (toms aren't told apart from snares - their notes fall back to snare / kick). The score favours hits near their group's middle (not two
    drums at once) and not much quieter than the loudest."""
    n = len(feats)
    F = np.array([[f[0], f[1] * 4, f[3] * 4] for f in feats], float)
    if n < 9:
        return [_rule(f) for f in feats], [0.0] * n
    Z = (F - F.mean(0)) / (F.std(0) + 1e-9)
    order = np.argsort(F[:, 0])
    C = np.array([Z[order[int(n * q)]] for q in (0.15, 0.5, 0.85)])
    for _ in range(20):
        lab = np.argmin(((Z[:, None, :] - C[None]) ** 2).sum(-1), axis=1)
        C = np.array([Z[lab == k].mean(0) if (lab == k).any() else C[k] for k in range(3)])
    rank = np.argsort([F[lab == k, 0].mean() if (lab == k).any() else 99 for k in range(3)])
    name = {int(rank[0]): "kick", int(rank[1]): "snare", int(rank[2]): "hat"}
    for k in range(3):                                  # a middle group that clearly sounds like kicks or hats (a
        if name[k] == "snare" and (lab == k).any():     # kit with no snare) isn't called snare
            mean = tuple(np.mean([feats[i][j] for i in np.flatnonzero(lab == k)]) for j in range(5))
            if mean[1] > 0.65 and mean[3] < 0.05:      # nearly all low end, no top: kicks
                name[k] = "kick"
            elif mean[3] > 0.55 and mean[1] < 0.08:    # nearly all top end: hats
                name[k] = "hat"
    dist = np.sqrt(((Z - C[lab]) ** 2).sum(-1))
    top = max(peaks) if peaks else 0.0
    labels, scores = [], []
    for i, f in enumerate(feats):
        c = name[int(lab[i])]
        if c == "hat" and f[4] >= 0.12:
            c = "open" if f[4] < 0.45 else "crash"
        labels.append(c)
        scores.append(round(float(-dist[i] - max(0.0, top - peaks[i] - 6) / 6), 3))
    return labels, scores


def make(song, stem, slice_="hits", t_from=None, t_to=None, limit=1024):
    """-> {"chops": [[t0, t1, class, peak_db]], "words": [...] (for "words")}: the stem's non-silent pieces."""
    from . import library
    if slice_ not in SLICES:
        raise ValueError(f"slice must be one of {', '.join(SLICES)}")
    info = library.read(song)
    if not info:
        raise ValueError("no such song")
    y = library.read_stem(song, stem, info)
    m = _mono(y)
    dur = len(m) / SR
    a_lim, b_lim = max(0.0, float(t_from or 0)), min(dur, float(t_to) if t_to else dur)
    db = _env_db(m)
    top = float(np.percentile(db, 98))
    thr = max(-55.0, top - 32.0)                       # "active": within 32 dB of the stem's loud parts
    act = db > thr
    fps = SR / HOP
    words = None

    def trim(a, b):
        """Pull the end in to where the sound has died away (24 dB under the chop's peak, for 50 ms or more)."""
        i0, i1 = int(a * fps), max(int(a * fps) + 1, int(b * fps))
        seg = db[i0:i1]
        if not len(seg):
            return b, -120.0
        pk = float(seg.max())
        live = seg > max(thr, pk - 24)
        on = np.flatnonzero(live)
        if not len(on):
            return b, pk
        j, quiet, gap = on[0], 0, int(0.05 * fps)
        last = j
        while j < len(live):
            if live[j]:
                last, quiet = j, 0
            else:
                quiet += 1
                if quiet >= gap:
                    break
            j += 1
        end = a + (last * HOP + 1024) / SR             # a frame looks 1024 samples ahead
        return min(b, max(a + 0.03, end)), pk

    def attack(t):
        """Where a hit really starts: the first sample within 30 ms either side that reaches a tenth of its peak."""
        s0, s1 = max(0, int((t - 0.03) * SR)), min(len(m), int((t + 0.06) * SR))
        env = np.abs(m[s0:s1])
        if not len(env) or env.max() <= 0:
            return t
        k = int(np.argmax(env > 0.1 * env.max()))
        return max(0.0, (s0 + k) / SR - 0.001)

    raw = []
    if slice_ == "hits":
        ons = _onsets(m, act)
        maxlen = 0.8 if "drum" in str(stem).lower() or stem in (info.get("drum_stems") or []) else 1.5
        for i, t in enumerate(ons):
            nxt = ons[i + 1] if i + 1 < len(ons) else dur
            a = attack(t)
            raw.append((a, min(nxt - 0.002, a + maxlen)))
    elif slice_ == "words":
        words = _lyrics_words(song)
        if not words:
            raise ValueError("this song has no word timing yet - press 'Find its words now', or try hits or phrases")
        raw = [(max(0.0, t0 - 0.02), t1) for t0, t1, _ in words]
    elif slice_ in ("beats", "half", "bars"):
        beats = np.asarray(info.get("beats") or [], float)
        if slice_ == "bars":
            idx = [b for b in (info.get("bars") or []) if b < len(beats)]
            pts = beats[idx] if idx else beats[::int(info.get("bpb") or 4)]
        elif slice_ == "half":
            pts = np.sort(np.concatenate([beats, (beats[:-1] + beats[1:]) / 2]))
        else:
            pts = beats
        raw = [(float(pts[i]), float(pts[i + 1])) for i in range(len(pts) - 1)]
    else:                                              # phrases: runs of sound with gaps under 0.3 s merged
        runs, start, quiet = [], None, 0
        for i, on in enumerate(act):
            if on:
                if start is None:
                    start = i
                quiet = 0
            elif start is not None:
                quiet += 1
                if quiet > 0.3 * fps:
                    runs.append((start, i - quiet + 1))
                    start, quiet = None, 0
        if start is not None:
            runs.append((start, len(act)))
        for s, e in runs:
            a, b = s / fps, e / fps
            while b - a > 8.0:
                raw.append((a, a + 8.0))
                a += 8.0
            if b - a >= 0.25:
                raw.append((a, b))
    out = []
    for i, (a, b) in enumerate(raw):
        if b <= a_lim or a >= b_lim or b - a < 0.02:
            continue
        a, b = max(a, a_lim), min(b, b_lim)
        i0, i1 = int(a * fps), max(int(a * fps) + 1, int(b * fps))
        if slice_ in ("beats", "half", "bars", "phrases"):
            if act[i0:i1].mean() < (0.3 if slice_ != "phrases" else 0.0):
                continue
            pk = float(db[i0:i1].max())
        else:
            b, pk = trim(a, b)
            if pk < thr + 3:
                continue
        if slice_ in ("words", "phrases"):           # word times can be early: start where the sound does
            x = np.abs(m[int(a * SR):int(b * SR)])
            if len(x) and x.max() > 0:
                pk_abs = x.max()
                last = len(x) - 1 - int(np.argmax(x[::-1] > 0.03 * pk_abs))      # and end where it does
                b0 = a
                a = min(b - 0.03, a + max(0.0, int(np.argmax(x > 0.1 * pk_abs)) / SR - 0.004))
                b = max(a + 0.03, min(b, b0 + last / SR + 0.01))
        row = [round(a, 4), round(b, 4), "", round(pk, 1)]
        if words is not None:
            row.append(words[i][2])
        out.append(row)
        if len(out) >= limit:
            break
    feats = [_features(m, r[0], r[1], db) for r in out]
    ok = [i for i, f in enumerate(feats) if f is not None]
    if slice_ == "hits":
        labels, scores = _sort_hits([feats[i] for i in ok], [out[i][3] for i in ok])
        for i, c, sc in zip(ok, labels, scores):
            out[i][2] = c
            out[i].append(sc)                          # [t0, t1, class, peak_db, score]
    else:
        for i in ok:
            out[i][2] = _rule(feats[i])
    for r in out:
        r[2] = r[2] or "snare"
    return {"chops": out, "seconds": round(dur, 2)}


# -------------------------------------------------------------------- which chop a note plays
def _drum_lists(chops):
    by = {c: [] for c in CLASSES}
    for i, ch in enumerate(chops):
        by.setdefault(ch[2], []).append(i)
    def score(i):                                       # the best examples first (make()'s score, else loudness)
        ch = chops[i]
        return float(ch[4]) if len(ch) > 4 and isinstance(ch[4], (int, float)) else float(ch[3]) / 10
    for c in by:                                        # 4 of each, taken in turn
        by[c] = sorted(by[c], key=lambda i: -score(i))[:4]
    return by


def index_for(inst, pitch, state):
    """The chop a note-on plays (None = nothing). `state` carries the counters (in order, shuffled, round robin)."""
    chops = inst.get("chops") or []
    if not chops:
        return None
    mode = inst.get("play") or "keys"
    if mode == "keys":
        i = int(pitch) - int(inst.get("base", 36))
        return i if 0 <= i < len(chops) else None
    if mode == "cue":
        i = (inst.get("assign") or {}).get(str(int(pitch)))
        if i is None:                                  # a note added since: still something, the same every time
            i = (int(pitch) - int(inst.get("base", 36))) % len(chops)
        return int(i) if 0 <= int(i) < len(chops) else None
    if mode in ("order", "shuffle"):
        k = state.get("n", 0)
        state["n"] = k + 1
        seq = inst.get("sequence") if mode == "order" else None
        if seq:                                          # "make it say": the words in the order typed
            if inst.get("seq_once") and k >= len(seq):
                return None
            i = int(seq[k % len(seq)])
            return i if 0 <= i < len(chops) else None
        if mode == "order":
            return k % len(chops)
        if "perm" not in state or len(state["perm"]) != len(chops):
            state["perm"] = np.random.default_rng(int(inst.get("seed", 1) or 1)).permutation(len(chops)).tolist()
        return state["perm"][k % len(chops)]
    by = state.get("by") or state.setdefault("by", _drum_lists(chops))
    cls = DRUM_OF.get(int(pitch))
    if cls is None:
        return None
    for c in [cls] + FALLBACK.get(cls, []):
        if by.get(c):
            rr = state.setdefault("rr", {})
            j = rr.get(c, 0)
            rr[c] = j + 1
            return by[c][j % len(by[c])]
    return None


def _norm(w):
    import re
    return re.sub(r"[^a-z0-9]", "", str(w).lower().replace("'", "").replace("’", ""))


def say(chops, text):
    """A sentence -> the word chops that say it: {"sequence": [chop index], "found": [[word, sung as]], "missing":
    [words]}. Exact words first (the loudest take, then the next take of a word used again); then a near miss
    (one letter off, or "runnin" for "running"); then two sung words that make it ("nowhere" = "no" + "where")."""
    import difflib
    pool = {}
    for i, c in enumerate(chops):
        if len(c) > 4 and isinstance(c[4], str) and _norm(c[4]):
            pool.setdefault(_norm(c[4]), []).append(i)
    for k in pool:
        pool[k].sort(key=lambda i: -float(chops[i][3]))
    used = {}

    def take(k):
        n = used.get(k, 0)
        used[k] = n + 1
        return pool[k][n % len(pool[k])]

    def near(w):
        if w.endswith("ing") and w[:-1] in pool:
            return w[:-1]
        if w + "g" in pool or (w.endswith("in") and w + "g" in pool):
            return w + "g"
        got = difflib.get_close_matches(w, list(pool), n=1, cutoff=0.8 if len(w) >= 4 else 0.99)
        return got[0] if got else None

    seq, found, missing = [], [], []
    for raw in str(text).split():
        w = _norm(raw)
        if not w:
            continue
        if w in pool:
            seq.append(take(w)); found.append([raw, chops[seq[-1]][4]])
            continue
        k = near(w)
        if k:
            seq.append(take(k)); found.append([raw, chops[seq[-1]][4]])
            continue
        split = next(((w[:j], w[j:]) for j in range(2, len(w) - 1) if w[:j] in pool and w[j:] in pool), None)
        if split:
            for part in split:
                seq.append(take(part))
            found.append([raw, " + ".join(chops[i][4] for i in seq[-2:])])
            continue
        missing.append(raw)
    return {"sequence": seq, "found": found, "missing": missing, "words": len(pool)}


def ratio_for(inst, pitch):
    if inst.get("pitch") is True and (inst.get("play") in ("order", "shuffle")):
        return 2.0 ** ((int(pitch) - int(inst.get("base", 60) or 60)) / 12.0)
    return 1.0


# -------------------------------------------------------------------- tuning chops to the notes
_f0 = {}


def f0_track(inst):
    """(f0 Hz per frame, loudness dB per frame, frame hop) for the whole stem - one YIN pass, cached."""
    from . import library
    f = library.song_dir(inst["song"]) / "stems" / f"{inst['stem']}.flac"
    key = (inst["song"], inst["stem"], f.stat().st_mtime_ns if f.exists() else 0)
    if key not in _f0:
        import librosa
        if len(_f0) > 6:
            _f0.clear()
        m = _mono(source(inst))
        hop = 512
        f0 = librosa.yin(m, fmin=65.0, fmax=1050.0, sr=SR, frame_length=2048, hop_length=hop)
        db = _env_db(m, hop=hop, win=2048)
        n = min(len(f0), len(db))
        _f0[key] = (f0[:n], db[:n], hop)
    return _f0[key]


def chop_pitches(inst):
    """Each chop's pitch as a MIDI note number (fractional), or None where it has no clear pitch (drums, breaths,
    noise): the median of its loud frames' pitch, if those agree within a few semitones."""
    f0, db, hop = f0_track(inst)
    thr = max(-50.0, float(np.percentile(db, 98)) - 30.0)
    out = []
    for c in inst.get("chops") or []:
        i0, i1 = int(float(c[0]) * SR / hop), max(int(float(c[0]) * SR / hop) + 1, int(float(c[1]) * SR / hop))
        fr, lv = f0[i0:i1], db[i0:i1]
        ok = (lv > thr) & (fr > 66) & (fr < 1040)
        if ok.sum() < 3:
            out.append(None)
            continue
        midi = 69 + 12 * np.log2(fr[ok] / 440.0)
        q1, q3 = np.percentile(midi, [25, 75])
        out.append(round(float(np.median(midi)), 2) if q3 - q1 < 4 else None)
    return out


def tune_semis(inst, i, target):
    """How far chop i moves to land on MIDI note `target` (0 if it has no clear pitch). "tune_octave" (default on)
    keeps it in the singer's own octave: never more than 6 semitones either way."""
    ps = inst.get("pitches") or []
    p = ps[i] if 0 <= i < len(ps) else None
    if p is None:
        return 0.0
    semis = float(target) - float(p)
    if inst.get("tune_octave", True):
        semis = ((semis + 6) % 12) - 6
    return semis


def shift(seg, semis, sr=SR):
    """Pitch-shift keeping the length and the voice's character (Rubber Band, formants kept)."""
    if abs(semis) < 0.03 or seg.shape[1] < 256:
        return seg
    import pedalboard
    y = pedalboard.time_stretch(np.ascontiguousarray(seg, np.float32), float(sr), stretch_factor=1.0,
                                pitch_shift_in_semitones=float(semis), preserve_formants=True, high_quality=True)
    y = np.asarray(y, np.float32)
    n = seg.shape[1]
    return y[:, :n] if y.shape[1] >= n else np.pad(y, ((0, 0), (0, n - y.shape[1])))


def tuned(inst):
    return inst.get("pitch") == "tune" and (inst.get("play") or "keys") not in ("drums", "flow")


# -------------------------------------------------------------------- sound
_src = {}


def source(inst):
    from . import library
    f = library.song_dir(inst["song"]) / "stems" / f"{inst['stem']}.flac"
    key = (inst.get("song"), inst.get("stem"), f.stat().st_mtime_ns if f.exists() else 0)   # a re-rendered part
    if key not in _src:
        if len(_src) > 6:
            _src.clear()
        _src[key] = np.asarray(library.read_stem(inst["song"], inst["stem"]), np.float32)
    return _src[key]


def chop_buffer(inst, idx, ratio=1.0, sr=SR, fades=True):
    """One chop as (2, n) float32, with short fades so it never clicks; varispeed when ratio != 1."""
    y = source(inst)
    a, b = float(inst["chops"][idx][0]), float(inst["chops"][idx][1])
    seg = y[:, int(a * sr):max(int(a * sr) + 32, int(b * sr))]
    if abs(ratio - 1.0) > 1e-4 and seg.shape[1] > 4:
        L = seg.shape[1]
        pos = np.arange(int(L / ratio)) * ratio
        i = np.minimum(pos.astype(np.int64), L - 2)
        fr = (pos - i).astype(np.float32)
        seg = seg[:, i] * (1 - fr) + seg[:, i + 1] * fr
    seg = np.array(seg, np.float32)
    if not fades:
        return seg
    n = seg.shape[1]
    fi, fo = min(n // 4, int(0.002 * sr)), min(n // 3, int(0.012 * sr))
    if fi > 0:
        seg[:, :fi] *= np.linspace(0, 1, fi, dtype=np.float32)
    if fo > 0:
        seg[:, n - fo:] *= np.linspace(1, 0, fo, dtype=np.float32)
    return seg


def fit_of(inst):
    return inst.get("fit") or ("cut" if inst.get("gate") else "whole")


XF = 0.006                                              # seconds of crossfade where one piece meets the next


def _fade_in(y, sec, sr=SR):
    f = min(y.shape[1] // 3, int(sec * sr))
    if f > 0:
        y[:, :f] *= np.linspace(0, 1, f, dtype=np.float32)
    return y


def _join(a, b, sr=SR):
    """a then b, overlapped by a short crossfade (no dip, no click)."""
    f = min(int(XF * sr), a.shape[1] // 2, b.shape[1] // 2)
    if f <= 0:
        return np.concatenate([a, b], axis=1)
    ramp = np.linspace(0, 1, f, dtype=np.float32)
    mid = a[:, -f:] * (1 - ramp) + b[:, :f] * ramp
    return np.concatenate([a[:, :-f], mid, b[:, f:]], axis=1)


def loop_cycle(seg, sr=SR):
    """(first, cycle): play `first` once, then `cycle` over and over - each round crossfaded into the next, so a
    looping chop has no dip and no click at the seam."""
    n = seg.shape[1]
    f = min(int(XF * sr), n // 2)
    if f <= 0:
        return seg, seg
    ramp = np.linspace(0, 1, f, dtype=np.float32)
    cycle = seg[:, :n - f].copy()
    cycle[:, :f] = seg[:, :f] * ramp + seg[:, n - f:] * (1 - ramp)
    return seg[:, :n - f], cycle


def repeat_to(seg, L, sr=SR):
    """The (unfaded) chop played over and over to fill L samples, each repeat crossfaded into the next."""
    if seg.shape[1] >= L:
        return _fade_in(seg[:, :L].copy(), 0.002, sr)
    first, cycle = loop_cycle(seg, sr)
    reps = -(-(L - first.shape[1]) // max(1, cycle.shape[1])) + 1
    out = np.concatenate([first, np.tile(cycle, (1, reps))], axis=1)[:, :L]
    return _fade_in(out.copy(), 0.002, sr)


def stretch_to(seg, L, sr=SR):
    """The chop time-stretched to exactly L samples, pitch kept (Rubber Band, through Pedalboard)."""
    full = seg.shape[1]
    if full < 64 or abs(L - full) < 0.01 * full:
        return repeat_to(seg, L, sr) if L > full else seg[:, :L].copy()
    factor = float(np.clip(full / L, 0.02, 50.0))       # pedalboard: > 1 shortens, < 1 lengthens
    try:
        import pedalboard
        y = pedalboard.time_stretch(np.ascontiguousarray(seg, np.float32), float(sr), stretch_factor=factor,
                                    high_quality=True, transient_mode="mixed")
    except Exception as e:                              # no stretcher: varispeed (pitch moves) - and it says so
        from . import components
        components.fallback("pedalboard", "chops stretched to the note are sped up / slowed down instead, so their "
                                          "pitch moves", problem=f"its time-stretch failed ({type(e).__name__}: {e})")
        pos = np.linspace(0, full - 1, L)
        i = np.minimum(pos.astype(np.int64), full - 2)
        fr = (pos - i).astype(np.float32)
        y = seg[:, i] * (1 - fr) + seg[:, i + 1] * fr
    y = np.asarray(y, np.float32)
    if y.shape[1] < L:
        y = np.pad(y, ((0, 0), (0, L - y.shape[1])))
    y = y[:, :L].copy()
    f = min(L // 3, int(0.004 * sr))
    if f > 0:
        y[:, :f] *= np.linspace(0, 1, f, dtype=np.float32)
    return y


def cue_take(inst, i, L, sr=SR):
    """L samples of the stem from chop i's start on (past the chop's end). With "skip_gaps" (the default) the stem's
    pauses are left out, so a held note keeps sounding; otherwise it plays the stem as it is, pauses and all."""
    y = source(inst)
    a = int(float(inst["chops"][i][0]) * sr)
    if inst.get("skip_gaps", True):
        sp = active_spans(inst, sr)
        if sp:
            pos, acc = None, 0
            for s0, s1 in sp:                           # where the cue point sits in the stream of sound
                if a < s1:
                    pos = acc + max(0, a - s0)
                    break
                acc += s1 - s0
            if pos is not None:
                return _read_spans(y, sp, pos, L, sr)
    seg = np.array(y[:, a:a + L], np.float32)
    if seg.shape[1] < L:
        seg = np.pad(seg, ((0, 0), (0, L - seg.shape[1])))
    return _fade_in(seg, 0.002, sr)


def assign_default(chops, pitches, how="low", seed=1):
    """{pitch: chop index} for the pitches a part uses: "low" = lowest note gets the first chop, "first" = in the
    order the notes are first played (pitches given in that order), "shuffle" = random chops (seeded)."""
    ps = list(dict.fromkeys(int(p) for p in pitches))
    if how == "low":
        ps = sorted(ps)
    n = max(1, len(chops))
    if how == "shuffle":
        idx = np.random.default_rng(int(seed or 1)).permutation(n).tolist()
        return {str(p): int(idx[k % n]) for k, p in enumerate(ps)}
    return {str(p): k % n for k, p in enumerate(ps)}


def _spans(inst, sr=SR):
    """The chops laid end to end for "flow": [(start sample, end sample)] in time order, overlaps merged."""
    out = []
    for c in sorted(inst.get("chops") or [], key=lambda c: c[0]):
        a, b = int(float(c[0]) * sr), int(float(c[1]) * sr)
        if b - a < 64:
            continue
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def _read_spans(y, sp, cur, L, sr=SR):
    """L samples of the spans [(a, b)] of y laid end to end, from position `cur` into that stream (wraps round),
    each join crossfaded."""
    total = sum(b - a for a, b in sp)
    if total <= 0:
        return np.zeros((2, L), np.float32)
    cur %= total
    k, acc = 0, 0
    while acc + (sp[k][1] - sp[k][0]) <= cur:
        acc += sp[k][1] - sp[k][0]
        k += 1
    off = cur - acc
    out = None
    while out is None or out.shape[1] < L:             # every piece adds at least half its length
        a, b = sp[k % len(sp)]
        have = 0 if out is None else out.shape[1]
        piece = np.array(y[:, a + off:min(b, a + off + (L - have) + 2 * int(XF * sr))], np.float32)
        out = piece if out is None else _join(out, piece, sr)
        off = 0
        k += 1
    if out.shape[1] < L:
        out = np.pad(out, ((0, 0), (0, L - out.shape[1])))
    return _fade_in(out[:, :L].copy(), 0.002, sr)


def stream_take(inst, cur, L, sr=SR):
    """L samples of the chops laid end to end, from position `cur` (samples into that stream; wraps round)."""
    sp = _spans(inst, sr)
    if not sp:
        return np.zeros((2, L), np.float32)
    return _read_spans(source(inst), sp, cur, L, sr)


_active = {}


def active_spans(inst, sr=SR, gap=0.08):
    """Where the stem has sound: [(start, end)] samples, pauses of `gap` s or more left out (10 ms kept each side)."""
    from . import library
    f = library.song_dir(inst["song"]) / "stems" / f"{inst['stem']}.flac"
    key = (inst["song"], inst["stem"], f.stat().st_mtime_ns if f.exists() else 0)
    if key not in _active:
        if len(_active) > 8:
            _active.clear()
        m = _mono(source(inst))
        db = _env_db(m)
        thr = max(-55.0, float(np.percentile(db, 98)) - 32.0)
        act = db > thr
        runs, start, quiet, g = [], None, 0, max(1, int(gap * sr / HOP))
        for i, on in enumerate(act):
            if on:
                if start is None:
                    start = i
                quiet = 0
            elif start is not None:
                quiet += 1
                if quiet >= g:
                    runs.append((start, i - quiet + 1))
                    start, quiet = None, 0
        if start is not None:
            runs.append((start, len(act)))
        pad = int(0.01 * sr)
        sp = []
        for a, b in runs:       # frame j covers [j*HOP, j*HOP+1024]: sound starts ~a*HOP+768, ends ~b*HOP
            a, b = max(0, a * HOP + 768 - pad), min(len(m), b * HOP + pad)
            if sp and a <= sp[-1][1]:
                sp[-1] = (sp[-1][0], max(sp[-1][1], b))
            else:
                sp.append((a, b))
        _active[key] = sp
    return _active[key]


def render(inst, notes, n, sr=SR):
    """notes [(t_on, t_off, pitch, vel)] in output seconds -> (2, n)."""
    out = np.zeros((2, n), np.float32)
    if not inst.get("chops"):
        raise RuntimeError("the chop kit has no chops yet: pick a song and stem and press Cut it")
    mode = inst.get("play") or "keys"
    if tuned(inst) and len(inst.get("pitches") or []) != len(inst["chops"]):
        inst = dict(inst, pitches=chop_pitches(inst))   # not measured in the dialog: measure now
    mono = bool(inst.get("mono", mode not in ("drums", "cue")))
    fit = mode if mode in ("flow", "cue") else fit_of(inst)
    rel = max(1, int(float(inst.get("release", 0.03) or 0.03) * sr))
    vel = bool(inst.get("velocity", mode == "drums"))
    xf = int(XF * sr)
    state = {}
    voices = []
    notes = sorted(notes, key=lambda e: (e[0], e[2]))
    k = 0
    while k < len(notes):                              # notes starting together (a chord) are one trigger in
        grp = [notes[k]]                               # "order"/"shuffle", so a chord doesn't eat several chops
        while k + len(grp) < len(notes) and notes[k + len(grp)][0] - notes[k][0] < 0.005:
            grp.append(notes[k + len(grp)])
        k += len(grp)
        if mode == "flow":
            voices.append((max(grp, key=lambda e: e[1] - e[0]), -1))
        elif mode in ("order", "shuffle"):
            top = max(grp, key=lambda e: e[3])
            i = index_for(inst, top[2], state)
            voices.append((top, i))
        else:
            for e in grp:
                voices.append((e, index_for(inst, e[2], state)))
    voices = [(e, i) for e, i in voices if i is not None]
    starts = [int(round(e[0] * sr)) for e, _ in voices]
    cur = 0
    for j, (e, i) in enumerate(voices):
        s0 = starts[j]
        if s0 >= n:
            continue
        held = max(1, int(round((e[1] - e[0]) * sr)))
        nxt = next((s for s in starts[j + 1:] if s > s0), None)
        end = s0 + held                                 # where the note wants it to stop
        if mono and nxt is not None:
            end = min(end, nxt) if fit != "whole" else nxt
        joined = nxt is not None and fit != "whole" and nxt - end < xf     # the next one starts as this ends
        want = end - s0 + (xf if joined else 0)
        if fit == "flow":
            seg = stream_take(inst, cur, want, sr)
            cur += end - s0
        elif fit == "cue":
            seg = cue_take(inst, i, want, sr)
            if tuned(inst):
                seg = shift(seg, tune_semis(inst, i, e[2]), sr)
        else:
            raw = chop_buffer(inst, i, ratio_for(inst, e[2]), sr, fades=(fit in ("whole", "cut")))
            if tuned(inst):
                raw = shift(raw, tune_semis(inst, i, e[2]), sr)
            if fit == "repeat":
                seg = repeat_to(raw, want, sr)
            elif fit == "stretch":
                seg = _fade_in(stretch_to(raw, want, sr), 0.002, sr)
            else:
                seg = raw
        full = seg.shape[1]
        if fit == "whole":
            L = min(full, (end - s0) if (mono and nxt is not None) else full)
        elif fit == "cut":
            L = min(full, want if joined else end - s0 + rel)
        else:
            L = min(full, want)
        L = min(L, n - s0)
        if L <= 0:
            continue
        seg = seg[:, :L].copy()
        if joined:
            f = min(L, xf)                              # crossfades with the next note's start
        elif L < full or fit in ("repeat", "stretch", "flow", "cue"):
            f = min(L, rel if fit != "whole" else int(0.006 * sr))
        else:
            f = 0
        if f > 0:
            seg[:, L - f:] *= np.linspace(1, 0, f, dtype=np.float32)
        out[:, s0:s0 + L] += seg * ((e[3] / 127.0) if vel else 1.0)
    return out
