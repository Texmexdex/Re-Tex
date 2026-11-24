"""MIDI as instructions for a project's lanes (besides glitch scores, glitch.py):

Arrangement maps - each MIDI part is tied to a lane; wherever the part is silent for a bar, the lane drops out. A
famous song's arrangement (when its bass, drums, lead come in and out) shapes a mashup of other songs.

Chord following - the chords of a MIDI song (its pitched parts) are read bar by bar; the chosen lanes are cut at the
bar lines and each piece is pitch-shifted (the clip's own pitch, rendered in HQ) so the stem's own chord root lands
on the MIDI's. Song B's guitar then plays song A's progression. The stem's root per bar comes from its audio (chroma);
"key" mode uses the song's key instead (steadier for riffs that sit on one note).

Both read MIDI beats as project beats (bar 1 of the MIDI range lands on the chosen project bar) and give back the
project's whole new clip list, which the UI applies as one undo step.
"""
import math

import numpy as np

from .config import SR
from .glitch import _cut, _uid, notes_in_project

NOTE = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def _bars(start, length, bpb):
    return [(start + k * bpb, start + (k + 1) * bpb) for k in range(int(round(length / bpb)))]


# -------------------------------------------------------------------- arrangement maps
def arrangement(project, mapping, notes_by_part, start, length, min_notes=1):
    """mapping {part id: track id}; notes_by_part {part id: [(s, e, p, v)] project beats}. -> (clips, report)."""
    bpb = int(project.get("bpb") or 4)
    clips = list(project.get("clips", []))
    report = {}
    for pid, tid in mapping.items():
        if not tid:
            continue
        ns = notes_by_part.get(pid) or []
        silent = []
        for a, b in _bars(start, length, bpb):
            n = sum(1 for s, e, _p, _v in ns if s < b - 1e-9 and e > a + 1e-9)
            if n < min_notes:
                if silent and abs(silent[-1][1] - a) < 1e-9:
                    silent[-1] = (silent[-1][0], b)
                else:
                    silent.append((a, b))
        lane = [c for c in clips if c.get("track") == tid]
        cur = list(lane)
        for a, b in silent:
            nxt = []
            for c in cur:
                if c.get("warps") or c["start"] + c["len"] <= a + 1e-9 or c["start"] >= b - 1e-9:
                    nxt.append(c)
                    continue
                left, right = _cut(project, c, a, b)
                if right is not None:
                    right["fade_in"] = 0.03                 # comes back in smoothly
                nxt += [x for x in (left, right) if x]
            cur = nxt
        clips = [c for c in clips if c.get("track") != tid] + cur
        report[tid] = round(sum(b - a for a, b in silent) / bpb)
    return clips, report


# -------------------------------------------------------------------- chords
_MAJ, _MIN = np.zeros(12), np.zeros(12)
_MAJ[[0, 4, 7]] = [1.0, 0.8, 0.9]
_MIN[[0, 3, 7]] = [1.0, 0.8, 0.9]
TEMPLATES = [(r, "maj", np.roll(_MAJ, r)) for r in range(12)] + [(r, "min", np.roll(_MIN, r)) for r in range(12)]


def chord_of(chroma, bass=None):
    """(root pitch class, "maj"/"min", score) of a 12-bin chroma (the bass note, if known, counts extra)."""
    c = np.asarray(chroma, float)
    if bass is not None:
        c = c + 0.5 * np.asarray(bass, float)
    if c.sum() <= 0:
        return None
    c = c / (np.linalg.norm(c) + 1e-9)
    best = max(TEMPLATES, key=lambda t: float(np.dot(c, t[2] / np.linalg.norm(t[2]))))
    return best[0], best[1], float(np.dot(c, best[2] / np.linalg.norm(best[2])))


def midi_chords(notes, bars):
    """[(root, quality) or None] per bar, from note durations (and the lowest note sounding, as the bass)."""
    out = []
    for a, b in bars:
        ch, bass = np.zeros(12), np.zeros(12)
        inside = [(max(s, a), min(e, b), p) for s, e, p, _v in notes if s < b and e > a]
        for s, e, p in inside:
            ch[p % 12] += e - s
        steps = np.arange(a, b, 0.5)
        for t in steps:                                  # the bass: the lowest note sounding each 1/8
            low = [p for s, e, p in inside if s <= t < e]
            if low:
                bass[min(low) % 12] += 0.5
        got = chord_of(ch, bass)
        out.append(None if got is None else got[:2])
    return out


def _clip_source_span(project, c, a, b, info):
    """Seconds of the clip's song (as recorded) that the clip plays between project beats a and b."""
    from . import project as P, warps, midi
    mult = P.song_mult(project, c["song"])
    ps = warps.pieces(c, mult, None)
    beats = np.asarray(info["beats"], float)

    def sec(t):
        u = min(max(float(t) - float(c["start"]), 0.0), float(c["len"]))
        w = float(c["src"])
        for u0, u1, w0, w1, _r, _k in ps:
            if u0 <= u <= u1:
                w = w0 + (w1 - w0) * (u - u0) / max(u1 - u0, 1e-9)
                break
        else:
            if ps:
                w = ps[-1][3]
        return float(midi.beat_time(beats, w))
    return sec(a), sec(b)


def stem_root(c, project, a, b, cache):
    """The chord root of what clip c plays in project beats [a, b) (None if it's unclear)."""
    from . import library
    import librosa
    info = cache.setdefault(("info", c["song"]), library.read(c["song"]))
    key = ("y", c["song"], c["stem"])
    if key not in cache:
        cache[key] = np.asarray(library.read_stem(c["song"], c["stem"], info), np.float32).mean(axis=0)
    y = cache[key]
    s0, s1 = _clip_source_span(project, c, max(a, c["start"]), min(b, c["start"] + c["len"]), info)
    x = y[int(s0 * SR):int(s1 * SR)]
    if len(x) < SR // 8 or float(np.max(np.abs(x))) < 1e-3:
        return None
    chroma = librosa.feature.chroma_stft(y=x, sr=SR, n_fft=8192, hop_length=4096).mean(axis=1)
    got = chord_of(chroma)
    return None if got is None or got[2] < 0.55 else got[0]


def split_at(project, c, t):
    """Clip c cut at project beat t into two butt-joined parts (no fades at the cut)."""
    left, right = _cut(project, c, t, t)
    if left is None or right is None:
        return [c]
    left["fade_out"] = 0
    right["fade_in"] = 0
    return [left, right]


def follow(project, tracks, chords, bars, mode="audio", max_shift=6):
    """Cut each lane at the bar lines and shift each bar's piece so its root lands on that bar's MIDI chord root.
    -> (clips, {track: [shift per bar]})."""
    from . import library
    clips = list(project.get("clips", []))
    cache, report = {}, {}
    edges = sorted({a for a, _ in bars} | {b for _, b in bars})
    for tid in tracks:
        pieces = [c for c in clips if c.get("track") == tid]
        for t in edges:                                # every clip cut at every bar line
            nxt = []
            for c in pieces:
                if not c.get("warps") and c["start"] < t - 1e-6 < c["start"] + c["len"] - 2e-6:
                    nxt += split_at(project, c, t)
                else:
                    nxt.append(c)
            pieces = nxt
        shifts, prev = [], 0.0
        for (a, b), ch in zip(bars, chords):
            moved = None
            for k, c in enumerate(pieces):
                if ch is None or c.get("warps") or c["start"] < a - 1e-6 or c["start"] >= b - 1e-6:
                    continue
                if mode == "key":
                    kk = (cache.setdefault(("info", c["song"]), library.read(c["song"])) or {}).get("key") or {}
                    root = kk.get("tonic") if isinstance(kk, dict) else None
                else:
                    root = stem_root(c, project, a, b, cache)
                if root is None:
                    continue
                d0 = (ch[0] - root) % 12
                cands = [x for x in (d0, d0 - 12) if abs(x) <= max_shift] or [d0 - 12 if d0 > 6 else d0]
                d = min(cands, key=lambda x: (abs(x - prev), abs(x)))    # the smallest jump from the last bar
                pieces[k] = {**c, "pitch": round(float(c.get("pitch", 0) or 0) + d, 3)}
                moved = d
            if moved is not None:
                prev = moved
            shifts.append(moved)
        clips = [c for c in clips if c.get("track") != tid] + pieces
        report[tid] = shifts
    return clips, report


# -------------------------------------------------------------------- groove transplant
def groove_template(notes, first_beat, bpb):
    """How far a (drum) part's notes sit off the grid at each step of the bar, on average (beats), how loud each step
    is relative to the part's average, and the grid (steps per beat): 1/8 for parts with about two notes a beat or
    fewer (so swung 8ths read as late 8ths, not early 16ths), else 1/16. -> (offsets, levels, steps per beat)."""
    onsets = sorted({round(n[0], 4) for n in notes})
    span = (onsets[-1] - onsets[0]) if len(onsets) > 1 else 1.0
    res = 2 if len(onsets) / max(span, 1.0) <= 2.2 else 4
    steps = bpb * res
    off, vel = [[] for _ in range(steps)], [[] for _ in range(steps)]
    for s0, _l, _p, v in notes:
        x = (s0 - first_beat) * res
        g = round(x)
        if abs(x - g) > 0.45:                          # between two steps: says nothing about the groove
            continue
        st = int(g) % steps
        off[st].append(s0 - first_beat - g / res)
        vel[st].append(v)
    vall = np.mean([v for vs in vel for v in vs]) if any(vel) else 100.0
    return ([round(float(np.mean(o)), 4) if o else 0.0 for o in off],
            [round(float(np.mean(v) / vall), 3) if v else 1.0 for v in vel], res)


_ons = {}


def _onsets_of(song, stem):
    from . import chops, library
    key = (song, stem)
    if key not in _ons:
        if len(_ons) > 8:
            _ons.clear()
        m = chops._mono(library.read_stem(song, stem))
        db = chops._env_db(m)
        act = db > max(-55.0, float(np.percentile(db, 98)) - 32.0)
        _ons[key] = np.array([chops._onsets(m, act)]).ravel()
    return _ons[key]


def groove_lane(project, tid, offsets, levels, amount, start, length, use_levels=False, res=4):
    """The lane's hits moved onto the groove: each clip is sliced at its hits; a slice whose hit sits near a 1/16
    step starts that step's offset later / earlier. Nothing is stretched: a slice moved later leaves a sliver of
    silence before it, one moved earlier is cut a little short. -> the lane's new clips."""
    from . import library, midi, project as P
    bpb = int(project.get("bpb") or 4)
    steps = len(offsets)
    out = []
    tol = 0.45
    for c in [c for c in project.get("clips", []) if c.get("track") == tid]:
        if c.get("warps") or c["start"] + c["len"] <= start or c["start"] >= start + length:
            out.append(c)
            continue
        info = library.read(c["song"])
        beats = np.asarray(info["beats"], float)
        mult, rate = P.song_mult(project, c["song"]), float(c.get("rate", 1) or 1)
        k = rate / mult
        w0 = float(c["src"])
        t_of = lambda sec: c["start"] + (float(midi.time_beat(beats, sec)) - w0) / k
        w_of = lambda t: w0 + (t - c["start"]) * k
        s_a, s_b = float(midi.beat_time(beats, w_of(c["start"]))), float(midi.beat_time(beats, w_of(c["start"] + c["len"])))
        hits = [t_of(x) for x in _onsets_of(c["song"], c["stem"]) if s_a + 0.02 < x < s_b - 0.02]
        pts = [c["start"]] + [h for h in hits if start <= h < start + length] + [c["start"] + c["len"]]
        moves = [0.0]
        gains = [1.0]
        for h in pts[1:-1]:
            g = round(h * res)
            if abs(h * res - g) > tol:
                moves.append(0.0); gains.append(1.0)
                continue
            st = int(g) % steps
            moves.append((g / res + offsets[st] - h) * amount)
            gains.append(1 + (levels[st] - 1) * amount if use_levels else 1.0)
        moves.append(0.0); gains.append(1.0)
        if all(abs(x) < 1e-4 for x in moves) and all(g == 1.0 for g in gains):
            out.append(c)                              # nothing to move: leave the clip whole
            continue
        for i in range(len(pts) - 1):
            a, b = pts[i], pts[i + 1]
            if b - a < 1e-3:
                continue
            na, nb = a + moves[i], b + moves[i + 1]
            ln = min(b - a, nb - na)
            if ln < 1e-3:
                continue
            nc = {**c, "id": c["id"] if i == 0 else _uid(), "start": round(na, 6), "len": round(ln, 6),
                  "src": round(float(w_of(a)), 6), "fade_in": 0 if i == 0 else 0.0, "fade_out": 0.01 if ln < b - a - 1e-4 else 0}
            if gains[i] != 1.0:
                nc["gain_db"] = round(float(c.get("gain_db", 0) or 0) + 20 * math.log10(max(0.1, gains[i])), 2)
            if i == len(pts) - 2:
                nc["fade_out"] = c.get("fade_out", 0.03)
            out.append(nc)
    return out


def register(app):
    from fastapi import HTTPException, Request
    import asyncio
    from . import library, midi

    def place(b, proj):
        """The MIDI song's notes per part, placed in the project; plus (start beat, length beats)."""
        bpb = int(proj.get("bpb") or 4)
        try:
            mi = midi.read(b["midi"])
        except (KeyError, TypeError):
            raise HTTPException(404, "no such MIDI song")
        if mi is None:
            raise HTTPException(404, "no such MIDI song")
        info = library.read(b["midi"]) or {}
        fb = int(info.get("first_beat", 0) or 0)
        start = (max(1, int(b.get("start_bar", 1))) - 1) * bpb
        length = max(1, int(b.get("bars", 8))) * bpb
        allnotes = [n for p in mi["parts"] for n in p["notes"]]
        if not allnotes:
            raise HTTPException(400, "that MIDI song has no notes")
        m_from = b.get("midi_bar")
        m_from = fb + (int(m_from) - 1) * bpb if m_from else math.floor((min(n[0] for n in allnotes) - fb) / bpb) * bpb + fb
        m_bars = max(1, int(b.get("midi_bars") or b.get("bars", 8)))
        loop = bool(b.get("loop", True))
        by = {p["id"]: notes_in_project(p["notes"], m_from, m_bars, start, length, loop, bpb) for p in mi["parts"]}
        return mi, by, start, length, bpb

    @app.post("/api/midi/arrange")
    async def midi_arrange(request: Request):
        """Body: {project, midi, map: {part: track}, start_bar, bars, midi_bar, midi_bars, loop, min_notes}
        -> {clips, dropped: {track: bars dropped}}."""
        b = await request.json()
        proj = b.get("project") or {}
        mi, by, start, length, bpb = place(b, proj)
        clips, rep = await asyncio.to_thread(arrangement, proj, b.get("map") or {}, by, start, length,
                                             int(b.get("min_notes", 1) or 1))
        return {"clips": clips, "dropped": rep}

    @app.post("/api/midi/chord_follow")
    async def midi_chord_follow(request: Request):
        """Body: {project, midi, parts?: [ids] (default: every pitched part), tracks, start_bar, bars, midi_bar,
        midi_bars, loop, mode: "audio" | "key"} -> {clips, chords: ["Am", ...], shifts: {track: [semitones]}}."""
        b = await request.json()
        proj = b.get("project") or {}
        mi, by, start, length, bpb = place(b, proj)
        want = b.get("parts") or [p["id"] for p in mi["parts"] if not p.get("drum")]
        notes = [n for pid in want for n in by.get(pid, [])]
        bars = _bars(start, length, bpb)
        chords = midi_chords(notes, bars)
        tracks = [t for t in (b.get("tracks") or []) if any(c.get("track") == t for c in proj.get("clips", []))]
        if not tracks:
            raise HTTPException(400, "pick at least one lane with clips on it")
        try:
            clips, rep = await asyncio.to_thread(follow, proj, tracks, chords, bars, b.get("mode", "audio"))
        except (OSError, ValueError) as e:
            raise HTTPException(400, str(e))
        names = [None if c is None else NOTE[c[0]] + ("m" if c[1] == "min" else "") for c in chords]
        return {"clips": clips, "chords": names, "shifts": rep}

    @app.post("/api/midi/groove_audio")
    async def midi_groove_audio(request: Request):
        """Body: {project, midi, part, tracks, start_bar, bars, amount (0-100), levels: bool} -> {clips, groove}: the
        lanes' hits moved onto the part's groove (its swing / push per 1/16 step)."""
        b = await request.json()
        proj = b.get("project") or {}
        try:
            mi = midi.read(b["midi"])
            part = midi.part_of(mi, b["part"])
        except (KeyError, TypeError):
            raise HTTPException(404, "no such MIDI part")
        info = library.read(b["midi"]) or {}
        bpb = int(proj.get("bpb") or 4)
        offs, lv, res = groove_template(part["notes"], int(info.get("first_beat", 0) or 0), bpb)
        if max(abs(o) for o in offs) * 60 / float(proj.get("bpm") or 120) < 0.004:
            raise HTTPException(400, "that part plays straight on the grid - it has no swing or push to take")
        start = (max(1, int(b.get("start_bar", 1))) - 1) * bpb
        length = max(1, int(b.get("bars", 8))) * bpb
        amt = max(0.0, min(1.0, float(b.get("amount", 100)) / 100))
        tracks = [t for t in (b.get("tracks") or []) if any(c.get("track") == t for c in proj.get("clips", []))]
        if not tracks:
            raise HTTPException(400, "pick at least one lane with clips on it")

        def go():
            clips = list(proj.get("clips", []))
            for t in tracks:
                new = groove_lane({**proj, "clips": clips}, t, offs, lv, amt, start, length, bool(b.get("levels")), res)
                clips = [c for c in clips if c.get("track") != t] + new
            return clips
        try:
            clips = await asyncio.to_thread(go)
        except (OSError, ValueError) as e:
            raise HTTPException(400, str(e))
        return {"clips": clips, "grid": "1/8" if res == 2 else "1/16",
                "groove": [round(o * 1000 * 60 / float(proj.get("bpm") or 120)) for o in offs]}
