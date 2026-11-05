"""Your lyrics for a song: typed once, lined up with the song's own vocal stem (align.py - forced alignment, the words
are never changed) and kept in songs/<id>/lyrics_user.json. Wherever the app reads a song's words - the timeline's
lyrics overlay, the song form's chorus finder, Reimagine - yours win over the Whisper transcript (lyrics.json, which
is left as it is; the frozen releases read only that one).

    {version, text, words: [{t, t1, w, conf, line}], lines: [{t0, t1, text, conf, words: [{t, w}]}], stem_mtime}
"""
import json
import os
import time

import numpy as np

from . import library

VERSION = 1


def path(sid):
    return library.song_dir(sid) / "lyrics_user.json"


def _stem(sid):
    return library.song_dir(sid) / "stems" / "vocals.flac"


def _stem_mtime(sid):
    p = _stem(sid)
    return int(p.stat().st_mtime) if p.exists() else 0


def load(sid):
    """Your lyrics for the song, or None. `stale` = the vocal stem changed since they were lined up (re-separated)."""
    try:
        d = json.loads(path(sid).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if d.get("version") != VERSION:
        return None
    d["stale"] = bool(d.get("stem_mtime")) and d["stem_mtime"] != _stem_mtime(sid)
    return d


def _write(sid, d):
    p = path(sid)
    tmp = p.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, p)


def save(sid, text, progress=lambda f, m="": None):
    """Keep `text` as the song's lyrics and line every word up with its vocal stem. Empty text = forget them (the
    Whisper transcript is used again). -> the saved dict, or None when cleared."""
    from . import align
    if not align.text_lines(text):
        path(sid).unlink(missing_ok=True)
        return None
    stem = _stem(sid)
    if not stem.exists():
        raise ValueError("this song has no vocal stem to line the words up with")
    progress(0.1, "lining your words up with the vocals")
    words = align.align_file(stem, text)
    d = dict(version=VERSION, text=str(text), words=words, lines=align.lines(words, text), stem_mtime=_stem_mtime(sid),
             aligned=round(time.time()))
    _write(sid, d)
    progress(1.0, "lined up")
    d["stale"] = False
    return d


def segments(sid):
    """The song's sung lines with word times (song seconds), yours first, else Whisper's. -> ([{t0, t1, text,
    words: [{t, w}]}], "yours" | "whisper" | None)"""
    u = load(sid)
    if u and u.get("lines"):
        return u["lines"], "yours"
    try:
        L = json.loads((library.song_dir(sid) / "lyrics.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return [], None
    return L.get("segments") or [], "whisper"


def mtime(sid):
    """Changes whenever the song's words do (yours or Whisper's) - for caches built on them."""
    out = 0
    for p in (path(sid), library.song_dir(sid) / "lyrics.json"):
        if p.exists():
            out = max(out, int(p.stat().st_mtime))
    return out


# ---------------------------------------------------------------- the words of a mix
# A clip plays a stretch of a song's audio somewhere in the project: song seconds -> source beat (the song's own beat
# grid) -> where that beat lands (the timing spec the mixer plays it with) -> clip offset (its warps) -> project beat.
# The same chain the mixer and the timeline use, so typed words land exactly where they're heard in the bounce.

def _clip_map(proj, c):
    """song seconds -> mix seconds for clip c, or None outside the clip."""
    from . import project as P, timing, variant, warps
    info = library.read(c["song"]) or {}
    beats = np.asarray(info.get("beats") or [], float)
    if len(beats) < 2:
        return lambda t: None
    tspec = P.song_tspec(proj, c["song"])
    landing = None
    if tspec:
        W = timing.positions(variant.beats_for(info, tspec), timing.parse(tspec))
        landing = (np.asarray(W, float), np.arange(len(W), dtype=float))
    ps = [p for p in warps.pieces(c, P.song_mult(proj, c["song"]), landing) if p[3] > p[2]]
    spb = 60.0 / float(proj["bpm"])

    def f(t):
        b = float(P._time_to_beat(t, beats))
        w = P._interp_ext(b, landing[1], landing[0]) if landing else b
        for u0, u1, w0, w1, _, _ in ps:
            if w0 - 1e-6 <= w <= w1 + 1e-6:
                return (float(c["start"]) + u0 + (w - w0) * (u1 - u0) / (w1 - w0)) * spb
        return None
    return f


def mix_lines(proj):
    """The typed lines heard in a mix, in mix seconds: every audible vocal clip whose song has your lyrics, cut to
    the words the clip plays. -> (lines [{t0, t1, text, words, song}], songs heard singing without typed lyrics)"""
    tracks = {t["id"]: t for t in proj.get("tracks", [])}
    solo = any(t.get("solo") for t in tracks.values())
    out, missing = [], set()
    for c in proj.get("clips", []):
        t = tracks.get(c.get("track"))
        if c.get("stem") != "vocals" or c.get("mute") or float(c.get("len", 0)) <= 0 or t is None:
            continue
        if t.get("mute") or (solo and not t.get("solo")) or not library.read(c["song"]):
            continue
        u = load(c["song"])
        if not u or not u.get("words"):
            missing.add(c["song"])
            continue
        f = _clip_map(proj, c)
        by_line = {}
        for w in u["words"]:
            a, b = f(w["t"]), f(w["t1"])
            if a is None:
                continue
            by_line.setdefault(w["line"], []).append(dict(t=round(a, 3), t1=round(b if b is not None else a + 0.2, 3),
                                                          w=w["w"]))
        for ws in by_line.values():
            out.append(dict(t0=ws[0]["t"], t1=ws[-1]["t1"], text=" ".join(w["w"] for w in ws),
                            words=[dict(t=w["t"], w=w["w"]) for w in ws], song=c["song"]))
    out.sort(key=lambda l: l["t0"])
    return out, sorted(missing)
