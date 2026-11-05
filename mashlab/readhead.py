"""ReadHead inside Re-Tex: automatic, beat-aligned chops and vocal arranging that come out as
ordinary Re-Tex clips (so they play, render and export through the normal mixer, untouched).

Two modes, both run on the PROJECT's grid (its tempo, bar 1 = beat 0):
  arrange  the vocal song on a lane is re-arranged against the rest of the project: it reads
           forward through the vocal with loops, alternations, stammers, ratchets and hook loops,
           keeping the singer's own bar phase and cutting only between syllables ("Flow", learned
           from the author's own hand-made edit), or places whole lines per section ("Lines")
  glitch   the lane's current audio is chopped in place: stutters, rolls, repeats, skips, gates,
           displacements, phrase-aware, every phrase back in sync

The heavy lifting is ReadHead's tested JS engine (readhead_js/, run with Node). This module only
builds its inputs from the warped renders Re-Tex already made, and turns the edit it returns back
into clips. It reads the library; it writes nothing there (inputs + feature caches go to the temp
folder). The frontend (web/js/readhead.js) shows the result and applies it as one undo step.
"""
import asyncio
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from . import library, project as P, timing, variant, warp
from .config import SR

JS_DIR = Path(__file__).with_name("readhead_js")
TMP = Path(tempfile.gettempdir()) / "retex_readhead"
ANALYSIS_SR = SR // 2           # the backing only feeds structure/energy analysis: half rate is plenty


def node_exe():
    n = shutil.which("node")
    if n:
        return n
    for p in (r"C:\Program Files\nodejs\node.exe", r"C:\Program Files (x86)\nodejs\node.exe"):
        if os.path.exists(p):
            return p
    return None


# ------------------------------------------------------------------ where a song's beats land
def _landing(project, sid):
    """W (song beat index -> warped beat) exactly as the mixer computes it, or None (beat j lands on j)."""
    tspec = P.song_tspec(project, sid)
    if not tspec:
        return None
    info = library.read(sid) or {}
    beats = variant.beats_for(info, tspec) if info.get("beats") else [0.0, 1.0]
    return np.asarray(timing.positions(beats, timing.parse(tspec)), float)


def _w_of(W, src):
    return float(src) if W is None else float(np.interp(src, np.arange(len(W), dtype=float), W))


def _src_of(W, w):
    if W is None:
        return float(w)
    idx = np.arange(len(W), dtype=float)
    if w <= W[0]:
        return float(w - W[0])            # before the first beat: same spacing
    if w >= W[-1]:
        return float(idx[-1] + (w - W[-1]))
    return float(np.interp(w, W, idx))


def _render(project, c):
    """(buffer, t0, mult) of the warped render this clip plays, or None if Re-Tex hasn't made it yet."""
    req = P.request(project, c)
    r = warp.open_render(*req)
    if r is None:
        return None
    return r[0], float(r[1]), P.song_mult(project, c["song"])


def _mono(buf, a, b):
    """Mono float32 of buffer samples [a, b) (zero-padded outside the buffer)."""
    n = b - a
    out = np.zeros(n, np.float32)
    lo, hi = max(0, a), min(buf.shape[1], b)
    if hi > lo:
        seg = np.asarray(buf[:, lo:hi], np.float32)
        out[lo - a:hi - a] = seg.mean(axis=0)
    return out


def _lane_audio(project, clips, T, length_samples, missing):
    """The clips as they sound on the timeline (rate-1 clips only), mono."""
    y = np.zeros(length_samples, np.float32)
    for c in clips:
        if c.get("mute") or abs(float(c.get("rate", 1) or 1) - 1) > 1e-6 or c.get("warps"):
            continue
        r = _render(project, c)
        if r is None:
            missing.add(c["song"])
            continue
        buf, t0, mult = r
        W = _landing(project, c["song"])
        base = _w_of(W, c["src"]) * mult * T - t0 * SR     # buffer sample at the clip start (mixer.py)
        d0, d1 = int(round(c["start"] * T)), int(round((c["start"] + c["len"]) * T))
        d1 = min(d1, length_samples)
        if d1 <= d0:
            continue
        g = 10 ** (float(c.get("gain_db", 0) or 0) / 20)
        a = int(round(base))
        y[d0:d1] += _mono(buf, a, a + (d1 - d0)) * g
    return y


def _decimate2(x):
    n = len(x) // 2 * 2
    return ((x[0:n:2] + x[1:n:2]) * 0.5).astype(np.float32)


def _write_input(arr):
    """Write a mono float32 input named by its content hash (the node side caches features per file)."""
    TMP.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha1(arr.tobytes()).hexdigest()[:20]
    p = TMP / f"in_{h}.f32"
    if not p.exists():
        arr.astype(np.float32).tofile(p)
    return str(p)


# ------------------------------------------------------------------ main entry
def run(project, track_id, mode="arrange", opts=None, rng=None):
    """Returns {clips, replace, phrases, plan, summary, warnings} for the frontend to apply."""
    opts = dict(opts or {})
    node = node_exe()
    if not node:
        raise RuntimeError("ReadHead needs Node.js (https://nodejs.org) - it wasn't found on this PC")
    bpm, bpb = float(project["bpm"]), int(project.get("bpb", 4))
    T = SR * 60.0 / bpm                                       # samples per project beat
    clips = [c for c in project.get("clips", []) if not c.get("mute")]
    lane = [c for c in clips if c["track"] == track_id]
    if not lane:
        raise ValueError("That lane has no clips")
    other = [c for c in clips if c["track"] != track_id]
    end_beats = max(c["start"] + c["len"] for c in clips)
    L = int(round(end_beats * T))
    warnings, missing = [], set()

    req = {"bpm": bpm, "bpb": bpb, "lengthBeats": end_beats, "mode": mode, "opts": opts,
           "sections": project.get("sections") or []}
    groups = {"drums": [], "bass": [], "other": []}
    for c in other:
        groups["drums" if c["stem"] == "drums" else "bass" if c["stem"] == "bass" else "other"].append(c)
    backing = []
    for role, cs in groups.items():
        if not cs:
            continue
        y = _lane_audio(project, cs, T, L, missing)
        if np.abs(y).max() > 0:
            backing.append({"role": role, "path": _write_input(_decimate2(y)), "sr": ANALYSIS_SR})
    if not backing:
        raise ValueError("Nothing else plays in this project to arrange against - add the instrumental first")
    req["backing"] = backing

    if mode == "arrange":
        # the vocal song on this lane (the one with the most vocal clips there)
        voc = [c for c in lane if c["stem"] == "vocals"] or lane
        counts = {}
        for c in voc:
            counts[(c["song"], c["stem"])] = counts.get((c["song"], c["stem"]), 0) + c["len"]
        sid, stem = max(counts, key=counts.get)
        ref = next(c for c in voc if c["song"] == sid and c["stem"] == stem)
        r = _render(project, ref)
        if r is None:
            raise RuntimeError("The vocal isn't rendered at this tempo yet - press play once so Re-Tex renders it, then try again")
        buf, t0, mult = r
        # source axis q (project beats): song warped beat w sits at q = w * mult; q = 0 at sample 0
        a = int(round(-t0 * SR))                              # buffer sample of q = 0 (usually before the buffer)
        src = _mono(buf, a, buf.shape[1])
        req["source"] = {"role": "vocal", "path": _write_input(src), "sr": SR}
        W = _landing(project, sid)
        src_len_q = len(src) / T
    else:
        y = _lane_audio(project, lane, T, L, missing)
        role = {"drums": "drums", "bass": "bass", "vocals": "vocal"}.get(max(set(c["stem"] for c in lane), key=lambda s: sum(c["len"] for c in lane if c["stem"] == s)), "other")
        req["source"] = {"role": role, "path": _write_input(y), "sr": SR}
        skipped = [c for c in lane if abs(float(c.get("rate", 1) or 1) - 1) > 1e-6 or c.get("warps")]
        if skipped:
            warnings.append(f"{len(skipped)} clip(s) on this lane have a speed change or stretch and were left as they are")
    if missing:
        warnings.append("Not rendered yet (left out of the analysis): " + ", ".join(sorted(missing)) + " - press play once and try again")

    TMP.mkdir(parents=True, exist_ok=True)
    rq = TMP / f"req_{os.getpid()}_{abs(hash(json.dumps(opts, sort_keys=True))) % 10**8}.json"
    rq.write_text(json.dumps(req), encoding="utf-8")
    try:
        p = subprocess.run([node, str(JS_DIR / "cli.mjs"), str(rq)], capture_output=True, text=True, timeout=600,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    finally:
        try:
            rq.unlink()
        except OSError:
            pass
    if p.returncode != 0:
        raise RuntimeError("ReadHead engine failed: " + (p.stderr.strip().splitlines() or ["?"])[-1][:300])
    res = json.loads(p.stdout)
    events = res["lane"]

    rng_lo = float(rng[0]) if rng else 0.0
    rng_hi = float(rng[1]) if rng else end_beats
    if mode == "arrange":
        new = _vocal_clips(events, sid, stem, W, mult, src_len_q, rng_lo, rng_hi)
        replace = {"track": track_id, "song": sid, "stem": stem, "start": rng_lo, "end": rng_hi}
    else:
        keep = [c for c in lane if not (abs(float(c.get("rate", 1) or 1) - 1) > 1e-6 or c.get("warps"))]
        new = _timeline_clips(events, keep, project, rng_lo, rng_hi)
        replace = {"track": track_id, "song": None, "start": rng_lo, "end": rng_hi,
                   "ids": [c["id"] for c in keep]}
    # clips that stick out of the range keep their outside parts (src moved with the cut, landing-exact)
    trims = []
    hit = (lambda c: c["id"] in set(replace["ids"])) if replace.get("ids") is not None else \
          (lambda c: c["song"] == replace["song"] and c["stem"] == replace["stem"])
    for c in lane:
        if not hit(c) or c["start"] + c["len"] <= rng_lo + 1e-9 or c["start"] >= rng_hi - 1e-9:
            continue
        left = rng_lo - c["start"] if c["start"] < rng_lo - 1e-9 else None
        right = None
        if c["start"] + c["len"] > rng_hi + 1e-9 and not c.get("warps"):
            Wc, mc = _landing(project, c["song"]), P.song_mult(project, c["song"])
            k = float(c.get("rate", 1) or 1)
            w = _w_of(Wc, c["src"]) + (rng_hi - c["start"]) * k / mc
            right = {**{kk: vv for kk, vv in c.items() if kk != "id"}, "start": rng_hi, "len": c["start"] + c["len"] - rng_hi,
                     "src": round(_src_of(Wc, w), 6), "fade_in": 0}
        trims.append({"id": c["id"], "left_len": left, "right": right})
    replace["trims"] = trims
    summary = {"mode": mode, "clips": len(new), "ms": res.get("ms"), "vocal": res.get("vocal"), "counts": res.get("counts")}
    return {"clips": new, "replace": replace, "phrases": res["phrases"], "plan": res.get("plan", []),
            "summary": summary, "warnings": warnings}


def _segments(events, lo, hi):
    """[(t0, t1, off, gain)] the edit plays over [lo, hi). Before the first event: untouched (off 0, gain 1)."""
    ev = sorted(events, key=lambda e: e["t"])
    out, st, t = [], {"off": 0.0, "gain": 1.0, "rate": 1}, lo
    for e in ev:
        if e["t"] <= lo:
            st = e
            continue
        if e["t"] >= hi:
            break
        if e["t"] > t:
            out.append((t, e["t"], float(st["off"]), float(st["gain"]), int(st.get("rate", 1))))
        st, t = e, e["t"]
    if hi > t:
        out.append((t, hi, float(st["off"]), float(st["gain"]), int(st.get("rate", 1))))
    return out


def _clip(track, song, stem, start, src, length, gain=1.0):
    return {"song": song, "stem": stem, "start": round(start, 6), "src": round(src, 6), "len": round(length, 6),
            "gain_db": round(20 * np.log10(max(gain, 1e-4)), 2) if gain < 0.999 else 0, "fade_in": 0, "fade_out": 0.03,
            "mute": False, "rate": 1, "pitch": 0, "keep_pitch": True, "readhead": True}


def _vocal_clips(events, sid, stem, W, mult, src_len_q, lo, hi):
    out = []
    for t0, t1, off, gain, rate in _segments(events, lo, hi):
        if gain <= 0 or rate != 1 or t1 - t0 < 1e-4:
            continue
        q0 = t0 - off
        if q0 >= src_len_q or q0 + (t1 - t0) <= 0:
            continue
        out.append(_clip(None, sid, stem, t0, _src_of(W, q0 / mult), t1 - t0, gain))
    return out


def _timeline_clips(events, lane, project, lo, hi):
    """Glitch mode: the edit reads the lane's own timeline; map each piece back to the clip(s) it came from."""
    lane = sorted(lane, key=lambda c: c["start"])
    land = {c["song"]: _landing(project, c["song"]) for c in lane}
    out = []
    for t0, t1, off, gain, rate in _segments(events, lo, hi):
        if gain <= 0 or rate != 1:
            continue
        p0, p1 = t0 - off, t1 - off                           # timeline beats being read
        for c in lane:
            a, b = max(p0, c["start"]), min(p1, c["start"] + c["len"])
            if b - a < 1e-4:
                continue
            mult = P.song_mult(project, c["song"])
            W = land[c["song"]]
            w = _w_of(W, c["src"]) + (a - c["start"]) / mult
            nc = _clip(None, c["song"], c["stem"], a + off, _src_of(W, w), b - a, gain)
            nc["gain_db"] = round(nc["gain_db"] + float(c.get("gain_db", 0) or 0), 2)
            out.append(nc)
    return out


def register(app):
    from fastapi import HTTPException, Request

    @app.post("/api/readhead/run")
    async def readhead_run(request: Request):
        """Body: {project, track, mode: "arrange"|"glitch", opts, range: [start, end] beats | null}.
        Returns clips for the frontend to apply as one undo step; changes nothing by itself."""
        b = await request.json()
        try:
            return await asyncio.to_thread(run, b["project"], b["track"], b.get("mode", "arrange"), b.get("opts") or {}, b.get("range"))
        except (ValueError, RuntimeError) as e:
            raise HTTPException(400, str(e))
