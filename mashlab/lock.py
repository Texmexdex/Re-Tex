"""🔒 Lock to marks: a song whose own timing drifts (a live take, a re-upload with its speed altered) made steady once,
so it sits on the grid; toggled per song in a project.

make(sid) warps each of the song's stems so every one of its marks (yours if you edited them, else the detector's -
those snapped onto the drum hit next to them first, tighten()) lands on a steady grid at the song's average mark spacing - Rubber Band R3 with the click-pilot re-timing, pitch kept
(warp.render) - and saves them next to the originals (stems_locked/). The originals are never touched. song.json
"lock" = {rev, sig, bpm, beats (where each of the song's beats lands in the locked audio), scale, ...}.

A lock belongs to the marks, stems and beat grid it was made from (sig): edit the marks, re-separate or re-analyze the
song and the lock is stale - the song plays as recorded until it's locked again (project.song_tspec checks).
"""
import json
import math
import shutil
import time
from pathlib import Path

import numpy as np

from .config import SR, STEMS


def marks_of(info, marks=None):
    """The marks a lock follows: `marks` (a project's own timing for the song) if given, else yours if you edited
    them, else the detector's (inside the song)."""
    um = marks if marks and marks.get("t") else info.get("user_marks")
    if um and um.get("t"):
        return [float(x) for x in um["t"]]
    dur = float(info.get("duration") or info["beats"][-1])
    return [float(x) for x in info["beats"] if -1e-6 <= x <= dur + 1e-6]


def sig(info, marks=None):
    """Identifies the marks + stems a lock is made from (web/js computes the same: timing.js lockSig)."""
    m, b = marks_of(info, marks), info["beats"]
    ms = lambda x: math.floor(x * 1000 + 0.5)            # (same rounding as JS Math.floor(x * 1000 + 0.5))
    return f"{len(m)}-{sum(ms(x) for x in m)}-r{int(info.get('stems_rev', 0) or 0)}-b{len(b)}-{ms(b[0])}"


def find(info, marks=None):
    """The lock made from these marks (a project's own timing, or by default the song's grid) and the song's current
    stems: its current lock ("lock", stems_locked/) or one kept in "locks" (stems_locked/<rev>/). None if none."""
    s = sig(info, marks)
    for lk in [info.get("lock") or {}, *(info.get("locks") or {}).values()]:
        if lk.get("sig") == s and lk.get("rev"):
            return lk
    return None


def current(info):
    """The song's lock if it matches its current marks and stems, else None."""
    lk = info.get("lock")
    return lk if lk and lk.get("sig") == sig(info) and lk.get("rev") else None


def in_use(sid, lock_sig):
    """Does a saved project play this song locked to the grid with this signature?"""
    from . import library
    from . import project as P
    info = library.read(sid)
    for f in P.PROJECTS.glob("*.json"):
        try:
            s = (json.loads(f.read_text(encoding="utf-8")).get("songs") or {}).get(sid)
        except (OSError, ValueError):
            continue
        if s and s.get("lock") and sig(info, s.get("marks")) == lock_sig:
            return True
    return False


def keep_current(sid, info):
    """Move the song's current lock out of the way, into "locks" and stems_locked/<rev>/ - a project still plays it
    (its grid is the one this lock was made from). Writes and returns the updated info."""
    from . import library
    lk = info.get("lock")
    if not lk or not lk.get("rev"):
        return info
    d = library.song_dir(sid) / "stems_locked"
    dst = d / lk["rev"]
    dst.mkdir(parents=True, exist_ok=True)
    for f in d.glob("*.flac"):
        f.replace(dst / f.name)
    info.setdefault("locks", {})[lk["sig"]] = lk
    info.pop("lock", None)
    library.write(sid, info)
    return info


SNAP = 0.04         # s: the detector's marks are snapped onto the drum hit next to them (16ths are ~150 ms away)
SNAP_CLEAR = 1.5    # ... when that hit stands out: at least 1.5 x the local median onset strength


def tighten(info, marks, drums):
    """The detector's marks sit a few dozen ms off the hits here and there; a lock would bake that in. Snap each
    one onto the strongest snare / kick onset within SNAP of it (drum stem, 150-1500 Hz band, as beats.body_env),
    when there is a clear one. Marks you placed yourself (not on one of the detector's beats) stay exactly put.
    -> (marks, how many moved)."""
    import librosa
    import scipy.signal as sg
    y = sg.sosfilt(sg.butter(4, [150, 1500], "band", fs=SR, output="sos"), drums).astype(np.float32)
    hop = 64
    env = librosa.onset.onset_strength(y=y, sr=SR, hop_length=hop)
    fps = SR / hop
    det = np.asarray(info["beats"], float)
    out, moved = marks.copy(), 0
    w = int(SNAP * fps)
    for j, t in enumerate(marks):
        if np.min(np.abs(det - t)) > 0.005:
            continue                                        # one of yours: leave it
        i = int(round(t * fps))
        a, b = max(0, i - w), min(len(env), i + w + 1)
        if b - a < 3:
            continue
        seg = env[a:b]
        k = int(np.argmax(seg))
        ctx = env[max(0, i - 4 * w):min(len(env), i + 4 * w + 1)]
        if seg[k] < SNAP_CLEAR * (np.median(ctx) + 1e-9) or k in (0, len(seg) - 1):
            continue
        out[j] = (a + k) / fps
        moved += 1
    order = np.argsort(out)                                 # (never cross a neighbour)
    return out[order], moved


def _render_stem(sid, stem, marks, bpm, out_path):
    """Worker process: one stem warped so mark j lands at j * period - t0 (warp.render on the marks) -> (t0, n)."""
    from . import audio_io, warp
    from .config import song_path
    info = json.loads((song_path(sid) / "song.json").read_text(encoding="utf-8"))
    y = audio_io.load(song_path(sid) / "stems" / f"{stem}.flac", SR) / float(info.get("stem_scale", 1.0) or 1.0)
    out, t0, method = warp.render(y, marks, bpm, 1.0, 0.0, stem, "hq", "")
    audio_io.save_npy_atomic(Path(out_path), out.astype(np.float32))
    return float(t0), int(out.shape[1]), method


def check(sid, marks=None):
    """Before a lock: the stretches it would play at the wrong speed (beatfix.slips - a lock pulls every part to the
    song's main tempo) and whether Repair clears them. -> {slips: [[start s, end s, ratio]], yours: the marks are the
    user's own, repair_clears, left: what Repair would leave}. Repair is only worked out for the detector's marks:
    marks lined up by ear are never replaced from here (the beat grid window has Repair, undoable). marks: a
    project's own timing for the song (always "yours")."""
    from . import beatfix, library
    info = library.read(sid)
    marks_ = np.asarray(marks_of(info, marks), float)
    sw = beatfix.slips(marks_) if len(marks_) >= 8 else []
    yours = bool((marks or {}).get("t") or (info.get("user_marks") or {}).get("t"))
    out = dict(slips=[[float(a), float(b), float(k)] for a, b, k in sw], yours=yours, repair_clears=False, left=[])
    if sw and not yours:
        new, _ = beatfix.as_marks(info, library.load_stem(sid, "drums").mean(0), SR)
        left = beatfix.slips(np.asarray(new["t"], float))
        out.update(repair_clears=not left, left=[[float(a), float(b), float(k)] for a, b, k in left])
    return out


def repair_and_make(sid):
    """Repair the detector's marks (beatfix: half / double / 3:2 miscounts re-counted where the drums confirm it,
    every beat on its drum hit), keep them as the song's marks, then lock to them. Refuses marks of the user's own."""
    from . import beatfix, library
    info = library.read(sid)
    if (info.get("user_marks") or {}).get("t"):
        raise ValueError("this song has marks you made - repair them in the Beat grid window")
    new, _ = beatfix.as_marks(info, library.load_stem(sid, "drums").mean(0), SR)
    info = library.read(sid)
    info["user_marks"] = new
    library.write(sid, info)
    return make(sid)


def make(sid, check=True, marks=None):
    """Start (or join) the background job that locks a song to its marks - or to `marks`, a project's own timing for
    it (that lock is kept next to the song's own, so each project plays the grid it was made with). -> the job
    (dict). check=False skips the wrong-speed check: for a grid that's meant to pull parts of the song to another
    speed (a Reimagine take's grid is the mix's, mapped onto the take bar for bar - where YuE2 rushed or dragged,
    the lock is what undoes it)."""
    from . import jobs, library, warp
    info = library.read(sid)
    for j in jobs.active():
        if j.get("kind") == "lock" and j.get("ref") == sid and j.get("status") in ("queued", "running"):
            return j
    own = marks if (marks and marks.get("t") and sig(info, marks) != sig(info)) else None   # a project's own timing

    def work(job):
        info = library.read(sid)
        marks_sig = sig(info, own)
        marks = np.asarray(marks_of(info, own), float)
        if len(marks) < 8:
            raise RuntimeError("not enough marks to lock to")
        from . import beatfix
        sw = beatfix.slips(marks) if check else []
        if sw:          # locking these would play that stretch at half or double speed (Oct 15: HORSE the Band)
            a, b, k = sw[0]
            raise RuntimeError(f"the beat marks at {int(a // 60)}:{int(a % 60):02d}-{int(b // 60)}:{int(b % 60):02d} run at "
                               f"{k:.2f}x the song's tempo - locking would play that part at the wrong speed. Open Beat grid, "
                               "press Repair (or line it up by ear), save, then lock")
        job.update(0.01, "lining the marks up with the drum hits", force=True)
        marks0 = marks.copy()
        marks, moved = tighten(info, marks, library.load_stem(sid, "drums").mean(0))
        period = float(marks[-1] - marks[0]) / (len(marks) - 1)          # the marks' average spacing
        bpm = 60.0 / period
        tmp = library.song_dir(sid) / "stems_locked.tmp"
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir()
        job.update(0.02, "warping the stems onto a steady grid (one time)", force=True)
        pool = warp._get_pool()
        futs = {s: pool.submit(_render_stem, sid, s, marks.tolist(), bpm, str(tmp / f"{s}.npy"), priority=2) for s in STEMS}
        done, res = 0, {}
        while len(res) < len(futs):
            for s, f in futs.items():
                if s not in res and f.done():
                    res[s] = f.result()
                    done += 1
                    job.update(0.05 + 0.85 * done / len(futs), f"locked {done} of {len(futs)} stems")
            time.sleep(0.3)
        t0 = res[STEMS[0]][0]
        ys = {s: np.load(tmp / f"{s}.npy") for s in STEMS}
        n = min(v.shape[1] for v in ys.values())
        peak = max(float(np.abs(v[:, :n]).max()) for v in ys.values())
        scale = round(min(1.0, library.STEM_PEAK / peak), 6) if peak > 0 else 1.0
        from . import audio_io
        rev = f"L{int(time.time())}"
        root = library.song_dir(sid) / "stems_locked"
        if own:                                           # a project's own timing: kept next to the song's lock
            out = root / rev
        else:                                             # the song's own grid: stems_locked/ as always - the lock
            info = library.read(sid)                      # there now is kept first if a project still plays it
            old = info.get("lock") or {}
            if old.get("rev") and old.get("sig") != marks_sig and in_use(sid, old.get("sig")):
                info = keep_current(sid, info)
            out = root
        out.mkdir(parents=True, exist_ok=True)
        for f in out.glob("*.flac"):                      # (never the kept locks in its subfolders)
            f.unlink()
        for s, y in ys.items():
            audio_io.save(out / f"{s}.flac", y[:, :n] * scale, SR, subtype="PCM_24")
        shutil.rmtree(tmp, ignore_errors=True)
        # where each of the song's beats lands in the locked audio: mark j sits at j * period - t0 (straight on at the
        # edge spacing past the first / last mark, as warp._geometry extends). Mapped from the marks as they were
        # BEFORE the snap: the detector's beat j IS mark j, so it lands exactly on steady beat j (its audio is the
        # hit the mark was snapped to).
        tl = np.arange(len(marks)) * period - t0
        b = np.asarray(info["beats"], float)
        lb = np.interp(b, marks0, tl)
        s0 = (tl[1] - tl[0]) / (marks0[1] - marks0[0]); s1 = (tl[-1] - tl[-2]) / (marks0[-1] - marks0[-2])
        lb = np.where(b < marks0[0], tl[0] + (b - marks0[0]) * s0, lb)
        lb = np.where(b > marks0[-1], tl[-1] + (b - marks0[-1]) * s1, lb)
        lock = dict(rev=rev, sig=marks_sig, bpm=round(bpm, 6), period=period, t0=t0,
                    beats=[round(float(x), 5) for x in lb], scale=scale, n=int(n), marks=int(len(marks)),
                    method=res[STEMS[0]][2], tightened=int(moved), t=time.time())
        if own:                                           # (published, so open pages switch over to it)
            locks = dict(library.read(sid).get("locks") or {})
            locks[marks_sig] = lock
            return library.summary(library._set(sid, locks=locks))
        return library.summary(library._set(sid, lock=lock))

    return jobs.submit("lock", f"🔒 Locking {info['title']} to its marks", work, ref=sid).to_dict()
