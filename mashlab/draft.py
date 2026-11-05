"""Quick stand-ins for warps that are still rendering.

After a tempo / transpose / half-double-time change every stem has to be re-warped in high quality, which
takes about a minute. Until then clips used to play the original stem varispeeded onto the grid - right
timing but wrong pitch (a song pulled from 80 to 105 BPM played 4.7 semitones sharp), snapping back
stem by stem as the HQ renders landed.

Instead clips play drafts: Rubber Band R2 renders of 16-beat chunks of the future HQ warp (a chunk takes
well under a second), at the right pitch, re-timed with the same click-pilot trick as the HQ warp so the
beats land on exactly the same samples. Chunks at the playhead render first; the mixer splits a clip at
chunk boundaries (6 ms crossfades) and swaps each piece to its draft as it lands, then to the HQ warp.
Neither swap changes the pitch.
"""
import hashlib
import json
import math
import os
import threading

import numpy as np
import soundfile as sf

from . import clipfx, library, procpool, warp
from .config import CACHE, SR, song_path

CONTEXT = SR                  # source samples rendered either side of a slice, so R2 starts up off-screen
CHUNK_BEATS = 16
MARGIN = int(0.3 * SR)        # each chunk extends this far past its 16 beats (room for crossfades)
XF = 256                      # crossfade between neighbouring chunks (timeline samples)
_lock = threading.Lock()
_open = {}
_geo = {}


def _map(x, xp, fp):
    """Piecewise-linear map through (xp, fp), extended linearly past both ends."""
    x = np.asarray(x, np.float64)
    y = np.interp(x, xp, fp)
    s0 = (fp[1] - fp[0]) / (xp[1] - xp[0])
    s1 = (fp[-1] - fp[-2]) / (xp[-1] - xp[-2])
    y = np.where(x < xp[0], fp[0] + (x - xp[0]) * s0, y)
    return np.where(x > xp[-1], fp[-1] + (x - xp[-1]) * s1, y)


def _n_in(sid, stem, tspec=""):
    from . import variant
    path = variant.stem_file(sid, stem, tspec)
    info = sf.info(str(path))
    return info.frames if info.samplerate == SR else int(round(info.frames * SR / info.samplerate))


def geometry(sid, stem, bpm, mult, tspec=""):
    """(t0, src, dst, n_out) of the HQ warp for this stem - the grid a draft must land on. Cached."""
    k = (sid, stem, warp.stems_rev(sid), round(bpm, 4), mult, tspec)
    with _lock:
        if k in _geo:
            return _geo[k]
    info = library.read(sid)
    if not info or not info.get("beats") or len(info["beats"]) < 3:
        return None
    try:
        n_in = _n_in(sid, stem, tspec)
    except Exception:
        return None
    from . import variant
    beats = variant.beats_for(info, tspec)
    bt, W = warp.timed(beats, tspec) if tspec else (beats, None)
    n_in -= _cut_samples(tspec)
    _, t0, src, dst, n_out = warp._geometry(bt, n_in, bpm, mult, W)
    g = (t0, src, dst, n_out)
    with _lock:
        _geo[k] = g
        if len(_geo) > 256:
            _geo.pop(next(iter(_geo)))
    return g


def pitch_accurate_preview(g, semis):
    """True when the plain varispeed preview is already (nearly) at the right pitch: no draft needed."""
    t0, src, dst, n_out = g
    r = np.diff(src) / np.diff(dst)
    return float(np.max(np.abs(r - 1))) < 0.0006 and abs(semis) < 0.01      # < ~1 cent


def _cut_samples(tspec):
    """How much shorter the source gets when the spec's cuts are taken out (timing.apply_cuts is exact)."""
    from . import timing
    spec = timing.parse(tspec)
    if not spec or not spec.get("c"):
        return 0
    tot, last = 0, -1
    for a, b in sorted((int(round(a * SR)), int(round(b * SR))) for a, b in spec["c"]):
        a = max(a, last)
        if b > a:
            tot += b - a
            last = b
    return tot


def key_for(req, A, B, rate=1.0, tp=0.0):
    """Readable prefix (the HQ warp's cache key) so a landed warp can clear its own drafts."""
    sid, stem, bpm, mult, semis, q = req[:6]
    tspec = req[6] if len(req) > 6 else ""
    extra = "" if abs(rate - 1) < 1e-6 and abs(tp) < 0.004 else "_" + hashlib.sha1(
        json.dumps([round(rate, 5), round(tp, 3)]).encode()).hexdigest()[:8]
    return f"{warp.cache_key(stem, bpm, mult, semis, q, tspec)}_{int(A)}_{int(B)}{extra}"


def path_for(sid, key):
    return CACHE / sid / "drafts" / f"{key}{warp.stems_rev(sid)}.npy"


def open_render(sid, key):
    with _lock:
        if key in _open:
            return _open[key]
    p = path_for(sid, key)
    if not p.exists():
        return None
    try:
        arr = np.load(p, mmap_mode="r")
    except (OSError, ValueError):                         # still being written / replaced: not ready yet
        return None
    with _lock:
        _open[key] = arr
        if len(_open) > 128:
            _open.pop(next(iter(_open)))
    return arr


def _worker(sid, stem, bpm, mult, semis, A, B, rate, tp, out_path, tspec=""):
    """Separate process: samples [A, B) of the would-be HQ warp, rendered from the source with R2. A clip
    with its own speed (keep pitch) / pitch gets that applied too, like clipfx does to an HQ warp: then
    output sample 0 == slice start and the clip reads it at 1x."""
    import pedalboard
    from . import audio_io
    if os.path.exists(out_path):                           # a duplicate request: already rendered
        return str(out_path)
    from . import variant
    info = json.loads((song_path(sid) / "song.json").read_text(encoding="utf-8"))
    path = variant.stem_file(sid, stem, tspec)
    fi = sf.info(str(path))
    full = None
    spec = None
    if tspec:
        from . import timing
        spec = timing.parse(tspec)
    if fi.samplerate == SR and not (spec and spec.get("c")):
        n_in = fi.frames
    else:                                                  # resample, or take the timing cuts out first
        full = audio_io.load(path, SR)
        if spec and spec.get("c"):
            full = timing.apply_cuts(full, spec["c"], SR)
        n_in = full.shape[1]
    beats = variant.beats_for(info, tspec)
    bt, W = warp.timed(beats, tspec) if tspec else (beats, None)
    _, t0, src, dst, n_out = warp._geometry(bt, n_in, bpm, mult, W)
    B = max(int(B), int(A))
    s0 = int(max(0, math.floor(float(_map(A, dst, src))) - CONTEXT))
    s1 = int(min(n_in, math.ceil(float(_map(B, dst, src))) + CONTEXT))
    out = np.zeros((2, B - A), np.float32)
    if s1 - s0 > 64:
        if full is None:
            y, _ = sf.read(str(path), start=s0, stop=s1, dtype="float32", always_2d=True)
            y = y.T
            y = np.repeat(y, 2, axis=0) if y.shape[0] == 1 else y[:2]
        else:
            y = full[:, s0:s1]
        y = np.ascontiguousarray(y, np.float32) / variant.scale_for(info, tspec)   # undo save headroom
        # per-input-sample stretch factor, as warp._stretch_array but for this stretch only
        ratios = np.diff(src) / np.diff(dst)
        idx = np.clip(np.round(src).astype(np.int64), 0, n_in)
        j = np.clip(np.searchsorted(idx, np.arange(s0, s1), side="right") - 1, 0, len(ratios) - 1)
        f = ratios[j]
        opts = dict(transient_mode="crisp" if stem == "drums" else "mixed", preserve_formants=(stem == "vocals"))
        z = pedalboard.time_stretch(y, SR, stretch_factor=f, pitch_shift_in_semitones=semis,
                                    high_quality=False, **opts)
        o = int(round(float(_map(s0, src, dst))))              # where source sample s0 lands
        z = _retime(z, f, semis, opts, src, dst, s0, s1, o)
        if stem != "drums":
            # R2 in "mixed" transient mode comes out ~2 dB quieter than the source (and the HQ warp)
            ry, rz = float(np.sqrt(np.mean(y ** 2))), float(np.sqrt(np.mean(z ** 2)))
            if ry > 1e-4 and rz > 1e-6:
                z = z * float(np.clip(ry / rz, 0.5, 2.0))
        a = A - o                                              # slice start inside z
        lo, hi = max(0, a), min(z.shape[1], a + (B - A))
        if hi > lo:
            out[:, lo - a:hi - a] = z[:, lo:hi]
    if abs(rate - 1) > 1e-6 or abs(tp) > 0.004:
        T = 60.0 / bpm
        nb = int(math.ceil(out.shape[1] / SR / T)) + 3
        out, _, _ = warp.render(out, (np.arange(nb) - 1) * T, bpm * rate, 1.0, tp, stem, "fast")
        out = np.ascontiguousarray(out, np.float32)
    return audio_io.save_npy_atomic(out_path, out)


def _retime(z, f, semis, opts, src, dst, s0, s1, o):
    """R2 wanders by +-15 ms. Push clicks on the source beats through the identical stretch, find where they
    landed and resample z so every beat sits exactly on the grid (warp.render does the same for HQ)."""
    import pedalboard
    from scipy.ndimage import median_filter
    jb = np.nonzero((src >= s0) & (src < s1 - 8))[0]
    if len(jb) < 2:
        return z
    pilot = np.zeros(s1 - s0, np.float32)
    click = np.array([1.0, 0.8, 0.5, 0.25], np.float32)
    for j in jb:
        i = int(round(src[j])) - s0
        pilot[i:i + 4] += click
    pz = pedalboard.time_stretch(pilot[None], SR, stretch_factor=f, pitch_shift_in_semitones=semis,
                                 high_quality=False, **opts)[0]
    e = np.convolve(pz.astype(np.float64) ** 2, np.ones(32) / 32, mode="same")
    expect = dst[jb] - o                                    # where each beat belongs, in z's samples
    meas = warp._measure(e, expect + 1.5, np.ones(len(jb), bool)) - 1.5
    good = np.isfinite(meas)
    if good.sum() < 2:
        return z
    err = meas[good] - expect[good]
    trend = median_filter(err, size=min(9, len(err) | 1), mode="nearest")
    keep = np.abs(err - trend) < 0.03 * SR
    if keep.sum() < 2:
        return z
    return warp.resample_map(z, meas[good][keep], expect[good][keep], z.shape[1])


def chunk_range(k, T, n_out):
    """Samples [A, B) of the HQ warp that draft chunk k covers (its 16 beats + margins)."""
    C = CHUNK_BEATS * T
    return max(0, int(k * C) - MARGIN), min(int(n_out), int((k + 1) * C) + MARGIN)


def _discard_partial(key):
    for f in CACHE.glob(f"*/drafts/{key}*.tmp.npy"):
        try:
            f.unlink()
        except OSError:
            pass


_renders = procpool.Renders(warp._get_pool, priority=0, kill_running=False, on_dropped=_discard_partial)


def queue(pending, on_done, start=True):
    """pending: Mixer.drafts [(clip_id, req, A, B, rate, tp, key, t0, t1)], most urgent first (t0, t1: where
    it plays on the timeline). Drafts nobody needs any more (the HQ warp landed, or the tempo moved on
    again) are dropped if they haven't started."""
    wanted = []
    for clip_id, req, A, B, rate, tp, key, _, _ in pending:
        sid, stem, bpm, mult, semis, _q = req[:6]
        out = path_for(sid, key)
        if out.exists():                                   # landed a moment ago: nothing to do
            continue
        wanted.append((key, _worker, (sid, stem, bpm, mult, semis, int(A), int(B), float(rate), float(tp),
                                      str(out), req[6] if len(req) > 6 else "")))
    _renders.sync(wanted, on_done, start)


def sweep(keep_reqs):
    """Delete warp drafts that nothing will play again: every stem whose HQ warp has landed, or that the
    arrangement stopped using (tempo moved on). keep_reqs: the warps the active project still waits for.
    Clip drafts (clipfx, named RENDER_VERSION...) are left to purge_old(). Returns how many it had to leave
    for now (still open, or still being written)."""
    keep = tuple(warp.cache_key(*r[1:]) + "_" for r in keep_reqs)
    left = 0
    for f in CACHE.glob("*/drafts/*.npy"):
        name = f.name
        if name.startswith(warp.RENDER_VERSION) or (keep and name.startswith(keep)):
            continue
        if name.endswith(".tmp.npy"):
            left += 1
            continue
        with _lock:
            for k in [k for k in _open if name.startswith(k)]:
                del _open[k]                  # release the memory map so Windows lets us delete the file
        try:
            f.unlink()
        except OSError:
            left += 1                         # still mapped by the mixer for a moment: try again shortly
    return left


def purge_old(max_age_s=86400):
    """Startup housekeeping: drafts and clip drafts older than a day."""
    import time
    now = time.time()
    for f in list(CACHE.glob("*/drafts/*.npy")):
        try:
            if now - f.stat().st_mtime > max_age_s:
                f.unlink()
        except OSError:
            pass
