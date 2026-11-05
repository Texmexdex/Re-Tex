"""Per-clip stretch + pitch.

A clip can play its source faster/slower than the grid (`rate`, >1 = faster), with or without keeping
pitch (`keep_pitch`), plus a free pitch offset in semitones (`pitch`).

  * keep_pitch off, pitch 0: pure varispeed, done live in the mixer (instant, exact).
  * anything else: a high-quality render of just the clip's slice of the already-warped stem
    (Rubber Band R3 + the same pilot re-timing as the main warp, so beats stay sample-accurate).
    Slices are block-aligned (8 beats) so trimming/moving a clip usually reuses the same render.
    Until the render lands, the mixer plays a varispeed preview with the correct timing.
"""
import hashlib
import json
import math
import os
import threading

import numpy as np

from . import procpool, warp
from .config import CACHE, SR

BLOCK_BEATS = 8
_lock = threading.Lock()
_open = {}


def total_pitch(c):
    r = float(c.get("rate", 1.0) or 1.0)
    p = float(c.get("pitch", 0.0) or 0.0)
    return p + (0.0 if c.get("keep_pitch", True) else 12 * math.log2(r))


def needs_hq(c):
    """Pure varispeed (pitch follows speed, no extra pitch) is done live; everything else is rendered."""
    r = float(c.get("rate", 1.0) or 1.0)
    extra = abs(float(c.get("pitch", 0.0) or 0.0)) > 0.004
    if c.get("keep_pitch", True):
        return extra or abs(r - 1) > 1e-4
    return extra


def segment(base_len, off, n_out, rate, T):
    """Block-aligned slice [A, B) of the base buffer the clip reads (+ a little margin)."""
    blk = BLOCK_BEATS * T
    a = off - 0.3 * SR * rate
    b = off + n_out * rate + 0.3 * SR * rate
    A = max(0, int(math.floor(a / blk) * blk))
    B = min(int(base_len), int(math.ceil(b / blk) * blk))
    return A, max(B, A + 1)


def key_for(base_req, A, B, rate, tp, draft=False):
    """draft=True: the quick Rubber Band R2 version that plays (at the right pitch) until the HQ one lands."""
    br = list(base_req[:6]) + ([base_req[6]] if len(base_req) > 6 and base_req[6] else [])   # plain grid: old keys
    s = json.dumps([br, int(A), int(B), round(rate, 5), round(tp, 3)])
    return warp.RENDER_VERSION + hashlib.sha1(s.encode()).hexdigest()[:18] + ("_d" if draft else "")


def path_for(sid, key):
    # drafts live with the warp drafts: short-lived, cleaned up by draft.purge_old()
    return CACHE / sid / ("drafts" if key.endswith("_d") else "clips") / f"{key}{warp.stems_rev(sid)}.npy"


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
        if len(_open) > 64:
            _open.pop(next(iter(_open)))
    return arr


def _worker(base_path, A, B, bpm, rate, tp, stem, quality, out_path):
    """Separate process: stretch/pitch the slice so output sample 0 == slice start."""
    from . import audio_io
    if os.path.exists(out_path):                           # a duplicate request: already rendered
        return str(out_path)
    base = np.load(base_path, mmap_mode="r")
    y = np.ascontiguousarray(base[:, A:B], dtype=np.float32)
    T = 60.0 / bpm
    nb = int(math.ceil(y.shape[1] / SR / T)) + 3
    beats = (np.arange(nb) - 1) * T               # uniform grid: the base buffer is already on-tempo
    out, _, _ = warp.render(y, beats, bpm * rate, 1.0, tp, stem, quality)
    return audio_io.save_npy_atomic(out_path, out.astype(np.float32))


def _job_args(p, quality=None):
    """p: pending tuple from Mixer.load -> worker args."""
    clip_id, base_req, A, B, rate, tp, stem, key = p[:8]
    sid, st, bpm, mult, semis, base_quality = base_req[:6]
    tspec = base_req[6] if len(base_req) > 6 else ""
    base_path = warp.cache_path(sid, warp.cache_key(st, bpm, mult, semis, base_quality, tspec))
    return (str(base_path), A, B, bpm, rate, tp, stem, quality or base_quality, str(path_for(sid, key)))


def queue(pending, drafts, on_done, start=True):
    """Render the project's pending clip slices in the background: `drafts` (quick R2 versions, first) and
    `pending` (HQ), as listed by Mixer.load - everything the arrangement still needs. Anything in flight
    that no clip wants any more (re-pitched, re-stretched, new tempo, deleted) is dropped, HQ renders even
    mid-render. start=False only drops."""
    ready = lambda p: path_for(p[1][0], p[7]).exists()     # landed a moment ago: nothing to do
    _drafts.sync([(p[7], _worker, _job_args(p, "fast")) for p in drafts if not ready(p)], on_done, start)
    _hq.sync([(p[7], _worker, _job_args(p)) for p in pending if not ready(p)], on_done, start)


def _discard_partial(key):
    """A render stopped mid-write leaves its temp file behind; remove it (skipped if a newer render of
    the same slice is writing it right now - Windows won't delete an open file)."""
    for f in CACHE.glob(f"*/clips/{key}*.tmp.npy"):
        try:
            f.unlink()
        except OSError:
            pass


def render_blocking(pending, progress=lambda f, m="": None):
    """For export: render every pending slice now."""
    todo = list({p[7]: p for p in pending}.values())
    if not todo:
        return
    pool = warp._get_pool()
    futs = [pool.submit(_worker, *_job_args(p)) for p in todo]
    for i, f in enumerate(futs):
        f.result()
        progress((i + 1) / len(futs), f"stretching clips {i + 1}/{len(futs)}")


_hq = procpool.Renders(warp._get_pool, priority=1, kill_running=True, on_dropped=_discard_partial)
_drafts = procpool.Renders(warp._get_pool, priority=0, kill_running=False)
