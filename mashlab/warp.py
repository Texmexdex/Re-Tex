"""Warp a stem onto the project grid: every source beat lands exactly on a project beat.

Engine (per stem, cached on disk):
  * tiny change (<0.35 % speed, no transpose): pure varispeed along the beat map - no phase-vocoder
    artifacts at all, pitch moves < 6 cents.
  * otherwise: Rubber Band R3 (pedalboard) with a per-beat stretch map + pitch shift. R3 drifts in time
    internally (up to tens of ms), so we push a click "pilot" through the identical stretch, measure
    where each beat really landed, and re-time the output with a windowed-sinc resampler (corrections
    are ~0.1 %, i.e. a couple of cents). Result: R3 quality with sample-accurate beats.

Rendered buffer coordinates: source beat j (index into the song's extended beat list) sits at sample
    (j * Tb - t0) * SR,   Tb = 60 / bpm * mult,   t0 = where source time 0 lands.
"""
import json
import os
import shutil
import threading
from concurrent.futures import as_completed

import numpy as np
import numba

from . import procpool, timing
from .config import CACHE, SR, song_path

VARISPEED_MAX = 0.0006   # tempo changes below ~1 cent are plain resampling; anything more is time-stretched, pitch kept
_pool = None
_pool_lock = threading.Lock()
_open = {}
_open_lock = threading.Lock()


# ---------------------------------------------------------------- sinc resampling along a time map

_TAPS = 32
_PHASES = 2048


def _make_table():
    half = _TAPS // 2
    ph = np.arange(_PHASES) / _PHASES
    k = np.arange(-half + 1, half + 1)
    x = k[None, :] - ph[:, None]
    w = np.kaiser(_TAPS, 9.0)
    t = np.sinc(x * 0.97) * 0.97 * w[None, :]       # 0.97: tiny low-pass margin against aliasing
    return (t / t.sum(1, keepdims=True)).astype(np.float32)


_TABLE = _make_table()


@numba.njit(cache=True, fastmath=True)
def _interp(x, pos, table):
    ch, n = x.shape
    m = pos.shape[0]
    taps = table.shape[1]
    half = taps // 2
    nph = table.shape[0]
    out = np.zeros((ch, m), np.float32)
    for i in range(m):
        p = pos[i]
        i0 = int(np.floor(p))
        fr = p - i0
        row = table[int(fr * nph) % nph]
        base = i0 - half + 1
        if base < 0 or base + taps > n:
            for c in range(ch):
                acc = 0.0
                for t in range(taps):
                    j = base + t
                    if 0 <= j < n:
                        acc += x[c, j] * row[t]
                out[c, i] = acc
        else:
            for c in range(ch):
                acc = 0.0
                for t in range(taps):
                    acc += x[c, base + t] * row[t]
                out[c, i] = acc
    return out


def resample_map(x, src_pts, dst_pts, n_out):
    """Output sample n reads input position interp(n, dst_pts, src_pts) (piecewise linear, extrapolated)."""
    n = np.arange(n_out, dtype=np.float64)
    pos = np.interp(n, dst_pts, src_pts)
    # linear extrapolation beyond the ends
    if len(dst_pts) > 1:
        s0 = (src_pts[1] - src_pts[0]) / (dst_pts[1] - dst_pts[0])
        s1 = (src_pts[-1] - src_pts[-2]) / (dst_pts[-1] - dst_pts[-2])
        lo, hi = n < dst_pts[0], n > dst_pts[-1]
        pos[lo] = src_pts[0] + (n[lo] - dst_pts[0]) * s0
        pos[hi] = src_pts[-1] + (n[hi] - dst_pts[-1]) * s1
    return _interp(np.ascontiguousarray(x, np.float32), pos, _TABLE)


# ---------------------------------------------------------------- the warp itself

def _geometry(beats, n_in, bpm, mult, W=None):
    """W: where each beat lands, in beats (timing.positions); None = beat j on beat j (auto-align)."""
    b = np.asarray(beats, float)
    W = np.arange(len(b), dtype=float) if W is None else np.asarray(W, float)
    Tb = 60.0 / bpm * mult
    w0 = W[0] - b[0] / (b[1] - b[0]) * (W[1] - W[0])     # where source time 0 lands, in beats
    t0 = w0 * Tb
    dst = (W * Tb - t0) * SR                            # where each beat must land in the output
    src = b * SR
    slope = (W[-1] - W[-2]) / (b[-1] - b[-2])
    n_out = int(np.ceil((n_in / SR - b[-1]) * slope * Tb * SR + dst[-1])) + 1
    return Tb, t0, src, dst, max(1, n_out)


def timed(beats, tspec):
    """(beat times in the cut source, W) for a timing spec, with beats a cut removed entirely dropped
    (their audio is gone; the clips that start there play from the next beat)."""
    spec = timing.parse(tspec)
    W = timing.positions(beats, spec)
    bt = timing.cut_map(spec.get("c", []))(beats) if spec else np.asarray(beats, float)
    keep = [0]
    for i in range(1, len(bt)):
        if bt[i] - bt[keep[-1]] > 1e-5 and W[i] - W[keep[-1]] > 1e-6:
            keep.append(i)
    return bt[keep], W[keep]


def _stretch_array(src, n_in, dst):
    """Per-input-sample stretch factor (>1 = faster) so beat j..j+1 fills exactly one target beat."""
    f = np.empty(n_in)
    idx = np.clip(np.round(src).astype(np.int64), 0, n_in)
    ratios = np.diff(src) / np.diff(dst)
    f[: idx[0]] = ratios[0]
    for j in range(len(ratios)):
        f[idx[j]: idx[j + 1]] = ratios[j]
    f[idx[-1]:] = ratios[-1]
    return f


def _pilot_positions(f, n_in, src, semis, stem_opts, n_expect):
    """Stretch a click track with the same settings; return measured output position of each click."""
    import pedalboard
    pilot = np.zeros(n_in, np.float32)
    idx = np.round(src).astype(np.int64)
    ok = (idx >= 0) & (idx < n_in - 8)
    click = np.array([1.0, 0.8, 0.5, 0.25], np.float32)
    for i in idx[ok]:
        pilot[i: i + 4] += click
    out = pedalboard.time_stretch(pilot[None], SR, stretch_factor=f, pitch_shift_in_semitones=semis,
                                  high_quality=True, **stem_opts)[0]
    e = np.convolve(out.astype(np.float64) ** 2, np.ones(32) / 32, mode="same")
    return out, e, ok


def _measure(env, expect, ok):
    """Where did each pilot click actually land? Sub-sample energy peak, nan if not found.

    R3's internal drift can grow past 50 ms, so the search follows it: each window is centred on
    where the previous click landed (expected + current drift) and spans +-30 % of a beat - wide
    enough for any drift, narrow enough that it can only ever contain one click."""
    res = np.full(len(expect), np.nan)
    n = len(env)
    beat = float(np.median(np.diff(expect))) if len(expect) > 1 else SR * 0.5
    win = int(np.clip(0.3 * beat, 0.02 * SR, 0.3 * SR))
    peaks = [env[int(x)] for x, o in zip(expect, ok) if o and 0 <= int(x) < n]
    floor = 0.02 * float(np.median(peaks)) if peaks else 0.0
    drift = 0.0
    for j, x in enumerate(expect):
        if not ok[j]:
            continue
        c = x + drift
        a, b = int(c - win), int(c + win)
        if a < 1 or b >= n - 1:
            continue
        seg = env[a:b]
        k = int(np.argmax(seg))
        if seg[k] <= max(1e-8, floor):
            continue
        i = a + k
        y0, y1, y2 = env[i - 1], env[i], env[i + 1]
        den = y0 - 2 * y1 + y2
        res[j] = i + (0.5 * (y0 - y2) / den if den != 0 else 0.0)
        drift = res[j] - x
    return res


def render(y, beats, bpm, mult=1.0, semis=0.0, stem="other", quality="hq", tspec=""):
    """y: (2, n) source stem. tspec: timing.spec_for() ("" = auto-align). Returns (out (2, m), t0 s, method)."""
    import pedalboard
    spec = timing.parse(tspec) if tspec else None
    if spec and abs(float(spec.get("s", 1.0)) - 1.0) > 1e-9:
        # screwed (project.screw, chopped & screwed): the song as it plays unscrewed, then slowed down like a
        # turntable - plain resampling, so time and pitch drop together with no stretching of their own
        s = float(spec["s"])
        base = {k: v for k, v in spec.items() if k != "s"}
        out, t0, method = render(y, beats, bpm / s, mult, semis - 12.0 * np.log2(s), stem, quality,
                                 json.dumps(base, sort_keys=True, separators=(",", ":")))
        n = int(np.ceil(out.shape[1] / s))
        return resample_map(out, np.array([0.0, float(out.shape[1])]), np.array([0.0, out.shape[1] / s]), n), t0 / s, method + "+screw"
    if tspec:
        cuts = timing.parse(tspec).get("c", [])
        if cuts:
            y = timing.apply_cuts(y, cuts, SR)
        beats, W = timed(beats, tspec)
    else:
        W = None
    n_in = y.shape[1]
    Tb, t0, src, dst, n_out = _geometry(beats, n_in, bpm, mult, W)
    ratios = np.diff(src) / np.diff(dst)

    if np.max(np.abs(ratios - 1)) < VARISPEED_MAX and abs(semis) < 0.005:
        # pristine path: resample along the beat map (pitch moves by the tiny tempo change only)
        return resample_map(y, src, dst, n_out), t0, "varispeed"

    f = _stretch_array(src, n_in, dst)
    opts = dict(transient_mode="crisp" if stem == "drums" else "mixed",
                preserve_formants=(stem == "vocals"))
    if quality == "fast":
        out = pedalboard.time_stretch(y, SR, stretch_factor=f, pitch_shift_in_semitones=semis,
                                      high_quality=False, **opts)
        # R2 keeps time well; still correct the (small) start offset with the pilot-free map
        return _fit_len(out, n_out), t0, "r2"

    out = pedalboard.time_stretch(y, SR, stretch_factor=f, pitch_shift_in_semitones=semis,
                                  high_quality=True, **opts)
    _, env, ok = _pilot_positions(f, n_in, src, semis, opts, n_out)
    # the pilot's click reference: its own energy peak offset in the unprocessed signal (~1.5 samples)
    ref = 1.5
    meas = _measure(env, dst + ref, ok) - ref
    good = ok & np.isfinite(meas)
    if good.sum() < 4:
        return _fit_len(out, n_out), t0, "r3-uncorrected"
    # reject outliers (missed detections) against the LOCAL trend of the error. (A global median threw
    # away every correct measurement once R3's drift had grown past 30 ms - the rest of the song then
    # went uncorrected, e.g. a drum render 45 ms early for its last minute.)
    from scipy.ndimage import median_filter
    err = meas[good] - dst[good]
    trend = median_filter(err, size=9, mode="nearest")
    keep = np.abs(err - trend) < 0.03 * SR
    d_pts, m_pts = dst[good][keep], meas[good][keep]
    fixed = resample_map(out, m_pts, d_pts, n_out)
    return fixed, t0, "r3"


def _fit_len(y, n):
    if y.shape[1] >= n:
        return np.ascontiguousarray(y[:, :n])
    return np.pad(y, ((0, 0), (0, n - y.shape[1])))


# ---------------------------------------------------------------- cache + process pool

# bump when the warp algorithm changes: older renders are ignored and cleaned up
RENDER_VERSION = "v4"      # v4: varispeed only within 1 cent of the original pitch (was 6 cents)
# v3: outliers judged against the local drift trend (v2 stopped correcting past 30 ms)
# v2: pilot tracking follows R3 drift of any size (v1 lost beats past 45 ms)


def cache_key(stem, bpm, mult, semis, quality, tspec=""):
    return f"{stem}_{bpm:.4f}_{mult:g}_{semis:+.3f}_{quality}_{RENDER_VERSION}{timing.spec_key(tspec)}"


def purge_old_versions():
    """Delete renders made by older warp versions (skips files another process still has open)."""
    freed = 0
    for p in list(CACHE.glob("*/*.npy")) + list(CACHE.glob("*/clips/*.npy")):
        if ("_" + RENDER_VERSION) in p.stem or (p.parent.name == "clips" and p.stem.startswith(RENDER_VERSION)):
            continue
        try:
            size = p.stat().st_size
            p.unlink()
            p.with_suffix(".json").unlink(missing_ok=True)
            freed += size
        except OSError:
            pass
    return freed


def stems_rev(sid):
    """'' or '_rN': bumped when a song's stems are re-separated, so renders of the old stems are never reused."""
    try:
        r = int(json.loads((song_path(sid) / "song.json").read_text(encoding="utf-8")).get("stems_rev", 0))
    except Exception:
        r = 0
    return f"_r{r}" if r else ""


def cache_path(sid, key):
    return CACHE / sid / f"{key}{stems_rev(sid)}.npy"


def _worker(sid, stem, bpm, mult, semis, quality, tspec=""):
    """Runs in a separate process: load stem + beats from disk, render, write .npy + .json."""
    from . import audio_io
    from . import variant
    info = json.loads((song_path(sid) / "song.json").read_text(encoding="utf-8"))
    y = audio_io.load(variant.stem_file(sid, stem, tspec), SR) / variant.scale_for(info, tspec)
    out, t0, method = render(y, variant.beats_for(info, tspec), bpm, mult, semis, stem, quality, tspec)
    key = cache_key(stem, bpm, mult, semis, quality, tspec)
    p = cache_path(sid, key)
    audio_io.save_npy_atomic(p, out.astype(np.float32))
    p.with_suffix(".json").write_text(json.dumps(dict(t0=t0, method=method, n=int(out.shape[1]))))
    return key


def kill_pool():
    """Stop all worker processes now (used on shutdown)."""
    global _pool
    p, _pool = _pool, None
    if p is not None:
        p.kill()


def _get_pool():
    """Worker processes shared by warps and clip renders; running tasks can be aborted (procpool.Pool)."""
    global _pool
    with _pool_lock:
        if _pool is None:
            _pool = procpool.Pool(max(2, min(12, (os.cpu_count() or 4) - 4)),
                                  preload=("mashlab.warp", "mashlab.draft", "mashlab.clipfx", "pedalboard", "scipy.ndimage"))
        return _pool


def discard_partial(sid, stem, bpm, mult, semis, quality="hq", tspec=""):
    """A warp stopped mid-write leaves its temp file behind; remove it (skipped if a newer render of the
    same stem is writing it right now - Windows won't delete an open file)."""
    final = cache_path(sid, cache_key(stem, bpm, mult, semis, quality, tspec))
    for f in final.parent.glob(final.stem + ".*tmp.npy"):
        try:
            f.unlink()
        except OSError:
            pass


def is_cached(sid, stem, bpm, mult, semis, quality="hq", tspec=""):
    p = cache_path(sid, cache_key(stem, bpm, mult, semis, quality, tspec))
    return p.exists() and p.with_suffix(".json").exists()


def render_many(reqs, progress=lambda f, m="": None):
    """reqs: list of (sid, stem, bpm, mult, semis, quality). Renders missing ones in parallel."""
    todo = [r for r in dict.fromkeys(reqs) if not is_cached(*r)]
    if not todo:
        return 0
    pool = _get_pool()
    futs = {pool.submit(_worker, *r): r for r in todo}
    done = 0
    progress(0.0, f"warping {len(todo)} stems")
    for fut in as_completed(futs):
        fut.result()
        done += 1
        progress(done / len(todo), f"warped {done}/{len(todo)} stems")
    return len(todo)


def open_render(sid, stem, bpm, mult, semis, quality="hq", tspec=""):
    """Memory-mapped rendered buffer + t0, or None if not rendered yet."""
    key = (sid, cache_key(stem, bpm, mult, semis, quality, tspec))
    with _open_lock:
        if key in _open:
            return _open[key]
    p = cache_path(sid, key[1])
    if not (p.exists() and p.with_suffix(".json").exists()):
        return None
    try:
        meta = json.loads(p.with_suffix(".json").read_text())
        arr = np.load(p, mmap_mode="r")
    except (OSError, ValueError):                         # still being written: not ready yet
        return None
    val = (arr, meta["t0"])
    with _open_lock:
        _open[key] = val
        if len(_open) > 96:
            _open.pop(next(iter(_open)))
    return val


def drop_song(sid):
    with _open_lock:
        for k in [k for k in _open if k[0] == sid]:
            del _open[k]
    shutil.rmtree(CACHE / sid, ignore_errors=True)


def prune_cache(max_gb=20):
    """Delete the least recently used warped stems once the cache passes max_gb. Files still open
    (being played, or mapped by another Mashup Lab window) are skipped - Windows can't delete them."""
    try:
        files = sorted(CACHE.glob("*/*.npy"), key=lambda p: p.stat().st_atime)
        total = sum(p.stat().st_size for p in files)
    except OSError:
        return
    open_paths = set()
    with _open_lock:
        for (sid, key) in _open:
            open_paths.add(cache_path(sid, key))
    for p in files:
        if total <= max_gb * 1e9:
            break
        if p in open_paths:
            continue
        try:
            size = p.stat().st_size
            p.unlink()
        except OSError:
            continue                          # in use: keep it (and its .json) for now
        total -= size
        p.with_suffix(".json").unlink(missing_ok=True)
