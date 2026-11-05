"""Lyrical layer (design A.1): transcribe a song's clean vocal stem into lyrics.json with the pinned
transcript-confidence authority `asr_conf`.

The ASR model itself lives behind the backend interface (mashlab.backends.asr, FEAT-001): this module loads
the clean vocal stem, routes the transcription through the chosen AsrBackend (local faster-whisper by
default), measures `voiced_coverage` against the song's own pYIN voicing (mash.npz f0, mashfeat), computes the
pinned `quality` block, and writes lyrics.json with the project's write-then-rename discipline.

`asr_conf` is a single, library-independent, pinned formula (design A.1 / MEDIUM-1) — the only authority for
transcript trust. `judgeable` (concept.py) and the A.5 neutral fallback both read it, so they never disagree.
"""
import json
import os

import numpy as np

from . import library, mashfeat
from .config import SR

VERSION = 1

# --- pinned asr_conf constants (design A.1, declared FIXED — tunable only here, never per-library) ----------
_LOGPROB_OFFSET = 0.6          # sigmoid((mean_logprob + 0.6) / 0.4): logprob -0.6 -> 0.5
_LOGPROB_SCALE = 0.4
_VOICED_SAT = 0.4              # voiced_coverage saturates once ASR covers >= 40% of voiced time
# usable=false thresholds (instrumental / garbled)
_USABLE_MIN_COVERAGE = 0.15
_USABLE_MIN_LOGPROB = -1.1

_FPS = 100.0                   # mash.npz f0 frame rate (mashfeat stores midi at 100 fps)


def _clip01(x):
    return float(np.clip(x, 0.0, 1.0))


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _duration_weighted(segments, field):
    """Segment-duration-weighted mean of a per-segment faster-whisper field (avg_logprob / no_speech_prob).
    Falls back to a plain mean when every segment has zero/absent duration, and to 0 when there are none."""
    num = 0.0
    den = 0.0
    plain = []
    for s in segments or []:
        val = s.get(field)
        if val is None:
            continue
        val = float(val)
        plain.append(val)
        dur = float(s.get("t1", 0.0)) - float(s.get("t0", 0.0))
        if dur > 0:
            num += val * dur
            den += dur
    if den > 0:
        return num / den
    if plain:
        return sum(plain) / len(plain)
    return 0.0


def voiced_coverage(segments, mash):
    """(ASR-covered voiced time) / (total voiced time from mash.npz f0 voicing, design A.1 / F4).

    Total voiced time = count of voiced f0 frames / fps. ASR-covered voiced time = voiced frames that also
    fall inside an ASR segment [t0, t1). Returns 0.0 when there is no voiced time (instrumental) or no mash."""
    if mash is None or "f0" not in mash:
        return 0.0
    f0 = np.asarray(mash["f0"])
    voiced = np.isfinite(f0)
    total = int(voiced.sum())
    if total == 0:
        return 0.0
    n = len(f0)
    covered = np.zeros(n, dtype=bool)
    for s in segments or []:
        a = int(round(float(s.get("t0", 0.0)) * _FPS))
        b = int(round(float(s.get("t1", 0.0)) * _FPS))
        a = max(0, min(n, a))
        b = max(0, min(n, b))
        if b > a:
            covered[a:b] = True
    covered_voiced = int((voiced & covered).sum())
    return covered_voiced / total


def quality_block(segments, mash):
    """The pinned lyrics.json `quality` block (design A.1). All terms clipped; `asr_conf` is the product of
    the three clipped terms VERBATIM."""
    mean_logprob = _duration_weighted(segments, "avg_logprob")
    no_speech = _duration_weighted(segments, "no_speech_prob")
    cov = voiced_coverage(segments, mash)
    asr_conf = (_clip01(_sigmoid((mean_logprob + _LOGPROB_OFFSET) / _LOGPROB_SCALE))
                * (1.0 - _clip01(no_speech))
                * _clip01(cov / _VOICED_SAT))
    usable = not (cov < _USABLE_MIN_COVERAGE or mean_logprob < _USABLE_MIN_LOGPROB)
    return dict(mean_logprob=round(float(mean_logprob), 4),
                voiced_coverage=round(float(cov), 4),
                asr_conf=round(float(asr_conf), 4),
                usable=bool(usable))


# ----------------------------------------------------------------- cache key / staleness

def _vocal_stem_path(sid):
    return library.song_dir(sid) / "stems" / "vocals.flac"


def cache_key(sid, backend):
    """Keyed on vocal-stem mtime + separation tag + ASR backend/model version (design A.1 / F15). Re-separating
    to Ultra/SW (a new separation tag + a new stem mtime) therefore re-runs ASR."""
    p = _vocal_stem_path(sid)
    mt = int(p.stat().st_mtime) if p.exists() else 0
    info = library.read(sid) or {}
    sep = info.get("separation", "?")
    bk = getattr(backend, "name", backend.__class__.__name__) if backend is not None else "?"
    return f"{VERSION}:{bk}:{sep}:{mt}"


def _path(sid):
    return library.song_dir(sid) / "lyrics.json"


def load(sid, backend=None):
    """Read a fresh lyrics.json for `sid`, or None when missing/stale. When `backend` is given the ASR-version
    part of the cache key is checked so a backend change invalidates the cache."""
    p = _path(sid)
    if not p.exists():
        return None
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return None
    if d.get("version") != VERSION:
        return None
    if backend is not None and d.get("cache_key") != cache_key(sid, backend):
        return None
    return d


def _write(sid, data):
    """Atomic write-then-rename (design F4): a partial/interrupted write never presents as complete."""
    p = _path(sid)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, p)
    return p


def build(sid, backend, mash=None):
    """Transcribe song `sid`'s clean vocal stem through `backend` (an AsrBackend) and assemble lyrics.json.

    GPU is owned by the backend; we unload() it afterwards (design A.1 / E.2). `voiced_coverage` is measured
    against the song's own pYIN voicing (mash.npz). Returns the lyrics.json dict (also written to disk)."""
    if mash is None:
        mash = mashfeat.load(library.song_dir(sid))
    y = library.load_stem(sid, "vocals")                 # (2, n) float32 at SR, true level
    audio = np.asarray(y, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=0)
    try:
        raw = backend.transcribe(audio, SR)
    finally:
        try:
            backend.unload()
        except Exception:
            pass
    segments = raw.get("segments", []) or []
    data = dict(version=VERSION,
                backend=raw.get("backend", getattr(backend, "name", "?")),
                lang=raw.get("lang"),
                segments=segments,
                quality=quality_block(segments, mash),
                cache_key=cache_key(sid, backend))
    _write(sid, data)
    return data


def ensure(sid, backend_factory=None):
    """Return a fresh lyrics.json for `sid`, transcribing on a cache miss. `backend_factory` is called with no
    args to build the AsrBackend only when a (GPU) transcription is actually needed (lazy); it defaults to the
    configured local ASR backend. Routes GPU through the backend and unloads after (design A.1)."""
    if backend_factory is None:
        from .backends.asr import local_asr
        backend_factory = local_asr
    # cheap staleness check without constructing the (heavy) backend: compare only the version+sep+mtime parts
    cached = load(sid)
    if cached is not None:
        backend = backend_factory()
        if cached.get("cache_key") == cache_key(sid, backend):
            try:
                backend.unload()
            except Exception:
                pass
            return cached
        return build(sid, backend)
    return build(sid, backend_factory())
