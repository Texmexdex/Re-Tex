"""FEAT-005 — the no-regression gate (Verification 7 / A8).

A Phase-2-free project (plain manual arrangement: no automation lane, no Phase-2 desc, pure varispeed clips)
must export BYTE-IDENTICALLY across runs. This is the invariant that lets the manual timeline keep opening
autonomous output and lets every Phase-2 addition stay strictly additive: mixer.bounce is the single export
path and nothing FEAT-002..005 added perturbs it.

The test builds a real (synthetic) song in the data_sandbox, renders its one warp (the pristine varispeed
path — pure numpy resampling, no RNG), bounces the arrangement twice, and asserts the two WAVs are sha256-
equal (and that the null-sum of the two signals is exactly zero).
"""
import hashlib
import json

import numpy as np
import pytest

import mashlab.audio_io as audio_io
import mashlab.library as library
import mashlab.project as project
import mashlab.warp as warp
import mashlab.variant as variant
import mashlab.mixer as mixer
from mashlab.config import SR, STEMS


SID = "sREGRESS01"


@pytest.fixture
def sandbox_paths(data_sandbox, monkeypatch):
    """Redirect the module-level SONGS/CACHE that library + warp captured via `from .config import ...` so
    stems, song.json and the render cache all live in the throwaway sandbox."""
    import mashlab.config as config
    monkeypatch.setattr(library, "SONGS", data_sandbox["SONGS"], raising=True)
    monkeypatch.setattr(config, "SONGS", data_sandbox["SONGS"], raising=True)     # config.song_path: every module's way in
    monkeypatch.setattr(warp, "CACHE", data_sandbox["CACHE"], raising=True)
    data_sandbox["CACHE"].mkdir(parents=True, exist_ok=True)
    library._stem_cache.clear()
    try:
        warp._open.clear()
    except Exception:
        pass
    return data_sandbox


def _make_song(paths):
    """Write a 6-second four-stem song + a steady 120 BPM grid into the sandbox and return its info."""
    d = paths["SONGS"] / SID
    (d / "stems").mkdir(parents=True, exist_ok=True)
    dur = 6.0
    n = int(dur * SR)
    t = np.arange(n) / SR
    rng = np.random.default_rng(1234)
    tones = {"drums": 2.0, "bass": 60.0, "other": 220.0, "vocals": 330.0}
    for s in STEMS:
        y = (0.2 * np.sin(2 * np.pi * tones[s] * t)).astype(np.float32)
        y += (0.001 * rng.standard_normal(n)).astype(np.float32)
        stereo = np.stack([y, y * 0.98])
        audio_io.save(d / "stems" / f"{s}.flac", stereo, SR, subtype="PCM_24")
    beats = [round(0.5 * i, 5) for i in range(12)]        # 120 BPM
    info = {
        "id": SID, "title": "Regression Song", "bpm": 120.0, "bpb": 4,
        "beats": beats, "bars": list(range(0, 12, 4)), "bpb": 4,
        "first_beat": 0, "last_beat": len(beats) - 1, "duration": dur,
        "segments": [[0, len(beats)]], "stable_tempo": True,
        "key": {"tonic": 0, "mode": "maj", "conf": 0.9, "name": "C maj", "camelot": "8B"},
        "stem_scale": 1.0, "separation": "best",
    }
    (d / "song.json").write_text(json.dumps(info), encoding="utf-8")
    return info


def _project(info):
    p = project.empty(name="regression", bpm=info["bpm"], bpb=info["bpb"])
    p["songs"][SID] = {"transpose": 0, "mult": 1.0}
    # one plain clip per stem, native speed / pitch -> the pure varispeed render path (no clipfx)
    for i, stem in enumerate(STEMS):
        tid = p["tracks"][i]["id"]
        p["clips"].append(project.clip(tid, SID, stem, start=0.0, src=0.0, length=8.0))
    return p


def _bounce_to(p, path):
    mixer.bounce(p, str(path))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_phase2_free_project_exports_byte_identical(sandbox_paths, tmp_path):
    info = _make_song(sandbox_paths)
    p = _project(info)

    # render every warp this arrangement needs, in-process (no worker pool), via the pristine varispeed path
    for req in project.render_requests(p):
        warp._worker(*req)
    assert all(warp.open_render(*req) is not None for req in project.render_requests(p))

    out_a = tmp_path / "a.wav"
    out_b = tmp_path / "b.wav"
    sha_a = _bounce_to(p, out_a)
    warp._open.clear()                      # force the 2nd bounce to re-read the cache from disk
    sha_b = _bounce_to(p, out_b)

    assert sha_a == sha_b, "a Phase-2-free export is not byte-identical across runs"

    # and the signals null-sum to exactly zero
    ya = audio_io.load(out_a, SR)
    yb = audio_io.load(out_b, SR)
    assert ya.shape == yb.shape
    assert float(np.abs(ya.astype(np.float64) - yb.astype(np.float64)).max()) == 0.0


def test_bounce_has_no_phase2_fields(sandbox_paths, tmp_path):
    """Sanity: the project we export carries no automation / mashup desc — it is genuinely Phase-2-free."""
    info = _make_song(sandbox_paths)
    p = _project(info)
    assert "mashup" not in p
    assert all("automation" not in t for t in p["tracks"])
