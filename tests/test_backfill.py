"""FEAT-005 — analyze.ensure_beat_content back-fill (Verification 7).

Older analyses predate the per-beat `beat_content` list. The helper re-derives it from the STORED beat grid
plus the drums stem's onset energy — no re-separation, no re-detection — and writes ONLY that one field back
into song.json (write-then-rename). When the grid or the drums stem is missing it must DEFER (signal 'needs a
full re-analyze') and write NOTHING.

The drums stem is provided via a monkeypatched library.load_stem returning a synthetic (2, n) pulse train, so
the test is CPU-only and never touches a real separator/FLAC. library.read / library.write do real disk I/O
inside the data_sandbox so the write-then-rename path is exercised for real.
"""
import json

import numpy as np
import pytest

import mashlab.analyze as analyze
import mashlab.library as library
from mashlab.config import SR


SID = "sBACKFILL1"


@pytest.fixture
def song_store(data_sandbox, monkeypatch):
    """Point library's module-level SONGS at the sandbox (library captured it via `from .config import
    SONGS`, so monkeypatching config alone isn't enough) and clear the in-memory stem cache."""
    monkeypatch.setattr(library, "SONGS", data_sandbox["SONGS"], raising=True)
    library._stem_cache.clear()
    return data_sandbox


def _write_song(info):
    d = library.song_dir(SID)
    (d / "stems").mkdir(parents=True, exist_ok=True)
    library.write(SID, info)
    return d


def _grid_info(n_beats=24, period=0.5):
    """A steady 120 BPM grid, 24 beats, first/last framing the whole thing. No beat_content yet."""
    beats = [round(period * i, 5) for i in range(n_beats)]
    return {
        "id": SID, "title": "Backfill Test", "bpm": 120.0, "bpb": 4,
        "beats": beats, "first_beat": 0, "last_beat": n_beats - 1,
        "duration": beats[-1] + period, "key": {"camelot": "8B"}, "separation": "best",
    }


def _pulse_drums(info):
    """A (2, n) float32 drums stem with a sharp snare-band burst on every beat (so body_env + beat_content
    find a clear pulse)."""
    beats = info["beats"]
    n = int((beats[-1] + 1.0) * SR)
    y = np.zeros(n, np.float32)
    rng = np.random.default_rng(0)
    for t in beats:
        s = int(t * SR)
        w = int(0.04 * SR)
        if s + w <= n:
            # a short 500 Hz-ish decaying burst sits inside body_env's 150-1500 Hz band
            k = np.arange(w)
            burst = np.sin(2 * np.pi * 500 * k / SR) * np.exp(-k / (0.01 * SR))
            y[s:s + w] += burst.astype(np.float32)
    y += 0.001 * rng.standard_normal(n).astype(np.float32)
    return np.stack([y, y])


@pytest.fixture
def stub_drums(monkeypatch):
    """Install a load_stem that returns the synthetic drums for SID / 'drums'; raise for anything else."""
    store = {}

    def install(stem):
        store["stem"] = stem

    def fake_load_stem(sid, s, tspec=""):
        if sid == SID and s == "drums" and store.get("stem") is not None:
            return store["stem"]
        raise FileNotFoundError(f"no stem {sid}/{s}")

    monkeypatch.setattr(library, "load_stem", fake_load_stem, raising=True)
    return install


def test_backfill_populates_beat_content(song_store, stub_drums):
    info = _grid_info()
    _write_song(info)
    stub_drums(_pulse_drums(info))

    assert library.read(SID).get("beat_content") is None            # precondition: field absent

    res = analyze.ensure_beat_content(SID)
    assert res["backfilled"] is True
    assert res["id"] == SID

    saved = library.read(SID)
    bc = saved["beat_content"]
    assert bc is not None
    assert len(bc) == len(info["beats"])                            # one entry per beat
    assert all(isinstance(v, int) for v in bc)                      # ints, 0..100 scale
    assert all(0 <= v <= 100 for v in bc)
    assert max(bc) > 12                                             # a real pulse scores above the suppression floor
    # ONLY beat_content was added — every other field is untouched
    for k, v in info.items():
        assert saved[k] == v


def test_backfill_accepts_song_dir_and_info(song_store, stub_drums):
    info = _grid_info()
    d = _write_song(info)
    stub_drums(_pulse_drums(info))
    # the design-stated (song_dir, info) call shape
    res = analyze.ensure_beat_content(d, info)
    assert res["backfilled"] is True
    assert len(library.read(SID)["beat_content"]) == len(info["beats"])


def test_backfill_defers_when_grid_missing(song_store, stub_drums):
    info = _grid_info()
    del info["beats"]                                               # no stored grid
    _write_song(info)
    stub_drums(_pulse_drums(_grid_info()))

    res = analyze.ensure_beat_content(SID)
    assert res["defer"] is True
    assert res["backfilled"] is False
    # nothing written: beat_content stays absent
    assert library.read(SID).get("beat_content") is None


def test_backfill_defers_when_first_last_missing(song_store, stub_drums):
    info = _grid_info()
    del info["last_beat"]
    _write_song(info)
    stub_drums(_pulse_drums(_grid_info()))
    res = analyze.ensure_beat_content(SID)
    assert res["defer"] is True
    assert library.read(SID).get("beat_content") is None


def test_backfill_defers_when_drums_unloadable(song_store, stub_drums):
    info = _grid_info()
    _write_song(info)
    # stub_drums NOT armed with a stem -> load_stem raises -> defer, write nothing
    res = analyze.ensure_beat_content(SID)
    assert res["defer"] is True
    assert res["backfilled"] is False
    assert library.read(SID).get("beat_content") is None
