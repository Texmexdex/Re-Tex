"""FEAT-002 A.1 — the pinned asr_conf authority (design A.1 / MEDIUM-1).

All CPU, no GPU, no model: we feed synthetic faster-whisper-shaped segments + a synthetic mash (f0 voicing)
straight into mashlab.lyrics.quality_block and assert the hand-computed asr_conf / usable flags.
"""
import math

import numpy as np
import pytest

from mashlab import lyrics


def _mash(voiced_frames, total_frames):
    """A synthetic mash dict: `voiced_frames` finite f0 frames followed by NaN out to total_frames (100 fps)."""
    f0 = np.full(total_frames, np.nan, dtype=np.float32)
    f0[:voiced_frames] = 60.0
    return {"f0": f0}


def _seg(t0, t1, logp, nosp):
    return dict(t0=t0, t1=t1, text="x " * 5, avg_logprob=logp, no_speech_prob=nosp, words=[])


def _expected(mean_logprob, no_speech, cov):
    sig = 1.0 / (1.0 + math.exp(-(mean_logprob + 0.6) / 0.4))
    return (min(max(sig, 0), 1)) * (1 - min(max(no_speech, 0), 1)) * min(max(cov / 0.4, 0), 1)


@pytest.mark.parametrize("logp,approx", [(-0.6, 0.5), (-0.2, 0.73), (-1.1, 0.22)])
def test_asr_conf_logprob_anchors(logp, approx):
    """The sigmoid term alone (no_speech=0, voiced_coverage saturated) hits ~0.5/0.73/0.22 at -0.6/-0.2/-1.1."""
    # one 10 s segment fully covering 10 s of voiced time (cov = 1.0 -> coverage term saturates at 1)
    mash = _mash(voiced_frames=1000, total_frames=1000)
    segs = [_seg(0.0, 10.0, logp, 0.0)]
    q = lyrics.quality_block(segs, mash)
    assert q["asr_conf"] == pytest.approx(approx, abs=0.02)
    assert q["asr_conf"] == pytest.approx(_expected(logp, 0.0, 1.0), abs=1e-3)


def test_no_speech_drives_to_zero():
    """no_speech ~ 1 collapses asr_conf to ~0 regardless of the logprob."""
    mash = _mash(1000, 1000)
    segs = [_seg(0.0, 10.0, -0.2, 0.99)]
    q = lyrics.quality_block(segs, mash)
    assert q["asr_conf"] < 0.02


def test_voiced_coverage_saturates_at_0p4():
    """The coverage term is clip(voiced_coverage/0.4,0,1): coverage 0.4 and 0.8 give the same asr_conf."""
    # 1000 voiced frames total; cover 400 (cov=0.4) vs 800 (cov=0.8). Segment logprob/no_speech identical.
    q04 = lyrics.quality_block([_seg(0.0, 4.0, -0.2, 0.0)], _mash(1000, 1000))
    q08 = lyrics.quality_block([_seg(0.0, 8.0, -0.2, 0.0)], _mash(1000, 1000))
    assert q04["voiced_coverage"] == pytest.approx(0.4, abs=1e-6)
    assert q08["voiced_coverage"] == pytest.approx(0.8, abs=1e-6)
    assert q04["asr_conf"] == pytest.approx(q08["asr_conf"], abs=1e-6)   # both saturate the coverage term


def test_usable_false_on_low_coverage():
    """voiced_coverage < 0.15 -> instrumental -> usable=false."""
    # cover 100 of 1000 voiced frames -> cov 0.10
    q = lyrics.quality_block([_seg(0.0, 1.0, -0.2, 0.0)], _mash(1000, 1000))
    assert q["voiced_coverage"] == pytest.approx(0.10, abs=1e-6)
    assert q["usable"] is False


def test_usable_false_on_garbled_logprob():
    """mean_logprob < -1.1 -> garbled (screams/heavy processing) -> usable=false, even with full coverage."""
    q = lyrics.quality_block([_seg(0.0, 10.0, -1.2, 0.0)], _mash(1000, 1000))
    assert q["usable"] is False


def test_usable_true_on_clean_vocal():
    """A clean, well-covered vocal is usable."""
    q = lyrics.quality_block([_seg(0.0, 8.0, -0.3, 0.03)], _mash(1000, 1000))
    assert q["usable"] is True


def test_duration_weighted_means():
    """mean_logprob / no_speech are segment-duration-weighted, not plain means."""
    # 9 s @ -0.2 and 1 s @ -2.0 -> weighted mean = (9*-0.2 + 1*-2.0)/10 = -0.38
    segs = [_seg(0.0, 9.0, -0.2, 0.0), _seg(9.0, 10.0, -2.0, 0.0)]
    q = lyrics.quality_block(segs, _mash(1000, 1000))
    assert q["mean_logprob"] == pytest.approx(-0.38, abs=1e-6)


def test_build_writes_lyrics_json(data_sandbox, song_dict, stub_asr, monkeypatch):
    """lyrics.build routes through a stub AsrBackend (no GPU), writes lyrics.json atomically, and unloads."""
    import json
    import mashlab.library as library
    import mashlab.lyrics as lyr

    sid = song_dict["id"]
    d = library.song_dir(sid)
    (d / "stems").mkdir(parents=True, exist_ok=True)
    library.write(sid, dict(song_dict, title="Test Track", status="ready", separation="ultra"))

    # avoid any audio IO: feed a tiny mono stem and a tiny mash directly
    monkeypatch.setattr(library, "load_stem", lambda s, stem, tspec="": np.zeros((2, 100), np.float32))
    mash = {"f0": np.concatenate([np.full(500, 60.0, np.float32), np.full(500, np.nan, np.float32)])}

    out = lyr.build(sid, stub_asr, mash=mash)
    assert stub_asr.unloaded is True
    p = d / "lyrics.json"
    assert p.exists()
    on_disk = json.loads(p.read_text(encoding="utf-8"))
    assert on_disk["version"] == lyr.VERSION
    assert "quality" in on_disk and set(on_disk["quality"]) >= {"asr_conf", "voiced_coverage", "usable"}
    assert on_disk["cache_key"] == out["cache_key"]
    # no leftover temp file under the final name's directory
    assert not list(d.glob("lyrics.*.tmp"))
