"""FEAT-002 A.2/A.3 — judgeable, mood graph, the lyrics_params() gate, and the NRC-VAD integrity gate.

All CPU: judgeable/mood/gate are pure math on dicts; the lexicon test writes a synthetic TSV into the
data_sandbox models dir (no download).
"""
import numpy as np
import pytest

from mashlab import concept


# ----------------------------------------------------------------- judgeable (A.2)

def _lyrics(text, asr_conf=0.9):
    return {"version": 1, "segments": [{"t0": 0, "t1": 3, "text": text}],
            "quality": {"asr_conf": asr_conf, "voiced_coverage": 0.6, "usable": True}}


def test_judgeable_rich_transcript():
    text = " ".join(f"word{i}" for i in range(60))     # 60 distinct words -> token term saturates
    j = concept.judgeable(_lyrics(text, asr_conf=0.9))
    assert j == pytest.approx(0.9, abs=1e-6)            # 0.9 * 1 * 1


def test_judgeable_empty_collapses():
    assert concept.judgeable(_lyrics("", asr_conf=0.9)) == pytest.approx(0.0, abs=1e-9)


def test_judgeable_all_repeated_collapses():
    # 30 copies of one word: n_unique=1, repeat_ratio ~ 0.967 -> judgeable near 0
    j = concept.judgeable(_lyrics("na " * 30, asr_conf=0.9))
    assert j < 0.01


def test_judgeable_low_asr_conf_scales_down():
    text = " ".join(f"word{i}" for i in range(60))
    assert concept.judgeable(_lyrics(text, asr_conf=0.3)) == pytest.approx(0.3, abs=1e-6)


# ----------------------------------------------------------------- mood graph (A.3, NIT-3)

def test_moods_opposite_both_orderings_defiant_numb():
    assert concept.moods_opposite("defiant", "numb") is True
    assert concept.moods_opposite("numb", "defiant") is True


def test_mood_bonus_values():
    assert concept.mood_bonus("defiant", "defiant") == pytest.approx(0.15)
    assert concept.mood_bonus("defiant", "numb") == pytest.approx(-0.10)
    assert concept.mood_bonus("defiant", "euphoric") == pytest.approx(0.0)


# ----------------------------------------------------------------- lyrics_params gate (A.3, Verification 2d)

def test_lyrics_params_safe_default_when_absent():
    alpha, clamp, (lo, hi) = concept.lyrics_params({"lyrics_calib": None})
    assert alpha == pytest.approx(0.06)
    assert clamp == (0.95, 1.05)
    assert (lo, hi) == (0.15, 0.65)


def test_lyrics_params_safe_at_n_pairs_40():
    calib = {"lyrics_calib": {"lo": 0.14, "hi": 0.63, "n_pairs": 40}}
    alpha, clamp, (lo, hi) = concept.lyrics_params(calib)
    assert alpha == pytest.approx(0.06)
    assert clamp == (0.95, 1.05)
    assert (lo, hi) == (0.15, 0.65)


def test_lyrics_params_strong_at_n_pairs_60():
    calib = {"lyrics_calib": {"lo": 0.14, "hi": 0.63, "n_pairs": 60}}
    alpha, clamp, (lo, hi) = concept.lyrics_params(calib)
    assert alpha == pytest.approx(0.12)
    assert clamp == (0.90, 1.12)
    assert (lo, hi) == (0.14, 0.63)       # the RECORDED endpoints


def test_lyrics_params_narrow_span_stays_safe():
    # n_pairs OK but span 0.11 < 0.15 -> stays on the safe default (MEDIUM-6)
    calib = {"lyrics_calib": {"lo": 0.40, "hi": 0.51, "n_pairs": 200}}
    alpha, clamp, (lo, hi) = concept.lyrics_params(calib)
    assert alpha == pytest.approx(0.06)
    assert (lo, hi) == (0.15, 0.65)


def test_lyrics_params_reads_appcfg(data_sandbox):
    """With no config.json, appcfg.lyrics_calib() is None -> the safe default."""
    alpha, clamp, _ = concept.lyrics_params()
    assert alpha == pytest.approx(0.06)
    assert clamp == (0.95, 1.05)


# ----------------------------------------------------------------- resonance / contrast (A.3)

def _concept(vec, mood=None, valence=None, arousal=None):
    v = np.zeros(384, dtype=np.float32)
    v[: len(vec)] = vec
    n = np.linalg.norm(v)
    if n > 0:
        v = v / n
    return {"concept_vec": v.tolist(), "tags": {"mood": mood}, "valence": valence, "arousal": arousal}


def test_resonance_high_for_aligned_vectors():
    a = _concept([1.0, 0.0], mood="defiant")
    b = _concept([1.0, 0.0], mood="defiant")
    # cos = 1 -> res_base clips to 1; equal mood +0.15 -> Lraw clips to 1
    assert concept.resonance(a, b) == pytest.approx(1.0, abs=1e-6)


def test_contrast_high_for_orthogonal_opposite():
    a = _concept([1.0, 0.0], mood="defiant")
    b = _concept([0.0, 1.0], mood="numb")
    # cos = 0 -> res_base = clip((0-0.15)/0.5,0,1)=0 ; contrast base (1-0)=1 ; opposite mood +0.15 -> clips 1
    assert concept.contrast(a, b) == pytest.approx(1.0, abs=1e-6)


def test_vad_term_zero_when_absent():
    a = _concept([1.0, 0.0], mood="defiant", valence=None, arousal=None)
    b = _concept([1.0, 0.0], mood="defiant", valence=0.2, arousal=0.9)
    # one side null -> vad term dropped (0), resonance still = 1.0 (cos 1 + equal mood)
    assert concept.resonance(a, b) == pytest.approx(1.0, abs=1e-6)


# ----------------------------------------------------------------- NRC-VAD integrity gate (Verification 1b)

def _write_lexicon(path, n_rows, good=True):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["Word\tValence\tArousal\tDominance"]
    for i in range(n_rows):
        if good:
            lines.append(f"word{i}\t0.5\t0.5\t0.5")
        else:
            lines.append(f"word{i}\t2.0\t0.5\t0.5")   # value out of [0,1]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_valid_lexicon_loads_and_is_used(data_sandbox, monkeypatch):
    concept._reset_vad_cache()
    # pin the expected row count small so the test TSV is cheap
    monkeypatch.setattr(concept.config, "NRC_VAD_ROWS", 3, raising=True)
    monkeypatch.setattr(concept.config, "NRC_VAD_SHA256", None, raising=True)
    _write_lexicon(data_sandbox["MODELS"] / "nrc_vad.tsv", 3, good=True)
    table = concept.load_vad_lexicon(force=True)
    assert table is not None and "word0" in table and table["word0"] == (0.5, 0.5)
    # a transcript whose words are in the lexicon yields a real valence/arousal
    ly = {"segments": [{"text": "word0 word1 word2"}]}
    v, a = concept.mean_vad(ly)
    assert v == pytest.approx(0.5) and a == pytest.approx(0.5)
    concept._reset_vad_cache()


def test_corrupted_lexicon_treated_as_absent(data_sandbox, monkeypatch, caplog):
    import logging
    concept._reset_vad_cache()
    monkeypatch.setattr(concept.config, "NRC_VAD_ROWS", 19971, raising=True)   # file will have the wrong count
    monkeypatch.setattr(concept.config, "NRC_VAD_SHA256", None, raising=True)
    path = data_sandbox["MODELS"] / "nrc_vad.tsv"
    _write_lexicon(path, 5, good=True)        # 5 rows != 19,971 -> gate fails
    with caplog.at_level(logging.WARNING):
        table = concept.load_vad_lexicon(force=True)
    assert table is None
    assert sum("treated as absent" in r.message for r in caplog.records) == 1
    v, a = concept.mean_vad({"segments": [{"text": "word0"}]})
    assert v is None and a is None
    # the file we wrote is the pre-existing one; the loader never writes a partial file under the final name
    assert path.exists()                       # (our fixture wrote it; loader created nothing new)
    concept._reset_vad_cache()


def test_out_of_range_values_treated_as_absent(data_sandbox, monkeypatch):
    concept._reset_vad_cache()
    monkeypatch.setattr(concept.config, "NRC_VAD_ROWS", 3, raising=True)
    monkeypatch.setattr(concept.config, "NRC_VAD_SHA256", None, raising=True)
    _write_lexicon(data_sandbox["MODELS"] / "nrc_vad.tsv", 3, good=False)
    assert concept.load_vad_lexicon(force=True) is None
    concept._reset_vad_cache()
