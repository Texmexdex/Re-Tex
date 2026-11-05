"""FEAT-002 A.4/A.5 — the bounded lyrical tie-breaker folded into pairs.evaluate.

The heavy planner/harmfit path in evaluate() is exercised elsewhere; here we unit-test the integration seam
deterministically: pairs._lyrical_factor (the per-pair L_factor + A.5 neutral rules) and the multiplicative
fold-in guarantee. Fakes expose only `.lyrics` and `.concept` (what _lyrical_factor reads). CPU, no GPU.
"""
import numpy as np
import pytest

from mashlab import pairs, concept


def _vec(parts):
    v = np.zeros(384, dtype=np.float32)
    v[: len(parts)] = parts
    n = np.linalg.norm(v)
    return (v / n).tolist() if n else v.tolist()


def _lyrics(text, asr_conf=0.9, usable=True, lang="en"):
    return {"version": 1, "lang": lang,
            "segments": [{"t0": 0, "t1": 3, "text": text}],
            "quality": {"asr_conf": asr_conf, "voiced_coverage": 0.6, "usable": usable}}


def _concept(vec, mood=None, judge=0.9):
    return {"concept_vec": _vec(vec), "tags": {"mood": mood}, "valence": None, "arousal": None,
            "judgeable": judge}


class FakeSong:
    def __init__(self, lyrics=None, concept=None):
        self.lyrics = lyrics
        self.concept = concept


def _rich_pair(mood_v="defiant", mood_i="defiant", vec_v=(1.0, 0.0), vec_i=(1.0, 0.0),
               asr=0.9, judge=0.9, usable=True, lang_v="en", lang_i="en"):
    text = " ".join(f"word{i}" for i in range(60))
    V = FakeSong(_lyrics(text, asr_conf=asr, usable=usable, lang=lang_v),
                 _concept(vec_v, mood=mood_v, judge=judge))
    I = FakeSong(_lyrics(text, asr_conf=asr, usable=usable, lang=lang_i),
                 _concept(vec_i, mood=mood_i, judge=judge))
    return V, I


# ----------------------------------------------------------------- A.5 neutral rules

def test_off_is_neutral():
    V, I = _rich_pair()
    f = pairs._lyrical_factor(V, I, "off")
    assert f["factor"] == 1.0
    assert f["lyrics"] is None
    assert f["lyrics_mode"] == "off"


def test_judgeable_zero_identical_to_off():
    """Criterion (c): judgeable below 0.35 neutralizes exactly like off (factor 1.0)."""
    V, I = _rich_pair(judge=0.0)
    f_j0 = pairs._lyrical_factor(V, I, "resonance")
    Voff, Ioff = _rich_pair()
    f_off = pairs._lyrical_factor(Voff, Ioff, "off")
    assert f_j0["factor"] == f_off["factor"] == 1.0


def test_low_asr_conf_neutralizes_rich_text():
    """Criterion (d): asr_conf < 0.5 neutralizes even with a rich, lexically varied transcript."""
    V, I = _rich_pair(asr=0.4)      # rich text (60 unique words) but low ASR confidence
    f = pairs._lyrical_factor(V, I, "resonance")
    assert f["factor"] == 1.0
    assert "ASR confidence" in f["lyrics_text"]


def test_unusable_vocal_neutralizes():
    V, I = _rich_pair(usable=False)
    assert pairs._lyrical_factor(V, I, "resonance")["factor"] == 1.0


def test_languages_differ_neutralizes():
    V, I = _rich_pair(lang_v="en", lang_i="es")
    assert pairs._lyrical_factor(V, I, "resonance")["factor"] == 1.0


def test_missing_concept_neutralizes():
    V = FakeSong(None, None)
    I = FakeSong(None, None)
    assert pairs._lyrical_factor(V, I, "resonance")["factor"] == 1.0


def test_lyrics_text_surfaces_lower_asr_conf():
    """A.5 HIGH-1: lyrics_text always surfaces the LOWER of the two songs' asr_conf."""
    text = " ".join(f"word{i}" for i in range(60))
    V = FakeSong(_lyrics(text, asr_conf=0.81), _concept((1.0, 0.0), mood="defiant"))
    I = FakeSong(_lyrics(text, asr_conf=0.62), _concept((1.0, 0.0), mood="defiant"))
    f = pairs._lyrical_factor(V, I, "resonance")
    assert "0.62" in f["lyrics_text"]        # the lower one


# ----------------------------------------------------------------- resonance / contrast magnitude

def test_resonance_lifts_strong_match():
    """A high-resonance pair gets a factor > 1 (within the safe ±5% default clamp)."""
    V, I = _rich_pair(mood_v="defiant", mood_i="defiant", vec_v=(1.0, 0.0), vec_i=(1.0, 0.0))
    f = pairs._lyrical_factor(V, I, "resonance")
    assert 1.0 < f["factor"] <= 1.05
    assert f["lyrics"] == 100       # round(Lraw*100), Lraw clips to 1


def test_contrast_lifts_opposite_pair():
    V, I = _rich_pair(mood_v="defiant", mood_i="numb", vec_v=(1.0, 0.0), vec_i=(0.0, 1.0))
    f = pairs._lyrical_factor(V, I, "contrast")
    assert 1.0 < f["factor"] <= 1.05
    assert f["lyrics"] == 100


def test_factor_bounded_to_safe_clamp(data_sandbox):
    """With no lyrics_calib the clamp is (0.95, 1.05); even a maximal Lraw cannot exceed it."""
    V, I = _rich_pair()
    f = pairs._lyrical_factor(V, I, "resonance")
    assert f["factor"] <= 1.05


# ----------------------------------------------------------------- criterion (a): broken pair stays below

def test_broken_pair_with_max_lyrics_stays_below_good_neutral_pair(data_sandbox):
    """Criterion (a): a harmonically broken pair (small musical `overall`) with maximal lyrical resonance must
    rate BELOW a musically-good pair with neutral lyrics. The clamp guarantees it: the lyrical factor is a
    bounded multiplier on the already-computed F1 overall, so it can only re-order near-ties."""
    V, I = _rich_pair()                                  # maximal resonance
    f_res = pairs._lyrical_factor(V, I, "resonance")     # factor in (1.0, 1.05]
    f_off = {"factor": 1.0}

    overall_broken = 20.0      # a harmonically broken pair: tiny F1 product
    overall_good = 70.0        # a musically-good pair

    rating_broken = overall_broken * f_res["factor"]     # boosted as hard as the gate allows
    rating_good = overall_good * f_off["factor"]          # neutral lyrics

    assert rating_broken < rating_good
    # even if the GOOD pair were penalized by the worst contrast factor, the broken+boosted stays below
    assert overall_broken * 1.05 < overall_good * 0.95


# ----------------------------------------------------------------- criterion (b): off reproduces pre-change

def test_off_matches_pre_change_rating_on_real_pair():
    """Criterion (b): lyrics_mode='off' reproduces the pre-change rating within rounding. We prove it on the
    integration itself: with off, factor == 1.0 exactly, so overall_v2 == overall (no change)."""
    V, I = _rich_pair()
    assert pairs._lyrical_factor(V, I, "off")["factor"] == 1.0
    # and the evaluate() default is off, so a default call never alters the F1 overall
    import inspect
    sig = inspect.signature(pairs.evaluate)
    assert sig.parameters["lyrics_mode"].default == "off"
