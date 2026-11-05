"""Unit tests for mashlab.orchestrate (design D) — the CPU-testable surface.

Covers the three pure, deterministic pieces per FEAT-004 acceptance:
  * the D.2 style -> parameter-bundle mapper (exact pinned bundles + unknown-text fallback with numeric cues),
  * the N>2 bed selection (max aggregate lyrics-aware fit, OD-4),
  * the 2-song deterministic near-tie rule (higher judgeable -> more usable bars -> lower id).

No GPU, no model loads, no network: pairs.evaluate is stubbed with synthetic scores.
"""
import pytest

from mashlab import orchestrate


# ------------------------------------------------------------------ D.2 style mapper (exact pinned bundles)

# The exact values pinned in design D.2 — this table is independent of the implementation's own copy, so a
# drift in either is caught.
PINNED = {
    "balanced": dict(vocal_density=0.50, chop=False, tempo_bias=0.50, global_rate=1.00, pitch_down=0,
                     swing=0.00, energy_floor=0.00, swap_vocals=False, add_sweeps=False, add_fills=False,
                     lyrics_mode="resonance"),
    "remix": dict(vocal_density=0.55, chop=False, tempo_bias=0.50, global_rate=1.00, pitch_down=0,
                  swing=0.00, energy_floor=0.00, swap_vocals=True, add_sweeps=True, add_fills=True,
                  lyrics_mode="resonance"),
    "chopped and screwed": dict(vocal_density=0.50, chop=True, tempo_bias=0.00, global_rate=0.80,
                                pitch_down=-3, swing=0.15, energy_floor=0.00, swap_vocals=False,
                                add_sweeps=False, add_fills=False, lyrics_mode="off"),
    "workout/hype": dict(vocal_density=0.55, chop=False, tempo_bias=0.65, global_rate=1.00, pitch_down=0,
                         swing=0.00, energy_floor=0.50, swap_vocals=True, add_sweeps=True, add_fills=True,
                         lyrics_mode="resonance"),
    "contrast/mashup-of-opposites": dict(vocal_density=0.50, chop=False, tempo_bias=0.50, global_rate=1.00,
                                         pitch_down=0, swing=0.00, energy_floor=0.00, swap_vocals=True,
                                         add_sweeps=False, add_fills=False, lyrics_mode="contrast"),
}


def _knobs(bundle):
    """Only the knob keys (drop the resolver's bookkeeping: source/prompt/target_bpm)."""
    return {k: bundle[k] for k in orchestrate.STYLE_KEYS}


@pytest.mark.parametrize("name", list(PINNED))
def test_preset_returns_exact_bundle(name):
    got = orchestrate.resolve_style(name)
    assert _knobs(got) == PINNED[name]
    assert got["source"] == "preset:" + name


def test_all_five_presets_present_and_full_keyset():
    assert set(orchestrate.STYLE_BUNDLES) == set(PINNED)
    for b in orchestrate.STYLE_BUNDLES.values():
        assert set(b.keys()) == set(orchestrate.STYLE_KEYS)


def test_no_prompt_is_balanced_5050():
    for p in (None, "", "   "):
        got = orchestrate.resolve_style(p)
        assert _knobs(got) == PINNED["balanced"]
        assert got["source"] == "default"


def test_aliases_resolve_to_presets():
    assert _knobs(orchestrate.resolve_style("chopped & screwed")) == PINNED["chopped and screwed"]
    assert _knobs(orchestrate.resolve_style("workout remix")) == PINNED["workout/hype"]
    assert _knobs(orchestrate.resolve_style("remixed")) == PINNED["remix"]
    assert _knobs(orchestrate.resolve_style("mashup of opposites")) == PINNED["contrast/mashup-of-opposites"]


def test_unknown_text_falls_back_to_balanced_then_overrides():
    # unknown free text with no classifier -> balanced base, then parsed cues override individual knobs.
    got = orchestrate.resolve_style("make it way slower and dreamy")
    assert got["source"] == "balanced+cues"
    assert got["tempo_bias"] <= 0.25           # "slower" lowers tempo_bias
    assert got["global_rate"] <= 0.90          # ...and global_rate
    # knobs with no matching cue stay at the balanced default
    assert got["vocal_density"] == PINNED["balanced"]["vocal_density"]
    assert got["pitch_down"] == PINNED["balanced"]["pitch_down"]


def test_half_time_cue_sets_tempo_bias_zero():
    got = orchestrate.resolve_style("a half-time flip")
    assert got["tempo_bias"] == 0.0


def test_bpm_cue_recorded():
    got = orchestrate.resolve_style("something around 90 bpm please")
    assert got["target_bpm"] == 90


def test_keyword_table_beats_classifier():
    # a keyword hit is resolved by the deterministic table FIRST (the classifier is only a fallback for
    # genuinely unknown text), so "pump it up" -> workout/hype as a preset, not via the classifier.
    def classifier(text, labels):
        raise AssertionError("classifier should not be consulted when a keyword matches")
    got = orchestrate.resolve_style("pump it up for the treadmill", classifier=classifier)
    assert got["source"] == "preset:workout/hype"


def test_zeroshot_classifier_fallback():
    # genuinely keyword-free text -> the injected classifier's nearest-preset label is honored.
    def classifier(text, labels):
        assert set(labels) == set(orchestrate.STYLE_BUNDLES)
        return "workout/hype"
    got = orchestrate.resolve_style("give it a treadmill feeling", classifier=classifier)
    assert got["source"] == "zeroshot:workout/hype"
    # cue overrides still apply on top of the chosen preset; full key set preserved
    assert set(got.keys()) >= set(orchestrate.STYLE_KEYS)


def test_resolve_style_never_mutates_canonical_bundle():
    got = orchestrate.resolve_style("balanced")
    got["vocal_density"] = 999
    assert orchestrate.STYLE_BUNDLES["balanced"]["vocal_density"] == 0.50


# ------------------------------------------------------------------ N>2 bed selection (OD-4)

def test_bed_selection_picks_max_aggregate_fit():
    ids = ["sA", "sB", "sC"]
    # fit[src][bed]: B is the best bed (high fit from both A and C as sources).
    fit = {
        ("sA", "sB"): 80.0, ("sC", "sB"): 75.0,     # bed B aggregate = 155
        ("sA", "sC"): 40.0, ("sB", "sC"): 45.0,     # bed C aggregate = 85
        ("sB", "sA"): 30.0, ("sC", "sA"): 35.0,     # bed A aggregate = 65
    }

    def scorer(src, bed):
        return dict(workable=True, rating=fit[(src, bed)])

    assert orchestrate.select_bed(ids, scorer) == "sB"


def test_bed_selection_unworkable_direction_contributes_nothing():
    ids = ["s1", "s2", "s3"]

    def scorer(src, bed):
        if bed == "s2":
            return dict(workable=False, reason="x")      # s2 as bed never fits -> aggregate 0
        return dict(workable=True, rating=50.0)

    assert orchestrate.select_bed(ids, scorer) != "s2"


def test_bed_selection_tie_break_uses_judgeable_then_bars_then_id():
    ids = ["sZ", "sY"]

    def scorer(src, bed):
        return dict(workable=True, rating=50.0)          # perfectly tied aggregates

    # higher judgeable wins the tie
    assert orchestrate.select_bed(ids, scorer, judge={"sZ": 0.9, "sY": 0.2}) == "sZ"
    # judgeable equal -> more bars wins
    assert orchestrate.select_bed(ids, scorer, judge={"sZ": 0.5, "sY": 0.5},
                                  bars={"sZ": 10, "sY": 40}) == "sY"
    # both equal -> lower id wins (sY < sZ lexicographically)
    assert orchestrate.select_bed(ids, scorer, judge={"sZ": 0.5, "sY": 0.5},
                                  bars={"sZ": 20, "sY": 20}) == "sY"


def test_select_bed_single_id():
    assert orchestrate.select_bed(["only"], lambda s, b: 0.0) == "only"


# ------------------------------------------------------------------ 2-song deterministic near-tie rule

def _res(score, workable=True):
    return dict(workable=workable, rating=score)


def test_direction_clear_winner_outside_tie_window():
    # gap > 1.0 -> pick the higher-scoring direction regardless of tie-break fields.
    choice = orchestrate.choose_direction(_res(80.0), _res(70.0), judge_v=0.1, judge_i=0.9,
                                           bars_v=1, bars_i=99, id_v="a", id_i="b")
    assert choice == "v_over_i"


def test_direction_near_tie_breaks_on_judgeable():
    # scores within 1.0 -> (a) higher judgeable of the leading vocal.
    choice = orchestrate.choose_direction(_res(75.0), _res(74.5), judge_v=0.80, judge_i=0.40,
                                           bars_v=5, bars_i=50, id_v="a", id_i="b")
    assert choice == "v_over_i"
    choice = orchestrate.choose_direction(_res(75.0), _res(74.5), judge_v=0.40, judge_i=0.80,
                                           bars_v=50, bars_i=5, id_v="a", id_i="b")
    assert choice == "i_over_v"


def test_direction_near_tie_judgeable_equal_breaks_on_bars():
    # judgeable within 0.02 -> (b) more usable vocal bars.
    choice = orchestrate.choose_direction(_res(60.0), _res(60.0), judge_v=0.50, judge_i=0.51,
                                           bars_v=40, bars_i=12, id_v="z", id_i="a")
    assert choice == "v_over_i"


def test_direction_near_tie_all_equal_breaks_on_lower_id():
    # judgeable + bars tied -> (c) lower song id (stable, reproducible).
    choice = orchestrate.choose_direction(_res(50.0), _res(50.0), judge_v=0.5, judge_i=0.5,
                                           bars_v=20, bars_i=20, id_v="s002", id_i="s001")
    assert choice == "i_over_v"     # i has the lower id
    choice = orchestrate.choose_direction(_res(50.0), _res(50.0), judge_v=0.5, judge_i=0.5,
                                           bars_v=20, bars_i=20, id_v="s001", id_i="s002")
    assert choice == "v_over_i"


def test_direction_swap_vocals_does_not_change_choice():
    base = orchestrate.choose_direction(_res(75.0), _res(74.5), judge_v=0.80, judge_i=0.40,
                                        bars_v=5, bars_i=50, id_v="a", id_i="b", swap_vocals=False)
    swap = orchestrate.choose_direction(_res(75.0), _res(74.5), judge_v=0.80, judge_i=0.40,
                                        bars_v=5, bars_i=50, id_v="a", id_i="b", swap_vocals=True)
    assert base == swap == "v_over_i"


# ------------------------------------------------------------------ D.1a VRAM guard helpers

def test_plan_batch_halves_on_low_vram():
    GiB = orchestrate.GiB
    # plenty of VRAM -> stock batch 4 for ultra
    assert orchestrate.plan_batch("ultra", 20 * GiB) == 4
    # ~half the stock requirement free -> batch 2
    assert orchestrate.plan_batch("ultra", 6 * GiB) == 2
    # only room for a single window -> batch 1
    assert orchestrate.plan_batch("ultra", 3 * GiB) == 1
    # not even batch 1 fits -> None (mandatory stage must fail loud)
    assert orchestrate.plan_batch("ultra", 1 * GiB) is None
    # no CUDA reading -> stock batch (guard is a no-op)
    assert orchestrate.plan_batch("ultra", None) == 4
    assert orchestrate.plan_batch("sw", None) == 2


def test_guard_mandatory_raises_stageoom_naming_figures():
    GiB = orchestrate.GiB
    with pytest.raises(orchestrate.StageOOM) as ei:
        orchestrate.guard_mandatory("separation", 2 * GiB, 10 * GiB)
    assert "separation" in str(ei.value)
    # enough VRAM -> no raise; None free -> no raise (no CUDA)
    orchestrate.guard_mandatory("asr", 20 * GiB, 3 * GiB)
    orchestrate.guard_mandatory("asr", None, 3 * GiB)


def test_guard_optional_skips_not_fails():
    GiB = orchestrate.GiB
    assert orchestrate.guard_optional(20 * GiB) is True       # room -> run the LLM tag stage
    assert orchestrate.guard_optional(2 * GiB) is False       # tight -> SKIP to zero-shot (never raises)
    assert orchestrate.guard_optional(None) is True           # no CUDA reading -> allow
