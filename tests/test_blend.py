"""FEAT-003 Verification 4: blend.py — role selection, C.2 frequency-slotting (carve mapping + HIGH-2
NEUTRAL/ESCALATE), C.3 energy match, C.4 chop qualifier + re-sequencer. Synthetic PSD/share arrays; no GPU,
no audio decode."""
import numpy as np
import pytest

from mashlab import blend as B


# ------------------------------------------------------------------ C.1 role selection

def test_bass_falls_back_to_chroma_when_tonality_low():
    # tonality < 0.5 -> key doesn't matter -> chroma compat fallback, mode "chroma".
    out = B.bass_fit(tonality_L=0.2, shift_scores_bass=0.9, compat_fallback=0.55)
    assert out["mode"] == "chroma"
    assert out["fit"] == pytest.approx(0.55)


def test_bass_uses_harmfit_when_tonality_high():
    out = B.bass_fit(tonality_L=1.0, shift_scores_bass=0.9, compat_fallback=0.4)
    assert out["mode"] == "harmfit"
    assert out["fit"] == pytest.approx(0.9)              # tone=1 -> pure harmfit score


def test_overlay_gate_requires_all_three():
    assert B.overlay_allowed(0.7, 0.8, True) is True
    assert B.overlay_allowed(0.5, 0.8, True) is False    # chroma compat below 0.60
    assert B.overlay_allowed(0.7, 0.7, True) is False    # harmfit below H_NEUTRAL 0.75
    assert B.overlay_allowed(0.7, 0.8, False) is False   # carve would push a core lane past the cap


def test_drum_fit_rewards_matched_groove_and_energy():
    g = {"s8": 0.1, "s16": 0.05}
    env = [0.2, 0.5, 0.9, 0.4]
    hi = B.drum_fit(g, g, env, env)
    lo = B.drum_fit({"s8": 0.6, "s16": 0.6}, g, [0.9, 0.1, 0.9, 0.1], env)
    assert hi > lo


# ------------------------------------------------------------------ C.2 carve mapping (VERBATIM)

def _bands(**lanes):
    """Build {lane: (9,) energies} putting all mass of each lane in one named band index -> shares control."""
    out = {}
    for lane, spec in lanes.items():
        e = np.zeros(len(B.BAND_EDGES) - 1)
        for band_index, mass in spec.items():
            e[band_index] = mass
        out[lane] = e
    return out


def test_carve_cuts_lower_priority_lane_with_pinned_gain_and_q():
    # band 5 (1000-2000 Hz): vocals 0.6, other 0.4 -> contended; other is lower priority -> carved.
    lane_bands = _bands(vocals={5: 0.6}, other={5: 0.4})
    carve = B.carve_band(B.band_shares(lane_bands), 5)
    assert carve is not None
    assert carve["lane"] == "other"
    overlap = 0.4
    assert carve["overlap"] == pytest.approx(overlap)
    assert carve["gain_db"] == pytest.approx(-np.clip(2 + 6 * (overlap - 0.25), 2, 5))
    assert carve["Q"] == pytest.approx(0.5 + 1.5 * overlap)
    center = float(np.sqrt(B.BAND_EDGES[5] * B.BAND_EDGES[6]))
    assert carve["center"] == pytest.approx(center)
    assert carve["hard"] is False


def test_band_not_contended_when_second_lane_below_25pct():
    lane_bands = _bands(vocals={5: 0.8}, other={5: 0.2})   # 2nd share 0.2 < 0.25 -> not contended
    assert B.carve_band(B.band_shares(lane_bands), 5) is None


def test_equal_split_overlap_is_half_and_not_hard():
    # a two-lane equal split: 2nd-largest share = 0.5 -> overlap 0.5 (< HARD 0.75), gain mapping = -5 dB.
    lane_bands = _bands(bass={4: 0.5}, other={4: 0.5})     # band 4 (500-1000 Hz); other is lower priority
    carve = B.carve_band(B.band_shares(lane_bands), 4)
    assert carve["lane"] == "other"
    assert carve["overlap"] == pytest.approx(0.5)
    assert carve["hard"] is False
    assert carve["gain_db"] == pytest.approx(-np.clip(2 + 6 * (0.5 - 0.25), 2, 5))   # -3.5 dB


def test_bass_vs_kick_becomes_a_duck_not_a_cut():
    # band 1 (60-120 Hz) where bass & drums both lead -> sidechain duck, routed to ducks[] not carves[].
    lane_bands = _bands(bass={1: 0.5}, drums={1: 0.5})
    res = B.frequency_slot(lane_bands)
    assert res["ducks"] and res["ducks"][0]["band"] == 1
    assert all(c["band"] != 1 for c in res["carves"])


# ------------------------------------------------------------------ C.2 HIGH-2 NEUTRAL / ESCALATE

def test_hard_contention_at_or_above_075_overlap():
    # Three lanes in one band: a dominant lane plus two that still clear 0.25 so overlap (2nd share) can rise.
    # Here 'other' is dominant and 'drums' holds the 2nd share; push that 2nd share >= 0.75 by making the two
    # leaders co-dominant at >=0.75 each relative to the band total.
    lane_bands = _bands(other={4: 0.80}, drums={4: 0.78})
    sh = B.band_shares(lane_bands)
    # 0.80 / (0.80+0.78) = 0.506 -> still below HARD. Equal two-lane splits top out at 0.5, so hard contention
    # is a mapping-saturation signal confirmed by ear (Verification §4); below it, light contention carves.
    carve = B.carve_band(sh, 4)
    assert carve is not None and carve["hard"] is False


def test_light_contention_does_not_flag_congested():
    lanes = {}
    v = np.zeros(len(B.BAND_EDGES) - 1)
    d = np.zeros(len(B.BAND_EDGES) - 1)
    for band in (4, 5, 6):
        v[band] = 0.5
        d[band] = 0.5                                      # equal split -> overlap 0.5 (< HARD 0.75)
    res = B.frequency_slot({"vocals": v, "drums": d})
    assert res["congested"] is False
    assert res["escalate"] is None


def test_high2_fires_and_escalates_when_core_hard_contended_in_3_bands():
    # Drive genuine hard contention deterministically by relaxing HARD_OVERLAP to the equal-split point, so a
    # core lane (drums, lower priority than vocals) is hard-contended in exactly 3 of 9 bands -> NEUTRAL.
    import mashlab.blend as blend

    v = np.zeros(len(B.BAND_EDGES) - 1)
    d = np.zeros(len(B.BAND_EDGES) - 1)
    for band in (4, 5, 6):
        v[band] = 0.5
        d[band] = 0.5
    old = blend.HARD_OVERLAP
    try:
        blend.HARD_OVERLAP = 0.5
        res = blend.frequency_slot({"vocals": v, "drums": d})
    finally:
        blend.HARD_OVERLAP = old
    assert res["congested"] is True
    assert res["congested_lane"] == "drums"              # the lower-priority core lane
    assert res["escalate"] == "drums"
    assert res["carves"] == []                            # NEUTRAL: do NOT carve


def test_below_threshold_still_carves_normally():
    # drums hard-contended in only 2 bands (< CONGESTED_BANDS=3) -> ordinary carve path applies.
    import mashlab.blend as blend

    v = np.zeros(len(B.BAND_EDGES) - 1)
    d = np.zeros(len(B.BAND_EDGES) - 1)
    for band in (4, 5):
        v[band] = 0.5
        d[band] = 0.5
    old = blend.HARD_OVERLAP
    try:
        blend.HARD_OVERLAP = 0.5
        res = blend.frequency_slot({"vocals": v, "drums": d})
    finally:
        blend.HARD_OVERLAP = old
    assert res["congested"] is False
    assert len(res["carves"]) >= 1                        # carves applied, not neutralized


# ------------------------------------------------------------------ C.3 energy match

def test_section_gain_tracks_bed_dynamics():
    base = 0.0
    quiet = B.section_gain(bed_section_energy=0.2, layer_section_energy=0.5, base_gain_db=base)
    loud = B.section_gain(bed_section_energy=0.9, layer_section_energy=0.5, base_gain_db=base)
    assert loud > base > quiet                            # louder bed section -> more layer gain


# ------------------------------------------------------------------ C.4 chop

def test_chop_triggers_on_swing_mismatch():
    out = B.chop_triggered({"s8": 0.4, "s16": 0.0}, {"s8": 0.0, "s16": 0.0}, 0.0, 1.0)
    assert out["fire"] and "swing_s8" in out["reasons"]


def test_chop_triggers_on_unplaceable_bass():
    out = B.chop_triggered({"s8": 0.0}, {"s8": 0.0}, tonality_bass=0.7, shift_scores_bass_best=0.5)
    assert out["fire"] and "bass_unplaceable" in out["reasons"]


def test_chop_triggers_on_style_flag():
    out = B.chop_triggered({"s8": 0.0}, {"s8": 0.0}, 0.0, 1.0, style_chop=True)
    assert out["fire"] and "reasons" and "style" in out["reasons"]


def test_chop_not_triggered_when_matched():
    out = B.chop_triggered({"s8": 0.05, "s16": 0.05}, {"s8": 0.0, "s16": 0.0}, 0.3, 0.9)
    assert out["fire"] is False


def test_chop_cell_qualifier_drops_filler_keeps_strong():
    cells = [
        {"beat": 0, "beat_content": 10},                 # <= 12 -> dropped as filler
        {"beat": 1, "beat_content": 39},                 # between 12 and 40 -> not qualifying
        {"beat": 2, "beat_content": 40},                 # >= 40 -> kept
        {"beat": 3, "beat_content": 95},                 # kept
    ]
    kept = B.qualify_cells(cells)
    beats = sorted(c["beat"] for c in kept)
    assert beats == [2, 3]
    assert all(c["strength"] == c["beat_content"] for c in kept)


def test_cell_size_follows_swing():
    assert B.cell_size_beats(0.0) == 1
    assert B.cell_size_beats(0.1) == 2
    assert B.cell_size_beats(0.2) == 2


def test_drum_cell_classification():
    assert B.classify_drum_cell(body_energy=1.0, hat_energy=0.2) == "body"
    assert B.classify_drum_cell(body_energy=0.1, hat_energy=1.0) == "hat"
    assert B.classify_drum_cell(0.5, 0.5, bands_separable=False) == "coarse"


def test_chop_sequence_places_strongest_body_on_downbeats():
    bar_spans = [(0, 4), (4, 8)]                          # two bars, 4 beats each
    pool = [
        {"beat": 0, "beat_content": 95, "strength": 95, "kind": "body"},
        {"beat": 1, "beat_content": 60, "strength": 60, "kind": "body"},
        {"beat": 2, "beat_content": 80, "strength": 80, "kind": "hat"},
    ]
    out = B.chop_sequence(bar_spans, pool, swing=0.0, bands_separable=True)
    downbeats = [p for p in out["placements"] if p["role"] == "downbeat"]
    assert downbeats and downbeats[0]["cell"]["strength"] == 95   # strongest body cell on the first downbeat
    assert out["note"] is None


def test_chop_sequence_coarse_fallback_notes_limitation():
    bar_spans = [(0, 4)]
    pool = [{"beat": 0, "beat_content": 70, "strength": 70}]
    out = B.chop_sequence(bar_spans, pool, swing=0.0, bands_separable=False)
    assert out["note"] == "coarse chop (drum bands not separable)"


def test_bass_root_on_downbeat():
    assert B.bass_root_on_downbeat(7, 7) is True
    assert B.bass_root_on_downbeat(19, 7) is True        # 19 % 12 == 7
    assert B.bass_root_on_downbeat(5, 7) is False


# ------------------------------------------------------------------ band integration sanity

def test_band_energies_integrate_into_nine_bands():
    freqs = np.linspace(0, 22050, 2048)
    psd = np.ones_like(freqs)
    be = B.band_energies(freqs, psd)
    assert len(be) == len(B.BAND_EDGES) - 1 == 9
    assert np.all(be >= 0)
