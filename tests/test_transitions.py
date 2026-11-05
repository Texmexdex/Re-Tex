"""FEAT-003 Verification 3 + 8/4: transitions.py — selection rule, fill detector, STRONG_Q, and the
offline sweep renderer's byte-identical determinism. Synthetic arrays only; no GPU, no audio decode."""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from mashlab import transitions as T

ROOT = Path(__file__).resolve().parent.parent


# ------------------------------------------------------------------ STRONG_Q (F11 / B.2)

def test_strong_q_is_in_range_nonbeatless_75th_percentile():
    # bars 0..9; beatless (<=12) tails at the edges must be excluded, as must out-of-range beats.
    bc = [5, 8, 40, 60, 80, 100, 70, 50, 9, 7]
    first, last = 2, 7                                   # in-range window = indices 2..7
    inside = np.array([40, 60, 80, 100, 70, 50])
    expect = float(np.quantile(inside, 0.75))
    assert T.strong_q(bc, first, last) == pytest.approx(expect)


def test_strong_q_drops_beatless_values():
    bc = [50, 12, 10, 90, 80]                            # 12 and 10 are <= CONTENT_NONE -> excluded
    sq = T.strong_q(bc, 0, 4)
    assert sq == pytest.approx(float(np.quantile([50, 90, 80], 0.75)))


def test_strong_q_all_beatless_is_zero():
    assert T.strong_q([5, 8, 12, 10], 0, 3) == 0.0


# ------------------------------------------------------------------ fill detector (B.2 type 5)

def _bed_info(beat_content, bpb=4, n_bars=8):
    n_beats = n_bars * bpb
    beats = [round(0.5 * i, 4) for i in range(n_beats)]
    bars = list(range(0, n_beats, bpb))
    return {"bars": bars, "beats": beats, "bpb": bpb, "beat_content": beat_content,
            "first_beat": 0, "last_beat": n_beats - 1}


def test_fill_detector_picks_the_qualifying_bar():
    # 8 bars of 4 beats. Make bar 3 a loud fill (mean 90), the rest quiet (mean 30).
    bc = []
    for bar in range(8):
        bc += [90, 90, 90, 90] if bar == 3 else [30, 30, 30, 30]
    info = _bed_info(bc)
    # boundary after bar 4 -> candidate fill bars are 0..3; bar 3 is the loud one and the only qualifier.
    assert T.fill_bar(info, boundary_bar=4) == 3


def test_fill_detector_none_when_no_bar_qualifies():
    bc = [20] * 32                                       # flat & below the 40 floor -> no qualifier
    info = _bed_info(bc)
    assert T.fill_bar(info, boundary_bar=6) is None


def test_fill_detector_respects_floor_even_with_high_ratio():
    # one bar 3x the median but still under the 40 absolute floor -> must NOT qualify
    bc = [10] * 28 + [30, 30, 30, 30]
    info = _bed_info(bc)
    assert T.fill_bar(info, boundary_bar=8) is None


# ------------------------------------------------------------------ selection rule (B.2 VERBATIM)

def test_section_build_with_usable_fill_is_type5():
    bc = []
    for bar in range(8):
        bc += [95, 95, 95, 95] if bar == 2 else [30, 30, 30, 30]
    info = _bed_info(bc)
    ev = {"kind": "section", "energy": 0.3, "next_energy": 0.8, "boundary_bar": 4}
    t, meta = T.select_type(ev, info)
    assert t == 5 and meta["fill_bar"] == 2 and meta["layer_sweep"] == 3


def test_section_build_without_fill_falls_to_type3():
    info = _bed_info([20] * 32)                          # nothing qualifies as a fill
    ev = {"kind": "section", "energy": 0.3, "next_energy": 0.8, "boundary_bar": 4}
    t, meta = T.select_type(ev, info)
    assert t == 3 and meta["fallback"] == "type5_no_fill"


def test_low_energy_section_merge_is_type2():
    info = _bed_info([30] * 32)
    ev = {"kind": "section", "energy": 0.2, "next_energy": 0.25}      # both below low_energy 0.40
    assert T.select_type(ev, info)[0] == 2


def test_ordinary_section_change_is_type3():
    info = _bed_info([30] * 32)
    ev = {"kind": "section", "energy": 0.6, "next_energy": 0.65}      # delta < 0.3, not both low
    assert T.select_type(ev, info)[0] == 3


def test_vocal_entry_exit_is_type4():
    info = _bed_info([30] * 32)
    assert T.select_type({"kind": "vocal_entry"}, info)[0] == 4
    assert T.select_type({"kind": "vocal_exit"}, info)[0] == 4


def test_vocal_swap_midline_is_type2():
    info = _bed_info([30] * 32)
    ev = {"kind": "vocal_swap", "c_in": 0.7, "c_out": 0.1, "bc_at": 90, "strong_q": 50}
    assert T.select_type(ev, info)[0] == 2            # cut cost > 0.5 forces the crossfade


def test_vocal_swap_clean_gap_strong_downbeat_is_type1():
    info = _bed_info([30] * 32)
    ev = {"kind": "vocal_swap", "c_in": 0.2, "c_out": 0.2, "bc_at": 90, "strong_q": 60}
    assert T.select_type(ev, info)[0] == 1            # clean gap on a strong downbeat -> hard cut


def test_vocal_swap_clean_gap_weak_beat_is_type2():
    info = _bed_info([30] * 32)
    ev = {"kind": "vocal_swap", "c_in": 0.2, "c_out": 0.2, "bc_at": 40, "strong_q": 60}
    assert T.select_type(ev, info)[0] == 2            # clean but not strong -> crossfade


# ------------------------------------------------------------------ build_plan: beat-alignment + cut cost

def test_build_plan_boundaries_land_on_a_downbeat_within_1ms():
    bpb = 4
    n_bars = 8
    n_beats = n_bars * bpb
    bpm = 120.0
    beat_period = 60.0 / bpm                              # 0.5 s
    beats = [round(beat_period * i, 6) for i in range(n_beats)]
    bars = list(range(0, n_beats, bpb))
    info = {"bars": bars, "beats": beats, "bpb": bpb, "beat_content": [30] * n_beats,
            "first_beat": 0, "last_beat": n_beats - 1}
    # events placed slightly off a downbeat; they must snap to the nearest downbeat bar.
    path = [
        {"kind": "section", "beat": 7.9, "energy": 0.6, "next_energy": 0.65, "boundary_bar": 8},
        {"kind": "vocal_swap", "beat": 16.1, "c_in": 0.2, "c_out": 0.2, "bc_at": 90, "strong_q": 60},
    ]
    plan = T.build_plan(path, info)
    for ev in plan["events"]:
        db = ev["downbeat_bar"]
        assert db in bars
        # time of the snapped downbeat vs the time implied by the raw event beat is within 1 ms of a real beat
        assert abs(beats[db] - round(beat_period * db, 6)) < 1e-3


def test_build_plan_never_hard_cuts_where_cut_cost_high():
    info = {"bars": [0, 4, 8, 12], "beats": [round(0.5 * i, 4) for i in range(16)], "bpb": 4,
            "beat_content": [90] * 16, "first_beat": 0, "last_beat": 15}
    # a mid-line swap (cut cost > 0.5) must NEVER be a hard cut (type 1), even on a strong downbeat.
    path = [{"kind": "vocal_swap", "beat": 4.0, "c_in": 0.8, "c_out": 0.9, "bc_at": 100, "strong_q": 10}]
    plan = T.build_plan(path, info)
    assert all(ev["type"] != 1 for ev in plan["events"])


def test_build_plan_merges_near_transitions():
    info = _bed_info([30] * 32)
    path = [
        {"kind": "vocal_entry", "beat": 4.0},
        {"kind": "vocal_swap", "beat": 4.3, "c_in": 0.2, "c_out": 0.2, "bc_at": 90, "strong_q": 60},
    ]
    plan = T.build_plan(path, info, params={"merge_beats": 1.0})
    assert len(plan["events"]) == 1                       # merged; the harder (lower type number) survives


# ------------------------------------------------------------------ sweep renderer determinism (B.4 §8/§4)

@pytest.fixture
def sweep_audio():
    n = 8192
    t = np.linspace(0, 1, n, endpoint=False)
    y = 0.3 * np.sin(2 * np.pi * 220 * t)
    return np.stack([y, y]).astype(np.float32)           # (2, n)


def test_sweep_byte_identical_in_one_process(sweep_audio):
    a = T.render_sweep(sweep_audio, "lpf", 20000, 400, "exp")
    b = T.render_sweep(sweep_audio, "lpf", 20000, 400, "exp")
    assert a.tobytes() == b.tobytes()


def test_noise_sweep_is_deterministic(sweep_audio):
    a = T.render_sweep(sweep_audio, "noise", 20, 8000, "exp")
    b = T.render_sweep(sweep_audio, "noise", 20, 8000, "exp")
    assert a.tobytes() == b.tobytes()


def test_cache_key_changes_with_param_or_renderer_version():
    spec1 = T.quantize_spec("lpf", 20000, 400, "exp", 8192)
    spec2 = T.quantize_spec("lpf", 20000, 401, "exp", 8192)      # a changed cutoff
    k1 = T.cache_key("sA", "other", 0, 8192, "lpf", spec1)
    k1_same = T.cache_key("sA", "other", 0, 8192, "lpf", spec1)
    k2 = T.cache_key("sA", "other", 0, 8192, "lpf", spec2)
    kv = T.cache_key("sA", "other", 0, 8192, "lpf", spec1, renderer_version="different")
    assert k1 == k1_same                                 # same tuple -> reuse
    assert k1 != k2                                       # changed param -> new key (no aliasing)
    assert k1 != kv                                       # changed renderer_version -> new key


_CROSS_PROC = """
import numpy as np
from mashlab import transitions as T
n = 8192
t = np.linspace(0, 1, n, endpoint=False)
y = 0.3 * np.sin(2 * np.pi * 220 * t)
audio = np.stack([y, y]).astype(np.float32)
out = T.render_sweep(audio, "lpf", 20000, 400, "exp")
import sys
sys.stdout.buffer.write(out.tobytes())
"""


def test_sweep_byte_identical_across_two_fresh_processes():
    py = ROOT / ".venv-phase2" / "Scripts" / "python.exe"
    if not py.exists():                                  # fall back to the current interpreter
        py = Path(sys.executable)
    r1 = subprocess.run([str(py), "-c", _CROSS_PROC], capture_output=True, cwd=str(ROOT))
    r2 = subprocess.run([str(py), "-c", _CROSS_PROC], capture_output=True, cwd=str(ROOT))
    assert r1.returncode == 0, r1.stderr.decode(errors="replace")
    assert r2.returncode == 0, r2.stderr.decode(errors="replace")
    assert r1.stdout and r1.stdout == r2.stdout
