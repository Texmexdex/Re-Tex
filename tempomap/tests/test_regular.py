"""regular.py: the user's rules for where the music doesn't say (Nov 16)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tempomap import meter, regular  # noqa: E402


def test_no_beat_spans_become_even():
    b = np.r_[np.cumsum(np.r_[0.0, 0.59 + 0.02 * np.sin(np.arange(15))]), 9.0 + 0.6 * np.arange(40)]
    b = np.sort(b)
    new = regular.even_gaps(b, [(0, 16)])                       # an ambient intro: carried back at the tempo after it
    assert np.allclose(np.diff(new[:17]), 0.6, atol=1e-6)
    mid = regular.even_gaps(b, [(20, 30)])                      # a break: same count, one speed between its ends
    d = np.diff(mid[19:31])
    assert np.allclose(d, d[0]) and mid[19] == b[19] and mid[30] == b[30]
    f = regular.warp_fn(b, new)                                 # subdivisions move with their beats
    assert abs(f([(b[3] + b[4]) / 2])[0] - (new[3] + new[4]) / 2) < 1e-6


def test_short_unsure_odd_meter_takes_the_main_meter():
    beats = np.arange(0, 200) * 0.5                             # 4/4 at 120, 2 subdivisions a beat
    tt = np.arange(0, 400) * 0.25
    starts = list(range(0, 160, 8)) + [160, 166] + list(range(172, 400, 8))   # one 6/8-ish bar pair
    runs = [dict(i0=0, i1=160, length=8, sub=2, confidence=0.8),
            dict(i0=160, i1=172, length=6, sub=2, confidence=0.3),
            dict(i0=172, i1=400, length=8, sub=2, confidence=0.8)]
    st, rn, sp = regular.bars(tt, beats, starts, runs, [], meter.label)
    assert len(rn) == 1 and rn[0]["length"] == 8                # one 4/4 run
    assert sp and sp[0]["kind"] == "assumed" and "doesn't come out even" in sp[0]["note"]   # 12 subdivisions = 6 beats


def test_a_sure_long_odd_meter_stays():
    beats = np.arange(0, 200) * 0.5
    tt = np.arange(0, 400) * 0.25
    starts = list(range(0, 100, 8)) + list(range(100, 240, 7)) + list(range(240, 400, 8))
    runs = [dict(i0=0, i1=100, length=8, sub=2, confidence=0.8),
            dict(i0=100, i1=240, length=7, sub=2, confidence=0.7),
            dict(i0=240, i1=400, length=8, sub=2, confidence=0.8)]
    st, rn, sp = regular.bars(tt, beats, starts, runs, [], meter.label)
    assert any(r["length"] == 7 for r in rn) and not sp
