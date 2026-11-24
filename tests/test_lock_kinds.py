"""The smooth kind of lock and the grid repair it goes with (mashlab/lock.py). Plain numbers, no library, no audio."""
import numpy as np

from mashlab import lock


def _info(beats, bpb=4):
    b = [round(float(x), 5) for x in beats]
    return dict(beats=b, duration=b[-1] + 1.0, bpb=bpb, bars=list(range(0, len(b), bpb)), bpm=120.0)


def test_a_smooth_lock_ignores_bunched_bars_and_jitter_but_follows_a_drift_that_lasts():
    rng = np.random.default_rng(3)
    n = 400
    true = 0.5 + np.arange(n) * 0.5
    m = true + rng.normal(0, 0.008, n)
    for b in (40, 120, 200, 280):                    # bunched bars: one long beat, then three short ones
        m[b + 1:b + 4] = m[b] + np.array([0.82, 1.22, 1.62])
    m = np.sort(m)
    sm, bends = lock.smooth_marks(m)
    assert len(sm) == n and sm[0] == m[0] and sm[-1] == m[-1] and np.all(np.diff(sm) > 0)
    assert np.abs(sm - true).max() < 0.03 and bends <= 8
    every, smooth = lock.would(m), lock.would_smooth(m)
    assert every["sec20"] > 3.0 and smooth["sec20"] == 0.0 and smooth["sec5"] == 0.0 and smooth["bends"] == bends
    per = np.r_[np.full(200, 0.5), np.linspace(0.5, 0.52, 200)]          # the second half slows down by 4 %
    d = 0.5 + np.r_[0.0, np.cumsum(per)]
    sd, _ = lock.smooth_marks(d)
    assert np.abs(sd - d).max() <= lock.SMOOTH_REACH[1]


def test_too_few_marks_are_left_as_they_are():
    m = 0.5 + np.arange(10) * 0.5
    sm, bends = lock.smooth_marks(m)
    assert np.allclose(sm, m) and bends == 0


def test_repair_re_counts_a_half_time_stretch_the_way_it_always_did():
    beats = np.r_[0.5 + np.arange(80) * 0.5, 40.5 + np.arange(40) * 1.0, 80.5 + np.arange(80) * 0.5]
    marks, rep = lock.repaired(_info(beats))
    assert rep["method"] == "recount" and rep["sec20"] == 0.0 and rep["asks"] == 0
    assert np.allclose(np.diff(marks["t"]), 0.5, atol=1e-3) and len(marks["t"]) == len(marks["d"]) == len(marks["s"])


def test_repair_evens_out_a_stretch_where_the_detector_lost_the_beat_and_marks_it_unsure():
    # 120 BPM, then 15 s counted 4 against 3 (no simple re-count fits), then 120 again
    lost = 100.5 + np.arange(1, 41) * 0.375
    beats = np.r_[0.5 + np.arange(201) * 0.5, lost, lost[-1] + np.arange(1, 161) * 0.5]
    info = _info(beats)
    before = lock.would(np.asarray(lock.marks_of(info)))["sec20"]
    marks, rep = lock.repaired(info)
    t = np.array(marks["t"])
    assert before > 10.0 and rep["method"] == "neighbours" and rep["sec20"] < 1.0 and rep["asks"] == 0
    assert np.allclose(np.diff(t), 0.5, atol=0.02)                        # the song plays on at 120 through it
    inside = (t > 101.0) & (t < 115.0)
    assert not np.array(marks["s"])[inside].any() and np.array(marks["s"])[~inside & (t < 99.0)].all()
    assert [x["kind"] for x in rep["sections"]] == ["count", "even", "count"] and len(rep["unsure"]) == 1


def test_a_steady_grid_needs_no_repair_and_gets_none():
    marks, rep = lock.repaired(_info(0.5 + np.arange(300) * 0.5))
    assert rep["method"] == "recount" and rep["recounted_beats"] == 0 and rep["sec20"] == 0.0
    assert np.allclose(marks["t"], 0.5 + np.arange(300) * 0.5, atol=1e-4)
