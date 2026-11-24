"""A smooth lock follows a tempo push that lasts a few bars (Nov 18: Hooker With A Penis sat 50-90 ms off the grid
for stretches of 4-10 s - red in the arrangement - because short speed changes were smoothed over up to ~100 ms)."""
import numpy as np

from mashlab import lock


def test_smooth_lock_follows_a_short_push():
    rng = np.random.default_rng(1)
    p = 0.326
    step = np.full(840, p)
    step[370:378] = p * 0.96              # 8 beats pushed 4 % faster, then back (the old smoothing: 102 ms off)
    m = np.cumsum(step) + rng.normal(0, 0.004, 840)
    sm, bends = lock.smooth_marks(m)
    assert np.abs(m - sm).max() < 0.03   # every mark within 30 ms of where the lock puts it on the grid
    assert bends < 40                    # and still a smooth line, not every mark


def test_smooth_lock_ignores_one_stray_mark():
    p = 0.326
    m = np.arange(400) * p
    m[200] += 0.06                        # one wrong mark
    sm, _ = lock.smooth_marks(m)
    assert abs(sm[200] - 200 * p) < 0.01
