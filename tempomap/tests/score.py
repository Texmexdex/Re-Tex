"""Score a map against a test song's truth: subdivisions, bar lines, tempo and bar names per stretch."""
import numpy as np


def near(a, b, tol):
    """Share of the times in a that have one in b within tol s."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if not len(a):
        return 1.0
    if not len(b):
        return 0.0
    j = np.clip(np.searchsorted(b, a), 1, len(b) - 1)
    d = np.minimum(np.abs(b[j] - a), np.abs(b[j - 1] - a))
    return float(np.mean(d <= tol))


def score(m, truth):
    tt = np.array(truth["tatums"])
    mt = np.array(m["tatums"])
    tb = tt[truth["bars"]]
    mb = np.array(m["bar_times"])
    labels = []
    for k, (L, s) in zip(truth["bars"], truth["labels"]):
        labels.append((tt[k], L, s))
    ok_lab = 0
    for t0, L, s in labels:                      # the map's bar at the truth's bar line: same length?
        j = np.searchsorted(mb, t0 - 0.03)
        if j < len(mb) - 1 and abs(mb[j] - t0) <= 0.03:
            n = int(np.sum((mt >= mb[j] - 1e-3) & (mt < mb[j + 1] - 1e-3)))
            ok_lab += int(n == L)
    return dict(
        tatums_found=near(tt, mt, 0.02), tatums_extra=1 - near(mt, tt, 0.02),
        bars_found=near(tb, mb, 0.03), bars_extra=1 - near(mb, tb, 0.03),
        bar_length_right=ok_lab / max(1, len(labels)),
    )
