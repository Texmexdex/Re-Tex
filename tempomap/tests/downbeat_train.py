"""Learn how bar 1 sounds from the MIDI answer key: a logistic model over what happens on each subdivision and
around it. Trains on one half of the key, reports on the other; prints coefficients for meter.DOWNBEAT."""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from keyset import Feat, halves, songs  # noqa: E402
from tempomap import meter  # noqa: E402


def song_rows(tr, npz):
    feat = Feat(npz)
    det = np.asarray(tr["beats"], float)
    dur = min(tr["duration"], feat.n / feat.fps)
    b = det[(det >= 0) & (det <= dur)]
    subs = meter.sub_runs(feat, b, 2)
    tt, ts, _ = meter.tatums(b, subs)
    on, ch = meter.describe(feat, tt)
    acc, chg = meter.accents(feat, tt, on, ch, ts)
    X = meter.downbeat_features(on, chg, ts)
    tb = np.array([x["t"] for x in tr["bars"] if x["t"] <= dur])
    step = np.median(np.diff(tt))
    y = np.zeros(len(tt))
    j = np.searchsorted(tt, tb)
    for k, t in zip(j, tb):
        for q in (k - 1, k):
            if 0 <= q < len(tt) and abs(tt[q] - t) < 0.3 * step:
                y[q] = 1
    return X, y


def main(keydir):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    items = songs(keydir)
    A, B = halves(items)
    XA, yA = map(np.concatenate, zip(*[song_rows(*x) for x in A]))
    XB, yB = map(np.concatenate, zip(*[song_rows(*x) for x in B]))
    m = LogisticRegression(C=0.3, class_weight="balanced", max_iter=2000).fit(XA, yA)
    print("AUC train", round(roc_auc_score(yA, m.decision_function(XA)), 3), "test", round(roc_auc_score(yB, m.decision_function(XB)), 3))
    # baseline: the hand-made accent (column order in downbeat_features: kick, snare, cym, bass, other ...)
    hand = XB[:, 0] + 0.7 * XB[:, 2] + 0.5 * XB[:, 3] + XB[:, meter.DB_CHG]
    print("AUC hand-made accent (test)", round(roc_auc_score(yB, hand), 3))
    full = LogisticRegression(C=0.3, class_weight="balanced", max_iter=2000).fit(np.vstack([XA, XB]), np.r_[yA, yB])
    print("COEF", [round(float(c), 4) for c in full.coef_[0]], "INTERCEPT", round(float(full.intercept_[0]), 4))
    print("COEF_A", [round(float(c), 4) for c in m.coef_[0]])


if __name__ == "__main__":
    main(sys.argv[1])
