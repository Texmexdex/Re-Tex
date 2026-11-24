"""Tune the bar finder's weights on one half of the MIDI key (coordinate search), report both halves."""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from keyset import Feat, score_bars, songs  # noqa: E402
from tempomap import barseg, meter  # noqa: E402


def prep(keydir):
    out = []
    for k, (tr, npz) in enumerate(songs(keydir)):
        feat = Feat(npz)
        det = np.asarray(tr["beats"], float)
        dur = min(tr["duration"], feat.n / feat.fps)
        b = det[(det >= 0) & (det <= dur)]
        subs = meter.sub_runs(feat, b, 2)
        tt, ts, _ = meter.tatums(b, subs)
        on, ch = meter.describe(feat, tt)
        acc_h, chg = meter.accents(feat, tt, on, ch, ts)
        acc_l = meter.downbeat_features(on, chg, ts) @ np.asarray(meter.DOWNBEAT_COEF)
        acc_l = (acc_l - acc_l.mean()) / (acc_l.std() + 1e-9)
        dl = feat.down_logits
        fr = np.clip(np.round(np.asarray(tt) * 50.0).astype(int), 0, len(dl) - 1)
        x = np.clip(np.maximum.reduce([dl[np.clip(fr + k, 0, len(dl) - 1)] for k in (-1, 0, 1)]), -8, 8)
        bt = (x - x.mean()) / (x.std() + 1e-9)
        tb = [x["t"] for x in tr["bars"] if x["t"] <= dur]
        out.append(dict(k=k, set=tr["set"], title=tr["title"], tt=tt, ts=np.asarray(ts), on=on, ch=ch, acc_h=acc_h,
                        acc_l=acc_l, bt=bt, tb=tb, dur=dur, eighth=np.median(np.diff(b)) / 2))
    return out


def evaluate(data, w, prior_scale=1.0, accent="l"):
    old = {k: dict(v) for k, v in barseg.PRIOR.items()}
    for k in barseg.PRIOR:
        barseg.PRIOR[k] = {L: c * prior_scale for L, c in old[k].items()}
    res = []
    try:
        for d in data:
            if accent.startswith("bt"):
                lw = float(accent[2:] or 0.5)
                acc = d["bt"] + lw * d["acc_l"]
            else:
                acc = d["acc_" + accent]
            st = barseg.segment(d["on"], d["ch"], acc, d["ts"], w)
            bt = d["tt"][st]
            sc = score_bars(bt, d["tb"], 0, d["dur"] - 1)
            L = np.diff(bt) / d["eighth"]
            # length right whatever the phase
            pb, tb = np.asarray(bt), np.asarray(d["tb"] + [d["dur"]])
            ok = tot = 0
            for a, b in zip(tb[:-1], tb[1:]):
                j = np.searchsorted(pb, 0.5 * (a + b)) - 1
                if 0 <= j < len(pb) - 1:
                    tot += 1
                    ok += abs((pb[j + 1] - pb[j]) - (b - a)) < 0.3 * d["eighth"] + 0.03
            sc["len_any"] = ok / max(1, tot)
            sc["odd"] = float(np.mean(np.abs(L - 8) > 0.6)) if d["set"] == "plain" and len(L) else 0.0
            res.append((d, sc))
    finally:
        for k in barseg.PRIOR:
            barseg.PRIOR[k] = old[k]
    return res


def objective(res):
    v = [0.5 * s["found"] + 0.5 * s["len_any"] - 0.3 * s["odd"] - 0.2 * s["extra"] for d, s in res]
    return float(np.mean(v))


def report(res, name):
    for st in ("odd", "plain"):
        r = [s for d, s in res if d["set"] == st]
        if r:
            print(f"   {name} {st:5s} n={len(r):2d} found {np.mean([x['found'] for x in r]):.3f} extra {np.mean([x['extra'] for x in r]):.3f} "
                  f"len_any {np.mean([x['len_any'] for x in r]):.3f}" + (f" odd {np.mean([x['odd'] for x in r]):.3f}" if st == "plain" else ""))


def main(keydir):
    data = prep(keydir)
    from keyset import halves
    A, B = halves(data, key=lambda d: d["set"])
    w = dict(barseg.W)
    ps, acc = 1.0, "bt0.5"
    grid = dict(rep=[0.5, 1, 2, 3, 4, 6], down=[0.2, 0.4, 0.6, 1.0, 1.5, 2.5], switch=[0.3, 0.8, 1.5, 3, 5, 8],
                chroma=[0, 0.25, 0.5, 1, 2], split=[0, 1, 2, 4, 8])
    best = objective(evaluate(A, w, ps, acc))
    print("start", round(best, 4), w, flush=True)
    for it in range(3):
        for key, vals in grid.items():
            for v in vals:
                ww = dict(w, **{key: v})
                o = objective(evaluate(A, ww, ps, acc))
                if o > best + 1e-4:
                    best, w = o, ww
                    print(" ", key, v, round(o, 4), flush=True)
        for v in (0.25, 0.5, 1, 1.5, 2, 3):
            o = objective(evaluate(A, w, v, acc))
            if o > best + 1e-4:
                best, ps = o, v
                print("  prior_scale", v, round(o, 4), flush=True)
        for a in ("l", "h", "bt0", "bt0.25", "bt0.5", "bt1"):
            o = objective(evaluate(A, w, ps, a))
            if o > best + 1e-4:
                best, acc = o, a
                print("  accent", a, round(o, 4), flush=True)
    print("BEST", w, "prior_scale", ps, "accent", acc)
    report(evaluate(A, w, ps, acc), "A (tuned)")
    report(evaluate(B, w, ps, acc), "B (check)")
    report(evaluate(B, dict(barseg.W), 1.0, "l"), "B before ")


if __name__ == "__main__":
    main(sys.argv[1])
