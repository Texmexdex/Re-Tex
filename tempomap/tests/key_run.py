"""Score the bar finder on the MIDI answer key.
python tests/key_run.py KEYDIR [half]   half = a (tune) / b (check) / all (default)
The detector's beats are replaced by the file's own quarter notes, so this measures the bars alone."""
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from keyset import Feat, score_bars, songs  # noqa: E402
from tempomap import build  # noqa: E402


def true_grid_map(feat, det, dur):
    """Bars on the file's own beats (the beat grid taken as known): measures the bar finder alone."""
    from tempomap import meter
    b = det[(det >= 0) & (det <= dur)]
    subs = meter.sub_runs(feat, b, 2)
    tt, ts, on = meter.tatums(b, subs)
    if meter.METHOD == "seg":
        starts, runs = meter.bars_seg(feat, tt, ts)
    else:
        starts, runs = meter.bars(feat, tt, ts)
    return dict(bar_times=[float(tt[k]) for k in starts], pulse_beats=list(b),
                meters=[dict(label=meter.label(r["length"], r["sub"])) for r in runs])


def run(keydir, half="all", verbose=True, grid="map"):
    rows = []
    items = songs(keydir)
    if half in ("a", "b"):
        items = [x for k, x in enumerate(items) if (k % 2 == 0) == (half == "a")]
    for tr, npz in items:
        t0 = time.time()
        feat = Feat(npz)
        det = np.asarray(tr["beats"], float)
        dur = min(tr["duration"], feat.n / feat.fps)
        det = det[det <= dur]
        if grid == "true":
            m = true_grid_map(feat, det, dur)
        elif grid == "bt":                            # the whole pipeline, on Beat This!'s own beats
            bb = np.asarray(feat.bt_beats, float)
            m = build.from_features(dict(beats=list(bb), bars=[]), feat, bb, dur)
        else:
            m = build.from_features(dict(beats=list(det), bars=[]), feat, det, dur)
        tb = [b["t"] for b in tr["bars"] if b["t"] <= dur]
        sc = score_bars(m["bar_times"], tb, 0.0, dur - 1.0)
        # wrong odd bars on a plain 4/4 song: predicted bars that aren't 8 eighths long
        L = np.diff(m["bar_times"]) / (np.median(np.diff(det)) / 2)
        sc["odd_bars"] = float(np.mean(np.abs(L - 8) > 0.6)) if tr["set"] == "plain" and len(L) else None
        sc["labels"] = len({x["label"] for x in m["meters"]})
        # bar length right whatever the phase: the predicted bar over each true bar's middle has its length
        pb, tb_ = np.asarray(m["bar_times"]), np.asarray(tb + [dur])
        ok, tot = 0, 0
        for a, b in zip(tb_[:-1], tb_[1:]):
            mid = 0.5 * (a + b)
            j = np.searchsorted(pb, mid) - 1
            if 0 <= j < len(pb) - 1:
                tot += 1
                ok += abs((pb[j + 1] - pb[j]) - (b - a)) < 0.3 * np.median(np.diff(det)) / 2 + 0.03
        sc["len_any"] = ok / max(1, tot)
        pbt = np.asarray(m["pulse_beats"])
        dd = det[(det > 2) & (det < dur - 2)]
        sc["beats"] = float(np.mean(np.min(np.abs(dd[:, None] - pbt[None, :]), axis=1) < 0.03)) if len(pbt) else 0.0
        rows.append((tr, sc))
        if verbose:
            print(f"{tr['set']:5s} {tr['title'][:34]:34s} found {sc['found']:.2f} extra {sc['extra']:.2f} "
                  f"len_ok {sc['length_ok']:.2f} len_any {sc['len_any']:.2f} beats {sc['beats']:.2f}" + (f" odd {sc['odd_bars']:.2f}" if sc['odd_bars'] is not None else "")
                  + f"  meters {sc['labels']:2d} (truth {len(tr['meters'])})  {time.time() - t0:.0f}s", flush=True)
    for st in ("odd", "plain"):
        r = [s for t, s in rows if t["set"] == st]
        if r:
            agg = {k: np.mean([x[k] for x in r]) for k in ("found", "extra", "length_ok", "len_any", "beats")}
            odd = [x["odd_bars"] for x in r if x["odd_bars"] is not None]
            print(f"== {st:5s} ({len(r)} songs): found {agg['found']:.3f} extra {agg['extra']:.3f} len_ok {agg['length_ok']:.3f} len_any {agg['len_any']:.3f} beats {agg['beats']:.3f}"
                  + (f"  wrong odd bars {np.mean(odd):.3f}" if odd else ""), flush=True)
    return rows


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "all", grid=sys.argv[3] if len(sys.argv) > 3 else "map")
