"""The MIDI answer key: load exported features (key/midi/*.npz + .json) and score a map's bars against the truth.
Features are rebuilt from the npz without the audio, so experiments run in seconds."""
import json
from pathlib import Path

import numpy as np

from tempomap import features


class Feat(features.Features):
    def __init__(self, npz):                         # noqa: D401 - no audio: restore the saved curves
        z = np.load(npz, allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        self.sr, self.fps, self.n, self.lag, self.cfps = meta["sr"], meta["fps"], meta["n"], meta["lag"], meta["cfps"]
        self.meta = meta
        self.parts = {k[5:]: z[k].astype(np.float64) for k in z.files if k.startswith("part_")}
        self.chroma = z["chroma"].astype(np.float64)
        zz = np.zeros(self.n)
        g = self.parts.get
        self.env = g("kick", zz) + g("snare", zz) + 0.5 * g("cymbal", zz) + 0.6 * g("bass", zz) + 0.6 * g("other", zz)
        self.hits = g("kick", zz) + g("snare", zz)
        if not self.hits.any():
            self.hits = self.env
        bt = Path(npz).parent.parent / "bt" / Path(npz).name
        if bt.exists():
            b = np.load(bt)
            self.down_logits = b["down_logits"].astype(np.float64)
            self.bt_beats = b["beats"]
            self.bt_downbeats = b["downbeats"]


def songs(root, which="midi"):
    d = Path(root) / which
    out = []
    for j in sorted(d.glob("*.json")):
        tr = json.loads(j.read_text(encoding="utf-8"))
        if tr.get("lag_ms") is not None and abs(tr["lag_ms"]) > 15:
            continue                                 # (the render doesn't line up with the file: no truth)
        if (d / (j.stem + ".npz")).exists():
            out.append((json.loads(j.read_text(encoding="utf-8")), d / (j.stem + ".npz")))
    return out


def score_bars(pred_bars, truth_bars, t0, t1, tol=0.07):
    """Bar lines (s) inside [t0, t1]: found = share of true lines with a predicted one within tol; extra = share of
    predicted lines with no true one; length_ok = share of true bars whose start AND end are both found (so that
    bar came out the right length, in the right place)."""
    p = np.asarray([x for x in pred_bars if t0 <= x <= t1])
    t = np.asarray([x for x in truth_bars if t0 <= x <= t1])
    if not len(t) or not len(p):
        return dict(found=0.0, extra=1.0 if len(p) else 0.0, length_ok=0.0, n=len(t))
    hit_t = np.min(np.abs(t[:, None] - p[None, :]), axis=1) <= tol
    hit_p = np.min(np.abs(p[:, None] - t[None, :]), axis=1) <= tol
    ok = hit_t[:-1] & hit_t[1:]
    return dict(found=float(hit_t.mean()), extra=float(1 - hit_p.mean()), length_ok=float(ok.mean()) if len(ok) else 0.0,
                n=int(len(t)))


def halves(items, key=lambda x: x[0]["set"]):
    """Tune half (A) and check half (B), alternating within each set so both halves are alike."""
    A, B = [], []
    for st in ("odd", "plain"):
        xs = [x for x in items if key(x) == st]
        A += xs[0::2]
        B += xs[1::2]
    return A, B
