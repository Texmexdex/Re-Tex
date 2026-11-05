"""Learned harmonic fit: does this vocal note sit right on these chords?

Trained on the library itself: every vocal over its OWN backing is a known-good combination, and the
same vocal over its own backing moved by k semitones is the same music "in the wrong key". A small
network learns from those pairs which note-against-chord situations sound written-together.

It is note-relative: for each note the singer holds on a beat (top 3 by weight) it looks at the bass,
the other instruments and the key context of the surrounding bars *from that note's point of view*
(everything rotated so the sung note is 0). That makes it transposition-invariant and lets it judge a
note against the whole chord (root, third, colour tones, clashes), not just note pairs.

For a beat it scores all 12 relative transpositions. "Regret" at a given transposition = how much
better the best transposition would have been for that beat (0 = nothing fits better). A melody that
fits several keys (pentatonic lines, chants, rap) has low regret in all of them - it IS easy to mash up.

Measured on this library (5-fold cross-validation by song, sung vocals): ranks a vocal's real key above
the clashing ones (half step, whole step, tritone) for 90 % of songs and 82 % of 8-bar phrases; the old
chroma metric managed 60 % / 52 %.
"""
import json
from pathlib import Path

import numpy as np
from scipy.special import erf

from . import harmony as H

MODEL_PATH = Path(__file__).resolve().parent / "models" / "harmfit.npz"
CTX_BEATS = 16
_model = None


def xcorr(v, b):
    """F[..., j] = sum_i v[..., i] * b[..., (i + j) % 12]: share of (vocal note, backing note j above)."""
    V = np.fft.rfft(v, axis=-1)
    B = np.fft.rfft(b, axis=-1)
    return np.fft.irfft(np.conj(V) * B, n=12, axis=-1)


def context(back, on, beats=CTX_BEATS):
    """Key context per beat: backing chroma smoothed over the surrounding bars (where the backing plays)."""
    from scipy.ndimage import uniform_filter1d
    return H.normalize_rows(uniform_filter1d(back * on[:, None], size=beats, axis=0, mode="nearest"))


def features(vocals, bass, other, ctx):
    """vocals: [tracked melody (n, 12), CQT chroma (n, 12)]; bass / other / ctx: (n, 12).
    -> (n, 5, 12) pitch-class distributions (the model input)."""
    parts = [H.normalize_rows(v) for v in vocals] + [H.normalize_rows(bass), H.normalize_rows(other), H.normalize_rows(ctx)]
    return np.stack(parts, 1).astype(np.float32)


def _gelu_np(x):
    return 0.5 * x * (1.0 + erf(x / np.sqrt(2.0)))


try:
    import math
    from numba import vectorize, float32

    @vectorize([float32(float32)], target="parallel", cache=True)
    def _gelu(x):
        return 0.5 * x * (1.0 + math.erf(x * 0.7071067811865476))
except Exception:           # numba missing: plain numpy
    _gelu = _gelu_np


class Model:
    """Note-relative MLP. Weights: W1 (h, 60) over [melody, chroma, bass, other, ctx] x 12, W2, W3."""

    def __init__(self, params, meta):
        self.p = {k: np.asarray(v, np.float32) for k, v in params.items()}
        self.meta = meta
        self.reps = meta["reps"]
        self.scale = float(meta.get("scale", 5.0))
        self.topk = int(meta.get("topk", 3))
        h = self.p["b1"].shape[0]
        W1 = self.p["W1"].reshape(h, 5, 12)
        self._Wv = W1[:, :2]           # (h, 2, 12)
        self._Wb = W1[:, 2:]           # (h, 3, 12)
        # backing rolled by q, projected: Wb . roll(b, q) == roll(Wb, -q) . b  -> stack all 12 q
        self._Wb_all = np.stack([np.roll(self._Wb, -q, axis=2).reshape(h, 36) for q in range(12)], 0)   # (12, h, 36)
        self._Wb_mat = np.ascontiguousarray(self._Wb_all.transpose(2, 0, 1).reshape(36, 12 * h))
        Wv_all = np.stack([np.roll(self._Wv, p, axis=2).reshape(h, 24) for p in range(12)], 0)    # (12, h, 24)
        self._Wv_mat = np.ascontiguousarray(Wv_all.transpose(2, 0, 1).reshape(24, 12 * h))

    def shift_scores(self, X, chunk=6000):
        """X: (n, 5, 12) -> (n, 12) scores; column k = backing moved up k semitones (= vocal down k)."""
        n = len(X)
        out = np.zeros((n, 12), np.float32)
        h = self.p["b1"].shape[0]
        W2, b2, W3, b3 = self.p["W2"], self.p["b2"], self.p["W3"], self.p["b3"]
        ks = np.arange(12)
        K = self.topk
        for a in range(0, n, chunk):
            x = np.ascontiguousarray(X[a:a + chunk], np.float32) * np.float32(self.scale)
            m = len(x)
            v = X[a:a + chunk, 0]                                          # note weights: tracked melody
            top = np.argsort(-v, axis=1)[:, :K]                            # (m, K) sung notes
            wv = np.take_along_axis(v, top, 1)
            wv = wv / (wv.sum(1, keepdims=True) + 1e-6)
            # vocal part rotated so note p -> 0:  Wv . roll(voc, -p) == roll(Wv, p) . voc
            A_all = (x[:, :2].reshape(m, 24) @ self._Wv_mat).reshape(m * 12, h)
            A = A_all[(np.arange(m)[:, None] * 12 + top).ravel()].reshape(m, K, 1, h)
            # backing rolled by q, for all q
            B = (x[:, 2:].reshape(m, 36) @ self._Wb_mat).reshape(m * 12, h)
            # z(p, k) = A_p + B_{(k - p) % 12}
            q = (ks[None, None, :] - top[:, :, None]) % 12                 # (m, K, 12)
            Bq = B[(np.arange(m)[:, None, None] * 12 + q).ravel()].reshape(m, K, 12, h)
            Bq += A
            Bq += self.p["b1"]
            z = _gelu(Bq).reshape(-1, h)
            z = _gelu((z @ W2.T + b2).astype(np.float32))
            f = (z @ W3.T + b3).reshape(m, K, 12)
            out[a:a + m] = (f * wv[:, :, None]).sum(1)
        return out

    def regret(self, S, vocal_shift):
        """S: (n, 12) shift scores; vocal_shift: semitones the vocal is moved relative to the backing."""
        k = (-int(vocal_shift)) % 12
        return S.max(1) - S[:, k]

    def fit(self, regret):
        """0..1 per beat: 1 = nothing would fit better."""
        return np.exp(-np.asarray(regret) / self.meta["tau"])


def get():
    global _model
    if _model is None:
        with np.load(MODEL_PATH, allow_pickle=False) as z:
            params = {k: z[k] for k in z.files if k != "meta"}
            meta = json.loads(bytes(z["meta"]).decode())
        _model = Model(params, meta)
    return _model


def save(params, meta, path=MODEL_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **{k: np.asarray(v, np.float32) for k, v in params.items()},
             meta=np.frombuffer(json.dumps(meta).encode(), np.uint8))
    global _model
    _model = None
