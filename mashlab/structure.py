"""Song structure from bar-level features: phrase-aligned sections, labels, energy, stem presence."""
import numpy as np
from scipy.ndimage import median_filter

from .config import STEMS


def bar_spans(bars, n_beats, bpb):
    """[(beat_start, beat_end)] for each bar; the last bar is capped at bpb beats."""
    out = []
    for i, b in enumerate(bars):
        e = bars[i + 1] if i + 1 < len(bars) else min(n_beats, b + bpb)
        out.append((b, min(e, b + bpb * 2)))
    return out


def per_bar(X, spans):
    """Mean of per-beat rows over each bar span. X: (n_beats, d) -> (n_bars, d)."""
    X = np.asarray(X)
    return np.array([X[a:b].mean(0) if b > a else np.zeros(X.shape[1:]) for a, b in spans])


def presence(rms_db, song_peak_db, mix_db):
    """Is the stem audibly playing? Within 24 dB of its own loud parts, above -52 dBFS, and not buried
    more than 22 dB under the full mix (so separation bleed doesn't count as a part)."""
    return (rms_db > song_peak_db - 24) & (rms_db > -52) & (rms_db > mix_db - 22)


def _novelty(F, k=4):
    """Checkerboard-kernel novelty over a bar self-similarity matrix. F: (n_bars, d), rows unit-norm."""
    n = len(F)
    S = F @ F.T
    nov = np.zeros(n)
    for i in range(1, n):
        a0, b1 = max(0, i - k), min(n, i + k)
        past, fut = S[a0:i, a0:i], S[i:b1, i:b1]
        cross = S[a0:i, i:b1]
        nov[i] = (past.mean() + fut.mean()) / 2 - cross.mean()
    return np.maximum(nov, 0)


def sections(bar_feats, energy, pres, seg_breaks, content, min_len=4):
    """Split bars into phrase-aligned sections.

    bar_feats: (n_bars, d) standardized features; energy: (n_bars,) 0..1; pres: dict stem->(n_bars,) bool
    seg_breaks: bar indices that must be boundaries (edits); content: (first_bar, last_bar) with music.
    """
    n = len(bar_feats)
    if n < 2:
        return [dict(start=0, end=n, label="A", name="section", energy=float(energy.mean()) if n else 0)]
    F = bar_feats / (np.linalg.norm(bar_feats, axis=1, keepdims=True) + 1e-9)
    nov = _novelty(F, 4) + 0.5 * _novelty(F, 8)
    nov = nov / (nov.max() + 1e-9)
    # energy / instrumentation changes are strong section cues
    jump = np.zeros(n)
    jump[1:] = np.abs(np.diff(median_filter(energy, 3)))
    for s in STEMS:
        p = pres[s].astype(float)
        jump[1:] += 0.35 * np.abs(np.diff(median_filter(p, 3)))
    score = nov + jump / (jump.max() + 1e-9)

    # phrase phase: which bar offset mod 4 carries the most boundary evidence
    c0, c1 = content
    phase = int(np.argmax([score[[i for i in range(c0, c1) if (i - c0) % 4 == ph]].sum() for ph in range(4)]))
    phase = (c0 + phase) % 4
    bonus = np.array([0.35 if (i - phase) % 8 == 0 else 0.2 if (i - phase) % 4 == 0 else -0.25 for i in range(n)])
    cand = score + bonus

    forced = sorted(set([0, n] + [b for b in seg_breaks if 0 < b < n] +
                        ([c0] if 0 < c0 < n else []) + ([c1] if 0 < c1 < n else [])))
    bounds = set(forced)
    # greedy: take strongest candidates that keep every section >= min_len bars
    thr = np.quantile(cand[1:], 0.75) if n > 8 else 0.5
    for i in np.argsort(-cand):
        if i == 0 or cand[i] < thr:
            continue
        if all(abs(i - b) >= min_len for b in bounds):
            bounds.add(int(i))
    bounds = sorted(bounds)
    # merge slivers created by forced boundaries
    merged = [bounds[0]]
    for b in bounds[1:]:
        if b - merged[-1] < 2 and b != n:
            continue
        merged.append(b)
    if merged[-1] != n:
        merged[-1] = n
    spans = list(zip(merged[:-1], merged[1:]))

    # label by similarity (A, B, C...): greedy clustering on mean section features
    means = [F[a:b].mean(0) for a, b in spans]
    labels, protos = [], []
    for m in means:
        m = m / (np.linalg.norm(m) + 1e-9)
        sims = [float(m @ p) for p in protos]
        if sims and max(sims) > 0.9:
            labels.append(int(np.argmax(sims)))
        else:
            protos.append(m)
            labels.append(len(protos) - 1)
    counts = np.bincount(labels)

    out = []
    for idx, ((a, b), lab) in enumerate(zip(spans, labels)):
        e = float(energy[a:b].mean())
        pr = {s: float(pres[s][a:b].mean()) for s in STEMS}
        out.append(dict(start=int(a), end=int(b), label=chr(65 + lab % 26), energy=round(e, 3),
                        stems={k: round(v, 2) for k, v in pr.items()}, repeats=int(counts[lab])))
    _name(out, c0, c1)
    return out


def _name(secs, c0, c1):
    if not secs:
        return
    es = np.array([s["energy"] for s in secs])
    hi = np.quantile(es, 0.7)
    for i, s in enumerate(secs):
        st = s["stems"]
        vox, drums = st["vocals"] > 0.5, st["drums"] > 0.5
        if s["end"] <= c0 or s["start"] >= c1:
            nm = "silence"
        elif i == 0 or (s["start"] <= c0 + 1 and s["energy"] < hi):
            nm = "intro"
        elif i == len(secs) - 1 and s["energy"] < hi:
            nm = "outro"
        elif s["energy"] >= hi and s["repeats"] > 1:
            nm = "chorus" if vox else "drop"
        elif not drums and s["energy"] < hi:
            nm = "break"
        elif vox:
            nm = "verse"
        else:
            nm = "instrumental"
        s["name"] = nm
