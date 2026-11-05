"""Same-song detection: duplicates, alternate versions / mixes / demos, remixes that keep the harmony.

A mashup of a song with (a version of) itself always "fits" perfectly, so those pairs have to be found
from the audio - titles don't help ("Ligeia" / "Tomb of Liegia", "Let's Go" / "On Letting Go").

Method (cover-song identification, Serra et al. style), on three views of each song:
  harmony  - beat chroma of all pitched stems           (re-recorded versions, covers, demos)
  backing  - beat chroma without the vocal              (instrumental versions of a vocal song)
Each view is time-delay embedded (a vector = a few bars *as they unfold*), compared in the 2 most likely
relative keys and at both beat scales when one grid runs at double speed. A mutual-nearest-neighbour
cross-recurrence matrix is searched for its longest gap-tolerant aligned path (Qmax). The same search
against the other song played BACKWARDS gives the chance level (a drone or a stock loop lines up either
way; a real shared passage only forwards), so evidence = (forward - backward) / shorter length (share).
Some songs (stock loops, one-chord jams) share a bit with everything - "hubs" - so each pair's share is
judged against both songs' own distribution over the whole library: a duplicate is an outlier for BOTH.
"""
import json
import numpy as np
from numba import njit

from . import harmony as H
from .config import DATA

VERSION = 4
KAPPA = 0.1            # nearest-neighbour share for the recurrence matrix
EMBED = {"harmony": 8, "backing": 8}      # beats per embedded vector
NEED = 3.3             # evidence (both views, in units of each song's own spread) to call it the same song
NEED_FAR_TEMPO = 6.0   # ... when the tempos don't match (versions rarely change tempo by > 15 %)
MIN_SHARE = 0.38       # and at least this share of the shorter song must be shared


def _unit(X):
    return X / (np.linalg.norm(X, axis=-1, keepdims=True) + 1e-9)


@njit(cache=True)
def _qmax(B, gap_o=5.0, gap_e=0.5):
    n, m = B.shape
    Q = np.zeros((n + 2, m + 2))
    best = 0.0
    for i in range(2, n + 2):
        for j in range(2, m + 2):
            if B[i - 2, j - 2]:
                Q[i, j] = max(Q[i - 1, j - 1], Q[i - 2, j - 1], Q[i - 1, j - 2]) + 1
            else:
                g1 = Q[i - 1, j - 1] - (gap_o if B[i - 3, j - 3] else gap_e) if i >= 3 and j >= 3 else 0.0
                g2 = Q[i - 2, j - 1] - (gap_o if B[i - 4, j - 3] else gap_e) if i >= 4 and j >= 3 else 0.0
                g3 = Q[i - 1, j - 2] - (gap_o if B[i - 3, j - 4] else gap_e) if i >= 3 and j >= 4 else 0.0
                Q[i, j] = max(0.0, g1, g2, g3)
            if Q[i, j] > best:
                best = Q[i, j]
    return best


def _half(X, mean=False):
    n = len(X) // 2 * 2
    Y = X[:n:2] + X[1:n:2]
    return Y / 2 if mean else Y


def fingerprint(S):
    """S: automash.SongData -> per-view beat sequences (and half-speed versions) + tempo."""
    f = S.f
    a, b = S.info["first_beat"], S.info["last_beat"] + 1
    norm = lambda X: (X / (X.max(1, keepdims=True) + 1e-9)).astype(np.float32)
    back = (H.normalize_rows(f["chroma_bass"]) + H.normalize_rows(f["chroma_other"]))[a:b]
    harm = back + 0.7 * H.normalize_rows(f["chroma_vocals"][a:b])
    return dict(bpm=float(S.bpm), harmony=(norm(harm), norm(_half(harm))), backing=(norm(back), norm(_half(back))))


def _embed(X, m, center=False):
    n = len(X) - m + 1
    if n <= 0:
        return np.zeros((0, X.shape[1] * m), np.float32)
    E = np.hstack([X[i:i + n] for i in range(m)])
    if center:
        E = E - E.mean(1, keepdims=True)
    return _unit(E)


def _oti(ca, cb, n=2):
    ga, gb = ca.mean(0), cb.mean(0)
    sc = [float(ga @ np.roll(gb, k)) for k in range(12)]
    return [int(k) for k in np.argsort(sc)[::-1][:n]]


def _q(Ea, Eb):
    M = Ea @ Eb.T
    ka, kb = max(1, int(KAPPA * M.shape[1])), max(1, int(KAPPA * M.shape[0]))
    ta = np.partition(M, -ka, axis=1)[:, -ka][:, None]
    tb = np.partition(M, -kb, axis=0)[-kb, :][None, :]
    return _qmax(np.ascontiguousarray((M >= ta) & (M >= tb)))


def compare(fa, fb):
    """-> {view: dict(z, share, shift)}: evidence of shared material, beyond chance, per view."""
    r = fa["bpm"] / max(1e-6, fb["bpm"])
    scales = [(0, 0)]
    if 1.6 < r < 2.5:
        scales.append((1, 0))                 # a's grid runs at double speed
    elif 1.6 < 1 / r < 2.5:
        scales.append((0, 1))
    out = {}
    for view, m in EMBED.items():
        chroma = True
        best = dict(z=0.0, share=0.0, shift=0)
        for ia, ib in scales:
            A, B = fa[view][ia], fb[view][ib]
            if min(len(A), len(B)) < 3 * m:
                continue
            for k in (_oti(A, B) if chroma else [0]):
                Bk = np.roll(B, k, axis=1) if chroma else B
                Ea, Eb = _embed(A, m, not chroma), _embed(Bk, m, not chroma)
                n = min(len(Ea), len(Eb))
                d = _q(Ea, Eb) - _q(Ea, Eb[::-1])
                if d / n > best["share"]:
                    best = dict(z=round(float(d / np.sqrt(n)), 2), share=round(float(d / n), 3),
                                shift=int(k if k <= 6 else k - 12))
        out[view] = best
    return out


# ------------------------------------------------------------------ library-wide cache

_PATH = DATA / "same_song.json"
_mem = None


def _load():
    global _mem
    if _mem is None:
        try:
            _mem = json.loads(_PATH.read_text(encoding="utf-8"))
            if _mem.get("version") != VERSION:
                _mem = None
        except Exception:
            _mem = None
        if _mem is None:
            _mem = {"version": VERSION, "pairs": {}}
    return _mem


def _save():
    tmp = _PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps({k: v for k, v in _mem.items() if not k.startswith("_")}), encoding="utf-8")
    tmp.replace(_PATH)


def _pk(a, b):
    return "|".join(sorted((a, b)))


def _compare_many(args):
    fps, pairs = args
    return [(a, b, compare(fps[a], fps[b])) for a, b in pairs]


def scan(fps, stamps, progress=lambda f, m="": None, pool=None):
    """fps: {id: fingerprint} or a function(ids) -> {id: fingerprint} (called only for songs that have
    pairs to compare); stamps: {id: str changing on re-analysis}. Compares every pair not cached yet
    (in parallel when a process pool is given). Returns the pair table."""
    mem = _load()
    ids = sorted(stamps)
    todo = []
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            r = mem["pairs"].get(_pk(a, b))
            if r is None or r.get("stamp") != stamps[a] + stamps[b]:
                todo.append((a, b))
    if not todo:
        return mem["pairs"]
    if callable(fps):
        fps = fps(sorted({s for p in todo for s in p}))
    chunks = [todo[i:i + 150] for i in range(0, len(todo), 150)]
    done = 0

    def take(res):
        nonlocal done
        for a, b, r in res:
            r["stamp"] = stamps[a] + stamps[b]
            r["tempo_ratio"] = round(fps[a]["bpm"] / max(1e-6, fps[b]["bpm"]), 4)
            mem["pairs"][_pk(a, b)] = r
        done += len(res)
        progress(done / len(todo), f"checking for duplicate songs ({done}/{len(todo)} pairs)")

    mem.pop("_dec", None)
    if pool is None or len(todo) < 300:
        for ch in chunks:
            take(_compare_many((fps, ch)))
    else:
        from concurrent.futures import as_completed
        futs = []
        for ch in chunks:
            need = {s for p in ch for s in p}
            futs.append(pool.submit(_compare_many, ({s: fps[s] for s in need}, ch)))
        for f in as_completed(futs):
            take(f.result())
    _save()
    return mem["pairs"]


def _decide():
    """Hub-normalized verdicts for every cached pair -> {pair key: dict(same, strength, share, view)}."""
    mem = _load()
    if mem.get("_dec_n") == len(mem["pairs"]) and "_dec" in mem:
        return mem["_dec"]
    per = {}
    for k, r in mem["pairs"].items():
        a, b = k.split("|")
        for v in EMBED:
            sh = r[v]["share"]
            per.setdefault((v, a), []).append(sh)
            per.setdefault((v, b), []).append(sh)
    stat = {key: (float(np.median(x)), float(np.quantile(x, 0.9))) for key, x in per.items()}
    dec = {}
    for k, r in mem["pairs"].items():
        a, b = k.split("|")
        norm = {}
        for v in EMBED:
            (ma, qa), (mb, qb) = stat[(v, a)], stat[(v, b)]
            spread = max(qa - ma, qb - mb, 0.03)
            norm[v] = (r[v]["share"] - max(qa, qb)) / spread
        strength = min(norm.values())
        share = max(r[v]["share"] for v in EMBED)
        ratio = r.get("tempo_ratio", 1.0)
        dev = min(abs(np.log(ratio * m)) for m in (0.5, 1.0, 2.0))
        need = NEED if dev < np.log(1.15) else NEED_FAR_TEMPO
        identical = all(r[v]["share"] >= 0.9 for v in EMBED)
        same = identical or (strength >= need and share >= MIN_SHARE)
        dec[k] = dict(same=bool(same), strength=round(float(strength), 2), share=round(float(share), 2))
    mem["_dec"], mem["_dec_n"] = dec, len(mem["pairs"])
    return dec


def verdict(a, b):
    """dict(same, strength, share) or None when the pair hasn't been compared."""
    return _decide().get(_pk(a, b))


def same_song(a, b):
    r = verdict(a, b)
    return bool(r and r["same"])


def groups(ids=None):
    """Connected groups of same-song pairs (for showing 'these look like duplicates')."""
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for k, r in _decide().items():
        if not r["same"]:
            continue
        a, b = k.split("|")
        if ids is not None and (a not in ids or b not in ids):
            continue
        parent[find(a)] = find(b)
    out = {}
    for x in list(parent):
        out.setdefault(find(x), []).append(x)
    return [sorted(g) for g in out.values() if len(g) > 1]
