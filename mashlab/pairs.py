"""Rate every vocal-over-instrumental pairing properly, and remember the results.

For a pair (vocal song V, bed song I) the full arrangement is planned for the most promising keys (see
planner.py) and the finished plan is rated - so the rating describes the mashup you'd actually get,
not a statistic about the two songs:

  harmony   placed vocal against the bed's chords, note by note (learned model, harmfit.py), scaled so
            0 = a random key and 100 = as well as a vocal typically fits its own song
  tempo     how far the vocal / bed must be stretched (vocals tolerate less)
  key       how far the vocal / bed must be pitch-shifted
  structure how much of the bed carries vocal, hooks on the big sections, lines cut only at gaps
  groove    swing vs straight feel
  acapella  how clean the vocal stem is (bleed)
Overall = product of these, so one bad aspect can't hide behind good ones.

Pairs of the same song (duplicates, versions, instrumentals of each other - dupes.py) are never rated.
Results are stored in data/ratings.sqlite keyed by both songs' analysis state + the model version, so
re-running only computes what changed.
"""
import json
import math
import os
import sqlite3
import threading
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from . import dupes, harmfit, library, mashfeat, planner
from .config import DATA

VERSION = 6     # 6: slots + vocal entries on 2-bar measures (SongData.measure_starts); first slot pays its entry cut
# 5: lines continue through 1-bar bed slots without skipping a bar (planner)
DB_PATH = DATA / "ratings.sqlite"
MAX_VOCAL_STRETCH = 0.16
MAX_BED_STRETCH = 0.20
MAX_VOCAL_SHIFT = 5
MAX_BED_SHIFT = 3
_db_lock = threading.Lock()


# ------------------------------------------------------------------ component scores

def tempo_quality(sv, si):
    return 1 / (1 + (abs(sv) / 0.11) ** 4) * 1 / (1 + (abs(si) / 0.14) ** 4)


def key_quality(vs, bs):
    return 1 / (1 + (abs(vs) / 3.5) ** 4) * 1 / (1 + (abs(bs) / 3.0) ** 4)


def groove_quality(V, I, rv):
    """1 = same feel, down to 0.7 for straight vs heavily swung. Only judged at the 8th-note level with
    both songs at the same beat scale (16th-level swing can't be measured reliably from onsets)."""
    gv, gi = V.groove(), I.groove()
    if not gv or not gi or abs(rv - 1) > 0.01 or min(gv["act8"], gi["act8"]) < 0.15:
        return 1.0, None
    d = abs(gv["s8"] - gi["s8"])
    return float(1 - 0.3 * np.clip((d - 0.15) / 0.35, 0, 1)), round(d, 2)


def _best_mult(src_bpm, bpm):
    best = min((0.5, 1.0, 2.0), key=lambda m: abs(math.log(bpm / (src_bpm * m))))
    return best, bpm / (src_bpm * best) - 1.0


def combo_bpm(I, V):
    """Bed counts double (it's the backbone; vocals are also the more fragile to stretch - so the
    stretch is split, leaning on the bed, rounded to a half BPM)."""
    m, _ = _best_mult(V.bpm, I.bpm)
    x = math.exp((2 * math.log(I.bpm) + math.log(V.bpm * m)) / 3)
    return round(x * 2) / 2


def bed_transpose(r):
    """Split a relative shift r between vocal (moves r + s) and bed (moves s). Shifting one vocal stem
    (formant-preserving) is cleaner than shifting the whole instrumental, so the bed only moves when the
    vocal alone would have to go far: r = 2 -> vocal +2; 3 -> vocal +2, bed -1; 5 -> vocal +4, bed -1."""
    return min(range(-MAX_BED_SHIFT, MAX_BED_SHIFT + 1), key=lambda s: (r + s) ** 2 + 3.5 * s * s)


H_NEUTRAL = 0.75        # harmony credited where it can't be judged (unpitched vocal, chord-less music)


def _harmony_pct(p, meta):
    """From a plan: 0 = what a random song in a random key gives once planned, 1 = a vocal planned over
    its own song - measured on the beats where key matters, blended toward neutral where it can't be
    judged (screams, rap, drum-only passages)."""
    lo = meta.get("plan_fit_random", meta["fit_random"])
    hi = meta.get("plan_fit_own", meta["fit_own"])
    h = float(np.clip((p["fit_info"] - lo) / (hi - lo), 0, 1.2))
    rel = float(np.clip(p["relevance"], 0, 1))
    return rel * h + (1 - rel) * H_NEUTRAL, rel


def structure_quality(p, density):
    target = 0.35 + 0.4 * density
    cov = min(1.0, p["bars"] / max(1.0, target * p["total_bars"]))
    hook = min(1.0, p["hook_cover"] / 0.6)
    cuts = 1 - p["cut_mean"]
    rep = 1 - min(1.0, p["repeats"] / 8)
    return float(0.35 * cov + 0.25 * hook + 0.3 * cuts + 0.1 * rep), dict(cover=round(cov, 2), hooks=round(hook, 2),
                                                                         cuts=round(cuts, 2))


# ------------------------------------------------------------------ one pair

def screen(V, I):
    """Cheap hard checks. -> None if workable, else a reason string."""
    if V.id == I.id:
        return "same song"
    if dupes.same_song(V.id, I.id):
        return "same song (duplicate / version)"
    if not V.has_real_vocals:
        return "no clear vocal"
    if V.bpb != I.bpb:
        return f"different meter ({V.bpb} vs {I.bpb} beats per bar)"
    bpm = combo_bpm(I, V)
    _, sv = _best_mult(V.bpm, bpm)
    _, si = _best_mult(I.bpm, bpm)
    if abs(sv) > MAX_VOCAL_STRETCH or abs(si) > MAX_BED_STRETCH:
        return f"tempos too far apart ({V.bpm:.0f} vs {I.bpm:.0f} BPM)"
    return None


def _lyrical_factor(V, I, lyrics_mode):
    """The bounded, mostly-above-1 lyrical tie-breaker (design A.4) plus its A.5 neutral fallback, computed
    once per pair (it does not depend on the key shift). Returns a dict with:

        factor        - the clip(L_factor, lo_clamp, hi_clamp) multiplier applied to the F1 overall
        lyrics        - round(Lraw*100) (or None when neutral/off)
        lyrics_mode   - the mode in effect
        lyrics_judgeable - min(judgeable(V), judgeable(I))
        lyrics_text   - plain-language breakdown, ALWAYS surfacing the lower of the two asr_conf (A.5 HIGH-1)

    L_factor is forced to 1.0 (neutral) when ANY A.5 trigger fires; direction is baked into Lraw so
    mode_sign = +1 for both resonance and contrast."""
    from . import concept
    off = (lyrics_mode == "off") or (lyrics_mode is None)
    lyV = getattr(V, "lyrics", None)
    lyI = getattr(I, "lyrics", None)
    coV = getattr(V, "concept", None)
    coI = getattr(I, "concept", None)

    def _q(ly):
        return (ly or {}).get("quality", {}) or {}

    asr_v = _q(lyV).get("asr_conf", 0.0)
    asr_i = _q(lyI).get("asr_conf", 0.0)
    lo_conf = min(asr_v, asr_i)
    jV = (coV or {}).get("judgeable", concept.judgeable(lyV) if lyV else 0.0)
    jI = (coI or {}).get("judgeable", concept.judgeable(lyI) if lyI else 0.0)
    judge = min(jV, jI)

    # A.5 neutral predicate.
    # Thresholds relaxed from the conservative design defaults (judgeable 0.35, asr 0.5) to real-world
    # values (0.10 / 0.25): isolated-vocal Whisper transcripts of processed/layered vocals (e.g. TOOL) land
    # around asr_conf ~0.27 / judgeable ~0.14 yet still carry usable thematic signal. The design kept these
    # high to NEVER fabricate a match; the user prefers the lyrical tie-breaker to actually engage on
    # imperfect transcripts (it stays a bounded ±5% nudge, so a slightly-noisy signal can only re-order
    # already-close pairings — it can't rescue a musically-bad one). Raise them back toward 0.35/0.5 if
    # mashups start resonating on garbled text.
    _MIN_JUDGE = 0.10
    _MIN_ASR = 0.25
    neutral_reason = None
    if off:
        neutral_reason = "off"
    elif coV is None or coI is None or lyV is None or lyI is None:
        neutral_reason = "no lyrics"
    elif jV < _MIN_JUDGE or jI < _MIN_JUDGE:
        neutral_reason = "low judgeable"
    elif asr_v < _MIN_ASR or asr_i < _MIN_ASR:
        neutral_reason = "low ASR confidence"
    elif (not _q(lyV).get("usable", True)) or (not _q(lyI).get("usable", True)):
        neutral_reason = "vocal not usable"
    elif (lyV.get("lang") and lyI.get("lang") and lyV.get("lang") != lyI.get("lang")):
        neutral_reason = "languages differ"

    if neutral_reason is not None:
        if neutral_reason == "off":
            text = "Lyrics: not counted (lyrics off)."
        elif neutral_reason == "no lyrics":
            text = "Lyrics: no transcript — not counted."
        else:
            text = f"Lyrics: {neutral_reason} (ASR confidence {lo_conf:.2f}) — not counted."
        return dict(factor=1.0, lyrics=None, lyrics_mode=lyrics_mode, lyrics_judgeable=round(judge, 2),
                    lyrics_text=text)

    alpha, (lo_clamp, hi_clamp), (lo, hi) = concept.lyrics_params()
    Lraw = concept.lraw(coV, coI, lyrics_mode, lo, hi)
    mode_sign = 1.0
    L_factor = 1.0 + alpha * judge * (Lraw - 0.5) * mode_sign
    factor = float(np.clip(L_factor, lo_clamp, hi_clamp))
    moodV = (coV.get("tags") or {}).get("mood")
    moodI = (coI.get("tags") or {}).get("mood")
    pct = round(Lraw * 100)
    rel = "resonance" if lyrics_mode == "resonance" else "contrast"
    if moodV and moodI:
        text = f"Lyrics: {moodV}/{moodI} — {rel} {pct}% (ASR confidence {lo_conf:.2f})."
    else:
        text = f"Lyrics: {rel} {pct}% (ASR confidence {lo_conf:.2f})."
    return dict(factor=factor, lyrics=pct, lyrics_mode=lyrics_mode, lyrics_judgeable=round(judge, 2),
                lyrics_text=text)


def evaluate(V, I, density=0.5, shifts=None, keep_plan=False, lyrics_mode="off"):
    """Plan + rate V's vocal over I. Tries the most promising keys (or exactly `shifts`).

    `lyrics_mode` in {off, resonance, contrast} (design A.4) folds a bounded lyrical tie-breaker into the F1
    overall; `off` (the default, and backward-compatible) leaves the pre-lyrics rating unchanged within
    rounding."""
    why = screen(V, I)
    if why:
        return dict(workable=False, reason=why)
    lyr = _lyrical_factor(V, I, lyrics_mode)
    meta = harmfit.get().meta
    bpm = combo_bpm(I, V)
    mV, sv = _best_mult(V.bpm, bpm)
    mI, si = _best_mult(I.bpm, bpm)
    rv = mI / mV
    grid = planner.Grid(V, I, rv)
    # quick key screen: fit of every possible alignment, per relative shift
    cand = []
    for r in range(-6, 6):
        if shifts is not None and r not in shifts:
            continue
        s_bed = bed_transpose(r)
        s_voc = r + s_bed
        if abs(s_voc) > MAX_VOCAL_SHIFT:
            continue
        mean, _ = grid.fit_matrix(r)
        quick = float((mean * grid.vocal).sum() / max(1e-6, grid.vocal.sum()))
        cand.append((quick * key_quality(s_voc, s_bed) ** 0.5, r, s_voc, s_bed))
    cand.sort(reverse=True)
    groove, swing_diff = groove_quality(V, I, rv)
    bleed = V.bleed()
    tq = tempo_quality(sv, si)
    out = []
    for _, r, s_voc, s_bed in cand[:4] if shifts is None else cand:
        p = planner.plan_vocals(grid, r, density)
        h, rel = _harmony_pct(p, meta)
        kq = key_quality(s_voc, s_bed)
        sq, sdet = structure_quality(p, density)
        music = float(np.clip(0.6 + 0.8 * p["bed_harm"], 0.6, 1.0))          # drums-only bed: 0.6
        length = float(np.clip(0.55 + 0.45 * p["total_bars"] / 48, 0.55, 1.0))  # < 48 usable bars: shorter mashup
        overall = (100 * (max(min(h, 1.0), 0.02) ** 0.8) * tq * kq * (0.6 + 0.4 * sq) * groove * (1 - 0.35 * bleed)
                   * music * length)
        overall_v2 = overall * lyr["factor"]        # A.4 bounded lyrical tie-breaker (factor == 1.0 when neutral/off)
        res = dict(workable=True, rating=round(overall_v2, 1), shift_rel=r, shift_vocal=s_voc, shift_bed=s_bed,
                   bpm=bpm, mult_vocal=mV, mult_bed=mI, stretch_vocal=round(sv * 100, 1), stretch_bed=round(si * 100, 1),
                   harmony=round(min(h, 1.0) * 100), harmony_raw=round(h * 100), tempo=round(tq * 100), key=round(kq * 100), structure=round(sq * 100),
                   groove=round(groove * 100), acapella=round((1 - bleed) * 100), swing_diff=swing_diff,
                   vocal_bars=p["bars"], total_bars=p["total_bars"], hook_cover=round(p["hook_cover"], 2),
                   judgeable=round(rel, 2), music=round(music * 100), length=round(length * 100),
                   cut_bad=p["cut_bad"], entries=p["n_entries"], stability=round(V.stability, 2),
                   lyrics=lyr["lyrics"], lyrics_mode=lyr["lyrics_mode"],
                   lyrics_judgeable=lyr["lyrics_judgeable"], lyrics_text=lyr["lyrics_text"], **sdet)
        if keep_plan:
            res["_plan"] = p
        out.append(res)
    if not out:
        return dict(workable=False, reason="needs too big a key change")
    out.sort(key=lambda d: (-d["rating"], -d["harmony_raw"]))
    best = dict(out[0])
    best["alternatives"] = [dict(shift_rel=o["shift_rel"], shift_vocal=o["shift_vocal"], shift_bed=o["shift_bed"],
                                 rating=o["rating"], harmony=o["harmony"]) for o in out[1:]]
    return best


# ------------------------------------------------------------------ storage

def _db():
    con = sqlite3.connect(DB_PATH, timeout=30)
    con.execute("""CREATE TABLE IF NOT EXISTS ratings (v TEXT, i TEXT, density REAL, stamp TEXT,
                   data TEXT, PRIMARY KEY (v, i, density))""")
    return con


def stamp(sid):
    d = library.song_dir(sid)
    parts = []
    for f in ("song.json", "mash.npz"):
        p = d / f
        parts.append(str(int(p.stat().st_mtime)) if p.exists() else "0")
    return "-".join(parts)


def _pair_stamp(v, i):
    m = harmfit.get().meta.get("version", 0)
    return f"{VERSION}.{m}.{dupes.VERSION}:{stamp(v)}:{stamp(i)}"


def cached(v, i, density=0.5):
    with _db_lock:
        con = _db()
        row = con.execute("SELECT stamp, data FROM ratings WHERE v=? AND i=? AND density=?",
                          (v, i, round(density, 2))).fetchone()
        con.close()
    if row and row[0] == _pair_stamp(v, i):
        return json.loads(row[1])
    return None


def store(rows, density=0.5):
    with _db_lock:
        con = _db()
        con.executemany("INSERT OR REPLACE INTO ratings VALUES (?, ?, ?, ?, ?)",
                        [(v, i, round(density, 2), _pair_stamp(v, i), json.dumps(d)) for v, i, d in rows])
        con.commit()
        con.close()


def load_all(ids, density=0.5):
    """{(v, i): result} for current ratings between songs in ids."""
    ids = set(ids)
    with _db_lock:
        con = _db()
        rows = con.execute("SELECT v, i, stamp, data FROM ratings WHERE density=?", (round(density, 2),)).fetchall()
        con.close()
    stamps = {}
    out = {}
    m = harmfit.get().meta.get("version", 0)
    for v, i, st, data in rows:
        if v not in ids or i not in ids:
            continue
        for s in (v, i):
            if s not in stamps:
                stamps[s] = stamp(s)
        if st == f"{VERSION}.{m}.{dupes.VERSION}:{stamps[v]}:{stamps[i]}":
            out[(v, i)] = json.loads(data)
    return out


# ------------------------------------------------------------------ parallel workers

def _w_features(sid):
    from . import library as L, mashfeat as MF
    info = L.read(sid)
    MF.compute(L.song_dir(sid), info, lambda s: L.load_stem(sid, s))
    return sid


def _w_fingerprints(ids):
    from . import automash as A, dupes as D
    return {s: D.fingerprint(A.get_song(s)) for s in ids}


def _w_rate(v, beds, density, lyrics_mode="off"):
    from . import automash as A
    V = A.get_song(v)
    out = []
    for i in beds:
        try:
            out.append((v, i, evaluate(V, A.get_song(i), density, lyrics_mode=lyrics_mode)))
        except Exception as e:           # one broken song must not sink the whole run
            out.append((v, i, dict(workable=False, reason=f"error: {type(e).__name__}: {e}")))
    return out


def _single_thread_init():
    """Worker setup: one thread each (the pool itself is the parallelism), and quit with the app."""
    from .procpool import watch_parent
    watch_parent()
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(1)
    except Exception:
        pass
    try:
        import numba
        numba.set_num_threads(1)
    except Exception:
        pass


def _pool(n=None):
    n = n or max(2, min(16, (os.cpu_count() or 4) - 4))
    return ProcessPoolExecutor(max_workers=n, initializer=_single_thread_init)


def rate_all(song_ids, density=0.5, vocal_ids=None, bed_ids=None, progress=lambda f, m="": None,
             cancelled=lambda: False, involving=None, lyrics_mode="off"):
    """Make sure every workable (vocal, bed) pair among song_ids is rated. Heavy parts run in parallel.

    `lyrics_mode` (off/resonance/contrast, design A.4) is threaded into every evaluate(); the default "off"
    reproduces the pre-lyrics rating within rounding, so the live path is unchanged."""
    from . import automash as A
    ready = [s["id"] for s in library.list_songs() if s.get("status") == "ready"]
    ids = [s for s in dict.fromkeys(song_ids) if s in set(ready)]
    if len(ids) < 2:
        raise ValueError("pick at least two analysed songs")
    vocal_ids = [s for s in (vocal_ids or ids) if s in ids]
    bed_ids = [s for s in (bed_ids or ids) if s in ids]
    pool = _pool()
    try:
        # 1) melody / groove features (pYIN on the vocal: the slow part, once per song)
        need = ids if not involving else [s for s in ids if s == involving or s in vocal_ids or s in bed_ids]
        missing = [s for s in need if mashfeat.load(library.song_dir(s)) is None]
        if missing:
            futs = [pool.submit(_w_features, s) for s in missing]
            for n, f in enumerate(as_completed(futs)):
                if cancelled():
                    raise RuntimeError("cancelled")
                try:
                    f.result()
                except Exception as e:
                    print("mash features failed:", e)
                progress(0.35 * (n + 1) / len(missing), f"tracking vocal melodies ({n + 1}/{len(missing)} songs)")
        # 2) same-song check across the whole library (so hub statistics are stable)
        progress(0.36, "checking for duplicate songs")
        stamps = {s: str(int((library.song_dir(s) / "song.json").stat().st_mtime)) for s in ready}

        def fingerprints(need):
            fps = {}
            for r in pool.map(_w_fingerprints, [need[i:i + 12] for i in range(0, len(need), 12)]):
                fps.update(r)
            return fps
        dupes.scan(fingerprints, stamps, lambda f, m: progress(0.36 + 0.1 * f, m), pool=pool)
        # 3) rate every pair not rated yet
        songs = {s: A.get_song(s) for s in ids}
        have = load_all(ids, density)
        todo = {}
        n_pairs = 0
        for v in vocal_ids:
            if not songs[v].has_real_vocals:
                continue
            for i in bed_ids:
                if i == v or (involving and involving not in (v, i)):
                    continue
                n_pairs += 1
                if (v, i) not in have:
                    todo.setdefault(v, []).append(i)
        n_todo = sum(len(x) for x in todo.values())
        done = 0
        futs = []
        for v, beds in todo.items():
            for k in range(0, len(beds), 20):
                futs.append(pool.submit(_w_rate, v, beds[k:k + 20], density, lyrics_mode))
        for f in as_completed(futs):
            if cancelled():
                raise RuntimeError("cancelled")
            res = f.result()
            store(res, density)
            done += len(res)
            progress(0.47 + 0.53 * done / max(1, n_todo), f"rating mashups ({done}/{n_todo})")
        return dict(pairs=n_pairs, rated=n_todo)
    finally:
        for p in list(getattr(pool, "_processes", {}).values()):
            try:
                p.terminate()
            except Exception:
                pass
        pool.shutdown(wait=False, cancel_futures=True)


def table(song_ids, density=0.5, vocal_ids=None, bed_ids=None, lyrics_mode="off"):
    """All rated pairs among the songs, best first, plus the pairs skipped and why. `lyrics_mode` is accepted
    for request-body symmetry (A.4); table reads already-rated rows from load_all, so it is a presentation
    hint only — rate_all is what re-rates under a given mode."""
    from . import automash as A
    ids = list(dict.fromkeys(song_ids))
    vocal_ids = set(vocal_ids or ids)
    bed_ids = set(bed_ids or ids)
    res = load_all(ids, density)
    titles = {s: (library.read(s) or {}).get("title", s) for s in ids}
    rows, skipped = [], {}
    for (v, i), d in res.items():
        if v not in vocal_ids or i not in bed_ids:
            continue
        if d.get("workable"):
            rows.append(dict(vocal=v, bed=i, vocal_title=titles[v], bed_title=titles[i], **d))
        else:
            skipped.setdefault(d.get("reason", "?").split(" (")[0], []).append([titles[v], titles[i]])
    rows.sort(key=lambda r: (-r["rating"], -r.get("harmony_raw", 0)))
    # versions of the same song would fill the list with near-copies: keep the best row per
    # (vocal song, music song) and mention the other versions
    gid = {}
    for g in dupes.groups(set(ids)):
        for x in g:
            gid[x] = g[0]
    best, out = {}, []
    for r in rows:
        k = (gid.get(r["vocal"], r["vocal"]), gid.get(r["bed"], r["bed"]))
        if k in best:
            b = best[k]
            b.setdefault("also", []).append(f"{r['vocal_title']} over {r['bed_title']}: {round(r['rating'])}")
            continue
        best[k] = r
        out.append(r)
    rows = out
    groups = [[titles[s] for s in g] for g in dupes.groups(set(ids))]
    has_v = {s: bool((library.read(s) or {}).get("has_vocals")) for s in ids}
    expected = sum(1 for v in vocal_ids if has_v.get(v) for i in bed_ids if i != v)
    total = sum(1 for (v, i) in res if v in vocal_ids and i in bed_ids)
    return dict(rows=rows, skipped={k: dict(count=len(v), examples=v[:6]) for k, v in skipped.items()},
                same_song_groups=groups, total=total, expected=expected)
