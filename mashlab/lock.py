"""🔒 Lock to marks: a song whose own timing drifts (a live take, a re-upload with its speed altered) made steady once,
so it sits on the grid; toggled per song in a project.

make(sid) warps each of the song's stems so every one of its marks (yours if you edited them, else the detector's -
those snapped onto the drum hit next to them first, tighten()) lands on a steady grid at the song's average mark spacing - Rubber Band R3 with the click-pilot re-timing, pitch kept
(warp.render) - and saves them next to the originals (stems_locked/). The originals are never touched. song.json
"lock" = {rev, sig, bpm, beats (where each of the song's beats lands in the locked audio), scale, ...}.

A lock belongs to the marks, stems and beat grid it was made from (sig): edit the marks, re-separate or re-analyze the
song and the lock is stale - the song plays as recorded until it's locked again (project.song_tspec checks).

Two kinds (lock["kind"]; the same marks, the same place on disk - re-locking the other way replaces it):
  "marks"   every mark lands on the grid (above). Right when the marks are right; where they aren't, the audio
            between them is played faster or slower than it was recorded, beat by beat.
  "smooth"  the song is pulled onto the grid along a smooth line through its marks (smooth_marks): it follows
            drift that lasts, never a single beat or bar. A bunched bar, a mark that sits off, the detector's
            jitter leave the audio alone. Nothing is snapped to drum hits.
"""
import json
import math
import shutil
import time
from pathlib import Path

import numpy as np

from .config import SR, STEMS


def marks_of(info, marks=None):
    """The marks a lock follows: `marks` (a project's own timing for the song) if given, else yours if you edited
    them, else the detector's (inside the song)."""
    um = marks if marks and marks.get("t") else info.get("user_marks")
    if um and um.get("t"):
        return [float(x) for x in um["t"]]
    dur = float(info.get("duration") or info["beats"][-1])
    return [float(x) for x in info["beats"] if -1e-6 <= x <= dur + 1e-6]


def sig(info, marks=None):
    """Identifies the marks + stems a lock is made from (web/js computes the same: timing.js lockSig)."""
    m, b = marks_of(info, marks), info["beats"]
    ms = lambda x: math.floor(x * 1000 + 0.5)            # (same rounding as JS Math.floor(x * 1000 + 0.5))
    return f"{len(m)}-{sum(ms(x) for x in m)}-r{int(info.get('stems_rev', 0) or 0)}-b{len(b)}-{ms(b[0])}"


def find(info, marks=None):
    """The lock made from these marks (a project's own timing, or by default the song's grid) and the song's current
    stems: its current lock ("lock", stems_locked/) or one kept in "locks" (stems_locked/<rev>/). None if none."""
    s = sig(info, marks)
    for lk in [info.get("lock") or {}, *(info.get("locks") or {}).values()]:
        if lk.get("sig") == s and lk.get("rev"):
            return lk
    return None


def current(info):
    """The song's lock if it matches its current marks and stems, else None."""
    lk = info.get("lock")
    return lk if lk and lk.get("sig") == sig(info) and lk.get("rev") else None


# ---------------------------------------------------------------- what a lock does to the song's speed
# A lock moves every mark onto a steady grid, so wherever the marks are wrong the audio between them is played
# faster or slower than it was recorded. Measured on the user's library (Nov 10): 20 of 77 locked songs played 2 s
# or more of the song over 20 % off - all from the detector's marks, most of them stretches too short or too
# uneven for beatfix.slips (it looks at whole sections of 12+ beats). These say what a lock will do before it's
# made (check), and what an existing one did (did).

OFF, WRONG = 0.05, 0.20        # off the song's own speed: by more than 5 %; plainly wrong: by more than 20 %
WRONG_SEC = 2.0                # this much of the song played plainly wrong, and the lock asks first (wrong_speed)
JOIN = 2                       # beats: wrong stretches this close together are listed as one
LIST_MAX = 8                   # the longest wrong stretches are listed; the rest are counted


def speed_report(t, land):
    """What a lock does to the song's speed. t: times of the song's beats (or marks) as recorded, s; land: where the
    lock puts them, s. Between two of them the audio plays (their spacing as recorded) / (their spacing locked)
    times as fast as recorded - taken against the song's main speed (the time-weighted median), so the tempo the lock
    settles on doesn't count, only what changes within the song.
    -> {seconds, sec5, sec20 (how much of the song plays more than 5 % / 20 % off), fastest, slowest (speed
    factors, 1 = as recorded), parts: [[start s, end s, speed, lowest, highest]] - the stretches over 20 % off, in
    order: `speed` is what most of the stretch plays at; lowest / highest differ from it where the beats inside are
    uneven (a bunched bar plays one beat slow and the next ones fast)}, or None with too few beats."""
    t, land = np.asarray(t, float), np.asarray(land, float)
    if len(t) != len(land) or len(t) < 9:
        return None
    d, e = np.diff(t), np.diff(land)
    ok = (d > 1e-4) & (e > 1e-4)
    if ok.sum() < 8:
        return None
    sp = np.ones(len(d))
    sp[ok] = d[ok] / e[ok]
    idx = np.flatnonzero(ok)
    order = idx[np.argsort(sp[idx])]
    cum = np.cumsum(d[order])
    main = float(sp[order][min(len(order) - 1, int(np.searchsorted(cum, cum[-1] / 2)))])
    rel = sp / main
    off = np.where(ok, np.abs(rel - 1), 0.0)
    parts, run = [], None
    for i in np.flatnonzero(off > WRONG):
        if run and i - run[1] <= JOIN:
            run[1] = int(i)
        else:
            run = [int(i), int(i)]
            parts.append(run)

    def part(a, b):
        w = np.arange(a, b + 1)
        w = w[off[w] > WRONG]                              # the beats of the stretch that are off (not the joins)
        o = w[np.argsort(rel[w])]
        c = np.cumsum(d[o])
        typical = rel[o][min(len(o) - 1, int(np.searchsorted(c, c[-1] / 2)))]
        return [round(float(t[a]), 2), round(float(t[b + 1]), 2), round(float(typical), 3),
                round(float(rel[w].min()), 3), round(float(rel[w].max()), 3)]
    parts = [part(a, b) for a, b in parts]
    return dict(seconds=round(float(d[ok].sum()), 1), sec5=round(float(d[off > OFF].sum()), 1),
                sec20=round(float(d[off > WRONG].sum()), 1), fastest=round(float(rel[ok].max()), 3),
                slowest=round(float(rel[ok].min()), 3), parts=parts)


def would(marks):
    """speed_report for a lock made from these marks (as they are: the lock also snaps the detector's marks onto
    the drum hit next to them, tighten(), which can add a few % from beat to beat)."""
    m = np.asarray(marks, float)
    if len(m) < 9:
        return None
    return speed_report(m, np.arange(len(m)) * float(m[-1] - m[0]) / (len(m) - 1))


def did(info, lk=None):
    """speed_report for a lock that exists (the song's current one, or `lk`): from where it put each of the song's
    own beats. None when the song's beats have changed since (the record no longer says)."""
    lk = lk or info.get("lock") or {}
    b, lb = np.asarray(info.get("beats") or [], float), np.asarray(lk.get("beats") or [], float)
    if len(b) != len(lb) or len(b) < 9:
        return None
    inside = (b >= -1e-6) & (b <= float(info.get("duration") or b[-1]) + 1e-6)
    return speed_report(b[inside], lb[inside])


def wrong_speed(marks):
    """The stretches a lock to these marks would play at the wrong speed, for the lock to ask about before it's
    made: beatfix.slips (a whole section counted at another tempo, or switched to half / double time) plus, beat by
    beat, whatever would play more than 20 % off - bunched or drawn-out bars, short miscounts - once that adds up to
    WRONG_SEC (the LIST_MAX longest). -> [(start s, end s, the marks' tempo there against the song's)], in order."""
    from . import beatfix
    m = np.asarray(marks, float)
    out = [tuple(x) for x in beatfix.slips(m)] if len(m) >= 8 else []
    rep = would(m)
    if rep and rep["sec20"] >= WRONG_SEC:
        new = [p for p in rep["parts"] if not any(p[0] < y + 1e-6 and x < p[1] + 1e-6 for x, y, _ in out)]
        for a, b, sp, _, _ in sorted(new, key=lambda p: p[0] - p[1])[:LIST_MAX]:
            out.append((a, b, round(1.0 / sp, 3)))
    return sorted(out)


def is_take(info):
    """A Reimagine take (reimagine.save): its lock is made to pull it into line with the mix, bar for bar."""
    return str(info.get("source") or "").replace("\\", "/").startswith("data/yue2/")


def report_all():
    """Every locked song with what its lock did to the song's speed (did) and the saved projects that have the lock
    switched on - to review locks made before the lock asked about wrong-speed stretches. Reimagine takes are left
    out (their locks are meant to change the speed). -> [{id, title, current: it still matches the song's marks,
    speed, projects: [names]}], the most changed first."""
    from . import library
    from . import project as P
    on = {}
    for f in P.PROJECTS.glob("*.json"):
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        used = {c.get("song") for c in d.get("clips") or []}
        for sid, s in (d.get("songs") or {}).items():
            if s.get("lock") and sid in used:
                on.setdefault(sid, []).append(d.get("name") or d.get("id"))
    out = []
    for d in library.SONGS.iterdir():
        info = library.read(d.name) if d.is_dir() else None
        if not info or not info.get("lock") or is_take(info):
            continue
        rep = did(info)
        if rep:
            row = dict(id=info["id"], title=info.get("title", info["id"]), current=current(info) is not None,
                       speed=rep, projects=sorted(on.get(info["id"], [])), kind=(info["lock"].get("kind") or "marks"),
                       yours=bool((info.get("user_marks") or {}).get("t")))
            row["fix"] = fix_of(info) if rep["sec20"] >= WRONG_SEC else None
            out.append(row)
    return sorted(out, key=lambda r: (-r["speed"]["sec20"], -r["speed"]["sec5"]))


def fix_of(info):
    """What "Repair, then lock" would leave of a lock that plays parts off speed: {method, sec20} for the
    detector's marks (worked out without the drum stem, so within a second or two of the real thing), None for
    marks of the user's own (those are only ever changed in the Beat grid window)."""
    if (info.get("user_marks") or {}).get("t") or not info.get("beats"):
        return None
    try:
        _, rp = repaired(info, None)
        return dict(method=rp["method"], sec20=rp["sec20"])
    except Exception:
        return None


def in_use(sid, lock_sig):
    """Does a saved project play this song locked to the grid with this signature?"""
    from . import library
    from . import project as P
    info = library.read(sid)
    for f in P.PROJECTS.glob("*.json"):
        try:
            s = (json.loads(f.read_text(encoding="utf-8")).get("songs") or {}).get(sid)
        except (OSError, ValueError):
            continue
        if s and s.get("lock") and sig(info, s.get("marks")) == lock_sig:
            return True
    return False


def keep_current(sid, info):
    """Move the song's current lock out of the way, into "locks" and stems_locked/<rev>/ - a project still plays it
    (its grid is the one this lock was made from). Writes and returns the updated info."""
    from . import library
    lk = info.get("lock")
    if not lk or not lk.get("rev"):
        return info
    d = library.song_dir(sid) / "stems_locked"
    dst = d / lk["rev"]
    dst.mkdir(parents=True, exist_ok=True)
    for f in d.glob("*.flac"):
        f.replace(dst / f.name)
    info.setdefault("locks", {})[lk["sig"]] = lk
    info.pop("lock", None)
    library.write(sid, info)
    return info


KINDS = ("marks", "smooth")
SMOOTH_REACH = (0.015, 0.05)   # s a mark may sit off the smooth line: twice the marks' own roughness, within these
SMOOTH_BEATS = 9               # the running median that takes single marks / a bunched bar out (two bars and a beat)
SMOOTH_WIDE = 0.03             # s: a short speed change is smoothed over only if the beats then stay this close (was
                               # 0.08 of a bar - ~100 ms at 184 BPM: whole stretches of Hooker With A Penis sat 50-90 ms
                               # off the grid, red in the arrangement, Nov 18; at 30 ms: 5 more bends, max 25 ms)


def smooth_marks(marks):
    """The marks as a smooth lock sees them: on a line with as few bends as keep every mark within reach, after a
    running median has taken out what only lasts a beat or a bar (gridmash.calm / fit_points / ease - the same
    following a fitted song gets in Pick stems). One time per mark, first and last unchanged, always rising.
    -> (smoothed times, how many times the line bends = changes of speed the lock will make)."""
    from . import gridmash as G
    m = np.asarray(marks, float)
    n = len(m)
    if n < 2 * SMOOTH_BEATS:
        return m.copy(), 0
    j = np.arange(n, dtype=float)
    p = float(np.median(np.diff(m)))
    base = m[0] + j * p
    t = base + G.calm(m - base, SMOOTH_BEATS)
    rough = np.median([abs(m[i] - (m[i - 4:i].sum() + m[i + 1:i + 5].sum()) / 8.0) for i in range(4, n - 4)])
    tol = float(np.clip(2.0 * rough, *SMOOTH_REACH))
    keep = G.ease(G.fit_points(j, t, tol), j, t, p, min(G.WIDE * 4 * p, SMOOTH_WIDE), short=8)
    out = np.interp(j, j[keep], t[keep])
    out[0], out[-1] = m[0], m[-1]
    return out, len(keep) - 2


def would_smooth(marks):
    """speed_report for a smooth lock made from these marks, plus `bends` (its changes of speed)."""
    m = np.asarray(marks, float)
    if len(m) < 9:
        return None
    sm, bends = smooth_marks(m)
    rep = speed_report(sm, np.arange(len(m)) * float(m[-1] - m[0]) / (len(m) - 1))
    if rep is not None:
        rep["bends"] = int(bends)
    return rep


SNAP = 0.005        # s: gentle transient snap (+-5 ms max) after Viterbi k=1 phase alignment (no beat-to-beat whiplash)


def tighten(info, marks, drums, mix=None):
    """Smoothly align detector marks onto zero-phase full-band kick+snare drum hits (gridaudit.align_marks_to_hits).

    Replaces the old causal 150-1500 Hz +-40 ms per-beat snapper (which cut out sub/kicks, lagged 6-9 ms
    behind real hits, and injected beat-to-beat speed whiplash). Marks placed by the user (user_marks) and
    Reimagine takes are left untouched.
    -> (marks, how many moved)."""
    from . import gridaudit
    m = np.asarray(marks, float)
    if len(m) < 8 or is_take(info):
        return m.copy(), 0
    if (info.get("user_marks") or {}).get("t"):
        return m.copy(), 0
    det = np.asarray(info.get("beats") or [], float)
    if len(det) >= 8:
        # If marks already differ substantially from detector beats (e.g. a project's own edited marks), leave them
        nearest = np.min(np.abs(det[:, None] - m[None, :]), axis=0) if len(m) <= 2000 else np.zeros(len(m))
        if np.mean(nearest > 0.005) > 0.15:
            return m.copy(), 0
    aligned, moved, _ = gridaudit.align_marks_to_hits(m, drums, sr=SR, mix_mono=mix, max_shift_s=SNAP)
    return aligned, moved


def _render_stem(sid, stem, marks, bpm, out_path, dst_beats=None):
    """Worker process: one stem warped so mark j lands at dst_beats[j] (or j * period - t0) -> (t0, n, method)."""
    from . import audio_io, warp
    from .config import song_path
    info = json.loads((song_path(sid) / "song.json").read_text(encoding="utf-8"))
    y = audio_io.load(song_path(sid) / "stems" / f"{stem}.flac", SR) / float(info.get("stem_scale", 1.0) or 1.0)
    if dst_beats is not None:
        dst = np.asarray(dst_beats, float)
        out, method = warp.render_to_grid(y, marks, dst, semis=0.0, stem=stem)
        t0 = -float(dst[0])
    else:
        out, t0, method = warp.render(y, marks, bpm, 1.0, 0.0, stem, "hq", "")
    audio_io.save_npy_atomic(Path(out_path), out.astype(np.float32))
    return float(t0), int(out.shape[1]), method


def verify_grid(sid):
    """Build a zero-phase full-band transient-verified grid proposal for `sid` (writes nothing).

    Fixes 2x/0.5x slips and isolated glitch beats, tracks k=1 drum-hit phase via Viterbi DP, locks steady
    songs onto a pure IRLS line, evens drumless breaks, picks kick-verified downbeats, and post-audits.
    -> ({t, d, s, bpb}, report)."""
    from . import gridaudit, library
    info = library.read(sid)
    if not info:
        raise FileNotFoundError(f"no such song: {sid}")
    drums, mix = gridaudit.load_song_drums_and_mix(sid)
    dur = float(info.get("duration") or (info["beats"][-1] if info.get("beats") else 0.0))
    bpb = int(info.get("bpb") or 4)
    raw_beats = info.get("beats") or []
    ref_bars_t = [float(raw_beats[j]) for j in (info.get("bars") or []) if 0 <= int(j) < len(raw_beats)]
    um_t = (info.get("user_marks") or {}).get("t")
    res = gridaudit.verify_and_fix(
        raw_beats, dur, drums, sr=SR, bpb=bpb,
        ref_bars_t=ref_bars_t or None, mix_mono=mix, user_marks_t=um_t,
    )
    n = len(res["beats"])
    bar_set = set(res["bars"])
    d = [1 if j in bar_set else 0 for j in range(n)]
    s = [1] * n
    marks = dict(t=res["beats"], d=d, s=s, bpb=res["bpb"])
    report = dict(
        method="verified",
        mode=res["mode"],
        bpm=res["bpm"],
        onset_source=res["onset_source"],
        before=len(raw_beats),
        after=n,
        raw_audit=res["raw_audit"],
        user_marks_audit=res["user_marks_audit"],
        post_audit=res["post_audit"],
        note=res["note"],
    )
    return marks, report


def _compact_audit(aud, onset_source="drums", note=None):
    """Lightweight audit dict suitable for song.json and summaries (without per-window sections array)."""
    if not aud:
        return None
    out = dict(
        status=aud.get("status", "ok"),
        median_offset_ms=aud.get("median_offset_ms", 0.0),
        max_offset_ms=aud.get("max_offset_ms", 0.0),
        signed_median_ms=aud.get("signed_median_ms", 0.0),
        active_windows=int(aud.get("active_windows", 0)),
        bad_windows=int(aud.get("bad_windows", 0)),
        drift_pct=aud.get("drift_pct", 0.0),
        max_straight_err_ms=aud.get("max_straight_err_ms", 0.0),
        needs_quantize=bool(aud.get("needs_quantize", False)),
        onset_source=onset_source,
    )
    if note:
        out["note"] = note
    return out


def audit_existing(sid, info=None, lk=None):
    """Measure real zero-phase drum-hit accuracy of an existing locked stem against its steady grid."""
    from . import audio_io, gridaudit, library, variant
    info = info or library.read(sid)
    if not info:
        return None
    lk = lk or info.get("lock")
    if not lk or not lk.get("rev"):
        return None
    tspec = json.dumps({"lock": lk["rev"]})
    p_drums = variant.stem_file(sid, "drums", tspec)
    if not p_drums.exists():
        return None
    scale = float(lk.get("scale", 1.0) or 1.0)
    drums = audio_io.load(p_drums, SR).mean(0) / scale
    mix = None
    if float(np.sqrt(np.mean(drums ** 2))) < 1e-4:
        parts = []
        for s in ("bass", "other", "vocals"):
            ps = variant.stem_file(sid, s, tspec)
            if ps.exists():
                parts.append(audio_io.load(ps, SR).mean(0) / scale)
        if parts:
            m_n = min(len(x) for x in parts)
            mix = sum(x[:m_n] for x in parts)
    body, _, _, fps, env_src = gridaudit.onset_curves(drums, sr=SR, mix_mono=mix)
    if lk.get("marks_t") and len(lk["marks_t"]) >= 8:
        grid_t = np.asarray(lk["marks_t"], float)
    elif lk.get("marks") and lk.get("period") and lk.get("t0") is not None:
        grid_t = np.arange(int(lk["marks"]), dtype=float) * float(lk["period"]) - float(lk["t0"])
    else:
        lb = np.asarray(lk.get("beats") or [], float)
        grid_t = lb[(lb >= 0.0) & (lb <= len(drums) / SR)]
    aud = gridaudit.audit_beats(grid_t, body, fps)
    note = (f"Measured on locked audio: {aud['median_offset_ms']:.1f} ms median "
            f"(max {aud['max_offset_ms']:.1f} ms across {aud['active_windows']} sections)")
    return dict(aud, onset_source=env_src, note=note)


def repaired(info, drums=None):
    """The detector's marks put right, the better of two ways:
      "recount"     beatfix: everything on ONE pulse - half / double time always re-counted, 3:2 only where the drum
                    hits agree, the rest left alone;
      "neighbours"  gridmash.tidy: every stretch judged against the stretch beside it, outward from the longest -
                    re-counted (3:2 too), kept as a real change of tempo, or, where the detector lost the beat
                    altogether, given even beats at its neighbour's tempo (those come back unsure, s = 0).
    The better one is the one a smooth lock would have to bend the least (seconds over 20 % off; "recount" unless
    "neighbours" saves a second or more). On the 16 detector-mark locks that played parts at the wrong speed
    (Nov 15): 1086 s over 20 % off as locked, 820 s after "recount" alone, about 280 s after the better of the two.
    drums: the drum stem (mono) - both ways then also sit each beat on its hit. Nothing is written.
    -> ({t, d, s, bpb}, report: beatfix.repair's, plus method, sec20, asks (what a smooth lock would still play
    wrong: seconds, stretches))."""
    from . import beatfix
    from . import gridmash as G

    def score(t):
        t = np.asarray(t, float)
        rp = would_smooth(t) or {}
        return float(rp.get("sec20", 0.0)), len(wrong_speed(smooth_marks(t)[0])) if len(t) >= 9 else 0
    best, best_rep = beatfix.as_marks(info, drums, SR)
    best_rep = dict(best_rep, method="recount")
    best_score = score(best["t"])
    ib = info["beats"]
    dur = float(info.get("duration") or ib[-1])
    det = np.array([x for x in ib if -1e-6 <= x <= dur + 1e-6])
    segs, main = G.stretches(det)
    if len(segs) > 1 and G.needs_tidy(det) != (0, 0):
        t = G.tidy(det)
        if drums is not None and len(t) > 20:
            t = beatfix._on_hits(t, drums, SR)
        if len(t) >= 8:
            bpb = int(info.get("bpb") or 4)
            downs = [ib[j] for j in info.get("bars") or [] if 0 <= j < len(ib) and -1e-6 <= ib[j] <= dur + 1e-6]
            span = (float(det[segs[main][0]]), float(det[segs[main][1]]))
            same = [(float(det[x[0]]), float(det[x[1]])) for i, x in enumerate(segs) if x[3] == ("count", 1.0) and i != main]
            lost = [(round(float(det[x[0]]), 2), round(float(det[x[1]]), 2)) for x in segs if x[3][0] == "even"]
            marks = dict(t=[round(float(x), 5) for x in t], d=G.bar_flags(t, downs, bpb, span, same),
                         s=[0 if any(a <= x <= b for a, b in lost) else 1 for x in t], bpb=bpb)
            sc = score(marks["t"])
            if sum(marks["d"]) >= 2 and sc[0] < best_score[0] - 1.0:
                best, best_score = marks, sc
                best_rep = dict(method="neighbours", pulse=round(float(segs[main][2]), 2), unsure=lost, before=len(det), after=len(t),
                                recounted_beats=int(sum(x[1] - x[0] for x in segs if x[3][0] == "count" and x[3][1] != 1.0)),
                                sections=[dict(start=round(float(det[x[0]]), 2), end=round(float(det[x[1]]), 2), bpm=round(x[2], 1),
                                               ratio=(x[3][1] if x[3][0] == "count" else None), kind=x[3][0]) for x in segs])
    best_rep.update(sec20=round(best_score[0], 1), asks=int(best_score[1]))
    return best, best_rep


def check(sid, marks=None):
    """Before a lock: the stretches it would play at the wrong speed (wrong_speed - a lock pulls every part to the
    song's main tempo) and whether Repair clears them. -> {slips: [[start s, end s, ratio]], yours: the marks are the
    user's own, repair_clears, left: what Repair would leave, speed: what the lock would do to the song's speed
    overall (speed_report; None with too few marks)}. Repair is only worked out for the detector's marks:
    marks lined up by ear are never replaced from here (the beat grid window has Repair, undoable). marks: a
    project's own timing for the song (always "yours"). Repair = repaired(), locked the smooth way: repair_clears
    / left say what that would still play wrong, `repair` = {method, sec20: seconds it would leave over 20 % off,
    was: the same for a lock to the marks as they are}. `smooth`: these marks as they are, locked the smooth way."""
    from . import gridaudit, library
    info = library.read(sid)
    marks_ = np.asarray(marks_of(info, marks), float)
    sw = wrong_speed(marks_)
    yours = bool((marks or {}).get("t") or (info.get("user_marks") or {}).get("t"))
    out = dict(slips=[[float(a), float(b), float(k)] for a, b, k in sw], yours=yours, repair_clears=False, left=[],
               speed=would(marks_))
    # the same marks locked the smooth way: what that would do to the song's speed, and what it would still play wrong
    sm = smooth_marks(marks_)[0] if len(marks_) >= 9 else marks_
    out["smooth"] = dict(speed=would_smooth(marks_), slips=[[float(a), float(b), float(k)] for a, b, k in wrong_speed(sm)])
    drums_mono = None
    try:
        drums_mono = library.load_stem(sid, "drums").mean(0)
        mix_mono = None
        if float(np.sqrt(np.mean(drums_mono ** 2))) < 1e-4:
            parts = []
            for st in ("bass", "other", "vocals"):
                try:
                    parts.append(library.load_stem(sid, st).mean(0))
                except Exception:
                    pass
            if parts:
                mn = min(len(x) for x in parts)
                mix_mono = sum(x[:mn] for x in parts)
        body, _, _, fps, env_src = gridaudit.onset_curves(drums_mono, sr=SR, mix_mono=mix_mono)
        aud = gridaudit.audit_beats(marks_, body, fps)
        out["audit"] = _compact_audit(aud, onset_source=env_src)
        out["off_beat"] = bool(aud["status"] == "off_beat")
    except Exception:
        out["audit"] = None
        out["off_beat"] = False
    if sw and not yours:
        if drums_mono is None:
            drums_mono = library.load_stem(sid, "drums").mean(0)
        new, rp = repaired(info, drums_mono)
        left = wrong_speed(smooth_marks(np.asarray(new["t"], float))[0])
        out.update(repair_clears=not left, left=[[float(a), float(b), float(k)] for a, b, k in left],
                   repair=dict(method=rp["method"], sec20=rp["sec20"], was=(out["speed"] or {}).get("sec20")))
    return out


def verify_and_make(sid, kind="marks", force=False, from_project=None):
    """Build a zero-phase full-band transient-verified grid, save it as the song's grid (freezing older
    projects so they keep their old timing), and lock to it cleanly."""
    from . import project as P
    new_marks, _ = verify_grid(sid)
    P.set_song_grid(sid, new_marks, from_project=from_project)
    return make(sid, check=not force, kind=kind)


def repair_and_make(sid, kind="smooth", force=False):
    """Repair the detector's marks (repaired(): miscounted stretches re-counted, stretches where the detector lost
    the beat evened out, every beat on its drum hit), keep them as the song's marks, then lock to them - the smooth
    way unless asked otherwise, since repaired stretches are right in count, not beat for beat. force: lock even
    if a stretch would still play at another speed (the user saw what is left). Refuses marks of the user's own."""
    from . import library
    info = library.read(sid)
    if (info.get("user_marks") or {}).get("t"):
        raise ValueError("this song has marks you made - repair them in the Beat grid window")
    new, _ = repaired(info, library.load_stem(sid, "drums").mean(0))
    info = library.read(sid)
    info["user_marks"] = new
    library.write(sid, info)
    return make(sid, check=not force, kind=kind)


def make(sid, check=True, marks=None, kind="marks"):
    """Start (or join) the background job that locks a song to its marks - or to `marks`, a project's own timing for
    it (that lock is kept next to the song's own, so each project plays the grid it was made with). -> the job
    (dict). kind: "marks" (every mark on the grid) or "smooth" (see the top of this file). check=False skips the
    wrong-speed check: for a grid that's meant to pull parts of the song to another
    speed (a Reimagine take's grid is the mix's, mapped onto the take bar for bar - where YuE2 rushed or dragged,
    the lock is what undoes it)."""
    from . import jobs, library, warp
    info = library.read(sid)
    kind = kind if kind in KINDS else "marks"
    for j in jobs.active():
        if j.get("kind") == "lock" and j.get("ref") == sid and j.get("status") in ("queued", "running"):
            return j
    own = marks if (marks and marks.get("t") and sig(info, marks) != sig(info)) else None   # a project's own timing

    def work(job):
        from . import gridaudit
        info = library.read(sid)
        marks_sig = sig(info, own)
        raw_marks = np.asarray(marks_of(info, own), float)
        marks = raw_marks.copy()
        if len(marks) < 8:
            raise RuntimeError("not enough marks to lock to")
        bends = 0
        if kind == "smooth":                               # from here on "the marks" are the smooth line through them
            marks, bends = smooth_marks(marks)
        sw = wrong_speed(marks) if check else []
        if sw:          # locking these would play that stretch at half or double speed (Oct 15: HORSE the Band)
            a, b, k = sw[0]
            raise RuntimeError(f"the beat marks at {int(a // 60)}:{int(a % 60):02d}-{int(b // 60)}:{int(b % 60):02d} run at "
                               f"{k:.2f}x the song's tempo - locking would play that part at the wrong speed. Open Beat grid, "
                               "press Repair (or Verify to drum hits), save, then lock")
        moved = 0
        if kind != "smooth":                               # (a smooth lock follows no single mark, so none is snapped)
            job.update(0.01, "lining the marks up with the drum hits", force=True)
            marks, moved = tighten(info, marks, library.load_stem(sid, "drums").mean(0))
        period = float(np.median(np.diff(marks)))          # main beat period (never skewed by slow intros/outros)
        bpm = 60.0 / period
        t0_dst = max(0.0, float(marks[0]))
        dst_beats = t0_dst + np.arange(len(marks), dtype=np.float64) * period
        tmp = library.song_dir(sid) / "stems_locked.tmp"
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir()
        job.update(0.02, "warping the stems onto a steady grid (one time)", force=True)
        pool = warp._get_pool()
        futs = {s: pool.submit(_render_stem, sid, s, marks.tolist(), bpm, str(tmp / f"{s}.npy"),
                               dst_beats.tolist(), priority=2) for s in STEMS}
        done, res = 0, {}
        while len(res) < len(futs):
            for s, f in futs.items():
                if s not in res and f.done():
                    res[s] = f.result()
                    done += 1
                    job.update(0.05 + 0.80 * done / len(futs), f"locked {done} of {len(futs)} stems")
            time.sleep(0.3)
        t0 = float(res[STEMS[0]][0])
        ys = {s: np.load(tmp / f"{s}.npy") for s in STEMS}
        n = min(v.shape[1] for v in ys.values())
        peak = max(float(np.abs(v[:, :n]).max()) for v in ys.values())
        scale = round(min(1.0, library.STEM_PEAK / peak), 6) if peak > 0 else 1.0

        # Post-lock transient verification on the actual rendered drums stem: remove any uniform R3 phase shift
        # (2-30 ms) so tl sits dead-center on the real drum hits, and record honest verification numbers.
        job.update(0.88, "verifying locked drum hits against steady grid")
        tl = np.arange(len(marks), dtype=np.float64) * period - t0
        q_drums = ys["drums"][:, :n].mean(0)
        q_mix = None
        if float(np.sqrt(np.mean(q_drums ** 2))) < 1e-4:
            q_mix = sum(ys[s][:, :n].mean(0) for s in ("bass", "other", "vocals"))
        body, _, _, fps, env_src = gridaudit.onset_curves(q_drums, sr=SR, mix_mono=q_mix)
        q_audit = gridaudit.audit_beats(tl, body, fps)
        res_ms = float(q_audit.get("signed_median_ms", 0.0) or 0.0)
        if q_audit["active_windows"] > 0 and 2.0 <= abs(res_ms) <= 30.0 and (-t0 + res_ms / 1000.0) >= 0.0:
            t0 = t0 - res_ms / 1000.0
            tl = np.arange(len(marks), dtype=np.float64) * period - t0
            q_audit = gridaudit.audit_beats(tl, body, fps)
        q_note = (f"Verified locked audio: {q_audit['median_offset_ms']:.1f} ms median "
                  f"(max {q_audit['max_offset_ms']:.1f} ms across {q_audit['active_windows']} sections)")

        from . import audio_io
        rev = f"L{int(time.time())}"
        root = library.song_dir(sid) / "stems_locked"
        if own:                                           # a project's own timing: kept next to the song's lock
            out = root / rev
        else:                                             # the song's own grid: stems_locked/ as always - the lock
            info = library.read(sid)                      # there now is kept first if a project still plays it
            old = info.get("lock") or {}
            if old.get("rev") and old.get("sig") != marks_sig and in_use(sid, old.get("sig")):
                info = keep_current(sid, info)
            out = root
        out.mkdir(parents=True, exist_ok=True)
        for f in out.glob("*.flac"):                      # (never the kept locks in its subfolders)
            f.unlink()
        for s, y in ys.items():
            audio_io.save(out / f"{s}.flac", y[:, :n] * scale, SR, subtype="PCM_24")
        shutil.rmtree(tmp, ignore_errors=True)

        # Map original audio times (info["beats"] and raw_marks) -> exact audio times in stems_locked using
        # `marks -> tl` (the actual warp map applied by R3, plus any post-lock residual phase lock on t0).
        s0 = (tl[1] - tl[0]) / max(1e-6, marks[1] - marks[0])
        s1 = (tl[-1] - tl[-2]) / max(1e-6, marks[-1] - marks[-2])

        def _to_locked(x):
            y_ = np.interp(x, marks, tl)
            y_ = np.where(x < marks[0], tl[0] + (x - marks[0]) * s0, y_)
            return np.where(x > marks[-1], tl[-1] + (x - marks[-1]) * s1, y_)

        b = np.asarray(info["beats"], float)
        lb = _to_locked(b)
        marks_t = _to_locked(raw_marks) if kind == "smooth" else tl
        lock = dict(rev=rev, sig=marks_sig, bpm=round(bpm, 6), period=period, t0=float(t0),
                    beats=[round(float(x), 5) for x in lb],
                    marks_t=[round(float(x), 5) for x in marks_t],
                    scale=scale, n=int(n), marks=int(len(marks)),
                    method=res[STEMS[0]][2], tightened=int(moved), t=time.time(), kind=kind, bends=int(bends),
                    speed=speed_report(marks, tl),
                    audit=_compact_audit(q_audit, onset_source=env_src, note=q_note))
        if own:                                           # (published, so open pages switch over to it)
            locks = dict(library.read(sid).get("locks") or {})
            locks[marks_sig] = lock
            return library.summary(library._set(sid, locks=locks))
        return library.summary(library._set(sid, lock=lock))

    return jobs.submit("lock", f"🔒 Locking {info['title']} {'smoothly' if kind == 'smooth' else 'to its marks'}", work, ref=sid).to_dict()
