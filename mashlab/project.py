"""Arrangement model (plain JSON) and helpers.

Units: clip.start / clip.len are PROJECT beats; clip.src is the SOURCE beat index (song's extended beat
list, fractional allowed). Per song in a project: transpose (semitones) and mult (project beats per
source beat: 2 = half-time song played in a double-time project, 0.5 = the reverse).
"""
import json
import math
import re
import time
import uuid

import numpy as np

from . import library
from .config import PROJECTS, PITCHED

TRACK_COLORS = {"drums": "#f0a04b", "bass": "#5fb3f0", "other": "#8ad17a", "vocals": "#e36fa6", "any": "#b39cf0"}


def new_id(prefix):
    return f"{prefix}{uuid.uuid4().hex[:8]}"


def empty(name="Untitled mashup", bpm=120.0, bpb=4):
    return dict(id=new_id("p"), name=name, bpm=float(bpm), bpb=int(bpb), quality="hq", songs={},
                tracks=[track(r.capitalize(), r) for r in ("drums", "bass", "other", "vocals")],
                clips=[], master=dict(gain_db=0.0, glue=True, limit=True, target_lufs=-12.0),
                loop=dict(on=False, start=0.0, end=16.0), created=time.time(), notes="")


def track(name, role="any", automation=None, **kw):
    """A timeline track. `automation` is an OPTIONAL list of time-varying FX events (design D/B.4): autonomous
    output attaches a per-track automation lane here, but it is ADDITIVE — omitted entirely when None, so a
    Phase-2-free project (empty()/the manual flow never passes it) is byte-identical to before."""
    t = dict(id=new_id("t"), name=name, role=role, color=TRACK_COLORS.get(role, TRACK_COLORS["any"]),
             gain_db=0.0, pan=0.0, mute=False, solo=False, hpf=0.0, lpf=0.0, duck=0.0, pocket=0.0)
    if automation is not None:
        t["automation"] = list(automation)
    t.update(kw)
    return t


def clip(track_id, song, stem, start, src, length, **kw):
    c = dict(id=new_id("c"), track=track_id, song=song, stem=stem, start=float(start), src=float(src),
             len=float(length), gain_db=0.0, fade_in=0.0, fade_out=0.0, mute=False, rate=1.0, pitch=0.0,
             keep_pitch=True)
    c.update(kw)
    return c


def _played_tuning(project, sid, seen):
    """Cents-level tuning (semitones from A440) a song plays at in this project, or None if unmeasured."""
    own = (library.read(sid) or {}).get("tuning_st")
    if own is None:
        return None
    to = (project["songs"].get(sid) or {}).get("tune_to")
    if not to or sid in seen:
        return float(own)
    if to == "a440":
        return 0.0
    if to not in project["songs"]:
        return float(own)
    t = _played_tuning(project, to, seen | {sid})
    return float(own) if t is None else t


def tuning_shift(project, sid):
    """Semitones a song is shifted by to match the tuning picked for it (songs[sid]["tune_to"]: "a440", or
    another song of the project, as that song plays). Off by default: nothing is retuned unless asked."""
    own = (library.read(sid) or {}).get("tuning_st")
    if own is None or not (project["songs"].get(sid) or {}).get("tune_to"):
        return 0.0
    return _played_tuning(project, sid, set()) - float(own)


def screw_semis(project):
    """Chopped & screwed: project["screw"] = the speed the whole mix plays at (0.75 = 25 % slower - the tempo is
    slowed by the same factor), its pitch dropping along like a slowed turntable: 12 * log2(screw) semitones."""
    r = float(project.get("screw") or 1.0)
    return 12.0 * math.log2(r) if r > 0 and abs(r - 1.0) > 1e-9 else 0.0


def _is_sample(sid):
    return str(sid).startswith("smp_")


def song_semis(project, sid, stem):
    """Total pitch shift for a stem: the song's transpose + the tuning match you picked (if any) + the screw. Drums
    only ever take the screw. A baked sample (sampler.py) is already screwed / tuned: only its own transpose."""
    if _is_sample(sid):
        return round(float((project["songs"].get(sid) or {}).get("transpose", 0.0)), 3)
    screw = screw_semis(project)
    if stem not in PITCHED:
        return round(screw, 3)
    s = project["songs"].get(sid, {})
    return round(float(s.get("transpose", 0.0)) + tuning_shift(project, sid) + screw, 3)


def song_mult(project, sid):
    return float(project["songs"].get(sid, {}).get("mult", 1.0))


def song_tspec(project, sid):
    """Every song plays as recorded: one fixed speed (project tempo / the song's tempo), its own timing kept -
    drifts, odd bars and tempo changes included. Beat j of the song lands at W[j] = fb + (t_j - t_fb) / p
    project beats (timing.positions "natural" with no pins; p = the song's beat period, fb = its first beat),
    so nothing is stretched beat by beat. web/js/timing.js landing() is the exact twin."""
    info = library.read(sid) or {}
    if not info.get("bpm") or not info.get("beats"):
        return ""
    spec = {"fb": int(info.get("first_beat", 0)), "m": "natural", "p": 60.0 / float(info["bpm"])}
    ps = project["songs"].get(sid) or {}
    if ps.get("lock"):
        from . import lock
        lk = lock.find(info, ps.get("marks"))   # the lock made from this project's grid for the song (its own, or
        if lk:                                  # the song's) - 🔒 the locked stems, on their own steady grid
            spec.update(lock=lk["rev"], p=60.0 / float(lk["bpm"]))
    screw = float(project.get("screw") or 1.0)
    if abs(screw - 1.0) > 1e-9 and not _is_sample(sid):   # screwed: rendered unscrewed, then resampled (warp.render)
        spec["s"] = screw
    return json.dumps(spec, sort_keys=True, separators=(",", ":"))


def request(project, c):
    """The warp a clip plays: (sid, stem, bpm, mult, semis, quality, timing spec)."""
    sid = c["song"]
    return (sid, c["stem"], round(float(project["bpm"]), 4), song_mult(project, sid),
            song_semis(project, sid, c["stem"]), project.get("quality", "hq"), song_tspec(project, sid))


def render_requests(project):
    """request() for every stem the arrangement uses."""
    reqs = []
    for c in project["clips"]:
        if library.read(c["song"]) is None:
            continue
        reqs.append(request(project, c))
    return list(dict.fromkeys(reqs))


def length_beats(project):
    return max([c["start"] + c["len"] for c in project["clips"]] + [0.0])


def save(project):
    project["modified"] = time.time()
    p = PROJECTS / f"{project['id']}.json"
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(project, indent=1), encoding="utf-8")
    tmp.replace(p)
    return p


def load(pid):
    return json.loads((PROJECTS / f"{pid}.json").read_text(encoding="utf-8"))


def list_projects():
    out = []
    for p in PROJECTS.glob("*.json"):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            out.append(dict(id=d["id"], name=d.get("name"), bpm=d.get("bpm"), modified=d.get("modified", 0),
                            songs=list(d.get("songs", {}).keys()), n_clips=len(d.get("clips", [])),
                            branch_of=d.get("branch_of")))
        except Exception:
            pass
    return sorted(out, key=lambda x: x["modified"], reverse=True)


_BRANCH = re.compile(r"^(.*?) \(branch (\d+)\)$")


def branch(pid, name=None):
    """A copy of saved project `pid` under a new id, to change while the original stays exactly as it is.
    Named "<name> (branch N)" (N = the next number for that name, so branching a branch doesn't stack
    suffixes); `branch_of` remembers where it came from. -> the new project (saved)."""
    src = load(pid)
    p = json.loads(json.dumps(src))
    base = _BRANCH.match(src.get("name") or "")
    base = base.group(1) if base else (src.get("name") or "Untitled mashup")
    taken = [int(m.group(2)) for q in list_projects() if (m := _BRANCH.match(q.get("name") or "")) and m.group(1) == base]
    p["id"] = new_id("p")
    p["name"] = (name or "").strip() or f"{base} (branch {max(taken, default=0) + 1})"
    p["branch_of"] = dict(id=src["id"], name=src.get("name"), at=time.time())
    p["created"] = time.time()
    save(p)
    return p


def delete(pid):
    (PROJECTS / f"{pid}.json").unlink(missing_ok=True)


def set_song_grid(sid, marks, from_project=None):
    """Save the song's own grid (its user_marks; None = the detector's again) - for projects from now on. Saved
    projects that followed the old grid get a copy of it (freeze_song_grid), and if one of them plays the song
    locked, the lock made from the old grid is kept for it (lock.keep_current). from_project: the project the
    change was made from - it follows the new grid. Written quietly: the page already has the marks, and a "song
    changed" event would make every open page reload the song's waveforms. -> (frozen project ids, the old grid)."""
    from . import lock, timing
    info = library.read(sid)
    frozen, old = [], None
    if lock.sig(info) != lock.sig(dict(info, user_marks=marks)):
        old = info.get("user_marks") or timing.detected_marks(info)
        frozen, locked = freeze_song_grid(sid, old, skip=from_project)
        cur = info.get("lock") or {}
        if locked and cur.get("rev") and cur.get("sig") == lock.sig(info):
            info = lock.keep_current(sid, info)
    info["user_marks"] = marks
    library.write(sid, info)
    return frozen, old


def freeze_song_grid(sid, old_marks, skip=None):
    """A song's grid is about to change (the beat grid window saved new marks): every saved project that uses the
    song and follows its grid gets a copy of the grid it has been playing (songs[sid].marks), so it keeps sounding
    exactly as it did - the new grid is for projects from now on (the user's rule, Oct 19). A project with its own
    timing for the song already has it; `skip` (the project the change was made from) follows the new grid. The
    files' "modified" stays as it was. -> ids of the projects frozen, and of those among them that play it locked."""
    frozen, locked = [], []
    for f in PROJECTS.glob("*.json"):
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        s = (d.get("songs") or {}).get(sid)
        if s is None or "marks" in s or d.get("id") == skip:
            continue
        s["marks"] = old_marks
        tmp = f.with_suffix(".tmp")
        tmp.write_text(json.dumps(d, indent=1), encoding="utf-8")
        tmp.replace(f)
        frozen.append(d["id"])
        if s.get("lock"):
            locked.append(d["id"])
    return frozen, locked


# ------------------------------------------------------------------ beat grid changes

def _beat_to_time(b, beats):
    """Fractional beat index -> seconds, extrapolating past either end at the edge tempo."""
    beats = np.asarray(beats, float)
    i = np.arange(len(beats))
    b = np.asarray(b, float)
    t = np.interp(b, i, beats)
    t = np.where(b < 0, beats[0] + b * (beats[1] - beats[0]), t)
    return np.where(b > len(beats) - 1, beats[-1] + (b - (len(beats) - 1)) * (beats[-1] - beats[-2]), t)


def _time_to_beat(t, beats):
    beats = np.asarray(beats, float)
    i = np.arange(len(beats))
    t = np.asarray(t, float)
    b = np.interp(t, beats, i)
    b = np.where(t < beats[0], (t - beats[0]) / (beats[1] - beats[0]), b)
    return np.where(t > beats[-1], len(beats) - 1 + (t - beats[-1]) / (beats[-1] - beats[-2]), b)


def _landing(beats, bpm):
    """(W, beat index): where each beat lands when the song plays as recorded (see song_tspec; the anchor beat
    only shifts W by a constant, which no clip offset or length depends on). None without a tempo."""
    if not bpm:
        return None
    b = np.asarray(beats, float)
    return (b - b[0]) / (60.0 / float(bpm)), np.arange(len(b), dtype=float)


def _interp_ext(x, xp, fp):
    """np.interp that carries on past either end at the edge slope (clips may run past a song's grid)."""
    x = float(x)
    if x < xp[0]:
        return float(fp[0] + (x - xp[0]) * (fp[1] - fp[0]) / max(xp[1] - xp[0], 1e-9))
    if x > xp[-1]:
        return float(fp[-1] + (x - xp[-1]) * (fp[-1] - fp[-2]) / max(xp[-1] - xp[-2], 1e-9))
    return float(np.interp(x, xp, fp))


def _clip_src_end(c, mult, landing):
    """The source beat a clip's end plays (its warps and un-stretched beats included)."""
    from . import warps
    ps = warps.pieces(c, mult, landing)
    if not ps:
        return float(c["src"])
    w = ps[-1][3]
    return _interp_ext(w, landing[0], landing[1]) if landing else w


def _clip_len_to(c, mult, landing, end):
    """The clip length (project beats) at which clip c ends exactly at source beat `end`."""
    from . import warps
    we = _interp_ext(end, landing[1], landing[0]) if landing else float(end)
    k = float(c.get("rate", 1.0) or 1.0) / float(mult)
    for u0, u1, w0, w1, r, kp in warps.pieces(dict(c, len=1e9), mult, landing):
        if w1 >= we:
            return u0 + max(0.0, we - w0) / (k * r)
    return float(c["len"])


def remap_song_beats(sid, old_beats, new_beats, old_bpm=None, new_bpm=None, old_bars=None, new_bars=None):
    """A song's beat grid changed (re-analysis): move every saved project's clips of that song so they
    start at the same moment of the audio, and rescale the song's speed factor when the song's MAIN beat
    level changed (e.g. an eighth-note track folded to quarters: old/new tempo = 2). The level comes from
    the main tempos, not from where the clips happen to sit - a section the old grid had mis-tracked at
    another level used to play at the wrong speed and simply plays right now.
    Songs play as recorded, so every clip keeps its exact moment of the audio (it used to be pulled onto the
    nearest new beat when it had started on one - that changed what you hear once nothing is warped any more).
    Each clip keeps the same stretch of audio: its end is carried over the same way (songs play as recorded,
    so its length only changes if the song's tempo estimate did). Its stretched / contracted parts (warps)
    stay on the same moments of the audio. Nothing else in the project moves. -> [project ids]."""
    to_new = lambda b: _time_to_beat(_beat_to_time(b, old_beats), new_beats)

    changed = []
    for p in PROJECTS.glob("*.json"):
        try:
            proj = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if sid not in proj.get("songs", {}):
            continue
        song = proj["songs"][sid]
        clips = [c for c in proj.get("clips", []) if c.get("song") == sid]
        old_mult, old_land = song_mult(proj, sid), _landing(old_beats, old_bpm)
        ends = [_clip_src_end(c, old_mult, old_land) for c in clips]
        if clips:
            src = np.array([c["src"] for c in clips], float)
            new_src = to_new(src)
            if old_bpm and new_bpm:
                f = float(new_bpm) / float(old_bpm)                  # new beats per old beat, main level
            else:                                                    # no tempos given: local estimate
                f = float(np.median(to_new(src + 1) - new_src)) if len(src) else 1.0
            level = 1.0
            for cand in (1.0, 0.5, 1 / 3, 0.25, 2.0, 3.0, 4.0):
                if abs(f / cand - 1) < 0.08:
                    level = cand
                    break
            for c, s_old, s in zip(clips, src, new_src):
                c["src"] = round(float(s), 4)
                for g in c.get("warps") or []:
                    g["a"], g["b"] = (round(float(x), 6) for x in to_new([g["a"], g["b"]]))
            song["mult"] = old_mult / level                          # project beats per (new) source beat
            new_land = _landing(new_beats, new_bpm)
            for c, e_old in zip(clips, ends):
                L = _clip_len_to(c, song["mult"], new_land, float(to_new(e_old)))
                if abs(L - float(c["len"])) > 1e-3:
                    c["len"] = round(max(0.05, L), 4)
        proj["modified"] = time.time()
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(proj, indent=1), encoding="utf-8")
        tmp.replace(p)
        changed.append(proj["id"])
    return changed
