"""Listening sessions: test results to hear and weigh in on, instead of numbers only.

Every audio test worth hearing becomes a folder in ROOT/listening/ (not in git, not in frozen releases):

    session.json   {id, title, created, goal, decision, questions: [..],
                    items: [{id, file, label, what, measured, mine}]}   - written once, by the test
    feedback.json  {items: {item id: {verdict, notes, at}}, overall: {notes, at}}   - yours, from the app
    *.flac         the audio (24-bit FLAC: lossless at the mixer's own resolution)

`what` says what the clip is, `measured` what the numbers said, `mine` what I made of it / decided; your verdict and
notes are read back before the next round, so an engineer's ears weigh in on what the measurements can't hear.
"""
import json
import os
import re
import shutil
import time
from pathlib import Path

from .config import ROOT

DIR = Path(os.environ.get("MASHLAB_LISTENING") or ROOT / "listening")
VERDICTS = ("right", "close", "wrong")          # sounds right / close, needs work / not there


def enabled():
    return DIR.is_dir()


def _read(p, default):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def _write(p, d):
    tmp = p.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, p)


def _dir(sid):
    d = DIR / Path(str(sid)).name
    if not (d / "session.json").is_file():
        raise KeyError(sid)
    return d


def sessions():
    """Every session, newest first, with your feedback so far and how many items still wait for it."""
    out = []
    if not enabled():
        return out
    for d in DIR.iterdir():
        s = _read(d / "session.json", None)
        if not s:
            continue
        fb = _read(d / "feedback.json", {"items": {}, "overall": {}})
        s["feedback"] = fb
        s["waiting"] = sum(1 for it in s.get("items", []) if not (fb.get("items", {}).get(it["id"]) or {}).get("verdict"))
        s["folder"] = str(d)
        out.append(s)
    return sorted(out, key=lambda s: s.get("created", 0), reverse=True)


def audio_path(sid, name):
    """An item's audio file (only files the session lists)."""
    d = _dir(sid)
    s = _read(d / "session.json", {})
    if name not in {it["file"] for it in s.get("items", [])}:
        raise KeyError(name)
    return d / name


def set_feedback(sid, item, verdict=None, notes=None):
    """Your verdict / notes on one item ("overall" = the session as a whole). -> feedback.json as saved."""
    d = _dir(sid)
    s = _read(d / "session.json", {})
    if item != "overall" and item not in {it["id"] for it in s.get("items", [])}:
        raise KeyError(item)
    if verdict is not None and verdict not in VERDICTS + ("",):
        raise ValueError(f"verdict must be one of {', '.join(VERDICTS)}")
    fb = _read(d / "feedback.json", {"items": {}, "overall": {}})
    cur = fb["overall"] if item == "overall" else fb["items"].setdefault(item, {})
    if verdict is not None:
        cur["verdict"] = verdict
    if notes is not None:
        cur["notes"] = str(notes)
    cur["at"] = time.strftime("%Y-%m-%d %H:%M")
    _write(d / "feedback.json", fb)
    return fb


def remove(sid):
    """Done with a session: its audio goes, its text (what was tried, measured, decided + your notes) is kept in
    listening/_archive/ (not listed) so nothing you wrote is lost."""
    d = _dir(sid)
    keep = DIR / "_archive" / d.name
    keep.mkdir(parents=True, exist_ok=True)
    for name in ("session.json", "feedback.json"):
        if (d / name).exists():
            shutil.copy2(d / name, keep / name)
    shutil.rmtree(d)


def reveal(sid):
    """Open the session's folder in Explorer (to drag the files into a DAW)."""
    os.startfile(_dir(sid))          # noqa: S606 - local app, the user's own folder


# ---------------------------------------------------------------- making one (from test scripts)

def create(title, goal, items, decision="", questions=()):
    """A new session. items: [{label, what, measured, mine, and src (an audio file) or audio (y (channels, n), sr)}].
    -> the session folder."""
    import numpy as np
    import soundfile as sf
    DIR.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:48]
    sid = f"{time.strftime('%Y-%m-%d-%H%M')}-{slug}"
    d = DIR / sid
    d.mkdir()
    listed = []
    for k, it in enumerate(items, 1):
        name = f"{k:02d}-{re.sub(r'[^a-z0-9]+', '-', it['label'].lower()).strip('-')[:40]}.flac"
        if "audio" in it:
            y, sr = it["audio"]
            y = np.asarray(y, np.float32)
            sf.write(d / name, (y.T if y.ndim == 2 else y), sr, subtype="PCM_24")
        elif str(it["src"]).lower().endswith(".flac"):
            shutil.copy2(it["src"], d / name)
        else:
            y, sr = sf.read(it["src"], dtype="float32", always_2d=True)
            sf.write(d / name, y, sr, subtype="PCM_24")
        listed.append(dict(id=f"i{k}", file=name, label=it["label"], what=it.get("what", ""),
                           measured=it.get("measured", ""), mine=it.get("mine", "")))
    _write(d / "session.json", dict(id=sid, title=title, created=time.time(), goal=goal, decision=decision,
                                    questions=list(questions), items=listed))
    return d
