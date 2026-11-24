"""TempoMap, the sidecar: a separate program (its own folder and code, run in its own process) that finds a song's
grid structure - tempo sections, beats, bars of any length - and gives it back as beat marks for the Beat grid
window to propose. Re-Tex only starts it and reads what it wrote; nothing of TempoMap runs inside Re-Tex, so if it
fails, Re-Tex carries on.

Where it lives: ROOT/tempomap (its own folder inside Re-Tex's; settings "tempomap_dir" to put it elsewhere). It
writes only to its own maps/ and cache/ folders there; it reads the library. A song's map is redone when the song changed after it was made (song.json is newer).
"""
import json
import os
import re
import subprocess
import sys
import threading
from pathlib import Path

from . import library
from .config import DATA, ROOT, read_settings

SID_OK = re.compile(r"^[A-Za-z0-9_\-]+$")
FILE_OK = re.compile(r"^(report\.html|map\.json|marks\.json|mix_[a-z]+\.mp3)$")
_locks, _locks_lock = {}, threading.Lock()


def home():
    """The TempoMap folder, or None when it isn't there."""
    p = Path(read_settings().get("tempomap_dir") or ROOT / "tempomap")
    return p if (p / "tempomap" / "cli.py").exists() else None


def out_dir(sid):
    h = home()
    return h / "maps" / sid if h else None


def check_sid(sid):
    if not SID_OK.match(sid or ""):
        raise ValueError("bad song id")
    if str(sid).startswith(("midi_", "smp_")):
        raise ValueError("MIDI songs and samples keep their own timing - the tempo map is for recordings")
    if not library.read(sid):
        raise ValueError("no such song")


def is_fresh(sid, need_audio=False):
    """A map made after the song last changed (and with its click tracks, if asked)."""
    d = out_dir(sid)
    if not d or not (d / "marks.json").exists() or not (d / "report.html").exists():
        return False
    made = (d / "marks.json").stat().st_mtime
    song = library.song_dir(sid) / "song.json"
    if song.exists() and song.stat().st_mtime > made:
        return False
    code = home() / "tempomap"                   # TempoMap itself changed since: map it again
    if code.is_dir() and any(p.stat().st_mtime > made for p in code.glob("*.py")):
        return False
    if need_audio:                               # click tracks made with this map (not left from an older one)
        mp3 = d / "mix_map.mp3"
        if not mp3.exists() or mp3.stat().st_mtime < (d / "report.html").stat().st_mtime - 120:
            return False
    return True


def run(sid, audio=True, force=False, timeout=1200):
    """Make the song's map (one at a time per song; a second caller waits for the first). -> dict(ok, log)."""
    check_sid(sid)
    h = home()
    if not h:
        return dict(ok=False, log="TempoMap isn't there (no tempomap folder in Re-Tex's folder, and no "
                                  "\"tempomap_dir\" in the settings)")
    with _locks_lock:
        lk = _locks.setdefault(sid, threading.Lock())
    with lk:
        if not force and is_fresh(sid, need_audio=audio):
            return dict(ok=True, log="(already made)")
        cmd = [sys.executable, "-m", "tempomap", "--id", sid, "--data", str(DATA), "--out", str(h / "maps"),
               "--no-open"] + ([] if audio else ["--no-audio"])
        env = dict(os.environ, PYTHONIOENCODING="utf-8")
        kw = {"creationflags": 0x08000000} if os.name == "nt" else {}       # CREATE_NO_WINDOW
        try:
            r = subprocess.run(cmd, cwd=str(h), capture_output=True, text=True, encoding="utf-8", errors="replace",
                               timeout=timeout, env=env, **kw)
        except subprocess.TimeoutExpired:
            return dict(ok=False, log=f"TempoMap took longer than {timeout // 60} minutes")
        log = (r.stdout or "")[-3000:] + (("\n" + r.stderr[-3000:]) if r.stderr else "")
        ok = r.returncode == 0 and (out_dir(sid) / "marks.json").exists()
        return dict(ok=ok, log=log.strip())


def marks(sid):
    """-> (marks {t, d, s, bpb, source}, summary of the map) from the map on disk."""
    d = out_dir(sid)
    mk = json.loads((d / "marks.json").read_text(encoding="utf-8"))
    m = json.loads((d / "map.json").read_text(encoding="utf-8"))
    secs = [dict(start=s["start"], end=s["end"], bpm=s["bpm"], how=s["how"]) for s in m.get("sections", [])]
    labels = []
    for x in m.get("meters", []):
        if x["label"] not in labels:
            labels.append(x["label"])
    if not m.get("downbeat_model"):
        from . import components
        components.fallback("beat_this", "the tempo map found bars from the onsets alone (less reliable): "
                                         + (m.get("downbeat_error") or "Beat This! didn't run"))
    return mk, dict(sections=secs, meters=labels, downbeat_model=bool(m.get("downbeat_model")),
                    downbeat_error=m.get("downbeat_error", ""))


def file_path(sid, name):
    """A file of the song's map, for the sidecar page (only the files TempoMap writes)."""
    if not SID_OK.match(sid or "") or not FILE_OK.match(name or ""):
        return None
    d = out_dir(sid)
    p = d / name if d else None
    return p if p and p.is_file() else None
