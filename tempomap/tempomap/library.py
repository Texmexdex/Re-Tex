"""Reading songs from the Re-Tex library - read only, nothing is ever written there.

The library folder comes from Re-Tex's own settings (%APPDATA%/MashupLab/settings.json "data"), else Re-Tex's data/ folder.
A song is songs/<id>/ with song.json (title, the detector's beats, bar lines, sections, any marks you saved) and
stems/*.flac.
"""
import json
import os
from pathlib import Path

import numpy as np
import soundfile as sf


def data_dir(override=None):
    """The library: an explicit folder, else the Re-Tex folder this sits in (its settings.json "data", else its
    data/ folder), else Re-Tex's settings in %APPDATA%."""
    if override:
        return Path(override)
    retex = Path(__file__).resolve().parents[2]
    for f in (retex / "settings.json", Path(os.environ.get("APPDATA", "")) / "MashupLab" / "settings.json"):
        try:
            s = json.loads(f.read_text(encoding="utf-8"))
            if s.get("data"):
                return Path(s["data"])
        except (OSError, ValueError):
            pass
        if f.parent == retex and (retex / "mashlab").exists() and (retex / "data").exists():
            return retex / "data"
    return retex / "data"


def songs(root):
    """[(id, title)] of the analysed songs (MIDI items left out)."""
    out = []
    d = Path(root) / "songs"
    for p in sorted(d.iterdir()) if d.exists() else []:
        f = p / "song.json"
        if not f.exists():
            continue
        try:
            info = json.loads(f.read_text(encoding="utf-8"))
        except ValueError:
            continue
        if info.get("beats") and info.get("kind") != "midi" and (p / "stems").exists():
            out.append((p.name, info.get("title") or p.name))
    return out


def find(root, words):
    """Songs whose id or title contains every one of the words (any case)."""
    w = [x.lower() for x in words]
    return [(i, t) for i, t in songs(root) if all(x in (t + " " + i).lower() for x in w)]


def load(root, sid):
    """-> (info dict, {stem: mono float32}, sample rate). Stems missing from the folder are left out."""
    d = Path(root) / "songs" / sid
    info = json.loads((d / "song.json").read_text(encoding="utf-8"))
    stems, sr = {}, None
    for name in ("drums", "bass", "other", "vocals"):
        f = d / "stems" / f"{name}.flac"
        if f.exists():
            y, sr = sf.read(str(f), dtype="float32", always_2d=True)
            y = y.mean(axis=1)
            scale = float(info.get("stem_scale", 1.0) or 1.0)
            stems[name] = y / scale if scale != 1.0 else y
    return info, stems, sr


def detector_beats(info):
    dur = float(info.get("duration") or info["beats"][-1])
    b = np.asarray(info["beats"], float)
    return b[(b >= -1e-6) & (b <= dur + 1e-6)], dur
