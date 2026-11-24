"""python -m tempomap [title words ...]   (run.bat does this)

Finds the song in the Re-Tex library by words of its title, measures its tempo map, and writes
maps/<song id>/ : report.html (open it), map.json, and MP3s of the song with a click on the map and on its current
grid. maps/index.html lists every song mapped so far. The library is only read.
"""
import argparse
import datetime
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from . import build, check, downbeats, library, render, report

HERE = Path(__file__).resolve().parent.parent


def _pool(x, fps, to=50.0):
    k = max(1, int(round(fps / to)))
    n = len(x) // k
    return [round(float(v), 3) for v in np.asarray(x[:n * k]).reshape(n, k).max(axis=1)], fps / k


def summary_text(m):
    secs = [s for s in m["sections"] if s["how"] != "none"] or m["sections"]
    tempo = " > ".join(f"{s['bpm']:.2f}" for s in secs) + " BPM"
    seen = []
    for x in m["meters"]:
        if x["label"] not in seen:
            seen.append(x["label"])
    return tempo, ", ".join(seen)


def steady_grid(info, det, dur):
    """What Re-Tex plays for an unlocked song: one fixed speed, a straight grid at the song's stored BPM, lined up
    with the detector's first beats. -> dict(bpm, beats, bars) or None."""
    bpm = float(info.get("bpm") or 0)
    b = np.asarray(det, float)
    if bpm <= 0 or len(b) < 8:
        return None
    p = 60.0 / bpm
    first = b[:32]
    ph = np.angle(np.mean(np.exp(2j * np.pi * (first - first[0]) / p))) / (2 * np.pi) * p
    a = first[0] + ph
    a -= np.floor(a / p) * p
    beats = np.arange(a, dur, p)
    j0 = int(np.argmin(np.abs(beats - b[0])))
    bars = beats[j0 % 4::4]
    return dict(bpm=bpm, beats=[round(float(x), 4) for x in beats], bars=[round(float(x), 4) for x in bars])


def process(root, sid, out_root, audio=True, log=print):
    t0 = time.time()
    info, stems, sr = library.load(root, sid)
    det, dur = library.detector_beats(info)
    log(f"  measuring ({len(stems)} stems, {dur / 60:.1f} min) ...")
    dl = downbeats.logits(stems, sr, Path(out_root).parent / "cache" / f"{sid}_downbeats.npz",
                          Path(root).parent)
    log("  Beat This! downbeats: " + ("yes" if dl is not None else
                                       f"NOT USED - {downbeats.last_error} - this is a fallback: bars from the onsets alone, less reliable"))
    m, feat = build.tempo_map(info, stems, sr, det, dur, dl)
    m["downbeat_model"] = dl is not None
    if dl is None:
        m["downbeat_error"] = downbeats.last_error
    m["id"] = sid
    d = Path(out_root) / sid
    d.mkdir(parents=True, exist_ok=True)
    grid_bars = [float(info["beats"][j]) for j in info.get("bars") or []
                 if 0 <= j < len(info["beats"]) and -1e-6 <= info["beats"][j] <= dur + 1e-6]
    grid = dict(beats=[round(float(x), 4) for x in det], bars=[round(x, 4) for x in grid_bars])
    um = info.get("user_marks") or {}
    marks = None
    if um.get("t"):
        t = np.asarray(um["t"], float)
        dd = list(um.get("d") or [])
        marks = dict(beats=[round(float(x), 4) for x in t], bars=[round(float(t[i]), 4) for i, x in enumerate(dd) if x and i < len(t)])
    steady = steady_grid(info, det, dur)
    lk = dict(map=check.lock(feat, m["pulse_beats"]), grid=check.lock(feat, det),
              marks=check.lock(feat, marks["beats"]) if marks else None,
              steady=check.lock(feat, steady["beats"]) if steady else None)
    fit = [("tempo map", check.summary(lk["map"])), ("current grid (detector)", check.summary(lk["grid"]))]
    if steady:
        fit.append((f"one speed at the stored {steady['bpm']:.2f} BPM", check.summary(lk["steady"])))
    if marks:
        fit.append(("your saved marks", check.summary(lk["marks"])))
    hits, efps = _pool(feat.hits, feat.fps)
    files = {"map": "mix_map.mp3", "grid": "mix_grid.mp3"}
    sources = [("map", "the tempo map"), ("grid", "the current grid (detector)")]
    if steady:
        files["steady"] = "mix_steady.mp3"
        sources.append(("steady", f"one speed at the stored {steady['bpm']:.2f} BPM (Re-Tex unlocked)"))
    if marks:
        files["marks"] = "mix_marks.mp3"
        sources.append(("marks", "your saved marks"))
    if audio:
        log("  writing the click tracks ...")
        mix = render.mix_of(stems)
        bt = set(np.round(m["bar_times"], 4))
        y = render.with_clicks(mix, sr, bars=m["bar_times"], beats=[b for b in m["beats"] if round(b, 4) not in bt],
                               ticks=m["tatums"])
        render.write_mp3(d / files["map"], y, sr)
        render.write_mp3(d / files["grid"], render.with_clicks(mix, sr, bars=grid["bars"], beats=grid["beats"]), sr)
        if steady:
            render.write_mp3(d / files["steady"], render.with_clicks(mix, sr, bars=steady["bars"], beats=steady["beats"]), sr)
        if marks:
            render.write_mp3(d / files["marks"], render.with_clicks(mix, sr, bars=marks["bars"], beats=marks["beats"]), sr)
    tempo, bars = summary_text(m)
    sub = f"{len(m['sections'])} tempo section{'s' if len(m['sections']) != 1 else ''}: {tempo} · bars: {bars} · measured from its stems"
    (d / "report.html").write_text(report.page(m, dict(grid=grid, marks=marks, env=dict(fps=efps, hits=hits), lock=lk, steady=steady,
                                                       fit=fit, files=files, sources=sources, subtitle=sub)), encoding="utf-8")
    m["fit"] = {k: v for k, v in fit}
    (d / "marks.json").write_text(json.dumps(build.retex_marks(m)), encoding="utf-8")
    m["made"] = datetime.datetime.now().isoformat(timespec="seconds")
    (d / "map.json").write_text(json.dumps(m, indent=1), encoding="utf-8")
    log(f"  done in {time.time() - t0:.0f} s: {tempo} | {bars}")
    return d / "report.html"


def write_index(out_root):
    rows = []
    for f in sorted(Path(out_root).glob("*/map.json")):
        try:
            m = json.loads(f.read_text(encoding="utf-8"))
        except ValueError:
            continue
        tempo, bars = summary_text(m)
        rows.append(dict(href=f"{f.parent.name}/report.html", title=m.get("title") or f.parent.name, tempo=tempo, bars=bars,
                         made=(m.get("made") or "").replace("T", " ")))
    p = Path(out_root) / "index.html"
    p.write_text(report.index(rows), encoding="utf-8")
    return p


def main(argv=None):
    ap = argparse.ArgumentParser(description="Tempo map of a Re-Tex library song (read only).")
    ap.add_argument("words", nargs="*", help="words of the song's title")
    ap.add_argument("--data", help="the Re-Tex library folder (default: Re-Tex's own setting)")
    ap.add_argument("--out", default=str(HERE / "maps"))
    ap.add_argument("--all", action="store_true", help="map every song that matches the words")
    ap.add_argument("--no-audio", action="store_true", help="skip the MP3s")
    ap.add_argument("--no-open", action="store_true", help="don't open the page when done")
    ap.add_argument("--id", help="map this song id (as Re-Tex calls it) - no prompts; exit code 0 = done")
    a = ap.parse_args(argv)
    root = library.data_dir(a.data)
    if a.id:                                     # Re-Tex's sidecar call: one song, by id, nothing interactive
        if not (Path(root) / "songs" / a.id / "song.json").exists():
            print(f"no song {a.id} in {root}")
            return 2
        process(root, a.id, a.out, audio=not a.no_audio)
        write_index(a.out)
        return 0
    words = a.words
    if not words:
        try:
            words = input("Song title (a few words): ").split()
        except EOFError:
            return 1
    found = library.find(root, words)
    if not found:
        print(f"No song in {root} has all of: {' '.join(words)}")
        return 1
    if len(found) > 1 and not a.all:
        for i, (sid, title) in enumerate(found[:30], 1):
            print(f"  {i:2d}  {title}")
        try:
            pick = input("Which one (number, or 'all')? ").strip().lower()
        except EOFError:
            return 1
        found = found if pick == "all" else [found[int(pick) - 1]] if pick.isdigit() and 0 < int(pick) <= len(found) else []
        if not found:
            return 1
    last = None
    for sid, title in found:
        print(f"{title}")
        try:
            last = process(root, sid, a.out, audio=not a.no_audio)
        except Exception as e:                                     # one song failing doesn't stop the rest
            print(f"  failed: {type(e).__name__}: {e}")
    idx = write_index(a.out)
    target = last if len(found) == 1 and last else idx
    print(f"\n{target}")
    if not a.no_open and target and os.name == "nt":
        os.startfile(str(target))
    return 0


if __name__ == "__main__":
    sys.exit(main())
