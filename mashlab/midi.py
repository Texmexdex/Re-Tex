"""MIDI songs: a MIDI file, a blank part, or a stem transcribed by MuScriptor, as a library item that behaves like a
song. Its parts are its stems: each part's notes are played by its instrument (instruments.py) into
stems/<part>.flac at the song's own timing, which is what previews, waveforms, the song view and ReadHead read.

What the mixer plays comes from the render worker as for any song (warp._worker -> warp_render here): the notes are
synthesized straight onto the project's tempo, timing spec (as recorded / locked / this project's marks) and
transpose - no time-stretching, so a MIDI part is always exactly on its grid and never smeared.

Files, in data/midi/midi_<id>/ (config.song_path):
  song.json  song-shaped (kind "midi"): beats (seconds of every beat - from the file's tempo map, or another song's
             grid), bpm, bpb, bars, sections, key, stems = part ids, stem_names, drum_stems, stems_rev
  midi.json  {"parts": [{id, name, channel, program, bank, drum, notes: [[start beat, length beats, pitch, velocity]],
             cc: {"pb": [[beat, value]], "<cc#>": [[beat, value]]}, instrument: {...}}]}
Notes are in the song's beats, so a note keeps its place in the music whatever the tempo or timing it plays at."""
import hashlib
import json
import math
import re
import time
from pathlib import Path

import numpy as np

from . import audio_io, instruments
from .config import MIDI_DIR, SR, song_path
from .events import bus

PEAKS_PER_SEC = 100
TAIL = 2.0                  # seconds rendered after the last note ends (release tails)
STEM_SCALE = 0.5            # part stems are saved 6 dB down (24-bit FLAC can't hold overs); readers undo it
KS_MAJ = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KS_MIN = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def is_midi(sid):
    return str(sid).startswith("midi_")


# -------------------------------------------------------------------- files
def read(sid):
    p = song_path(sid) / "midi.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def write(sid, mi):
    p = song_path(sid) / "midi.json"
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(mi), encoding="utf-8")
    tmp.replace(p)


def part_of(mi, pid):
    for p in mi["parts"]:
        if p["id"] == pid:
            return p
    raise KeyError(f"no part {pid}")


def list_songs():
    from . import library
    out = []
    if MIDI_DIR.is_dir():
        for d in MIDI_DIR.iterdir():
            if d.is_dir() and d.name.startswith("midi_"):
                info = library.read(d.name)
                if info:
                    out.append(library.summary(info))
    return out


# -------------------------------------------------------------------- beats <-> seconds
def beat_time(beats, b):
    """Seconds of (fractional) beat b on a beat grid, carried on past either end at the nearest beat's spacing."""
    bt = np.asarray(beats, float)
    b = np.asarray(b, float)
    idx = np.arange(len(bt), dtype=float)
    t = np.interp(b, idx, bt)
    lo, hi = b < 0, b > idx[-1]
    t = np.where(lo, bt[0] + b * (bt[1] - bt[0]), t)
    t = np.where(hi, bt[-1] + (b - idx[-1]) * (bt[-1] - bt[-2]), t)
    return t


def time_beat(beats, t):
    """The inverse: (fractional) beat at t seconds."""
    bt = np.asarray(beats, float)
    t = np.asarray(t, float)
    idx = np.arange(len(bt), dtype=float)
    b = np.interp(t, bt, idx)
    b = np.where(t < bt[0], (t - bt[0]) / (bt[1] - bt[0]), b)
    b = np.where(t > bt[-1], idx[-1] + (t - bt[-1]) / (bt[-1] - bt[-2]), b)
    return b


def _extrap(src, dst):
    """x -> y along matching points (src, dst), straight on past the ends."""
    src, dst = np.asarray(src, float), np.asarray(dst, float)

    def f(x):
        x = np.asarray(x, float)
        y = np.interp(x, src, dst)
        y = np.where(x < src[0], dst[0] + (x - src[0]) * (dst[1] - dst[0]) / (src[1] - src[0]), y)
        return np.where(x > src[-1], dst[-1] + (x - src[-1]) * (dst[-1] - dst[-2]) / (src[-1] - src[-2]), y)
    return f


def last_beat_used(mi):
    end = 0.0
    for p in mi["parts"]:
        for s, ln, _, _ in p["notes"]:
            end = max(end, s + ln)
    return end


# -------------------------------------------------------------------- reading MIDI files
def _text(t):
    """mido reads names as latin-1; most files are UTF-8 or an old Windows code page (Cyrillic, Central European...).
    Re-read the raw bytes as the likeliest of those."""
    try:
        raw = t.encode("latin-1")
    except UnicodeEncodeError:
        return t
    if not any(c >= 0x80 for c in raw):
        return t
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        pass
    u = raw.decode("cp1251", errors="replace")
    letters = [ch for ch in u if ch.isalpha()]
    if letters and sum("\u0400" <= ch <= "\u04ff" for ch in letters) / len(letters) > 0.5:
        return u                                        # mostly Cyrillic: a Russian / Ukrainian file
    return raw.decode("cp1252", errors="replace")


def parse_file(path):
    """A .mid file -> {ppq, beats_sec (tempo map, one per beat), bpm, bpb, parts}. One part per (track, channel) that
    plays notes; notes / controllers in beats (quarter notes)."""
    import mido
    mf = mido.MidiFile(str(path))
    ppq = mf.ticks_per_beat
    tempos, tsig, parts = [], None, {}
    for ti, tr in enumerate(mf.tracks):
        tick, name = 0, None
        for msg in tr:
            tick += msg.time
            if msg.is_meta:
                if msg.type == "set_tempo":
                    tempos.append((tick, msg.tempo))
                elif msg.type == "time_signature" and tsig is None:
                    tsig = (msg.numerator, msg.denominator)
                elif msg.type in ("track_name", "instrument_name") and not name:
                    name = _text(msg.name).strip() or None
                continue
            ch = getattr(msg, "channel", None)
            if ch is None:
                continue
            p = parts.setdefault((ti, ch), dict(track=ti, channel=ch, name=None, program=None, bank=0, notes=[],
                                                cc={}, open={}))
            if name and not p["name"]:
                p["name"] = name
            b = tick / ppq
            if msg.type == "program_change":
                if p["program"] is None:
                    p["program"] = msg.program
            elif msg.type == "control_change":
                if msg.control == 0:
                    p["bank"] = msg.value
                elif msg.control in (1, 7, 10, 11, 64, 71, 74, 91, 93):
                    p["cc"].setdefault(str(msg.control), []).append([round(b, 6), msg.value])
            elif msg.type == "pitchwheel":
                p["cc"].setdefault("pb", []).append([round(b, 6), msg.pitch])
            elif msg.type == "note_on" and msg.velocity > 0:
                p["open"].setdefault(msg.note, []).append((b, msg.velocity))
            elif msg.type in ("note_off", "note_on"):
                st = p["open"].get(msg.note)
                if st:
                    b0, v = st.pop(0)
                    p["notes"].append([round(b0, 6), round(max(b - b0, 1 / 64), 6), msg.note, v])
        for p in parts.values():                              # notes never let go: end them at the track's end
            if p["track"] == ti:
                for note, st in p["open"].items():
                    for b0, v in st:
                        p["notes"].append([round(b0, 6), round(max(tick / ppq - b0, 0.25), 6), note, v])
                p["open"] = {}
    # tempo map -> seconds of every beat
    tm = {}
    for tick, tempo in sorted(tempos):
        tm[tick] = tempo
    tm.setdefault(0, tm.get(min(tm), 500000) if tm else 500000)
    marks = sorted(tm.items())
    out_parts = []
    k = 0
    for (ti, ch), p in sorted(parts.items()):
        if not p["notes"]:
            continue
        k += 1
        drum = ch == 9
        prog = p["program"] or 0
        nm = p["name"] or ("Drums" if drum else instruments.GM[prog])
        p["notes"].sort(key=lambda n: (n[0], n[2]))
        for v in p["cc"].values():
            v.sort()
        out_parts.append(dict(id=f"p{k}", name=nm, channel=ch, program=prog, bank=p["bank"], drum=drum,
                              notes=p["notes"], cc=p["cc"]))
    end_beat = max((n[0] + n[1] for p in out_parts for n in p["notes"]), default=4.0)
    nb = int(math.ceil(end_beat)) + 9

    def tick_sec(T):
        t = 0.0
        for j, (mt, tempo) in enumerate(marks):
            if mt >= T:
                break
            nxt = marks[j + 1][0] if j + 1 < len(marks) else T
            t += (min(T, nxt) - mt) / ppq * tempo / 1e6
        return t
    beats_sec = [tick_sec(i * ppq) for i in range(nb)]
    # the song's tempo: the one held longest
    dur, end_tick = {}, max(1, int(end_beat * ppq))
    for j, (mt, tempo) in enumerate(marks):
        nxt = min(marks[j + 1][0] if j + 1 < len(marks) else end_tick, end_tick)
        dur[tempo] = dur.get(tempo, 0) + max(0, nxt - mt)
    bpm = 60e6 / max(dur, key=dur.get)
    bpb = 4
    if tsig:
        q = tsig[0] * 4 / tsig[1]
        bpb = int(q) if abs(q - round(q)) < 1e-6 and 2 <= q <= 12 else 4
    return dict(ppq=ppq, beats=beats_sec, bpm=bpm, bpb=bpb, parts=out_parts)


# -------------------------------------------------------------------- song records
def _key(parts):
    h = np.zeros(12)
    for p in parts:
        if p.get("drum"):
            continue
        for s, ln, pitch, v in p["notes"]:
            h[int(pitch) % 12] += ln * v
    if h.sum() <= 0:
        return dict(tonic=0, mode="major", conf=0.0, name="C major", camelot="8B")
    best = None
    for t in range(12):
        for mode, prof in (("major", KS_MAJ), ("minor", KS_MIN)):
            r = float(np.corrcoef(h, np.roll(prof, t))[0, 1])
            if best is None or r > best[0]:
                best = (r, t, mode)
    from . import harmony as H
    r, t, mode = best
    return dict(tonic=t, mode=mode, conf=round(max(0.0, r), 2), name=H.key_name(t, mode), camelot=H.camelot(t, mode))


def _slug(s):
    return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")[:40] or "midi"


def _info(sid, title, source, beats, bpm, bpb, parts, first_beat=0, sections=None, parent=None, created=None):
    nb = len(beats)
    nbars = max(1, (nb - first_beat) // bpb)
    bars = [first_beat + i * bpb for i in range(nbars + 1) if first_beat + i * bpb < nb]
    if sections is None:
        sections = [dict(start=0, end=len(bars), label="A", name="MIDI", energy=0.5, stems={p["id"]: 1.0 for p in parts},
                         repeats=1)]
    end = max((n[0] + n[1] for p in parts for n in p["notes"]), default=4.0)
    return dict(id=sid, title=title, kind="midi", status="rendering", error=None, source=source, created=created or time.time(),
                duration=round(float(beat_time(beats, end)) + TAIL, 4), bpm=round(float(bpm), 4), bpb=int(bpb),
                stable_tempo=bool(np.ptp(np.diff(beats)) < 1e-4), beats=[round(float(t), 6) for t in beats], bars=bars,
                first_beat=int(first_beat), last_beat=nb - 1, segments=[], sections=sections, key=_key(parts),
                tuning=0.0, tuning_st=0.0, separation="midi", has_vocals=False, peaks_per_sec=PEAKS_PER_SEC,
                grid_version=1, stems_rev=0, stem_scale=STEM_SCALE, stems=[p["id"] for p in parts],
                stem_names={p["id"]: p["name"] for p in parts}, drum_stems=[p["id"] for p in parts if p.get("drum")],
                parent=parent)


def create(title, source, beats, bpm, bpb, parts, first_beat=0, sections=None, parent=None, render=True):
    """A new MIDI song (parts already have instruments, or get the default for their program). -> sid."""
    from . import library
    MIDI_DIR.mkdir(parents=True, exist_ok=True)
    sid = f"midi_{_slug(title)}_{hashlib.sha1(f'{title}{time.time()}'.encode()).hexdigest()[:6]}"
    d = song_path(sid)
    (d / "stems").mkdir(parents=True, exist_ok=True)
    (d / "peaks").mkdir(exist_ok=True)
    for p in parts:
        p.setdefault("cc", {})
        p.setdefault("instrument", instruments.default_for(p.get("program", 0), p.get("drum", False), p.get("name")))
    write(sid, {"version": 1, "parts": parts})
    info = _info(sid, title, source, beats, bpm, bpb, parts, first_beat, sections, parent)
    library.write(sid, info)
    bus.publish("song", song=library.summary(info))
    if render:
        render_all(sid)
    return sid


def refresh_info(sid):
    """Keep song.json in step with midi.json (parts added / removed / renamed, length, key)."""
    from . import library
    info, mi = library.read(sid), read(sid)
    parts = mi["parts"]
    info["stems"] = [p["id"] for p in parts]
    info["stem_names"] = {p["id"]: p["name"] for p in parts}
    info["drum_stems"] = [p["id"] for p in parts if p.get("drum")]
    info["key"] = _key(parts)
    end = max((n[0] + n[1] for p in parts for n in p["notes"]), default=4.0)
    need = int(math.ceil(end)) + 9
    beats = list(info["beats"])
    while len(beats) < need:                                      # notes past the grid: carry it on
        beats.append(round(beats[-1] + (beats[-1] - beats[-2]), 6))
    info["beats"], info["last_beat"] = beats, len(beats) - 1
    info["bars"] = [b for b in range(info["first_beat"], len(beats), info["bpb"])]
    info["duration"] = round(float(beat_time(beats, end)) + TAIL, 4)
    secs = info.get("sections") or []
    for sec in secs:
        sec["stems"] = {p["id"]: 1.0 for p in parts}
    if secs:
        secs[-1]["end"] = max(secs[-1]["end"], len(info["bars"]))
    library.write(sid, info)
    return info


# -------------------------------------------------------------------- rendering
def _events_at(part, to_out):
    """The part's notes and controllers as (seconds where they play): to_out maps a beat -> output seconds."""
    notes = part["notes"]
    if notes:
        a = np.asarray(notes, float)
        on, off = to_out(a[:, 0]), to_out(a[:, 0] + a[:, 1])
        ev = [(float(on[i]), float(off[i]), int(a[i, 2]), int(a[i, 3])) for i in range(len(a))]
    else:
        ev = []
    cc = []
    for ctrl, pts in (part.get("cc") or {}).items():
        if not pts:
            continue
        p = np.asarray(pts, float)
        ts = to_out(p[:, 0])
        cc += [(float(ts[i]), ctrl if ctrl == "pb" else int(ctrl), int(p[i, 1])) for i in range(len(p))]
    return ev, cc


def render_natural(info, part):
    """The part at the song's own timing (its beat grid in seconds) -> (2, n)."""
    beats = info["beats"]
    end = max((n[0] + n[1] for n in part["notes"]), default=0.0)
    n = int((float(beat_time(beats, end)) + TAIL) * SR) + 1
    ev, cc = _events_at(part, lambda b: beat_time(beats, b))
    return instruments.render(part.get("instrument"), ev, cc, n)


def _fit(y, n):
    """y padded with silence or cut to n samples - every part of a MIDI song is exactly the song's length, so its
    stems can be added together like any song's (the audition, the song view's waveform, ReadHead...)."""
    y = np.asarray(y, np.float32)
    if n is None or y.shape[1] == n:
        return y
    return y[:, :n] if y.shape[1] > n else np.pad(y, ((0, 0), (0, n - y.shape[1])))


def _song_samples(info):
    return max(1, int(round(float(info.get("duration") or 1.0) * SR)))


def _save_part_audio(sid, pid, y, n=None):
    y = _fit(y, n)
    d = song_path(sid)
    audio_io.save(d / "stems" / f"{pid}.flac", np.clip(y * STEM_SCALE, -1, 1), SR, subtype="PCM_24")
    audio_io.peaks(y, SR, PEAKS_PER_SEC).tofile(d / "peaks" / f"{pid}.bin")
    for f in (d / "peaks").glob(f"{pid}.hi.bin"):
        f.unlink(missing_ok=True)


def _save_mix_peaks(sid, info):
    d = song_path(sid)
    ys = []
    for pid in info["stems"]:
        f = d / "stems" / f"{pid}.flac"
        if f.exists():
            ys.append(audio_io.load(f, SR) / STEM_SCALE)
    if ys:
        n = max(y.shape[1] for y in ys)
        mix = sum(np.pad(y, ((0, 0), (0, n - y.shape[1]))) for y in ys)
        audio_io.peaks(mix, SR, PEAKS_PER_SEC).tofile(d / "peaks" / "mix.bin")


def render_all(sid, pids=None, progress=lambda f, m="": None):
    """(Re-)render parts' stems at the song's own timing, then mark the song changed so every render made from the
    old stems is replaced (stems_rev, as when a song is re-separated). Old MIDI renders are deleted: they can't be
    played any more and are quick to remake."""
    from . import library, warp
    info, mi = library.read(sid), read(sid)
    todo = [p for p in mi["parts"] if pids is None or p["id"] in pids]
    errors = {}
    for i, p in enumerate(todo):
        progress(i / max(1, len(todo)), f"playing {p['name']}")
        try:
            _save_part_audio(sid, p["id"], render_natural(info, p), _song_samples(info))
        except Exception as e:
            errors[p["id"]] = f"{type(e).__name__}: {e}"
            _save_part_audio(sid, p["id"], np.zeros((2, int(SR)), np.float32), _song_samples(info))
    for pid in [f.stem for f in (song_path(sid) / "stems").glob("*.flac")]:
        if pid not in info["stems"]:
            (song_path(sid) / "stems" / f"{pid}.flac").unlink(missing_ok=True)
    import soundfile as sf
    n_all = _song_samples(info)
    for pid in info["stems"]:                      # parts not re-rendered: still the song's length (it may have grown)
        f = song_path(sid) / "stems" / f"{pid}.flac"
        if f.exists() and sf.info(str(f)).frames != n_all:
            _save_part_audio(sid, pid, audio_io.load(f, SR) / STEM_SCALE, n_all)
    _save_mix_peaks(sid, info)
    with library._lock:
        for k in [k for k in library._stem_cache if k[0] == sid]:
            del library._stem_cache[k]
    rev = int(info.get("stems_rev", 0) or 0) + 1
    warp.drop_song(sid)                     # renders of the old notes / instruments (and the open-file cache)
    info = library._set(sid, stems_rev=rev, status="ready", error=("; ".join(f"{k}: {v}" for k, v in errors.items()) or None),
                        part_errors=errors or None)
    return info


def warp_render(sid, info, stem, bpm, mult=1.0, semis=0.0, quality="hq", tspec=""):
    """warp.render for a MIDI part: (out (2, n), t0 s, method), synthesized at the target tempo / timing instead of
    stretched; None for timing it doesn't handle (cuts) - the worker then stretches the part's stem as for audio.
    Same geometry as warp.render (warp._geometry / warp.timed), so a MIDI clip sits where an audio clip would."""
    from . import timing, variant, warp
    spec = timing.parse(tspec) if tspec else None
    if spec and abs(float(spec.get("s", 1.0)) - 1.0) > 1e-9:          # screwed: unscrewed render, then resampled
        s = float(spec["s"])
        base = {k: v for k, v in spec.items() if k != "s"}
        r = warp_render(sid, info, stem, bpm / s, mult, semis - 12.0 * np.log2(s), quality,
                        json.dumps(base, sort_keys=True, separators=(",", ":")))
        if r is None:
            return None
        out, t0, method = r
        n = int(np.ceil(out.shape[1] / s))
        return warp.resample_map(out, np.array([0.0, float(out.shape[1])]), np.array([0.0, out.shape[1] / s]), n), t0 / s, method + "+screw"
    if spec and spec.get("c"):
        return None
    part = part_of(read(sid), stem)
    beats = np.asarray(variant.beats_for(info, tspec), float)
    W = None
    if tspec:
        bt, W = warp.timed(beats, tspec)
        if len(bt) != len(beats):
            return None
    end = max((n[0] + n[1] for n in part["notes"]), default=0.0)
    n_in = int((float(beat_time(beats, end)) + TAIL) * SR) + 1
    Tb, t0, src, dst, n_out = warp._geometry(beats, n_in, bpm, mult, W)
    to_out = _extrap(src, dst)
    shift = 0 if part.get("drum") else int(round(semis))
    ev, cc = _events_at(part, lambda b: to_out(beat_time(beats, b) * SR) / SR)
    ev = [(a, b, p + shift, v) for a, b, p, v in ev if 0 <= p + shift <= 127]
    out = instruments.render(part.get("instrument"), ev, cc, n_out)
    return out, t0, "midi"


# -------------------------------------------------------------------- import / edit
def import_file(path, title=None):
    """A .mid file -> a MIDI song, played by the default instruments. -> sid (renders in a background job)."""
    from . import jobs
    m = parse_file(path)
    if not m["parts"]:
        raise ValueError("that MIDI file has no notes")
    title = title or Path(path).stem
    sid = create(title, Path(path).name, m["beats"], m["bpm"], m["bpb"], m["parts"], render=False)
    job = jobs.submit("midi", f"MIDI: {title}", lambda job: _render_job(sid, None, job), ref=sid)
    return sid, job


def _render_job(sid, pids, job):
    from . import library
    info = render_all(sid, pids, job.sub(0.0, 1.0, "") if hasattr(job, "sub") else (lambda f, m="": None))
    return library.summary(info)


def blank(title, bpm, bpb=4, bars=8, program=0, drum=False):
    """An empty MIDI song with one part, for drawing notes in the piano roll. -> sid."""
    spb = 60.0 / float(bpm)
    beats = [round(i * spb, 6) for i in range(bars * bpb + 9)]
    part = dict(id="p1", name="Drums" if drum else instruments.GM[program], channel=9 if drum else 0, program=program,
                bank=0, drum=drum, notes=[], cc={})
    return create(title, None, beats, bpm, bpb, [part])


def save_part(sid, pid, notes=None, cc=None, name=None, instrument=None, drum=None):
    """Change a part (notes, controllers, name, instrument) and re-render it. -> the job."""
    from . import jobs
    mi = read(sid)
    p = part_of(mi, pid)
    if notes is not None:
        p["notes"] = sorted([[round(float(s), 6), round(max(float(ln), 1 / 256), 6), int(pt), int(v)]
                             for s, ln, pt, v in notes if 0 <= int(pt) <= 127], key=lambda n: (n[0], n[2]))
    if cc is not None:
        p["cc"] = {str(k): sorted([[round(float(b), 6), int(v)] for b, v in pts]) for k, pts in cc.items()}
    if name is not None:
        p["name"] = str(name)[:60]
    if instrument is not None:
        if instrument.get("type") == "chops" and instrument.get("song") == sid and instrument.get("stem") == pid:
            raise ValueError("a part can't be chopped from its own sound - pick another part or song")
        p["instrument"] = instrument
    if drum is not None:
        p["drum"] = bool(drum)
    write(sid, mi)
    refresh_info(sid)
    return jobs.submit("midi", f"MIDI: {p['name']}", lambda job: _render_job(sid, [pid], job), ref=sid)


def add_part(sid, name="New part", program=0, drum=False, copy_of=None):
    mi = read(sid)
    k = 1 + max([int(p["id"][1:]) for p in mi["parts"] if p["id"][1:].isdigit()] + [0])
    if copy_of:
        src = part_of(mi, copy_of)
        p = json.loads(json.dumps(src))
        p.update(id=f"p{k}", name=(name if name != "New part" else src["name"] + " (copy)"))
    else:
        p = dict(id=f"p{k}", name=name, channel=9 if drum else 0, program=program, bank=0, drum=drum, notes=[], cc={},
                 instrument=instruments.default_for(program, drum, name))
    mi["parts"].append(p)
    write(sid, mi)
    refresh_info(sid)
    return p["id"]


def delete_part(sid, pid):
    mi = read(sid)
    if len(mi["parts"]) <= 1:
        raise ValueError("a MIDI song keeps at least one part")
    mi["parts"] = [p for p in mi["parts"] if p["id"] != pid]
    write(sid, mi)
    refresh_info(sid)


def take_timing(sid, other):
    """Play this MIDI song on another song's beat grid (its drift, tempo changes and bar lines): note at beat k sounds
    where that song's beat k is. -> the render job."""
    from . import jobs, library
    o = library.read(other)
    if not o or not o.get("beats"):
        raise ValueError("that song has no beat grid")
    info = library.read(sid)
    info.update(beats=list(o["beats"]), bpm=o["bpm"], bpb=o["bpb"], first_beat=int(o.get("first_beat", 0)),
                bars=list(o["bars"]), stable_tempo=o.get("stable_tempo", False), timing_from=other)
    library.write(sid, info)
    refresh_info(sid)
    return jobs.submit("midi", f"MIDI: timing from {o['title']}", lambda job: _render_job(sid, None, job), ref=sid)


GM_GUESS = [("drum", None), ("perc", None), ("bass", 33), ("piano", 0), ("key", 4), ("organ", 16), ("guitar", 27),
            ("string", 48), ("violin", 40), ("cello", 42), ("brass", 61), ("trumpet", 56), ("sax", 65), ("flute", 73),
            ("voice", 53), ("vocal", 53), ("choir", 52), ("synth", 81), ("lead", 81), ("pad", 89)]


def from_transcription(parent_sid, stem, notes, label=None):
    """MuScriptor notes [[onset s, offset s, pitch, instrument]] of a stem -> a MIDI song on the parent song's own
    beat grid, so its clips line up with the stem's. -> sid."""
    from . import library
    info = library.read(parent_sid)
    beats = info["beats"]
    groups = {}
    for on, off, pitch, inst in notes:
        groups.setdefault(str(inst or "notes"), []).append((on, off, pitch))
    parts = []
    for k, (inst, ns) in enumerate(sorted(groups.items(), key=lambda kv: -len(kv[1])), 1):
        low = inst.lower()
        guess = next((g for w, g in GM_GUESS if w in low), "none")
        drum = guess is None
        prog = 0 if guess in (None, "none") else guess
        if guess == "none":
            prog = 33 if stem == "bass" else 0
        b_on = time_beat(beats, [n[0] for n in ns])
        b_off = time_beat(beats, [n[1] for n in ns])
        parts.append(dict(id=f"p{k}", name=f"{inst.title()} ({stem})", channel=9 if drum else 0, program=prog, bank=0,
                          drum=drum, cc={}, notes=sorted([[round(float(a), 6), round(max(float(b - a), 1 / 64), 6), int(n[2]), 96]
                                                          for a, b, n in zip(b_on, b_off, ns)])))
    if not parts:
        raise ValueError("no notes were found in that stem")
    secs = [dict(s, stems={p["id"]: 1.0 for p in parts}) for s in (info.get("sections") or [])] or None
    return create(label or f"MIDI: {info['title']} ({stem})", None, beats, info["bpm"], info["bpb"], parts,
                  first_beat=int(info.get("first_beat", 0)), sections=secs, parent=dict(song=parent_sid, stem=stem))


def transcribe(parent_sid, stem):
    """MuScriptor on a stem -> a MIDI song (a background job, GPU). -> the job."""
    from . import jobs, library, muscheck

    def work(job):
        job.update(0.02, "loading MuScriptor")
        y = library.read_stem(parent_sid, stem)
        tmp = song_path(parent_sid) / f"midi_in_{stem}.tmp.wav"
        audio_io.save(tmp, y, SR, subtype="FLOAT")
        try:
            info = library.read(parent_sid)
            dur = float(info.get("duration") or y.shape[1] / SR)
            notes = muscheck.transcribe(tmp, progress=lambda t: job.update(0.05 + 0.85 * min(1.0, t / max(1.0, dur)), "transcribing"))
        finally:
            tmp.unlink(missing_ok=True)
        sid = from_transcription(parent_sid, stem, notes)
        return {"sid": sid}
    return jobs.submit("midi", f"MIDI from {stem}", work, ref=parent_sid, gpu=True)


def to_midi_file(sid, path, pids=None):
    """Write the song (or some parts) as a standard .mid, with its tempo map."""
    import mido
    from . import library
    info, mi = library.read(sid), read(sid)
    ppq = 480
    beats = np.asarray(info["beats"], float)
    mf = mido.MidiFile(ticks_per_beat=ppq)
    tt = mido.MidiTrack()
    mf.tracks.append(tt)
    tt.append(mido.MetaMessage("time_signature", numerator=int(info["bpb"]), denominator=4, time=0))
    last, prev_tempo = 0, None
    for i in range(len(beats) - 1):
        tempo = int(round((beats[i + 1] - beats[i]) * 1e6))
        if tempo != prev_tempo:
            tt.append(mido.MetaMessage("set_tempo", tempo=max(1, tempo), time=i * ppq - last))
            last, prev_tempo = i * ppq, tempo
    for p in mi["parts"]:
        if pids and p["id"] not in pids:
            continue
        ch = 9 if p.get("drum") else min(15, max(0, int(p.get("channel", 0) or 0)))
        ch = 0 if ch == 9 and not p.get("drum") else ch
        tr = mido.MidiTrack()
        mf.tracks.append(tr)
        tr.append(mido.MetaMessage("track_name", name=p["name"].encode("utf-8").decode("latin-1"), time=0))   # UTF-8 bytes (_text reads them back)
        ev = [(0, 0, mido.Message("program_change", channel=ch, program=int(p.get("program", 0) or 0)))]
        for s, ln, pt, v in p["notes"]:
            ev.append((int(round(s * ppq)), 2, mido.Message("note_on", channel=ch, note=int(pt), velocity=int(v))))
            ev.append((int(round((s + ln) * ppq)), 1, mido.Message("note_off", channel=ch, note=int(pt), velocity=0)))
        for ctrl, pts in (p.get("cc") or {}).items():
            for b, v in pts:
                m = mido.Message("pitchwheel", channel=ch, pitch=int(v)) if ctrl == "pb" else \
                    mido.Message("control_change", channel=ch, control=int(ctrl), value=int(v))
                ev.append((int(round(b * ppq)), 0, m))
        ev.sort(key=lambda e: (e[0], e[1]))
        t = 0
        for tick, _, m in ev:
            tr.append(m.copy(time=tick - t))
            t = tick
    mf.save(str(path))
    return path
