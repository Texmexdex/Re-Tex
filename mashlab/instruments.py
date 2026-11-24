"""Instruments that play a MIDI part (mashlab/midi.py): SF2 soundfonts (FluidSynth), VST3 instruments (Pedalboard),
a sampler (any audio file, or a span of a song in the library) and a small built-in synth (always there, so a part
makes sound before anything is set up).

An instrument is a plain dict, stored with the part in midi.json:
  {"type": "sf2", "path": ".../x.sf2", "bank": 0, "program": 0, "name": "...", "gain_db": 0, "reverb": false}
  {"type": "vst3", "path": ".../Synth.vst3", "plugin_name": null, "state": "<base64 raw_state>", "name": "...", "gain_db": 0}
  {"type": "sampler", "path": ".../hit.wav" | null, "span": {"song", "stem", "t0", "t1"} | null, "root": 60,
   "mode": "pitched" | "oneshot", "attack": 0.003, "release": 0.12, "loop": false, "gain_db": 0, "name": "..."}
  {"type": "basic", "wave": "saw" | "square" | "sine" | "triangle", "attack": 0.005, "decay": 0.2, "sustain": 0.7,
   "release": 0.15, "cutoff": 6000, "gain_db": 0, "name": "..."}
  {"type": "chops", ...}: a stem cut into chops, one per note (chops.py)

Where to look for soundfonts and VST3 plugins: models/soundfonts and the standard VST3 folders, plus the folders in
data/config.json "midi": {"soundfont_dirs": [...], "vst3_dirs": [...]} (the instrument dialog's Folders... edits them).

render(inst, notes, cc, n) -> (2, n) float32. notes: [(t_on s, t_off s, pitch, velocity)], cc: [(t s, ctrl, value)]
with ctrl an int CC number or "pb" (pitch bend, -8192..8191). Times are where they play in the output, so the caller
(midi.py) has already put them on whatever timing the part plays at. Runs in render worker processes too."""
import base64
import glob
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from .config import ROOT, SR

DRUM_BANK = 128
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)

GM = ["Acoustic Grand Piano", "Bright Acoustic Piano", "Electric Grand Piano", "Honky-tonk Piano", "Electric Piano 1",
      "Electric Piano 2", "Harpsichord", "Clavinet", "Celesta", "Glockenspiel", "Music Box", "Vibraphone", "Marimba",
      "Xylophone", "Tubular Bells", "Dulcimer", "Drawbar Organ", "Percussive Organ", "Rock Organ", "Church Organ",
      "Reed Organ", "Accordion", "Harmonica", "Tango Accordion", "Acoustic Guitar (nylon)", "Acoustic Guitar (steel)",
      "Electric Guitar (jazz)", "Electric Guitar (clean)", "Electric Guitar (muted)", "Overdriven Guitar",
      "Distortion Guitar", "Guitar Harmonics", "Acoustic Bass", "Electric Bass (finger)", "Electric Bass (pick)",
      "Fretless Bass", "Slap Bass 1", "Slap Bass 2", "Synth Bass 1", "Synth Bass 2", "Violin", "Viola", "Cello",
      "Contrabass", "Tremolo Strings", "Pizzicato Strings", "Orchestral Harp", "Timpani", "String Ensemble 1",
      "String Ensemble 2", "Synth Strings 1", "Synth Strings 2", "Choir Aahs", "Voice Oohs", "Synth Voice",
      "Orchestra Hit", "Trumpet", "Trombone", "Tuba", "Muted Trumpet", "French Horn", "Brass Section", "Synth Brass 1",
      "Synth Brass 2", "Soprano Sax", "Alto Sax", "Tenor Sax", "Baritone Sax", "Oboe", "English Horn", "Bassoon",
      "Clarinet", "Piccolo", "Flute", "Recorder", "Pan Flute", "Blown Bottle", "Shakuhachi", "Whistle", "Ocarina",
      "Lead 1 (square)", "Lead 2 (sawtooth)", "Lead 3 (calliope)", "Lead 4 (chiff)", "Lead 5 (charang)",
      "Lead 6 (voice)", "Lead 7 (fifths)", "Lead 8 (bass + lead)", "Pad 1 (new age)", "Pad 2 (warm)",
      "Pad 3 (polysynth)", "Pad 4 (choir)", "Pad 5 (bowed)", "Pad 6 (metallic)", "Pad 7 (halo)", "Pad 8 (sweep)",
      "FX 1 (rain)", "FX 2 (soundtrack)", "FX 3 (crystal)", "FX 4 (atmosphere)", "FX 5 (brightness)", "FX 6 (goblins)",
      "FX 7 (echoes)", "FX 8 (sci-fi)", "Sitar", "Banjo", "Shamisen", "Koto", "Kalimba", "Bagpipe", "Fiddle", "Shanai",
      "Tinkle Bell", "Agogo", "Steel Drums", "Woodblock", "Taiko Drum", "Melodic Tom", "Synth Drum", "Reverse Cymbal",
      "Guitar Fret Noise", "Breath Noise", "Seashore", "Bird Tweet", "Telephone Ring", "Helicopter", "Applause",
      "Gunshot"]


# -------------------------------------------------------------------- where things are
def fluidsynth_exe():
    """FluidSynth's command-line player: the copy install.bat puts in tools/, else one on PATH."""
    for p in (ROOT / "tools" / "fluidsynth" / "bin" / "fluidsynth.exe", ROOT / "tools" / "fluidsynth" / "fluidsynth.exe"):
        if p.exists():
            return str(p)
    return shutil.which("fluidsynth")


def _midi_cfg():
    try:
        from . import appcfg
        return (appcfg.load_config() or {}).get("midi") or {}
    except Exception:
        return {}


def folders():
    """{"sf2": [...], "vst3": [...]}: the folders added in the dialog (the built-in ones aren't listed)."""
    c = _midi_cfg()
    return {"sf2": list(c.get("soundfont_dirs") or []), "vst3": list(c.get("vst3_dirs") or [])}


def set_folders(sf2=None, vst3=None):
    from . import appcfg
    cfg = appcfg.load_config() or {}
    m = dict(cfg.get("midi") or {})
    clean = lambda xs: list(dict.fromkeys(str(Path(x)) for x in xs if str(x).strip()))
    if sf2 is not None:
        m["soundfont_dirs"] = clean(sf2)
    if vst3 is not None:
        m["vst3_dirs"] = clean(vst3)
    cfg["midi"] = m
    appcfg.save_config(cfg)
    return folders()


def soundfont_dirs():
    return [ROOT / "models" / "soundfonts"] + [Path(p) for p in folders()["sf2"]]


def list_soundfonts(limit=3000):
    """Every .sf2 / .sf3 under the soundfont folders and their subfolders -> [{path, name, folder, mb}]."""
    out, seen = [], set()
    for root in soundfont_dirs():
        if not root.is_dir():
            continue
        for d, subs, files in os.walk(root):
            subs.sort()
            for f in sorted(files):
                if f.lower().endswith((".sf2", ".sf3")):
                    p = os.path.join(d, f)
                    k = os.path.normcase(os.path.abspath(p))
                    if k in seen:
                        continue
                    seen.add(k)
                    try:
                        mb = round(os.path.getsize(p) / 1e6, 1)
                    except OSError:
                        mb = None
                    rel = os.path.relpath(d, root)
                    out.append({"path": p, "name": os.path.splitext(f)[0], "mb": mb,
                                "folder": root.name if rel == "." else os.path.join(root.name, rel)})
                    if len(out) >= limit:
                        return out
    return out


def default_sf2(quiet=False):
    """The General MIDI soundfont parts start with: data/config.json "midi" {"sf2": path} if set, else the first
    .sf2 in models/soundfonts/ (install.bat puts GeneralUser GS there), else the first in your soundfont folders.
    Any step past the one you set is a fallback and says so (components.fallback), unless quiet (a status check)."""
    from . import appcfg, components
    p = ((appcfg.load_config() or {}).get("midi") or {}).get("sf2")
    if p and Path(p).exists():
        return str(p)
    d = ROOT / "models" / "soundfonts"
    found = next(iter((sorted(d.glob("*.sf2")) + sorted(d.glob("*.sf3"))) if d.is_dir() else []), None)
    if found is None:
        fonts = list_soundfonts(1)
        found = fonts[0]["path"] if fonts else None
    if p and not quiet:
        components.fallback("soundfont", f"the soundfont you set ({p}) is gone, so new parts use "
                                         f"{found or 'the built-in synth'} instead")
    elif found and not quiet and not str(found).startswith(str(d)):
        components.fallback("soundfont", f"there's no soundfont in {d}, so new parts use {found} from your soundfont "
                                         f"folders instead")
    return str(found) if found else None


def default_for(program=0, drum=False, name=None):
    """The instrument a new part gets: the GM sound for its program from the default soundfont, else the built-in synth."""
    sf2 = default_sf2()
    if sf2 and fluidsynth_exe():
        return {"type": "sf2", "path": sf2, "bank": DRUM_BANK if drum else 0, "program": 0 if drum else int(program or 0),
                "name": "Drum kit" if drum else GM[int(program or 0) % 128], "gain_db": 0, "reverb": False}
    from . import components
    components.fallback("fluidsynth" if sf2 else "soundfont",
                        "new MIDI parts get the built-in synth / drums instead of the General MIDI sounds")
    if drum:
        return {"type": "basic", "wave": "drums", "name": "Built-in drums", "gain_db": 0}
    p = int(program or 0)
    wave = "sine" if p < 8 or 88 <= p < 96 else "square" if 80 <= p < 88 else "triangle" if 32 <= p < 40 else "saw"
    return {"type": "basic", "wave": wave, "attack": 0.005, "decay": 0.25, "sustain": 0.6, "release": 0.15,
            "cutoff": 5000, "gain_db": 0, "name": f"Built-in {wave}"}


def describe(inst):
    t = (inst or {}).get("type")
    if t == "sf2":
        return f"{inst.get('name') or 'preset'} · {Path(inst.get('path') or '').stem}"
    if t == "vst3":
        return inst.get("name") or inst.get("plugin_name") or Path(inst.get("path") or "").stem
    if t == "sampler":
        return inst.get("name") or ("sample: " + (Path(inst["path"]).name if inst.get("path") else "song span"))
    if t == "chops":
        return inst.get("name") or f"chops: {inst.get('stem')} ({len(inst.get('chops') or [])})"
    return inst.get("name") or "Built-in synth"


# -------------------------------------------------------------------- render
def render(inst, notes, cc, n, sr=SR):
    inst = inst or default_for()
    t = inst.get("type")
    if not notes:
        return np.zeros((2, n), np.float32)
    if t == "sf2":
        out = _render_sf2(inst, notes, cc, n, sr)
    elif t == "vst3":
        out = _render_vst3(inst, notes, cc, n, sr)
    elif t == "sampler":
        out = _render_sampler(inst, notes, cc, n, sr)
    elif t == "chops":
        from . import chops
        out = chops.render(inst, notes, n, sr)
    else:
        out = _render_basic(inst, notes, cc, n, sr)
    g = 10 ** (float(inst.get("gain_db", 0) or 0) / 20)
    out = np.asarray(out, np.float32)
    if out.shape[0] == 1:
        out = np.repeat(out, 2, axis=0)
    if out.shape[1] < n:
        out = np.pad(out, ((0, 0), (0, n - out.shape[1])))
    return (out[:2, :n] * g).astype(np.float32)


def _events(notes, cc, chan):
    """[(t, mido.Message)] sorted; note-offs before note-ons at the same time (a repeated note retriggers)."""
    import mido
    ev = []
    for t_on, t_off, p, v in notes:
        p = int(min(127, max(0, p)))
        ev.append((float(t_on), 1, mido.Message("note_on", channel=chan, note=p, velocity=int(min(127, max(1, v))))))
        ev.append((float(max(t_off, t_on + 0.005)), 0, mido.Message("note_off", channel=chan, note=p, velocity=0)))
    for t, ctrl, val in cc or []:
        if ctrl == "pb":
            ev.append((float(t), 0, mido.Message("pitchwheel", channel=chan, pitch=int(min(8191, max(-8192, val))))))
        else:
            ev.append((float(t), 0, mido.Message("control_change", channel=chan, control=int(ctrl), value=int(min(127, max(0, val))))))
    ev.sort(key=lambda e: (e[0], e[1]))
    return ev


def _render_sf2(inst, notes, cc, n, sr):
    import mido
    import soundfile as sf
    exe = fluidsynth_exe()
    sf2 = inst.get("path")
    from . import components
    if not exe:
        raise components.Missing("fluidsynth")
    if not sf2 or not Path(sf2).exists():
        raise components.Missing("soundfont", f"the soundfont this part plays is gone: {sf2}")
    bank, prog = int(inst.get("bank", 0) or 0), int(inst.get("program", 0) or 0)
    chan = 9 if bank == DRUM_BANK else 0
    tpb, tempo = 9600, 500000                      # 120 BPM: 19,200 ticks a second (0.05 ms steps)
    tps = tpb * 1e6 / tempo
    mf = mido.MidiFile(ticks_per_beat=tpb)
    tr = mido.MidiTrack()
    mf.tracks.append(tr)
    tr.append(mido.MetaMessage("set_tempo", tempo=tempo, time=0))
    head = [] if chan == 9 else [mido.Message("control_change", channel=chan, control=0, value=min(127, bank // 128 if bank > 127 else bank), time=0),
                                 mido.Message("control_change", channel=chan, control=32, value=0, time=0)]
    head.append(mido.Message("program_change", channel=chan, program=prog, time=0))
    tr.extend(head)
    last = 0
    for t, _, m in _events(notes, cc, chan):
        tick = int(round(t * tps))
        tr.append(m.copy(time=max(0, tick - last)))
        last = max(last, tick)
    tr.append(mido.MetaMessage("end_of_track", time=int(round(max(0.0, n / sr - last / tps) * tps))))
    with tempfile.TemporaryDirectory(prefix="retex_sf2_") as td:
        mid, wav = os.path.join(td, "part.mid"), os.path.join(td, "part.wav")
        mf.save(mid)
        cmd = [exe, "-ni", "-q", "-R", "1" if inst.get("reverb") else "0", "-C", "0", "-g", "0.6", "-r", str(sr),
               "-T", "wav", "-O", "float", "-F", wav, sf2, mid]
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=900, creationflags=NO_WINDOW)
        if p.returncode != 0 or not os.path.exists(wav):
            raise RuntimeError("FluidSynth failed: " + ((p.stderr or p.stdout or "").strip().splitlines() or ["?"])[-1][:200])
        y, got_sr = sf.read(wav, dtype="float32", always_2d=True)
    y = y.T
    if got_sr != sr:
        import soxr
        y = soxr.resample(y.T, got_sr, sr).T
    return y


_plugins = {}


def load_plugin(inst):
    import pedalboard
    key = (inst["path"], inst.get("plugin_name"))
    pl = _plugins.get(key)
    if pl is None:
        kw = {"plugin_name": inst["plugin_name"]} if inst.get("plugin_name") else {}
        pl = pedalboard.load_plugin(inst["path"], **kw)
        _plugins[key] = pl
    if inst.get("state"):
        try:
            pl.raw_state = base64.b64decode(inst["state"])
        except Exception:
            pass
    return pl


def _render_vst3(inst, notes, cc, n, sr):
    pl = load_plugin(inst)
    if not getattr(pl, "is_instrument", False):
        raise RuntimeError(f"{describe(inst)} is an effect, not an instrument")
    # pedalboard: .time = seconds from the start. Half a second of lead-in, cut off afterwards: some plugins make a
    # click or a burst while they start up (FurSynth does), and that must not end up in the part
    pre = 0.5
    msgs = [m.copy(time=t + pre) for t, _, m in _events(notes, cc, 0)]
    out = pl(msgs, duration=n / sr + pre, sample_rate=sr, num_channels=2, buffer_size=256, reset=True)
    return np.asarray(out, np.float32)[:, int(round(pre * sr)):]


_sample_cache = {}


def _sample(inst, sr):
    span = inst.get("span")
    key = (inst.get("path"), json.dumps(span, sort_keys=True) if span else None)
    if key in _sample_cache:
        return _sample_cache[key]
    if inst.get("path"):
        from . import audio_io
        y = audio_io.load(inst["path"], sr)
    elif span:
        from . import library
        full = library.read_stem(span["song"], span["stem"])
        y = full[:, int(float(span["t0"]) * sr):int(float(span["t1"]) * sr)]
    else:
        raise RuntimeError("the sampler has no sample yet: pick an audio file or a span of a song")
    y = np.asarray(y, np.float32)
    if y.ndim == 1:
        y = np.stack([y, y])
    if len(_sample_cache) > 16:
        _sample_cache.clear()
    _sample_cache[key] = y
    return y


def _render_sampler(inst, notes, cc, n, sr):
    buf = _sample(inst, sr)
    L = buf.shape[1]
    out = np.zeros((2, n), np.float32)
    root = int(inst.get("root", 60) or 60)
    att = max(1, int(float(inst.get("attack", 0.003) if inst.get("attack") is not None else 0.003) * sr))
    rel = max(1, int(float(inst.get("release", 0.12) or 0.12) * sr))
    oneshot = inst.get("mode") == "oneshot"
    loop = bool(inst.get("loop"))
    for t_on, t_off, p, v in notes:
        s0 = int(round(t_on * sr))
        if s0 >= n:
            continue
        ratio = 1.0 if oneshot else 2.0 ** ((p - root) / 12.0)
        held = max(1, int(round((t_off - t_on) * sr)))
        length = int(L / ratio) if oneshot else held + rel
        if not loop:
            length = min(length, int(L / ratio))
        length = min(length, n - s0)
        if length <= 0:
            continue
        pos = np.arange(length) * ratio
        if loop:
            pos = np.mod(pos, L - 1)
        i = np.minimum(pos.astype(np.int64), L - 2)
        fr = (pos - i).astype(np.float32)
        seg = buf[:, i] * (1 - fr) + buf[:, i + 1] * fr
        env = np.ones(length, np.float32)
        a = min(att, length)
        env[:a] = np.arange(1, a + 1, dtype=np.float32) / a          # reaches full level on its last sample
        if not oneshot and held < length:
            r = length - held
            env[held:] *= np.linspace(1, 0, r, endpoint=True)[:length - held]
        out[:, s0:s0 + length] += seg * env * (v / 127.0)
    return out


def _render_basic(inst, notes, cc, n, sr):
    out = np.zeros((2, n), np.float32)
    wave = inst.get("wave", "saw")
    if wave == "drums":
        return _render_drums(notes, n, sr)
    A = max(1, int(float(inst.get("attack", 0.005)) * sr))
    D = max(1, int(float(inst.get("decay", 0.25)) * sr))
    S = float(inst.get("sustain", 0.6))
    R = max(1, int(float(inst.get("release", 0.15)) * sr))
    cutoff = float(inst.get("cutoff", 5000) or 5000)
    a_lp = float(np.exp(-2 * np.pi * cutoff / sr))
    for t_on, t_off, p, v in notes:
        s0 = int(round(t_on * sr))
        held = max(1, int(round((t_off - t_on) * sr)))
        L = min(held + R, n - s0)
        if L <= 0:
            continue
        f = 440.0 * 2 ** ((p - 69) / 12.0)
        ph = np.arange(L) * f / sr
        frac = ph - np.floor(ph)
        if wave == "sine":
            x = np.sin(2 * np.pi * ph)
        elif wave == "square":
            x = np.where(frac < 0.5, 1.0, -1.0)
        elif wave == "triangle":
            x = 4 * np.abs(frac - 0.5) - 1
        else:
            x = 2 * frac - 1
        env = np.full(L, S)
        k = np.arange(L)
        env[k < A] = k[k < A] / A
        dk = (k >= A) & (k < A + D)
        env[dk] = 1 - (1 - S) * (k[dk] - A) / D
        if held < L:
            env[held:] = env[min(held, L - 1)] * np.linspace(1, 0, L - held)
        y = (x * env * (v / 127.0) * 0.3).astype(np.float32)
        if wave != "sine":
            from scipy.signal import lfilter
            y = lfilter([1 - a_lp], [1, -a_lp], y).astype(np.float32)
        out[:, s0:s0 + L] += y
    return out


def _render_drums(notes, n, sr):
    out = np.zeros(n, np.float32)
    rng = np.random.default_rng(7)
    for t_on, t_off, p, v in notes:
        s0 = int(round(t_on * sr))
        if s0 >= n:
            continue
        g = v / 127.0
        if p in (35, 36):                                          # kick: a falling sine
            L = int(0.35 * sr)
            k = np.arange(L) / sr
            x = np.sin(2 * np.pi * (50 * k + 60 * (1 - np.exp(-k * 30)) / 30)) * np.exp(-k * 9)
        elif p in (38, 40, 37, 39):                                # snare / clap: noise + body
            L = int(0.25 * sr)
            k = np.arange(L) / sr
            x = (rng.standard_normal(L) * 0.7 + np.sin(2 * np.pi * 190 * k) * 0.5) * np.exp(-k * 18)
        elif p in (42, 44, 46, 49, 51, 52, 55, 57, 59):            # hats / cymbals: bright noise
            L = int((0.6 if p in (46, 49, 52, 55, 57) else 0.08) * sr)
            k = np.arange(L) / sr
            x = np.diff(rng.standard_normal(L + 1)) * 0.5 * np.exp(-k * (6 if L > 0.2 * sr else 45))
        else:                                                      # toms / other: a pitched thud
            L = int(0.3 * sr)
            k = np.arange(L) / sr
            x = np.sin(2 * np.pi * (80 + (p - 41) * 12) * k) * np.exp(-k * 12)
        L = min(L, n - s0)
        out[s0:s0 + L] += (x[:L] * g * 0.5).astype(np.float32)
    return np.stack([out, out])


# -------------------------------------------------------------------- SF2 presets
def sf2_presets(path):
    """[{bank, program, name}] from a soundfont's preset headers (RIFF pdta/phdr), without loading its samples."""
    out = []
    with open(path, "rb") as f:
        hdr = f.read(12)
        if hdr[:4] != b"RIFF" or hdr[8:12] not in (b"sfbk",):
            raise ValueError("not a SoundFont 2 file")
        end = 8 + struct.unpack("<I", hdr[4:8])[0]
        pos = 12
        while pos + 8 <= end:
            f.seek(pos)
            cid, size = struct.unpack("<4sI", f.read(8))
            if cid == b"LIST":
                kind = f.read(4)
                if kind == b"pdta":
                    sub, sub_end = pos + 12, pos + 8 + size
                    while sub + 8 <= sub_end:
                        f.seek(sub)
                        sid, ssize = struct.unpack("<4sI", f.read(8))
                        if sid == b"phdr":
                            data = f.read(ssize)
                            for i in range(0, ssize - 38, 38):           # the last record is the "EOP" terminator
                                name, prog, bank = struct.unpack("<20sHH", data[i:i + 24])
                                out.append({"bank": bank, "program": prog,
                                            "name": name.split(b"\0")[0].decode("latin-1").strip()})
                            break
                        sub += 8 + ssize + (ssize & 1)
                    break
            pos += 8 + size + (size & 1)
    return sorted(out, key=lambda p: (p["bank"], p["program"]))


# -------------------------------------------------------------------- VST3 plugins
VST3_DIRS = [r"C:\Program Files\Common Files\VST3", os.path.expandvars(r"%LOCALAPPDATA%\Programs\Common\VST3")]
SCAN_FILE = None


def vst3_dirs():
    return list(dict.fromkeys(VST3_DIRS + folders()["vst3"]))


def _scan_file():
    from .config import DATA
    return DATA / "vst3_plugins.json"


def plugin_binaries():
    """Every VST3 plugin under the plugin folders (subfolders too, e.g. a vendor's folder of several), each bundle
    resolved to the binary inside it (Pedalboard can't always open the bundle folder itself)."""
    found = []
    for root in vst3_dirs():
        if not os.path.isdir(root):
            continue
        for d, subs, files in os.walk(root):
            subs.sort()
            for sub in [x for x in subs if x.lower().endswith(".vst3")]:      # a bundle: its 64-bit Windows binary
                bins = sorted(glob.glob(os.path.join(d, sub, "Contents", "x86_64-win", "*.vst3")))
                found += [b for b in bins if os.path.isfile(b)][:1]
            subs[:] = [x for x in subs if not x.lower().endswith(".vst3")]     # never look inside a bundle
            found += [os.path.join(d, f) for f in sorted(files) if f.lower().endswith(".vst3")]
    return list(dict.fromkeys(found))


_PROBE = r"""
import json, sys, os
sys.stdout = open(os.devnull, "w")
import pedalboard
res = []
path = sys.argv[1]
try:
    names = pedalboard.VST3Plugin.get_plugin_names_for_file(path)
except Exception:
    names = [None]
for nm in names[:8] or [None]:
    try:
        kw = {"plugin_name": nm} if nm and len(names) > 1 else {}
        pl = pedalboard.load_plugin(path, **kw)
        res.append({"name": nm or pl.name, "plugin_name": nm if len(names) > 1 else None, "instrument": bool(pl.is_instrument)})
    except Exception as e:
        res.append({"name": nm or os.path.basename(path), "error": str(e).splitlines()[-1][:160]})
sys.stdout = sys.__stdout__
print("@@RETEX@@" + json.dumps(res), flush=True)
"""


def scan_vst3(progress=lambda f, m="": None):
    """Load every plugin once, each in its own short-lived process (a crashing plugin can't take Re-Tex down), and
    remember which are instruments. -> [{path, name, plugin_name, instrument | error}] (also saved for next time)."""
    bins = plugin_binaries()
    out = []
    for i, b in enumerate(bins):
        progress(i / max(1, len(bins)), f"checking {Path(b).stem}")
        try:
            p = subprocess.run([sys.executable, "-I", "-c", _PROBE, b], capture_output=True, text=True, timeout=45,
                               creationflags=NO_WINDOW)
            # plugins print their own chatter (Csound, licence notices...): our answer is the marked line
            mark = [ln for ln in (p.stdout or "").splitlines() if ln.startswith("@@RETEX@@")]
            if mark:
                rows = json.loads(mark[-1][len("@@RETEX@@"):])
            else:
                why = "crashed while loading (it may only work inside a DAW)" if p.returncode not in (0, None) else "no answer"
                rows = [{"name": Path(b).stem, "error": why}]
        except subprocess.TimeoutExpired:
            rows = [{"name": Path(b).stem, "error": "took too long to load"}]
        except Exception as e:
            rows = [{"name": Path(b).stem, "error": str(e)[:160]}]
        for r in rows:
            out.append({"path": b, **r})
    _scan_file().write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


def known_vst3():
    try:
        return json.loads(_scan_file().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def open_editor(inst):
    """Open the plugin's own window (blocks until it's closed) and return its new state (base64). Server process."""
    pl = load_plugin(inst)
    pl.show_editor()
    return base64.b64encode(bytes(pl.raw_state)).decode("ascii")
