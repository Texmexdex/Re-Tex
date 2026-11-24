"""Test songs with a KNOWN tempo map: drums, bass and chords played from a map, written like a Re-Tex library song
(stems + song.json with a detector-style beat list that has the detector's usual mistakes), plus truth.json.

A map is a list of sections: dict(bpm=, sub=2|3, bars=[bar lengths in subdivisions], groups={length: [groups]},
drums=True). bpm counts beats; a beat is `sub` subdivisions (sub 2: eighths, sub 3: triplet eighths). A 7/8 bar is
7 subdivisions at sub 2 - three and a half beats, so the beat goes on straight through it the way it does on a DAW
grid while the bar lines move.
"""
import json
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 22050
GROUPS = {4: [2, 2], 5: [2, 3], 6: [3, 3], 7: [2, 2, 3], 8: [2, 2, 2, 2], 9: [3, 3, 3], 10: [2, 2, 2, 2, 2],
          11: [3, 3, 3, 2], 12: [3, 3, 3, 3], 13: [3, 3, 3, 2, 2], 14: [2, 2, 2, 2, 2, 2, 2], 15: [3, 3, 3, 3, 3],
          16: [2] * 8}
ROOTS = [55.0, 65.41, 49.0, 73.42]          # A1 C2 G1 D2, one per bar in turn


def _env(n, decay):
    return np.exp(-np.arange(n) / (decay * SR))


def _kick(rng):
    n = int(0.25 * SR)
    f = 50 + 70 * _env(n, 0.03)
    return 0.9 * np.sin(2 * np.pi * np.cumsum(f) / SR) * _env(n, 0.09)


def _snare(rng):
    n = int(0.2 * SR)
    noise = rng.standard_normal(n)
    from scipy.signal import butter, sosfilt
    noise = sosfilt(butter(2, [300, 4000], "band", fs=SR, output="sos"), noise)
    return 0.6 * (noise * _env(n, 0.05) + 0.4 * np.sin(2 * np.pi * 190 * np.arange(n) / SR) * _env(n, 0.04))


def _hat(rng, loud=0.15, decay=0.02):
    n = int(0.08 * SR) if decay < 0.1 else int(1.2 * SR)
    from scipy.signal import butter, sosfilt
    noise = sosfilt(butter(2, 6000, "high", fs=SR, output="sos"), rng.standard_normal(n))
    return loud * noise * _env(n, decay)


def _add(buf, x, t):
    i = int(round(t * SR))
    if i < 0 or i >= len(buf):
        return
    m = min(len(x), len(buf) - i)
    buf[i:i + m] += x[:m]


def _tone(freq, n, harm=3, amp=0.2):
    k = np.arange(n) / SR
    y = sum(np.sin(2 * np.pi * freq * h * k) / h for h in range(1, harm + 1))
    a = np.minimum(1.0, np.arange(n) / (0.01 * SR)) * np.minimum(1.0, (n - np.arange(n)) / (0.02 * SR))
    return amp * y * a


def render(spec, seed=0):
    """-> (stems {name: mono}, truth dict). The song starts after 0.5 s of silence."""
    rng = np.random.default_rng(seed)
    t = 0.5
    tatums, subs, bar_starts, labels, sections = [], [], [], [], []
    for sec in spec:
        dt = 60.0 / sec["bpm"] / sec["sub"]
        s0 = t
        for L in sec["bars"]:
            bar_starts.append(len(tatums))
            labels.append((L, sec["sub"]))
            for k in range(L):
                tatums.append(t)
                subs.append(sec["sub"])
                t += dt
        sections.append(dict(start=s0, end=t, bpm=sec["bpm"], sub=sec["sub"], drums=sec.get("drums", True)))
    dur = t + 2.0
    n = int(dur * SR)
    drums, bass, other = np.zeros(n), np.zeros(n), np.zeros(n)
    kick, snare = _kick(rng), _snare(rng)
    tat = np.array(tatums + [t])
    bi = 0
    for si, sec in enumerate(spec):
        groups = sec.get("groups", {})
        for bj, L in enumerate(sec["bars"]):
            k0 = bar_starts[bi]
            g = groups.get(L) or GROUPS[L]
            starts = np.cumsum([0] + g[:-1])
            root = ROOTS[bi % 4]
            bar_end = tat[k0 + L]
            if sec.get("drums", True):
                for gi, st in enumerate(starts):
                    hit = kick if gi % 2 == 0 else snare
                    _add(drums, hit, tat[k0 + st])
                for k in range(L):
                    _add(drums, _hat(rng, 0.2 if k in starts else 0.1), tat[k0 + k])
                if bj % 4 == 0:
                    _add(drums, _hat(rng, 0.35, 0.6), tat[k0])           # a crash every 4 bars
            for gi, st in enumerate(starts):
                a, b = tat[k0 + st], tat[k0 + (starts[gi + 1] if gi + 1 < len(starts) else L)]
                f = root * (1.0 if gi % 2 == 0 else 1.5)
                _add(bass, _tone(f, int((b - a) * SR) - 50, 4, 0.3), a)
            m = int((bar_end - tat[k0]) * SR) - 50
            ch = sum(_tone(root * 4 * r, m, 3, 0.07) for r in (1.0, 1.26, 1.5))
            _add(other, ch, tat[k0])
            bi += 1
    stems = {"drums": drums.astype(np.float32), "bass": bass.astype(np.float32), "other": other.astype(np.float32)}
    beats = []                                  # the beat: every `sub` subdivisions, straight on through odd bars
    k = 0
    while k < len(tatums):
        beats.append(tatums[k])
        k += subs[k]
    truth = dict(tatums=[round(x, 5) for x in tatums], subs=subs, bars=bar_starts, labels=labels, sections=sections,
                 beats=[round(x, 5) for x in beats], duration=dur)
    return stems, truth


def detector(truth, spec, seed=1):
    """A beat list with the detector's usual mistakes: right where it's plain 4/4 (with a few ms of jitter), every
    two triplets in a triplet-feel section (1.5x the tempo), and guessing in a section with no drums."""
    rng = np.random.default_rng(seed)
    out = []
    tat, subs = np.array(truth["tatums"]), truth["subs"]
    for sec in truth["sections"]:
        a, b = sec["start"], sec["end"]
        inside = tat[(tat >= a - 1e-6) & (tat < b - 1e-6)]
        if not sec["drums"]:
            p = 60.0 / (sec["bpm"] * 0.8)
            out += list(np.arange(a, b, p) + rng.normal(0, 0.02, len(np.arange(a, b, p))))
        else:
            out += list(inside[::2] + rng.normal(0, 0.006, len(inside[::2])))
    out = np.sort(np.asarray(out))
    return out


def write_song(root, sid, title, spec, seed=0):
    """A library-style song folder root/songs/<sid>/ (stems/*.flac, song.json) and root/truth/<sid>.json."""
    stems, truth = render(spec, seed)
    det = detector(truth, spec, seed + 1)
    d = Path(root) / "songs" / sid
    (d / "stems").mkdir(parents=True, exist_ok=True)
    for k, y in stems.items():
        sf.write(d / "stems" / f"{k}.flac", y, SR)
    bpm = 60.0 / float(np.median(np.diff(det)))
    info = dict(id=sid, title=title, beats=[round(float(x), 5) for x in det], bars=list(range(0, len(det), 4)), bpb=4,
                duration=truth["duration"], bpm=round(bpm, 3), status="ready")
    (d / "song.json").write_text(json.dumps(info), encoding="utf-8")
    tdir = Path(root) / "truth"
    tdir.mkdir(parents=True, exist_ok=True)
    (tdir / f"{sid}.json").write_text(json.dumps(truth), encoding="utf-8")
    return d


SPECS = {
    # Tool-like: 4/4, then the same eighths grouped 7, 10 (5/4), 5+7, back to 4/4, then a real change of tempo
    "synth_odd_meters": ("Test: 4/4 > 7/8 > 5/4 > 5+7 > 4/4 > faster 4/4", [
        dict(bpm=130, sub=2, bars=[8] * 16),
        dict(bpm=130, sub=2, bars=[7] * 12),
        dict(bpm=130, sub=2, bars=[10] * 8),
        dict(bpm=130, sub=2, bars=[5, 7] * 6),
        dict(bpm=130, sub=2, bars=[8] * 12),
        dict(bpm=152, sub=2, bars=[8] * 16),
    ]),
    # Fear Inoculum-like: a quiet intro with no drums, then one tempo throughout while the feel goes triplet and back
    "synth_triplet_feel": ("Test: intro > 4/4 > triplet feel, same tempo > 4/4", [
        dict(bpm=90, sub=2, bars=[8] * 6, drums=False),
        dict(bpm=90, sub=2, bars=[8] * 20),
        dict(bpm=90, sub=3, bars=[12] * 14),
        dict(bpm=90, sub=2, bars=[8] * 12),
    ]),
    # 11/8, a real change of tempo into 4/4, then 6/8
    "synth_eleven": ("Test: 11/8 > slower 4/4 > 6/8", [
        dict(bpm=150, sub=2, bars=[11] * 18),
        dict(bpm=118, sub=2, bars=[8] * 16),
        dict(bpm=118, sub=3, bars=[6] * 16),
    ]),
    # a take with no click: 4/4 creeping from 88 to 94 BPM a little every bar (the map should follow it)
    "synth_drift": ("Test: no click, 88 > 94 BPM creeping", [
        dict(bpm=88 + 0.125 * i, sub=2, bars=[8]) for i in range(48)
    ]),
}


if __name__ == "__main__":
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else "synth_library"
    for sid, (title, spec) in SPECS.items():
        print(write_song(root, sid, title, spec))
