"""Build the MIDI answer key and export compact features (read-only on the library).

python tests/export_key.py midi   -> key/midi/<id>.npz + key/midi/<id>.json for the library's MIDI songs whose
                                     source .mid is in data/uploads (meter-changing ones, plus some plain 4/4)
python tests/export_key.py real words+of+title ...  -> key/real/<id>.npz (features only; no truth)

Truth comes from the .mid itself (its time signatures and tempo map); the audio is Re-Tex's own render of that file,
so the bar lines are exact. The json also records the measured lag between the file's drum notes and the render.
"""
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tempomap import features, library  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "key"


def save_feat(path, feat, extra=None):
    arr = {f"part_{k}": v.astype(np.float16) for k, v in feat.parts.items()}
    arr.update(chroma=feat.chroma.astype(np.float16), meta=np.array(json.dumps(dict(
        sr=feat.sr, fps=feat.fps, n=feat.n, lag=feat.lag, cfps=feat.cfps, **(extra or {})))))
    np.savez_compressed(path, **arr)


def midi_truth(path):
    import mido
    mf = mido.MidiFile(str(path), clip=True)
    ppq = mf.ticks_per_beat
    tempos, ts, drums, end = [(0, 500000)], [], [], 0
    for tr in mf.tracks:
        t = 0
        for msg in tr:
            t += msg.time
            if msg.type == "set_tempo":
                tempos.append((t, msg.tempo))
            elif msg.type == "time_signature":
                ts.append((t, msg.numerator, msg.denominator))
            elif msg.type == "note_on" and msg.velocity:
                end = max(end, t)
                if msg.channel == 9:
                    drums.append(t)
    tempos = sorted(dict(sorted(tempos)).items())
    if not ts or ts[0][0] > 0:
        ts.insert(0, (0, 4, 4))
    ts = sorted(ts)

    def sec(T):
        s, T = 0.0, float(T)
        for j, (mt, tp) in enumerate(tempos):
            if mt >= T:
                break
            nxt = tempos[j + 1][0] if j + 1 < len(tempos) else T
            s += (min(T, nxt) - mt) / ppq * tp / 1e6
        return s

    bars, tick, k = [], 0, 0
    while tick <= end:
        while k + 1 < len(ts) and ts[k + 1][0] <= tick:
            k += 1
        num, den = ts[k][1], ts[k][2]
        L = int(round(num * ppq * 4 / den))
        bars.append(dict(t=round(sec(tick), 5), num=num, den=den, eighths=round(num * 8 / den, 3)))
        tick += max(L, 1)
    beats = [round(sec(q * ppq), 5) for q in range(int(end // ppq) + 2)]
    return dict(bars=bars, beats=beats, drums=[round(sec(t), 5) for t in sorted(set(drums))],
                tempos=[round(60e6 / tp, 3) for _, tp in tempos])


def midi_stems(d, info):
    mj = json.loads((d / "midi.json").read_text(encoding="utf-8"))
    parts = {p["id"]: p for p in mj.get("parts", [])}
    drum_ids = set(info.get("drum_stems") or [pid for pid, p in parts.items() if p.get("drum")])
    out, sr = {}, None
    for pid in info.get("stems") or parts:
        f = d / "stems" / f"{pid}.flac"
        if not f.exists():
            continue
        y, sr = sf.read(str(f), dtype="float32", always_2d=True)
        y = y.mean(axis=1)
        p = parts.get(pid, {})
        inst = p.get("instrument") or {}
        prog = int(p.get("program") or inst.get("program") or 0)
        name = "drums" if pid in drum_ids else ("bass" if 32 <= prog <= 39 else "other")
        if name in out:
            m = max(len(out[name]), len(y))
            a = np.zeros(m, np.float32)
            a[:len(out[name])] += out[name]
            a[:len(y)] += y
            out[name] = a
        else:
            out[name] = y
    return out, sr


def lag_ms(feat, drum_times):
    """Where the render's drum hits sit against the file's drum notes (ms, + = render later)."""
    if len(drum_times) < 20:
        return None
    best, bl = -1, None
    h = feat.hits
    for lag in range(-40, 41):                       # 5 ms steps
        idx = np.round((np.asarray(drum_times) + lag * 0.005) * feat.fps).astype(int)
        idx = idx[(idx >= 0) & (idx < len(h))]
        s = float(h[idx].sum())
        if s > best:
            best, bl = s, lag * 5
    return bl


def do_midi(root):
    up = Path(root) / "uploads"
    od = OUT / "midi"
    od.mkdir(parents=True, exist_ok=True)
    rows = []
    for d in sorted((Path(root) / "midi").iterdir()):
        f = d / "song.json"
        if not f.exists():
            continue
        info = json.loads(f.read_text(encoding="utf-8"))
        if info.get("parent") or not info.get("source"):
            continue
        src = up / info["source"]
        if not src.exists():
            print("no source", d.name, flush=True)
            continue
        tr = midi_truth(src)
        meters = sorted({f"{b['num']}/{b['den']}" for b in tr["bars"]})
        changes = sum(1 for a, b in zip(tr["bars"], tr["bars"][1:]) if (a["num"], a["den"]) != (b["num"], b["den"]))
        rows.append((d, info, tr, meters, changes, src))
    odd = [r for r in rows if r[4] >= 3]
    plain = [r for r in rows if r[4] == 0 and r[3] == ["4/4"] and len(r[2]["drums"]) > 300][:24]
    for d, info, tr, meters, changes, src in odd + plain:
        if (od / f"{d.name}.npz").exists():
            continue
        try:
            stems, sr = midi_stems(d, info)
            if "drums" not in stems:
                print("no drums", d.name, flush=True)
                continue
            feat = features.Features(stems, sr)
        except Exception as e:  # noqa: BLE001
            print("failed", d.name, e, flush=True)
            continue
        lag = lag_ms(feat, tr["drums"])
        save_feat(od / f"{d.name}.npz", feat)
        tr.update(id=d.name, title=info.get("title"), source=src.name, meters=meters, changes=changes, lag_ms=lag,
                  duration=float(len(next(iter(stems.values()))) / sr), set="odd" if changes >= 3 else "plain")
        tr.pop("drums")
        (od / f"{d.name}.json").write_text(json.dumps(tr), encoding="utf-8")
        print(f"{d.name}: {changes} changes, lag {lag} ms", flush=True)


def do_real(root, args):
    od = OUT / "real"
    od.mkdir(parents=True, exist_ok=True)
    for a in args:
        found = library.find(root, a.split("+"))
        if not found:
            print("not found", a, flush=True)
            continue
        sid, title = found[0]
        info, stems, sr = library.load(root, sid)
        det, dur = library.detector_beats(info)
        feat = features.Features(stems, sr)
        save_feat(od / f"{sid}.npz", feat, dict(title=title, det=[float(x) for x in det], dur=dur,
                                                bpm=float(info.get("bpm") or 0),
                                                bars=[float(info["beats"][j]) for j in info.get("bars") or []
                                                      if 0 <= j < len(info["beats"])]))
        print("real", title, flush=True)


if __name__ == "__main__":
    r = library.data_dir()
    if sys.argv[1] == "midi":
        do_midi(r)
    else:
        do_real(r, sys.argv[2:])
