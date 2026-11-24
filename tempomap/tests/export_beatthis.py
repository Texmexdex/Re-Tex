"""Run Re-Tex's Beat This! (read-only import of mashlab.beats) on the key songs and save its beats, downbeats and
per-frame logits to key/bt/<id>.npz. MIDI songs: the sum of Re-Tex's rendered parts; real songs: the sum of stems.
python tests/export_beatthis.py midi|real [ids...]"""
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

HERE = Path(__file__).resolve().parent.parent
RETEX_ROOT = HERE.parent
sys.path.insert(0, str(RETEX_ROOT))
from mashlab import beats as bt  # noqa: E402
from tempomap.library import data_dir  # noqa: E402

DATA = data_dir()


def mix_of(files):
    y, sr = None, None
    for f in files:
        x, sr = sf.read(str(f), dtype="float32", always_2d=True)
        x = x.mean(axis=1)
        if y is None:
            y = x
        else:
            m = max(len(y), len(x))
            z = np.zeros(m, np.float32)
            z[:len(y)] += y
            z[:len(x)] += x
            y = z
    return y, sr


def main(which, ids):
    od = HERE / "key" / "bt"
    od.mkdir(parents=True, exist_ok=True)
    if not ids:
        ids = [p.stem for p in (HERE / "key" / which).glob("*.npz")]
    for sid in ids:
        if (od / f"{sid}.npz").exists():
            continue
        d = DATA / ("midi" if which == "midi" else "songs") / sid
        files = sorted((d / "stems").glob("*.flac"))
        if which == "real":
            files = [f for f in files if f.stem in ("drums", "bass", "other", "vocals")]
        if not files:
            print("no stems", sid, flush=True)
            continue
        y, sr = mix_of(files)
        b, dn, dl, bl = bt.detect_full(y, sr)
        np.savez_compressed(od / f"{sid}.npz", beats=b, downbeats=dn, down_logits=np.asarray(dl, np.float16).ravel(),
                            beat_logits=np.asarray(bl, np.float16).ravel(), fps=np.array(50.0))
        print(sid, len(b), "beats", len(dn), "downbeats", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
