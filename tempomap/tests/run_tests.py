"""Make the test songs (known tempo maps), map them, and check the maps against the truth.
python tests/run_tests.py   - prints a table; exits 1 if any check fails. Writes only to a temp folder."""
import json
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

import synth  # noqa: E402
from score import score  # noqa: E402
from tempomap import build, library  # noqa: E402

LIMITS = dict(tatums_found=0.95, tatums_extra=0.06, bars_found=0.85, bars_extra=0.1, bar_length_right=0.8)


def main():
    bad = 0
    with tempfile.TemporaryDirectory() as root:
        for sid, (title, spec) in synth.SPECS.items():
            synth.write_song(root, sid, title, spec)
            info, stems, sr = library.load(root, sid)
            det, dur = library.detector_beats(info)
            m, _ = build.tempo_map(info, stems, sr, det, dur)
            truth = json.loads((Path(root) / "truth" / f"{sid}.json").read_text())
            sc = score(m, truth)
            # every true tempo is in the map, within 0.05 BPM
            tempos = [s["bpm"] for s in m["sections"] if s["how"] != "none"]
            tempo_ok = sid.startswith("synth_drift") or all(min(abs(t - s["bpm"]) for t in tempos) < 0.05 for s in truth["sections"] if s["drums"])
            fails = [k for k, v in LIMITS.items() if (sc[k] < v if not k.endswith("extra") else sc[k] > v)]
            if not tempo_ok:
                fails.append("tempo")
            bad += bool(fails)
            print("   sections:", ", ".join(f"{x['bpm']:.2f} {x['how']}" for x in m["sections"]))
            print(f"{title}\n   " + "  ".join(f"{k} {v:.2f}" for k, v in sc.items())
                  + f"  tempos {'ok' if tempo_ok else 'WRONG'}" + (f"   FAILED: {', '.join(fails)}" if fails else "   ok"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
