"""Scratch: which of the library's source .mid files have meter changes (read-only)."""
import sys
from pathlib import Path

import mido


def main():
    up = Path(sys.argv[1])
    rows = []
    for f in sorted(up.glob("*.mid*")):
        try:
            m = mido.MidiFile(f, clip=True)
        except Exception as e:  # noqa: BLE001
            print("bad", f.name, e)
            continue
        ts, tempos, drums, t = [], [], 0, 0
        for msg in mido.merge_tracks(m.tracks):
            t += msg.time
            if msg.type == "time_signature":
                if not ts or ts[-1][1:] != (msg.numerator, msg.denominator):
                    ts.append((t, msg.numerator, msg.denominator))
            elif msg.type == "set_tempo":
                tempos.append(round(mido.tempo2bpm(msg.tempo), 2))
            elif msg.type == "note_on" and getattr(msg, "channel", 0) == 9 and msg.velocity:
                drums += 1
        meters = sorted({f"{n}/{d}" for _, n, d in ts})
        rows.append((len(ts), f.name[14:], meters, len(set(tempos)), drums, round(m.length / 60, 1)))
    rows.sort(key=lambda r: -r[0])
    for r in rows:
        print(f"{r[0]:3d} changes  {r[1][:48]:48s} meters {','.join(r[2])[:60]:60s} tempos {r[3]:3d}  drumhits {r[4]:6d}  {r[5]} min")


if __name__ == "__main__":
    main()
