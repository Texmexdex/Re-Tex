"""Map a few library songs without audio and print the sections, bars and how well each grid sits (scratch).
python tests/real.py fear+inoculum+audio golem   ('+' between the words of one title)"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tempomap import build, check, library  # noqa: E402


def main():
    root = library.data_dir()
    for arg in sys.argv[1:]:
        found = library.find(root, arg.split("+"))
        if not found:
            print("not found:", arg)
            continue
        sid, title = found[0]
        info, stems, sr = library.load(root, sid)
        det, dur = library.detector_beats(info)
        m, feat = build.tempo_map(info, stems, sr, det, dur)
        print(f"\n== {title}  ({dur / 60:.1f} min)", flush=True)
        for s in m["sections"]:
            print(f"   {s['start']:7.1f}-{s['end']:7.1f}  {s['bpm']:9.3f} BPM  {s['how']:5s}  det {s['detector_bpm']:7.2f}  clarity {s['clarity']:5.1f}  rms {s['rms_ms']}  share {s['share']}")
        print("   bars:", " | ".join(f"{x['start']:.0f}-{x['end']:.0f} {x['label']} ({x['confidence']:.2f})" for x in m["meters"]))
        um = (info.get("user_marks") or {}).get("t")
        for name, b in (("map", m["pulse_beats"]), ("detector", det)) + ((("marks", um),) if um else ()):
            print(f"   fit {name:9s}", check.summary(check.lock(feat, b)), flush=True)


if __name__ == "__main__":
    main()
