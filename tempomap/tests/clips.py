"""Map songs (report + full click tracks) and cut short listening clips from the click tracks (scratch).
python tests/clips.py fear+inoculum+audio@60,330,450 golem@40 ...   ('+' between title words, '@' clip starts in s)
Clips go to maps/clips/<n>_<song>_<start>_<grid>.mp3, 30 s each."""
import sys
from pathlib import Path

import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tempomap import cli, library, render  # noqa: E402

LEN = 30.0


def main():
    root = library.data_dir()
    here = Path(__file__).resolve().parent.parent
    out = here / "maps"
    cdir = out / "clips"
    cdir.mkdir(parents=True, exist_ok=True)
    for n, arg in enumerate(sys.argv[1:], 1):
        words, _, at = arg.partition("@")
        found = library.find(root, words.split("+"))
        if not found:
            print("not found:", words, flush=True)
            continue
        sid, title = found[0]
        print("==", title, flush=True)
        cli.process(root, sid, out, audio=True, log=lambda s: print(s, flush=True))
        short = "".join(c for c in words.replace("+", "-") if c.isalnum() or c == "-")[:24]
        for key in ("map", "steady", "grid"):
            f = out / sid / f"mix_{key}.mp3"
            if not f.exists():
                continue
            y, sr = sf.read(f, dtype="float32")
            for t in [float(x) for x in at.split(",") if x]:
                name = cdir / f"{n}_{short}_{int(t // 60)}m{int(t % 60):02d}_{key}.mp3"
                render.write_mp3(name, y, sr, t, t + LEN)
                print("  ", name.name, flush=True)
    cli.write_index(out)


if __name__ == "__main__":
    main()
