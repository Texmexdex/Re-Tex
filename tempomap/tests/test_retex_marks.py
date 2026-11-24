"""retex_marks: the map as Re-Tex's song-grid marks."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tempomap.build import retex_marks  # noqa: E402


def test_marks_from_a_simple_map():
    beats = [0.5 * k for k in range(17)]                 # 120 BPM
    m = dict(pulse_beats=beats, bar_times=[0.0, 2.0, 4.0, 5.75, 8.0],   # 4/4, 4/4, then a bar line off the beat
             meters=[dict(start=0.0, end=4.0, confidence=0.9), dict(start=4.0, end=8.0, confidence=0.2)],
             sections=[dict(start=0.0, end=8.0, how="main")])
    r = retex_marks(m)
    assert r["t"] == beats
    assert [i for i, x in enumerate(r["d"]) if x] == [0, 4, 8, 16]     # 5.75 falls between beats: no mark
    assert r["s"][:8] == [1] * 8 and r["s"][8:16] == [0] * 8            # the unsure run is amber
    assert r["bpb"] == 4
    assert len(r["t"]) == len(r["d"]) == len(r["s"])


def test_no_pulse_is_unsure():
    m = dict(pulse_beats=[0.0, 1.0, 2.0, 3.0], bar_times=[0.0], meters=[],
             sections=[dict(start=0.0, end=4.0, how="none")])
    r = retex_marks(m)
    assert r["s"] == [0, 0, 0, 0]
