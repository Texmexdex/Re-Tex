"""Reimagine plumbing that needs no models: reading a YuE2/SheetSage2 score's sections, filing the mix's sung lines
under the section they're sung in (all the words used to land in the first section), a half-time take's grid at
the mix's beat level, and the aligned grid's shape. CPU-only.
"""
import numpy as np

from mashlab import reimagine, takealign

ABC = """X:1
T:
M:4/4
L:1/16
Q:1/4=120
V: Vocal clef=treble name="Vocal Melody" snm="Vocal"
V: Ins clef=treble name="Ins Melody" snm="Inst."
K:Am
% intro
V: Vocal
"Am"z16|"F"z16|
V: Ins
A4A4A4A4|F4F4F4F4|
% verse
V: Vocal
"Am"c4c4c4c4|Z3|
V: Ins
Z4|
% chorus
V: Vocal
"C"e16|"G"d16|
V: Ins
Z2|
"""


def test_score_sections_count_bars_and_time_them():
    s = reimagine.score_sections(ABC)
    assert [x["label"] for x in s] == ["intro", "verse", "chorus"]
    assert [x["bars"] for x in s] == [2, 4, 2]                      # "Z3" is three whole bars
    assert s[1]["t0"] == 4.0 and s[1]["t1"] == 12.0                 # 2 bars of 4/4 at 120 = 4 s
    assert s[2]["t1"] == 16.0


def test_lines_go_under_the_section_they_are_sung_in():
    secs = reimagine.score_sections(ABC)
    lines = [dict(t0=5.0, t1=7.0, text="first verse line"), dict(t0=8.0, t1=11.5, text="second verse line"),
             dict(t0=12.5, t1=15.0, text="the hook")]
    lyr = reimagine.tag_lyrics(secs, lines)
    assert lyr.index("[Intro]") < lyr.index("[Verse]") < lyr.index("first verse line") < lyr.index("[Chorus]") < lyr.index("the hook")
    assert lyr.split("[Verse]")[0].strip() == "[Intro]"              # nothing sung there: an instrumental intro


def test_half_time_take_is_brought_to_the_mix_level():
    period = 60 / 140
    beats = np.arange(0, 40) * 2 * period                          # the take tracked at 70 BPM
    tb, bars = takealign._to_mix_level(dict(beats=beats, bars=list(range(0, 40, 4)), bpb=4), period, beats[-1] + 1)
    assert np.allclose(np.diff(tb), period)                        # beats at the mix's level
    assert bars[:4] == [0, 4, 8, 12]                               # each of its bars is two of the mix's


def test_aligned_grid_starts_on_mix_beat_zero_and_is_in_bar_phase():
    period = 60 / 140
    tau = 1.3 + np.arange(64) * period                             # mix beat k is at 1.3 s + k beats in the take
    g = takealign.grid(tau, 1.3 + 70 * period, 140)
    b = np.asarray(g["beats"])
    assert abs(b[g["first_beat"]] - 1.3) < 1e-6 and g["last_beat"] - g["first_beat"] == 63
    assert g["first_beat"] in g["bars"]                            # mix beat 0 is a downbeat of the take's grid
    assert b[0] >= 0 and np.all(np.diff(b) > 0)
