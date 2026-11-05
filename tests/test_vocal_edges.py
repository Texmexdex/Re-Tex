"""Vocal clip edges follow the vocal's own 10 ms voicing: a pickup opens in the breath before the line, not part-way
into the line before it (the per-beat presence check counted a 340 ms breath as 'singing' and reached a whole bar
back - Easy over Hard Refresh, a word chopped at 0:42), and a held word rings until it ends, then stops before the
next sound. CPU-only: a stub song, no audio.
"""
import numpy as np

from mashlab import automash
from mashlab import project as P

FPS = 100
IBI = 0.5                                       # 120 BPM, beat j at j * 0.5 s


class _Song:
    id = "sEDGES"
    nb = 64
    beats = np.arange(64) * IBI

    def __init__(self, runs):
        self._v = np.zeros(int(self.nb * IBI * FPS), bool)
        for a, b in runs:
            self._v[int(a * FPS):int(b * FPS)] = True

    def voicing(self):
        return self._v


def _edges(runs, clips):
    L = _Song(runs)
    return L, automash._vocal_edges(L, clips, 1.0, 4.0)


def test_pickup_opens_in_the_breath_not_in_the_previous_line():
    # previous line ends 15.7 s, 300 ms breath, this line sings from 16.0 s into the phrase start (beat 33 = 16.5 s)
    L, (c,) = _edges([(10.0, 15.7), (16.0, 20.3)], [P.clip("t", "sEDGES", "vocals", 40.0, 33.0, 8.0)])
    t0 = c["src"] * IBI
    assert 15.7 <= t0 < 16.0                    # in the breath (the old per-beat pickup started at beat 29 = 14.5 s)
    assert not L.voicing()[int(round(t0 * FPS))]
    assert c["start"] == 40.0 - (33.0 - c["src"])
    assert 0.0 < c["fade_in"] * IBI <= 0.1      # a short lead-in that is done before the first sound


def test_held_word_rings_out_and_stops_before_the_next_sound():
    # the word at the phrase end (beat 41 = 20.5 s) is held to 20.8 s; the next sound is at 21.3 s
    L, (c,) = _edges([(16.0, 20.8), (21.3, 25.0)], [P.clip("t", "sEDGES", "vocals", 40.0, 32.0, 9.0)])
    end = (c["src"] + c["len"]) * IBI
    assert 20.8 < end <= 21.3
    assert c["fade_out"] * IBI >= 0.2           # the fade starts about when the voice stops, not over the word


def _back_to_back(line_start):
    a = P.clip("t", "sEDGES", "vocals", 0.0, 8.0, 12.0)         # src 4.0-10.0 s, ends in the breath
    b = P.clip("t", "sEDGES", "vocals", 12.0, 21.0, 8.0)        # phrase start 10.5 s, right where a ends
    return _edges([(4.0, 9.8), (line_start, 14.0)], [a, b])


def test_abutting_clips_crossfade_instead_of_chopping():
    L, (a, b) = _back_to_back(10.35)                             # b's line starts 0.15 s before its phrase start
    assert b["src"] * IBI <= 10.35                               # it keeps its first sound ...
    overlap = a["start"] + a["len"] - b["start"]
    assert 0 < overlap <= automash.EDGE_OVERLAP + 1e-9           # ... by overlapping a, at most half a beat
    assert a["fade_out"] >= overlap - 1e-9                       # a is faded out by the end of the overlap


def test_line_start_out_of_reach_eases_in():
    L, (a, b) = _back_to_back(10.1)                              # 0.8 beats back: more than the overlap allows
    assert a["start"] + a["len"] - b["start"] <= automash.EDGE_OVERLAP + 1e-9
    missing = b["src"] - 10.1 / IBI                              # beats of the line that couldn't be reached
    assert missing > 0 and b["fade_in"] >= missing - 1e-9        # eases in over them rather than starting chopped
