"""Chopped & screwed: the screw drops every stem's pitch with the tempo, and a screwed song at its own steady tempo is
rendered as a plain resample (a slowed turntable), not a time-stretch."""
import math

import numpy as np

from mashlab import project as P, warp
from mashlab.config import SR


def test_screw_drops_every_stem_drums_too(monkeypatch):
    monkeypatch.setattr(P, "tuning_shift", lambda proj, sid: 0.0)
    proj = {"songs": {"a": {"transpose": 2}}, "bpm": 105.0}
    assert P.song_semis(proj, "a", "drums") == 0.0 and P.song_semis(proj, "a", "vocals") == 2.0     # not screwed
    proj["screw"] = 0.75
    drop = round(12 * math.log2(0.75), 3)
    assert P.song_semis(proj, "a", "drums") == drop
    assert P.song_semis(proj, "a", "vocals") == round(2 + 12 * math.log2(0.75), 3)
    proj["screw"] = 1.0
    assert P.song_semis(proj, "a", "drums") == 0.0


def test_a_screwed_render_is_the_unscrewed_one_resampled():
    beats = np.arange(0, 8.0, 0.5)                   # a steady 120 BPM song, 8 s, played in a 120 BPM project
    t = np.arange(8 * SR) / SR
    y = np.stack([np.sin(2 * np.pi * 440 * t)] * 2).astype(np.float32) * 0.3
    spec = '{"fb":0,"m":"natural","p":0.5}'
    plain, t0, m0 = warp.render(y, beats, 120.0, semis=0.0, stem="other", tspec=spec)
    r = 0.75                                         # screwed: 90 BPM, 5 semitones down
    screwed, t1, m1 = warp.render(y, beats, 120.0 * r, semis=12 * math.log2(r), stem="other",
                                  tspec='{"fb":0,"m":"natural","p":0.5,"s":0.75}')
    assert m1 == m0 + "+screw"
    assert abs(screwed.shape[1] - plain.shape[1] / r) <= 1 and abs(t1 - t0 / r) < 1e-9
    seg = screwed[0, SR:3 * SR]                      # the tone comes out at 440 x 0.75 = 330 Hz: a slowed turntable
    f = np.fft.rfftfreq(len(seg), 1 / SR)[np.argmax(np.abs(np.fft.rfft(seg)))]
    assert abs(f - 330) < 2


def test_tspec_carries_the_screw(monkeypatch):
    from mashlab import library
    monkeypatch.setattr(library, "read", lambda sid: {"bpm": 120.0, "beats": [0.0, 0.5, 1.0], "first_beat": 0})
    proj = {"songs": {"a": {}}}
    assert '"s"' not in P.song_tspec(proj, "a")
    proj["screw"] = 0.8
    assert '"s":0.8' in P.song_tspec(proj, "a")
