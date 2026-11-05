"""Perform mode (mashlab/perform_live.py): what each pad plays, and the clips a take becomes."""
import numpy as np

from mashlab import perform_live as PL


def src_at(g, t, bpb=4):
    for t0, t1, s0, gain in g.pieces(t - 1e-6, t + 1e-6, bpb):
        if t0 <= t < t1:
            return s0 + (t - t0), gain
    return None


def test_loop_pads_replay_the_last_beats():
    g = PL.Gesture(1, "t", "s", "repeat", 10, {}); g.R = 13
    assert src_at(g, 10.5) == (9.5, 1.0) and src_at(g, 11.25) == (9.25, 1.0) and src_at(g, 12.9)[0] == 9.9
    assert src_at(g, 13.1) is None                       # after R the song carries on (slip)


def test_gap_and_triplets():
    g = PL.Gesture(1, "t", "d", "repeat", 20, {"gate": 0.5}); g.R = 21
    assert src_at(g, 20.1)[1] == 1.0 and src_at(g, 20.3)[1] == 0.0 and src_at(g, 20.6)[1] == 1.0
    t = PL.Gesture(1, "t", "d", "repeat", 30, {"triplet": True}); t.R = 31
    assert abs(src_at(t, 30 + 1 / 3)[0] - (30 - 1 / 3)) < 1e-9


def test_roll_jump_chop_mute():
    r = PL.Gesture(1, "t", "j", "roll", 60, {}); r.R = 62.5
    assert abs(src_at(r, 60.3)[0] - 60.05) < 1e-9 and abs(src_at(r, 61.1)[0] - 60.1) < 1e-9 and abs(src_at(r, 62.1)[0] - 60.0375) < 1e-9
    j = PL.Gesture(1, "t", "g", "jump", 80, {}); j.R = 82
    assert src_at(j, 81) == (77, 1.0)
    c = PL.Gesture(1, "t", "k", "chop", 40, {}); c.R = 41
    assert src_at(c, 40.1) == (40.1, 1.0) and src_at(c, 40.3)[1] == 0.0
    m = PL.Gesture(1, "t", "l", "mute", 5, {}); m.R = 6
    assert src_at(m, 5.5)[1] == 0.0


def test_samples_loop_from_their_start_and_start_on_their_own_grid():
    g = PL.Gesture(1, "t", "z", "sample", 44, {}, {"track": "t", "start": 32, "len": 16}); g.R = 70.25
    assert src_at(g, 44) == (32, 1.0) and src_at(g, 60) == (32, 1.0) and abs(src_at(g, 70)[0] - 42) < 1e-9
    assert PL.sample_quant(16, "auto", 4) == 4 and PL.sample_quant(1, "auto", 4) == 1 and PL.sample_quant(0.25, "auto", 4) == 0.25
    assert PL.sample_quant(3, "bar", 4) == 4


def test_take_clips_map_back_and_trim(monkeypatch):
    from mashlab import readhead as RH
    monkeypatch.setattr(RH, "_landing", lambda project, sid: None)
    project = {"songs": {"a": {"mult": 1}}, "bpm": 120, "bpb": 4,
               "clips": [{"id": "c1", "track": "t", "song": "a", "stem": "drums", "start": 0, "src": 10, "len": 32, "gain_db": 0}]}
    g = PL.Gesture(1, "t", "s", "repeat", 8, {}); g.R = 10
    take = PL.take_clips(project, g)
    assert [(c["start"], c["src"], c["len"]) for c in take["clips"]] == [(8, 17, 1), (9, 17, 1)]
    tr = take["replace"]["trims"][0]
    assert tr["id"] == "c1" and tr["left_len"] == 8 and tr["right"]["start"] == 10 and tr["right"]["src"] == 20 and tr["left_fade"] == 0.03
    # a chop starts on the audio that was playing: the left part joins it with no fade
    c = PL.Gesture(2, "t", "k", "chop", 8, {}); c.R = 9
    assert PL.take_clips(project, c)["replace"]["trims"][0]["left_fade"] == 0
    # a sample goes on the layer lane, the lane it came from untouched
    s = PL.Gesture(3, "t", "z", "sample", 16, {}, {"track": "t", "start": 0, "len": 4}); s.R = 24
    st = PL.take_clips(project, s, "tL")
    assert st["replace"]["track"] == "tL" and not st["replace"]["trims"]
    assert [(c["start"], c["src"], c["len"]) for c in st["clips"]] == [(16, 10, 4), (20, 10, 4)]


def test_takes_on_one_lane_never_overlap_and_are_dropped_once_in_the_project():
    class P:
        lock = __import__("threading").RLock()
        playing, pos, _spb, _bpb = True, 0, 22050.0, 4
    pl, lv = P(), PL.Live()
    pl.pos = int(10.1 * pl._spb)
    a = lv.press(pl, "t", "s", {"quant": 0.25})
    assert a["T"] == 10.25
    pl.pos = int(10.6 * pl._spb)
    b = lv.press(pl, "t", "f", {"quant": 0.25})          # a second key on the same lane takes over
    ga = next(g for g in lv.g if g.id == a["id"])
    assert ga.R == b["T"] == 10.75
    pl.pos = int(11.0 * pl._spb)
    gb = lv.release(pl, b["id"], 0.25)
    assert gb.R == 11.0
    lv.on_project({"perform_seen": gb.id})
    assert not lv.g


def test_live_render_matches_plain_mix_when_nothing_is_held():
    from mashlab.mixer import Mixer
    m = Mixer()
    sr_buf = np.random.default_rng(0).standard_normal((2, 50000)).astype(np.float32) * 0.1
    m.tracks = [{"id": "t", "gain": 1.0, "duck": 0, "role": "drums", "lr": np.ones(2, np.float32)}]
    m.track_index = {"t": 0}
    m.clips = [dict(s0=0, s1=50000, g=1.0, fi=0, fo=0, ti=0, id="c", buf=sr_buf, off=0, step=1.0)]
    m.end_sample = 50000
    lv = PL.Live()
    far = PL.Gesture(1, "t", "s", "repeat", 999, {}); far.R = 1000
    lv.g = [far]
    assert lv.render(m, 1000, 1024, 11025.0, 4) is None
    # a held loop on the lane: the block is the loop, read from a beat earlier
    g = PL.Gesture(2, "t", "s", "repeat", 2, {}); lv.g = [g]; lv.last_end = None
    spb = 11025.0
    out = lv.render(m, int(2.5 * spb), 1024, spb, 4)
    a = int(1.5 * spb)
    assert np.allclose(out, sr_buf[:, a:a + 1024], atol=1e-6)
