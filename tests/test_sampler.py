"""The sample editor's edits (sampler.py) on a synthetic clip, and baking one into a sample."""
import json

import numpy as np
import pytest

from mashlab import sampler
from mashlab.config import SR

BPM = 120.0                       # 0.5 s a beat
BEAT = int(0.5 * SR)


@pytest.fixture
def sess(tmp_path, monkeypatch):
    monkeypatch.setattr(sampler, "EDITS", tmp_path / "_edit")
    monkeypatch.setattr(sampler, "SAMPLES", tmp_path)
    sampler._cache.clear()
    n = 8 * BEAT                                          # a 2-bar clip: each beat its own constant level
    y = np.zeros((2, n), np.float32)
    for k in range(8):
        y[:, k * BEAT:(k + 1) * BEAT] = (k + 1) / 10
    meta = dict(eid="e1", project="p1", clip="c1", song="s1", stem="vocals", title="Test", bpm=BPM, bpb=4, beats=8.0)
    (tmp_path / "_edit" / "e1").mkdir(parents=True)
    sampler._cache["e1"] = (meta, y)
    return y


def mid(y, beat):                                          # the level in the middle of a beat
    return float(y[0, int((beat + 0.5) * BEAT)])


def test_no_edits_is_the_clip(sess):
    _, y = sampler.render("e1", [])
    assert np.array_equal(y, sess)


def test_silence_and_reverse_stay_in_their_range(sess):
    _, y = sampler.render("e1", [{"op": "silence", "a": 2, "b": 3}])
    assert mid(y, 2) == 0 and mid(y, 1) == pytest.approx(0.2) and mid(y, 3) == pytest.approx(0.4)
    _, y = sampler.render("e1", [{"op": "reverse", "a": 4, "b": 6}])
    assert mid(y, 4) == pytest.approx(0.6) and mid(y, 5) == pytest.approx(0.5) and mid(y, 6) == pytest.approx(0.7)
    assert y.shape == sess.shape


def test_stutter_repeats_the_first_cell(sess):
    _, y = sampler.render("e1", [{"op": "repeat", "a": 4, "b": 6, "cell": 0.5}])       # beat 4's first half, x4
    for b in (4.0, 4.5, 5.0, 5.5):
        assert float(y[0, int((b + 0.25) * BEAT)]) == pytest.approx(0.5)
    assert mid(y, 6) == pytest.approx(0.7)                                           # the rest is untouched


def test_roll_halves_its_pieces(sess):
    _, y = sampler.render("e1", [{"op": "repeat", "a": 0, "b": 4, "cell": 1, "roll": True}])
    quiet = np.abs(y[0, :4 * BEAT]) < 0.02                                          # each piece fades out and in
    starts = np.flatnonzero(np.diff(quiet.astype(int)) == -1)                        # where a piece comes back in
    gaps = np.diff(starts)
    assert len(starts) >= 8 and gaps[-1] < gaps[0] / 4                               # more, shorter pieces toward the end


def test_paste_replaces_and_mixes(sess):
    _, y = sampler.render("e1", [{"op": "paste", "from_a": 0, "from_b": 1, "a": 6}])
    assert mid(y, 6) == pytest.approx(0.1) and mid(y, 7) == pytest.approx(0.8)
    _, y = sampler.render("e1", [{"op": "paste", "from_a": 0, "from_b": 1, "a": 6, "mode": "mix"}])
    assert mid(y, 6) == pytest.approx(0.8)                                           # 0.7 + 0.1


def test_echo_rings_on_past_the_clip_and_echo_out_stops_the_dry(sess):
    _, y = sampler.render("e1", [{"op": "echo", "a": 7, "b": 8, "time": 1, "feedback": 0.5, "mix": 1.0}])
    assert y.shape[1] > sess.shape[1]                                                # the tail goes past the clip's end
    assert abs(mid(y, 8)) > 0.2 and abs(mid(y, 9)) > 0.1                             # repeats, quieter each time
    _, z = sampler.render("e1", [{"op": "echo", "a": 3, "b": 4, "time": 1, "feedback": 0.3, "mix": 1.0, "out": True}])
    assert mid(z, 4) == pytest.approx(0.4, abs=0.03)       # after the range only the echo of beat 3 (0.4) - not 0.5 + echo
    assert mid(z, 5) == pytest.approx(0.12, abs=0.03)      # the next repeat, x0.3


def test_tape_stop_slows_to_nothing_and_start_arrives_on_time(sess):
    _, y = sampler.render("e1", [{"op": "tapestop", "a": 2, "b": 4}])
    assert float(np.abs(y[0, 4 * BEAT - 200:4 * BEAT - 10]).max()) < 0.05            # stopped by the end
    assert mid(y, 2) == pytest.approx(0.3, abs=0.02) and mid(y, 4) == pytest.approx(0.5)
    _, y = sampler.render("e1", [{"op": "tapestart", "a": 2, "b": 4}])
    assert float(y[0, 4 * BEAT - 50]) == pytest.approx(0.4, abs=0.02)                 # reaches beat 3's audio by beat 4


def test_apply_bakes_a_sample(sess, monkeypatch):
    from mashlab import library, config
    monkeypatch.setattr(config, "SAMPLES", sampler.SAMPLES)
    monkeypatch.setattr(library, "SAMPLES", sampler.SAMPLES)
    monkeypatch.setattr(library, "read", lambda sid, _r=library.read: _r(sid) if str(sid).startswith("smp_") else {})
    events = []
    monkeypatch.setattr(sampler.bus, "publish", lambda kind, **d: events.append((kind, d)))
    r = sampler.apply("e1", [{"op": "silence", "a": 0, "b": 1}])
    info = json.loads((sampler.SAMPLES / r["sid"] / "song.json").read_text(encoding="utf-8"))
    assert info["kind"] == "sample" and info["bpm"] == BPM and info["beats"][2] == pytest.approx(1.0)
    assert r["beats"] == pytest.approx(8.0, abs=0.05)
    y = library.read_stem(r["sid"], "vocals")
    assert float(np.abs(y[:, int(0.01 * SR):BEAT // 2]).max()) == 0                 # silent past the 4 ms crossfade
    assert mid(y, 3) == pytest.approx(0.4, abs=1e-3)
    assert [s["id"] for s in library.list_samples()] == [r["sid"]]
    assert events and events[-1][0] == "sample_applied" and events[-1][1]["clip"] == "c1"
