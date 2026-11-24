"""What a lock does to a song's speed (lock.would / did / wrong_speed / report_all), and library items that don't
live in songs/ keeping their folder.

On the user's library (Nov 10) 20 of 77 locked songs played 2 s or more of the song over 20 % off its own speed:
stretches the section-level check (beatfix.slips) couldn't see. The lock now asks about those too."""
import json

import numpy as np
import pytest

from mashlab import beatfix, library, lock
from mashlab import project as P


def steady(n, bpm=120.0, t0=0.5):
    return t0 + np.arange(n) * 60.0 / bpm


def bunched(n_before=48, bars=3, n_after=48):
    """A steady 120 BPM song where the detector bunched a few bars: one long beat, three short ones (the bar is the
    right length, its beats are not - Fear Inoculum 5:57 read 0.82 / 0.40 / 0.40 / 0.40 s)."""
    gaps = [0.5] * n_before + [0.86, 0.38, 0.38, 0.38] * bars + [0.5] * n_after
    return 0.5 + np.concatenate([[0.0], np.cumsum(gaps)])


def test_a_steady_song_is_left_alone():
    r = lock.would(steady(200))
    assert r["sec5"] == 0 and r["sec20"] == 0 and r["parts"] == []
    assert abs(r["fastest"] - 1) < 1e-6 and abs(r["slowest"] - 1) < 1e-6
    assert lock.wrong_speed(steady(200)) == []


def test_marks_that_only_jitter_dont_make_the_lock_ask():
    rng = np.random.default_rng(3)
    m = steady(300) + rng.uniform(-0.012, 0.012, 300)       # +-12 ms around a steady pulse
    r = lock.would(m)
    assert r["sec20"] == 0 and r["fastest"] < 1.2 and r["slowest"] > 0.8
    assert lock.wrong_speed(m) == []


def test_bunched_bars_are_caught_where_the_section_check_sees_nothing():
    m = bunched()
    assert beatfix.slips(m) == []                            # three bars are too short for a section-level check
    r = lock.would(m)
    assert r["sec20"] == pytest.approx(6.0, abs=0.05)        # all 12 beats of the 3 bars (2 s each) play 20 %+ off
    assert len(r["parts"]) == 1
    a, b, typical, lo, hi = r["parts"][0]
    assert a == pytest.approx(24.5, abs=0.01) and b == pytest.approx(30.5, abs=0.01)
    assert lo == pytest.approx(0.76, abs=0.01) and hi == pytest.approx(1.72, abs=0.01)   # uneven: short beats slow, long fast
    sw = lock.wrong_speed(m)
    assert len(sw) == 1 and sw[0][0] == pytest.approx(24.5, abs=0.01)


def test_one_odd_beat_is_not_worth_a_question():
    gaps = [0.5] * 60 + [0.8] + [0.5] * 60                   # a single drawn-out beat: 0.8 s over 20 % off
    m = np.concatenate([[0.0], np.cumsum(gaps)])
    assert 0 < lock.would(m)["sec20"] < lock.WRONG_SEC
    assert lock.wrong_speed(m) == []


def test_a_half_time_section_is_listed_once():
    p = 0.4
    a = np.arange(40) * p
    half = a[-1] + 2 * p * np.arange(1, 31)                  # 30 beats counted at half the tempo
    m = np.concatenate([a, half, half[-1] + p * np.arange(1, 41)])
    old = beatfix.slips(m)
    assert old
    sw = lock.wrong_speed(m)
    assert len(sw) == len(old)                               # the beat-by-beat pass adds nothing a slip already covers
    r = lock.would(m)
    assert r["sec20"] == pytest.approx(30 * 2 * p, abs=0.05) and r["parts"][0][2] == pytest.approx(2.0, abs=0.01)


def test_did_reads_an_existing_lock_and_goes_quiet_when_the_beats_changed():
    beats = list(steady(120))
    land = np.array(beats, float)
    land[80:] += np.arange(40) * 0.25 + 0.25                 # the lock put the last 40 beats 0.75 s apart: slow
    info = dict(id="x", beats=beats, duration=float(beats[-1] + 1), lock=dict(rev="L1", beats=list(land)))
    r = lock.did(info)
    assert r["sec20"] == pytest.approx(20.0, abs=0.05) and r["slowest"] == pytest.approx(2 / 3, abs=0.01)
    assert r["parts"][0][0] == pytest.approx(beats[79], abs=0.01)
    assert lock.did(dict(info, beats=beats[:-3])) is None    # re-analyzed since: the record no longer says
    assert lock.did(dict(info, lock=dict(rev="L1", beats=beats)))["sec5"] == 0


def test_lock_check_reports_the_speed_and_the_lock_refuses_bunched_bars(monkeypatch, data_sandbox):
    from mashlab import jobs
    m = bunched()
    (data_sandbox["SONGS"] / "b").mkdir(exist_ok=True)
    library.write("b", dict(id="b", title="b", beats=list(m), bars=list(range(0, len(m), 4)), bpb=4,
                            duration=float(m[-1] + 1)))
    drums = np.zeros((2, int((m[-1] + 1) * 44100)), np.float32)
    monkeypatch.setattr(library, "load_stem", lambda *a, **k: drums)
    c = lock.check("b")
    assert c["slips"] and not c["yours"] and c["speed"]["sec20"] == pytest.approx(6.0, abs=0.05)
    monkeypatch.setattr(jobs, "submit", lambda kind, title, work, **k: work(type("J", (), {"update": lambda *a, **k: None})()))
    with pytest.raises(RuntimeError, match="wrong speed"):
        lock.make("b")


def test_report_all_lists_changed_locks_with_their_projects_and_skips_takes(monkeypatch, data_sandbox):
    monkeypatch.setattr(P, "PROJECTS", data_sandbox["PROJECTS"])
    beats = list(steady(120))
    land = np.array(beats, float)
    land[80:] += np.arange(40) * 0.25 + 0.25

    def song(sid, lock_beats, **kw):
        (data_sandbox["SONGS"] / sid).mkdir(exist_ok=True)
        info = dict(id=sid, title=sid.upper(), beats=beats, duration=float(beats[-1] + 1), stems_rev=0, **kw)
        info["lock"] = dict(rev="L1", sig=lock.sig(info), bpm=120.0, beats=list(lock_beats))
        library.write(sid, info)
    song("bad", land)
    song("fine", beats)
    song("take", land, source="data/yue2/render-1/take-1.flac")
    p = P.empty("Mix")
    p.update(id="p1", songs={"bad": dict(lock=True), "fine": dict(lock=True)},
             clips=[P.clip("t", "bad", "drums", 0, 0, 8)])
    (P.PROJECTS / "p1.json").write_text(json.dumps(p), encoding="utf-8")
    rep = lock.report_all()
    assert [r["id"] for r in rep] == ["bad", "fine"]         # the most changed first; the take isn't listed
    assert rep[0]["projects"] == ["Mix"] and rep[0]["current"] and rep[0]["speed"]["sec20"] > 15
    assert rep[1]["projects"] == [] and rep[1]["speed"]["sec5"] == 0   # "fine" is in the project's songs, but unused


def test_an_item_outside_songs_keeps_its_folder(monkeypatch, data_sandbox, tmp_path):
    """MIDI songs live in their own folder (config.song_path): moving one into a library folder must stick."""
    midi = tmp_path / "midi"
    (midi / "midi_x_1").mkdir(parents=True)
    (data_sandbox["SONGS"] / "song_1").mkdir()
    real = library.song_path
    monkeypatch.setattr(library, "song_path", lambda sid: midi / sid if str(sid).startswith("midi_") else real(sid))
    monkeypatch.setattr(library, "FOLDERS", tmp_path / "folders.json")
    library.set_folders(dict(folders=[dict(id="f1", name="Tool")],
                             assign={"song_1": "f1", "midi_x_1": "f1", "midi_gone": "f1", "song_gone": "f1"}))
    assert library.get_folders()["assign"] == {"song_1": "f1", "midi_x_1": "f1"}   # what's gone is still dropped
