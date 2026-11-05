"""beatfix: a detector grid that switched to half / double time between sections, put back on one pulse."""
import numpy as np

from mashlab import beatfix


def steady(t0, n, bpm):
    return t0 + np.arange(n) * 60.0 / bpm


def test_half_and_double_time_stretches_come_back_on_one_pulse():
    # the music: 150 BPM throughout. The detector: right for 40 beats, then half time for 40 of the music's beats,
    # then double time for 40, then right again
    p = 60 / 150
    a = steady(0, 40, 150)
    half = a[-1] + p * 2 * np.arange(1, 21)                   # every other beat
    dbl = half[-1] + p / 2 * np.arange(1, 81)                 # twice as many
    tail = dbl[-1] + p * np.arange(1, 41)
    det = np.concatenate([a, half, dbl, tail])
    assert beatfix.slips(det)                                 # the lock would trip on this
    new, rep = beatfix.repair(det)
    assert abs(rep["pulse"] - 150) < 1
    gaps = np.diff(new)
    assert np.allclose(gaps, p, rtol=0.02)                    # one steady pulse all the way
    assert len(new) == 40 + 40 + 40 + 40
    assert beatfix.slips(new) == []


def test_a_real_tempo_change_stays():
    det = np.concatenate([steady(0, 64, 150), steady(64 * 0.4, 64, 180)])
    new, rep = beatfix.repair(det)
    assert beatfix.slips(det) == []                           # 150 -> 180 isn't half / double time
    assert len(new) == len(det)
    assert abs(60 / np.median(np.diff(new[-30:])) - 180) < 2


def test_marks_carry_the_bar_phase_and_unsure_stretches():
    info = dict(beats=list(steady(0.5, 64, 120)), bars=list(range(1, 64, 4)), bpb=4, duration=40.0)
    m, rep = beatfix.as_marks(info)
    assert len(m["t"]) == 64 and m["d"][1] == 1 and m["d"][5] == 1 and m["d"][0] == 0
    assert all(m["s"])


def _take_grid():
    # a Reimagine take's grid (the mix's, mapped onto the take bar for bar): steady 147 BPM, but YuE2 rushed six
    # beats of the lead-in into one second - the lock is what pulls them back out (smoke test, v2 release)
    p = 60 / 147
    return np.concatenate([steady(0.4, 7, 147), 2.851 + 0.163 * np.arange(1, 7), 3.83 + 0.245 + p * np.arange(0, 246)])


def _lock_check(monkeypatch, data_sandbox, check):
    from mashlab import jobs, library, lock

    class Stop(Exception):
        pass

    def tighten(*a, **k):
        raise Stop
    (data_sandbox["SONGS"] / "take").mkdir(exist_ok=True)
    library.write("take", dict(id="take", title="take", beats=list(_take_grid()), bars=[], bpb=4, duration=110.0))
    monkeypatch.setattr(lock, "tighten", tighten)
    monkeypatch.setattr(library, "load_stem", lambda *a, **k: np.zeros((2, 10), np.float32))
    monkeypatch.setattr(jobs, "submit", lambda kind, title, work, **k: work(type("J", (), {"update": lambda *a, **k: None})()))
    try:
        lock.make("take", check=check)
    except Stop:
        return "passed the check"
    except RuntimeError as e:
        return str(e)


def _half_double_song(data_sandbox, monkeypatch, user_marks=None):
    """A song whose detector counted half time then double time on a steady 150 BPM, with drums on every beat."""
    from mashlab import library
    p = 60 / 150
    a = steady(0, 40, 150)
    half = a[-1] + p * 2 * np.arange(1, 21)
    dbl = half[-1] + p / 2 * np.arange(1, 81)
    det = np.concatenate([a, half, dbl, dbl[-1] + p * np.arange(1, 41)])
    drums = np.zeros(int((det[-1] + 1) * 44100), np.float32)
    for x in np.arange(0, det[-1], p):
        drums[int(x * 44100):int(x * 44100) + 300] = 0.8
    (data_sandbox["SONGS"] / "hd").mkdir(exist_ok=True)
    info = dict(id="hd", title="hd", beats=list(det), bars=list(range(0, len(det), 4)), bpb=4, duration=float(det[-1] + 1))
    if user_marks:
        info["user_marks"] = user_marks
    library.write("hd", info)
    monkeypatch.setattr(library, "load_stem", lambda *a, **k: np.stack([drums, drums]))
    return det


def test_lock_check_offers_repair_only_where_it_fixes_the_detectors_marks(monkeypatch, data_sandbox):
    from mashlab import library, lock
    det = _half_double_song(data_sandbox, monkeypatch)
    c = lock.check("hd")
    assert c["slips"] and not c["yours"] and c["repair_clears"] and c["left"] == []
    made = []
    monkeypatch.setattr(lock, "make", lambda sid, check=True: made.append((sid, check)) or {"id": 1})
    lock.repair_and_make("hd")                              # the repaired marks are kept, then the lock runs
    m = library.read("hd")["user_marks"]
    assert made == [("hd", True)] and len(m["t"]) == 160 and beatfix.slips(np.array(m["t"])) == []
    # marks the user made are never replaced from the lock
    _half_double_song(data_sandbox, monkeypatch, user_marks=dict(t=list(det), d=[0] * len(det), s=[1] * len(det)))
    c = lock.check("hd")
    assert c["slips"] and c["yours"] and not c["repair_clears"]
    import pytest
    with pytest.raises(ValueError):
        lock.repair_and_make("hd")


def test_lock_anyway_skips_the_check(monkeypatch, data_sandbox):
    from mashlab import jobs, lock

    class Stop(Exception):
        pass

    def tighten(*a, **k):
        raise Stop
    _half_double_song(data_sandbox, monkeypatch)
    monkeypatch.setattr(lock, "tighten", tighten)
    monkeypatch.setattr(jobs, "submit", lambda kind, title, work, **k: work(type("J", (), {"update": lambda *a, **k: None})()))
    import pytest
    with pytest.raises(RuntimeError, match="wrong speed"):
        lock.make("hd")
    with pytest.raises(Stop):                               # past the check, into the lock itself
        lock.make("hd", check=False)


def test_lock_refuses_the_rushed_stretch_but_a_take_lock_skips_the_check(monkeypatch, data_sandbox):
    assert beatfix.slips(_take_grid())
    assert "wrong speed" in _lock_check(monkeypatch, data_sandbox, True)
    assert _lock_check(monkeypatch, data_sandbox, False) == "passed the check"
