"""Timing per project (the user's rules, Oct 19): saving a song's grid is for projects from then on - saved projects
that used the old grid keep it (a copy frozen into them, and the lock made from it kept for them), and a project can
have its own timing for a song (Marks mode), with its own lock."""
import json

import pytest

from mashlab import library, lock, timing, variant
from mashlab import project as P


@pytest.fixture
def song(data_sandbox, monkeypatch):
    monkeypatch.setattr(P, "PROJECTS", data_sandbox["PROJECTS"])
    sid = "s1"
    beats = [round(0.5 * j, 3) for j in range(40)]
    info = dict(id=sid, title="Song", status="ready", bpm=120.0, bpb=4, beats=beats, bars=list(range(0, 40, 4)),
                first_beat=0, last_beat=39, duration=20.0, stems_rev=0)
    d = library.song_dir(sid)
    (d / "stems_locked").mkdir(parents=True)
    for s in ("drums", "bass", "other", "vocals"):
        (d / "stems_locked" / f"{s}.flac").write_bytes(b"locked-" + s.encode())
    info["lock"] = dict(rev="L1", sig=lock.sig(info), bpm=120.0, beats=beats, scale=1.0)
    library.write(sid, info)
    return sid


def proj(pid, songs, modified=111.0):
    p = P.empty(pid)
    p.update(id=pid, songs=songs, modified=modified)
    (P.PROJECTS / f"{pid}.json").write_text(json.dumps(p), encoding="utf-8")
    return p


def load(pid):
    return json.loads((P.PROJECTS / f"{pid}.json").read_text(encoding="utf-8"))


def test_detected_marks_match_the_detector(song):
    m = timing.detected_marks(library.read(song))
    assert m["t"] == library.read(song)["beats"] and sum(m["d"]) == 10 and all(m["s"]) and m["bpb"] == 4


def test_saving_a_songs_grid_freezes_old_projects_and_keeps_their_lock(song):
    proj("pLocked", {song: {"mult": 1, "lock": True}})
    proj("pPlain", {song: {"mult": 1}})
    proj("pOther", {"x": {"mult": 1}})
    proj("pOpen", {song: {"mult": 1, "lock": True}})
    old = timing.detected_marks(library.read(song))
    new = dict(old, t=[t + 0.01 for t in old["t"]])                    # the user moved every mark 10 ms
    frozen, prev = P.set_song_grid(song, new, from_project="pOpen")
    assert sorted(frozen) == ["pLocked", "pPlain"] and prev == old
    assert load("pLocked")["songs"][song]["marks"] == old and load("pPlain")["songs"][song]["marks"] == old
    assert "marks" not in load("pOpen")["songs"][song]                 # the project it was saved from follows it
    assert load("pOther") == json.loads(json.dumps(load("pOther")))    # (untouched)
    assert load("pLocked")["modified"] == 111.0                        # list order unchanged
    info = library.read(song)
    assert info["user_marks"] == new and "lock" not in info            # the old lock is kept for pLocked ...
    kept = info["locks"][lock.sig(info, old)]
    assert kept["rev"] == "L1" and (library.song_dir(song) / "stems_locked" / "L1" / "drums.flac").read_bytes() == b"locked-drums"
    # ... and pLocked still plays it; pOpen (the new grid) has no lock for it yet - plays as recorded until re-locked
    tsL = P.song_tspec(load("pLocked"), song)
    assert variant.lock_rev(tsL) == "L1" and variant.stem_file(song, "drums", tsL).parent.name == "L1"
    assert variant.beats_for(info, tsL) == kept["beats"]
    assert variant.lock_rev(P.song_tspec(load("pOpen"), song)) == ""
    # saving the same grid again freezes nothing more
    assert P.set_song_grid(song, new)[0] == []


def test_without_a_locked_old_project_the_lock_just_goes_stale(song):
    proj("pPlain", {song: {"mult": 1}})
    old = timing.detected_marks(library.read(song))
    P.set_song_grid(song, dict(old, t=[t + 0.01 for t in old["t"]]))
    info = library.read(song)
    assert info["lock"]["rev"] == "L1" and not info.get("locks")       # nothing moved: no project needs it
    assert lock.current(info) is None


def test_a_projects_own_timing_finds_its_own_lock(song):
    info = library.read(song)
    own = timing.detected_marks(info)
    own["t"] = [t + 0.02 for t in own["t"]]
    info.setdefault("locks", {})[lock.sig(info, own)] = dict(rev="L2", sig=lock.sig(info, own), bpm=121.0,
                                                            beats=info["beats"], scale=0.9)
    library.write(song, info)
    p = {"bpm": 120, "songs": {song: {"mult": 1, "lock": True, "marks": own}}}
    ts = P.song_tspec(p, song)
    assert variant.lock_rev(ts) == "L2" and json.loads(ts)["p"] == pytest.approx(60 / 121.0)
    assert variant.scale_for(library.read(song), ts) == 0.9
    p["songs"][song].pop("marks")                                      # back to the song's grid: the song's lock
    assert variant.lock_rev(P.song_tspec(p, song)) == "L1"
    assert lock.in_use(song, lock.sig(info, own)) is False
    proj("pOwn", {song: {"mult": 1, "lock": True, "marks": own}})
    assert lock.in_use(song, lock.sig(info, own)) is True
