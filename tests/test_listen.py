"""listen.py (Music Flamingo describes songs by ear): stored next to the tags, its genre as the song's guess, quoted
in the DJ brief - and the whole describe() round trip against a stand-in worker (no model, no GPU)."""
import json
import sys
import textwrap

import numpy as np

from mashlab import djbrain, libmeta, listen


def test_heard_is_stored_and_its_genre_becomes_the_guess(tmp_path, monkeypatch):
    monkeypatch.setattr(libmeta, "META", tmp_path / "meta.json")
    libmeta.set_heard("s1", {"genre": "Metal", "subgenre": "Hardcore Punk", "sound": "Raw, fast, shouted."}, "mf")
    assert libmeta.heard("s1") == {"genre": "Metal", "subgenre": "Hardcore Punk", "sound": "Raw, fast, shouted."}
    e = json.loads((tmp_path / "meta.json").read_text(encoding="utf-8"))["songs"]["s1"]
    eff = libmeta._effective(e)
    assert eff["genre"] == "Metal" and eff["src"]["genre"] == "guess" and eff["heard"]["subgenre"] == "Hardcore Punk"
    e["user"] = {"genre": "Post-Rock"}                     # the user's own edit still wins
    assert libmeta._effective(e)["genre"] == "Post-Rock"
    assert libmeta.heard("nope") is None
    e = {"tags": {"genre": "Other"}, "guess": {"genre": "Hip-Hop"}}   # a placeholder file tag doesn't block it
    assert libmeta._effective(e)["genre"] == "Hip-Hop"
    e["tags"]["genre"] = "Alternative"                     # a real one still wins
    assert libmeta._effective(e)["genre"] == "Alternative"


def test_the_dj_brief_quotes_what_was_heard(monkeypatch):
    class S:
        id, title, bpm, info, concept = "s1", "Song", 150.0, {"key": {"name": "A major"}, "last_beat": 32}, {}
        has_real_vocals, bpb, bars, nb = False, 4, [0, 8, 16, 24], 32
    secs = [dict(id="A1", name="verse", m0=0, m1=4, energy=0.5, stems=dict(drums=1, bass=1, other=1, vocals=1), hook=False, lyric="")]
    monkeypatch.setattr(djbrain, "_heard", lambda sid: {"genre": "Electronic", "subgenre": "Glitch", "sound": "Chopped vocal samples."})
    text = djbrain._describe(S, "A", secs, [0, 1, 2, 3])
    assert "Sounds like (heard): Electronic / Glitch - Chopped vocal samples." in text.splitlines()[1]
    monkeypatch.setattr(djbrain, "_heard", lambda sid: None)
    assert "heard" not in djbrain._describe(S, "A", secs, [0, 1, 2, 3])


def test_a_reply_cut_off_mid_sound_keeps_what_it_said():
    from mashlab import listen_worker
    text = ('{"genre":"Metal","subgenre":"Mathcore","sound":"Chaotic, aggressive, chaotic, intense, raw, aggressive, '
            'dissonant riffs, screamed vocals, frantic, frantic, relentless, chaotic, abrasive, punishing, wild, extreme')
    r = listen_worker.salvage(text)
    assert r["genre"] == "Metal" and r["subgenre"] == "Mathcore"
    words = [w.strip(",.").lower() for w in r["sound"].split()]
    assert len(words) == len(set(words)) and r["sound"].endswith(".") and "chaotic" in words
    import pytest
    with pytest.raises(ValueError):
        listen_worker.salvage("not json at all")


def test_off_without_config(monkeypatch):
    monkeypatch.setattr(listen, "cfg", lambda: None)
    assert listen.available()[0] is False
    monkeypatch.setattr(listen, "cfg", lambda: {"python": "Z:/nowhere/python.exe", "model": "m"})
    ok, why = listen.available()
    assert not ok and "missing" in why


def test_describe_round_trip_with_a_stand_in_worker(tmp_path, monkeypatch):
    worker = tmp_path / "fake_worker.py"
    worker.write_text(textwrap.dedent('''
        import json, sys
        job = json.load(open(sys.argv[1], encoding="utf-8"))
        with open(sys.argv[2], "a", encoding="utf-8") as f:
            f.write(json.dumps({"loaded": 0.1}) + "\\n")
            for it in job["items"]:
                row = ({"id": it["id"], "error": "boom"} if it["id"] == "bad" else
                       {"id": it["id"], "genre": "Rock", "subgenre": "Indie", "sound": "heard " + it["about"]})
                f.write(json.dumps(row) + "\\n")
            f.write(json.dumps({"done": True}) + "\\n")
    '''), encoding="utf-8")
    stored = {}
    monkeypatch.setattr(listen, "WORKER", worker)
    monkeypatch.setattr(listen, "cfg", lambda: {"python": sys.executable, "model": "fake"})
    monkeypatch.setattr(listen, "_clip", lambda sid, path: path.write_bytes(b"RIFF"))
    monkeypatch.setattr(listen, "about", lambda sid, info=None, folders=None: f"title {sid}")
    monkeypatch.setattr(listen, "_free_gpu", lambda: None)
    monkeypatch.setattr(listen.library, "read", lambda sid: {"status": "ready"})
    monkeypatch.setattr(listen.library, "get_folders", lambda: {"folders": [], "assign": {}})
    monkeypatch.setattr(libmeta, "set_heard", lambda sid, h, model="": stored.setdefault(sid, (h, model)))
    monkeypatch.setattr(libmeta, "all_meta", lambda: {})
    out = listen.describe(["a", "bad", "b"])
    assert out["a"] == {"genre": "Rock", "subgenre": "Indie", "sound": "heard title a"}
    assert out["bad"] == {"error": "boom"}                  # one song failing doesn't stop the rest
    assert set(stored) == {"a", "b"} and stored["b"][1] == "fake"
