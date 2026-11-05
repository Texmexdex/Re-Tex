"""Placing typed lyrics: cutting sections where the singing starts / stops, regrouping the score, the lyrics text."""
import json

import numpy as np

from mashlab import align, reimagine


def test_a_sample_counts_as_the_song_it_was_cut_from(data_sandbox, monkeypatch):
    # a sample from the sample editor has no analysis of its own: Reimagine asked for its features.npz and crashed
    from mashlab import config
    samples = data_sandbox["DATA"] / "samples"
    monkeypatch.setattr(config, "SAMPLES", samples)
    for base, sid, info in ((data_sandbox["SONGS"], "song1", {"id": "song1", "status": "ready"}),
                            (samples, "smp_x", {"id": "smp_x", "kind": "sample", "parent": {"song": "song1"}}),
                            (samples, "smp_orphan", {"id": "smp_orphan", "kind": "sample", "parent": {"song": "gone"}})):
        (base / sid).mkdir(parents=True)
        (base / sid / "song.json").write_text(json.dumps(info), encoding="utf-8")
    proj = {"clips": [{"song": "smp_x", "len": 8.0}, {"song": "song1", "len": 4.0}, {"song": "smp_orphan", "len": 99.0},
                      {"song": "song1", "len": 99.0, "mute": True}]}
    assert reimagine._songs_of(proj) == ["song1"]

HEAD = ["X:1", "T:", "M:4/4", "L:1/16", "Q:1/4=120", 'V: Vocal clef=treble name="Vocal"', 'V: Ins clef=treble name="Ins"',
        "K:C"]


def score(groups):
    """groups: [(label or None, n bars)] -> a native two-voice score."""
    out, n = list(HEAD), 0
    for label, k in groups:
        if label:
            out.append(f"% {label}")
        out += ["V: Vocal", "|".join(f"c{n + i}" for i in range(k)) + "|", "V: Ins", "|".join(f"C{n + i}" for i in range(k)) + "|"]
        n += k
    return "\n".join(out) + "\n"


def test_text_lines_drop_tags_and_blanks():
    assert align.text_lines("[Verse]\nfirst line\n\n  second  \n[Chorus 2]\nthird") == ["first line", "second", "third"]


def test_spell_reads_digits_and_drops_punctuation():
    assert align._spell("don't!") == "DON'T"
    assert align._spell("2") == "TWO"
    assert align._spell("--") == ""


def test_resection_moves_section_starts_and_keeps_every_bar():
    abc = score([("intro", 4), ("verse", 4), ("chorus", 4)])
    out = reimagine.resection(abc, [(0, "intro"), (2, "verse"), (9, "chorus")])
    head, bars = reimagine._groups(out)
    assert head == HEAD
    assert [b["vocal"] for b in bars] == [f"c{i}" for i in range(12)]
    assert [b["ins"] for b in bars] == [f"C{i}" for i in range(12)]
    assert [(i, b["label"]) for i, b in enumerate(bars) if b["label"]] == [(0, "intro"), (2, "verse"), (9, "chorus")]


def test_resection_groups_at_most_four_bars():
    out = reimagine.resection(score([("verse", 4)] * 3), [(0, "verse")])
    assert all(l.count("|") <= 4 for l in out.splitlines() if l and l[0] in "cC")


def test_resection_expands_multibar_rests():
    abc = "\n".join(HEAD + ["% intro", "V: Vocal", "Z2|c|", "V: Ins", "C0|C1|C2|"]) + "\n"
    _, bars = reimagine._groups(reimagine.resection(abc, [(0, "intro"), (2, "verse")]))
    assert [b["vocal"] for b in bars] == ["Z", "Z", "c"] and bars[2]["label"] == "verse"


BAR = 2.0          # 4/4 at 120 bpm


def line(b0, b1, text="la"):
    return dict(t0=b0 * BAR, t1=b1 * BAR, text=text)


def test_singing_late_in_the_first_section_gets_an_intro():
    # the score calls bars 0-8 "verse", but the singing only starts at bar 4
    secs = [dict(label="verse", t0=0, t1=8 * BAR), dict(label="chorus", t0=8 * BAR, t1=16 * BAR)]
    cuts, owner = reimagine.plan_sections(secs, [line(4, 6, "a"), line(6, 8, "b"), line(8, 12, "c")], BAR)
    assert cuts == [(0, "intro"), (4, "verse"), (8, "chorus")]
    assert owner == {0: 1, 1: 1, 2: 2}


def test_a_pickup_belongs_to_the_next_bar():
    # a line that starts just before a section's downbeat is that section's first line, not the last of the one before
    secs = [dict(label="verse", t0=0, t1=8 * BAR), dict(label="chorus", t0=8 * BAR, t1=16 * BAR)]
    cuts, owner = reimagine.plan_sections(secs, [line(0, 7, "a"), line(7.9, 10, "hook")], BAR)
    assert cuts == [(0, "verse"), (8, "chorus")]
    assert owner == {0: 0, 1: 1}


def test_long_gap_inside_a_section_becomes_an_interlude():
    secs = [dict(label="verse", t0=0, t1=16 * BAR)]
    cuts, owner = reimagine.plan_sections(secs, [line(0, 3), line(9, 12)], BAR)
    assert cuts == [(0, "verse"), (3, "interlude"), (9, "verse")]
    assert owner == {0: 0, 1: 2}


def test_unsung_sections_are_instrumental_and_names_are_the_studios():
    secs = [dict(label="intro", t0=0, t1=4 * BAR), dict(label="drop", t0=4 * BAR, t1=8 * BAR),
            dict(label="post-chorus", t0=8 * BAR, t1=12 * BAR), dict(label="outro", t0=12 * BAR, t1=16 * BAR)]
    cuts, _ = reimagine.plan_sections(secs, [line(8, 11)], BAR)
    assert cuts == [(0, "intro"), (4, "instrumental"), (8, "chorus"), (12, "outro")]
    assert {l for _, l in cuts} <= set(reimagine.LABELS.values())


def test_tagged_lyrics_file_lines_under_their_sections():
    cuts = [(0, "intro"), (4, "verse"), (8, "pre-chorus")]
    lines = [dict(text="one"), dict(text="two"), dict(text="three")]
    txt = reimagine.tagged_lyrics(cuts, lines, {0: 1, 1: 1, 2: 2})
    assert txt == "[Intro]\n\n[Verse]\none\ntwo\n\n[Pre-Chorus]\nthree\n"


def test_typed_words_land_where_the_clip_plays_them(monkeypatch):
    from mashlab import library, userlyrics
    info = dict(bpm=120.0, beats=[i * 0.5 for i in range(64)], first_beat=0, status="ready")
    monkeypatch.setattr(library, "read", lambda sid: info if sid == "s1" else None)
    words = [dict(t=1.0, t1=1.4, w="before", line=0),          # beat 2: before the clip's stretch of the song
             dict(t=3.0, t1=3.4, w="in", line=1), dict(t=3.5, t1=3.9, w="the", line=1), dict(t=4.0, t1=4.4, w="clip", line=1),
             dict(t=7.0, t1=7.4, w="after", line=2)]            # beat 14: past its end
    monkeypatch.setattr(userlyrics, "load", lambda sid: {"words": words} if sid == "s1" else None)
    proj = dict(bpm=120.0, songs={"s1": {}, "s2": {}}, tracks=[dict(id="t1"), dict(id="t2", mute=True)],
                clips=[dict(id="c1", track="t1", song="s1", stem="vocals", start=8.0, src=4.0, len=8.0),
                       dict(id="c2", track="t2", song="s1", stem="vocals", start=0.0, src=0.0, len=32.0),   # muted track
                       dict(id="c3", track="t1", song="s1", stem="drums", start=0.0, src=0.0, len=32.0)])
    lines, missing = userlyrics.mix_lines(proj)
    assert missing == []
    assert [l["text"] for l in lines] == ["in the clip"]
    assert [w["t"] for w in lines[0]["words"]] == [5.0, 5.5, 6.0]      # song 3.0 s = its beat 6 = clip beat 2 = mix beat 10


def test_retime_moves_each_take_line_to_where_the_mix_sings_it():
    bpm = 120.0                                     # 0.5 s a beat; the take's grid is steady too: beat k at k/2 s
    beats = [k * 0.5 for k in range(400)]
    mix = [dict(t0=10.0, t1=12.0), dict(t0=12.5, t1=14.5), dict(t0=20.0, t1=22.0), dict(t0=30.0, t1=31.0)]
    take = [dict(line=0, t0=11.5, t1=13.5, conf=0.4),     # 1.5 s late
            dict(line=1, t0=3.0, t1=40.0, conf=0.3),      # smeared over half the song: not found
            dict(line=2, t0=21.0, t1=23.6, conf=0.5),     # 1 s late and 30 % longer: stretched back
            dict(line=3, t0=31.5, t1=32.5, conf=0.02)]    # hardly heard: not found
    plan = {p["line"]: p for p in reimagine.retime_plan(mix, take, beats, bpm)}
    assert set(plan) == {0, 1, 2, 3}
    p = plan[0]
    assert abs(p["start"] - (10.0 - reimagine.RETIME_PRE) * 2) < 1e-3 and p["rate"] == 1.0
    assert abs(p["src"] - (11.5 - reimagine.RETIME_PRE) * 2) < 1e-3
    assert abs((p["start"] + p["len"]) - (12.0 + reimagine.RETIME_POST) * 2) < 1e-3    # a breath after its last word
    tight = reimagine.retime_plan([dict(t0=10.0, t1=12.4), dict(t0=12.45, t1=14.0)], [dict(line=0, t0=11.5, t1=13.9, conf=0.4),
                                  dict(line=1, t0=13.95, t1=15.5, conf=0.4)], beats, bpm)
    assert abs((tight[0]["start"] + tight[0]["len"]) - 12.4 * 2) < 1e-3                # ...or up to where the next one starts
    assert abs(plan[1]["src"] - (12.5 + 1.25 - reimagine.RETIME_PRE) * 2) < 1e-3       # its neighbours' offset (median of 1.5, 1)
    assert abs(plan[2]["rate"] - 1.3) < 1e-3
    assert abs(plan[2]["src"] - (21.0 - reimagine.RETIME_PRE * 1.3) * 2) < 1e-3


def test_a_word_parked_in_silence_moves_to_where_the_voice_comes_in():
    voiced = np.zeros(6000, bool)                  # 60 s of 10 ms frames: silent until 40 s, singing 40-50 s
    voiced[4000:5000] = True
    words = [dict(t=0.0, t1=0.3, w="MTM"), dict(t=40.6, t1=41.0, w="hit"), dict(t=41.1, t1=41.4, w="em")]
    out = align.settle(words, voiced)
    assert out[0]["t"] == 40.0 and out[0]["t1"] == 40.3
    assert [w["t"] for w in out[1:]] == [40.6, 41.1]                  # words in the voice stay put


def test_a_line_sung_before_the_takes_first_beat_loses_its_head():
    beats = [0.6 + k * 0.5 for k in range(400)]     # the take's grid starts at 0.6 s
    plan = reimagine.retime_plan([dict(t0=40.5, t1=42.0)], [dict(line=0, t0=0.2, t1=1.7, conf=0.6)], beats, 120.0)
    p, = plan
    assert abs(p["src"]) < 1e-6                                       # starts on the first beat ...
    assert abs(p["start"] * 0.5 - (40.5 + 0.4)) < 1e-3                # ... where that moment of the line belongs
