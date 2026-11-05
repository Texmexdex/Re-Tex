"""djremix (DJ Mix, Remix style): whole vocal lines found from the vocal's level (never cut mid-word), which way round
the songs go, the plan format, and the engine's clip bookkeeping - without songs or a model."""
import numpy as np

from mashlab import djremix
from mashlab import project as P


def _env(spans, n=3000, loud=-15.0, quiet=-70.0):
    """A level curve (dB, 100 fps) that is loud inside each (start s, end s) span."""
    e = np.full(n, quiet)
    for a, b in spans:
        e[int(a * 100):int(b * 100)] = loud
    return e


def test_lines_end_where_the_voice_stops_even_when_pitch_tracking_drops_out():
    env = _env([(1.0, 4.0), (5.0, 7.5)])
    vm = env > -40
    vm[200:240] = False                     # pitch tracking loses a held note in the middle of the first line
    out = djremix.split_phrases(vm, env, bar_frames=[], words=[], beat_s=0.5)
    assert [(round(a / 100, 1), round(b / 100, 1)) for a, b in out] == [(1.0, 4.0), (5.0, 7.5)]


def test_an_unvoiced_attack_belongs_to_its_line():
    env = _env([(1.0, 4.0)])
    vm = env > -40
    vm[100:115] = False                     # "S" of "Sweet": sound, but no pitch yet
    a, b = djremix.split_phrases(vm, env, [], [], 0.5)[0]
    assert a <= 101 and b >= 399


def test_a_short_breath_does_not_split_a_line_but_a_real_one_does():
    env = _env([(1.0, 3.0), (3.1, 5.0), (5.6, 8.0)])     # 100 ms dip, then a 600 ms breath
    out = djremix.split_phrases(env > -40, env, [], [], 0.5)
    assert len(out) == 2 and out[0][1] >= 495 and out[1][0] >= 555


def test_only_very_long_lines_are_split_and_then_at_the_deepest_dip():
    # 20 s of continuous singing at 120 BPM (beat 0.5 s: 16 bars = 32 s - so no split) ...
    env = _env([(1.0, 21.0)])
    assert len(djremix.split_phrases(env > -40, env, [k * 200 for k in range(16)], [], 0.5)) == 1
    # ... but 40 s is split, at the one place the voice dips most
    env = _env([(1.0, 41.0)], n=5000)
    env[2390:2400] = -30
    bars = [k * 200 for k in range(25)]
    out = djremix.split_phrases(env > -40, env, bars, [], 0.5)
    assert len(out) == 2 and abs(out[0][1] - 2395) < 15


def test_line_break_strength_reads_the_words():
    w = lambda s, line=0: (0.0, s, line)
    assert djremix._strength(w("fire."), w("And")) == 2
    assert djremix._strength(w("baby,"), w("Jesus")) == 1
    assert djremix._strength(w("care"), w("I")) == 0       # "I" is capitalised mid-sentence too
    assert djremix._strength(w("over"), w("it", 1)) == 3    # a new transcript line


def test_the_bed_is_the_music_the_other_vocal_fits_best_and_keeps_its_key():
    fit = {"AB": dict(workable=True, rating=55.8, shift_rel=2, shift_bed=0),
           "BA": dict(workable=True, rating=77.5, shift_rel=-2, shift_bed=0)}
    roles, tr = djremix.roles_and_keys({}, fit, {"A": 0, "B": -2})
    assert roles == {"bed": "A", "guest": "B"} and tr == {"A": 0, "B": -2}
    fit["AB"]["rating"] = 90
    roles, tr = djremix.roles_and_keys({}, fit, {"A": 0, "B": 0}, donor="C")
    assert roles == {"bed": "B", "guest": "A", "donor": "C"} and tr == {"B": 0, "A": 2, "C": 0}
    big = {"AB": dict(workable=False), "BA": dict(workable=True, rating=60, shift_rel=6, shift_bed=-3)}
    assert djremix.roles_and_keys({}, big, {"A": 0, "B": 0})[1] == {"A": -3, "B": 3}   # a big move is split


def test_plan_format_only_names_real_sections_and_lines():
    s = djremix.schema(["A1", "A2"], ["A1", "B1", "B2"], has_donor=False)
    part = s["properties"]["parts"]["items"]["properties"]
    assert part["section"]["enum"] == ["A1", "A2", "continue"]
    assert part["drums"]["enum"] == ["bed", "none"]
    assert 2 in part["bars"]["enum"] and 6 in part["bars"]["enum"] and 1 not in part["bars"]["enum"]
    assert s["properties"]["lines"]["items"]["properties"]["line"]["enum"] == ["A1", "B1", "B2"]
    assert "donor" in djremix.schema(["A1"], ["B1"], has_donor=True)["properties"]["parts"]["items"]["properties"]["drums"]["enum"]


def test_a_lane_joins_audio_that_simply_carries_on_and_knows_where_it_is_free():
    proj = P.empty("t", 120, 4)
    lane = djremix._Lane(proj, proj["tracks"][0])
    a = lane.add("s", "drums", 0, 16, 10, 1.0)
    assert lane.add("s", "drums", 16, 32, 26, 1.0) is a and a["len"] == 32       # carries on: one clip
    b = lane.add("s", "drums", 32, 40, 100, 1.0)                                # a jump: a new clip
    assert b is not a and len(proj["clips"]) == 2
    assert not lane.free(30, 34) and lane.free(40, 48)
