"""gridmash (Auto-Mashup, "Pick stems"): any stems from any songs fitted to one master - meters that differ, bars
that drift, whole vocal lines on bar lines. Songs are plain numbers here (no library, no audio)."""
import numpy as np

from mashlab import gridmash as G
from mashlab import project as P
from mashlab import warps as W


def song(sid, bpm, bpb, bars, start=0.5, drift=0.0, sections=None, phrases=None, on=None, info_bpm=None):
    """A steady song: `bars` bars of `bpb` beats from `start` s. drift = s each beat runs longer than the tempo says."""
    n = bars * bpb + 1
    t = start + np.arange(n) * (60.0 / bpm + drift)
    d = [1 if i % bpb == 0 else 0 for i in range(n)]
    end = float(t[-1])
    secs = sections or [dict(t0=start, t1=end, name="verse", energy=0.5, stems={})]
    return G.Src(sid, sid, t, d, t, info_bpm or bpm, bpb=bpb, content=(start, end), sections=secs, on=on, phrases=phrases)


def src_time(L, proj, c, u):
    """The moment of song L (s) that clip c plays at project beat u - through the clip's stretches, the way the
    mixer plays them (warps.pieces on the song's as-recorded landing)."""
    mult = proj["songs"][c["song"]]["mult"]
    p = 60.0 / L.info_bpm
    idx = np.arange(-64, len(L.ib) + 64, dtype=float)
    t = np.array([float(P._beat_to_time(i, L.ib)) for i in idx])
    for u0, u1, w0, w1, r, kp in W.pieces(c, mult, (t / p, idx)):
        if u0 - 1e-9 <= u - c["start"] <= u1 + 1e-9:
            return (w0 + (u - c["start"] - u0) * r / mult) * p
    raise AssertionError("the clip does not play there")


def clips_of(proj, sid, stem):
    return sorted((c for c in proj["clips"] if c["song"] == sid and c["stem"] == stem), key=lambda c: c["start"])


def on_a_beat(L, t, tol):
    return float(np.min(np.abs(L.bt - t))) <= tol


def test_three_into_four_repeats_an_inner_beat_and_four_into_three_drops_the_last():
    assert G.beat_map(3, 4) == [0, 1, 2, 1]
    assert G.beat_map(4, 3) == [0, 1, 2]
    assert G.beat_map(4, 4) == [0, 1, 2, 3]
    assert G.beat_map(3, 5) == [0, 1, 2, 1, 2]


def test_same_meter_layers_run_at_half_or_double_time_when_that_stretches_less():
    L = song("l", 70.0, 4, 8)
    tf = G.tempo_fit(L, 138.0, 4)
    assert tf["fit"] == "beat" and tf["M"] == 2.0 and tf["per_bar"] == 0.5 and abs(tf["stretch"]) < 0.02


def test_a_six_eight_song_goes_bar_for_bar_over_four_four_instead_of_being_squashed():
    L = song("l", 180.0, 6, 8)                       # counted in eighths: 6 to the bar
    tf = G.tempo_fit(L, 120.0, 4)
    assert tf["fit"] == "bar" and abs(tf["M"] - 2 / 3) < 1e-9 and tf["per_bar"] == 1.0
    assert abs(tf["stretch"]) < 0.01                 # ... where beat for beat would mean +33 %
    assert G.tempo_fit(L, 120.0, 4, fit="beat")["fit"] == "beat"


def test_a_waltz_near_the_masters_tempo_lines_up_beat_for_beat():
    tf = G.tempo_fit(song("l", 130.0, 3, 8), 120.0, 4)
    assert tf["fit"] == "beat" and tf["M"] == 1.0 and tf["per_bar"] is None and tf["slots"] == 3


def test_same_meter_drums_are_one_run_cut_only_at_the_masters_sections_and_the_cuts_are_seamless():
    secs = [dict(t0=0.5, t1=0.5 + 16 * 0.5, name="intro", energy=0.3, stems={}),
            dict(t0=0.5 + 16 * 0.5, t1=0.5 + 64 * 0.5, name="verse", energy=0.7, stems={})]
    M = song("m", 120.0, 4, 16, sections=secs)
    D = song("d", 126.0, 4, 40)
    proj, desc = G.build_project({"m": M, "d": D}, [dict(song="m", stems=["bass", "other"]), dict(song="d", stems=["drums"])])
    assert desc["master"] == "m" and proj["bpm"] == 120.0 and proj["bpb"] == 4
    cl = clips_of(proj, "d", "drums")
    assert len(cl) == 2                              # one clip per master section, nothing else
    a, b = cl
    assert abs(b["start"] - (a["start"] + a["len"])) < 1e-6
    mult = proj["songs"]["d"]["mult"]                # the mixer's own test for "the same audio carries on" (_joins)
    assert abs(b["src"] - (a["src"] + a["len"] / mult)) < 1e-4 and a["fade_out"] == 0.0 and b["fade_in"] == 0.0
    for k in range(16):                              # every master bar line plays one of the drums' beats
        c = next(c for c in cl if c["start"] <= 4 * k + 1e-6 < c["start"] + c["len"])
        assert on_a_beat(D, src_time(D, proj, c, 4 * k), 1e-3)


def test_four_beat_drums_under_a_waltz_play_beats_one_two_three_of_each_bar():
    M = song("m", 120.0, 3, 12)
    D = song("d", 120.0, 4, 30)
    proj, desc = G.build_project({"m": M, "d": D}, [dict(song="m", stems=["other"]), dict(song="d", stems=["drums"])])
    lay = next(x for x in desc["layers"] if x["song"] == "d")
    assert lay["fit"] == "beat" and lay["beats"] == [1, 2, 3] and proj["bpb"] == 3
    cl = clips_of(proj, "d", "drums")
    assert len(cl) == 12                             # a clip per bar: its 4th beat is skipped every time
    for k, c in enumerate(cl):
        t = src_time(D, proj, c, 3 * k)              # the waltz's bar line plays a 4/4 bar's ONE
        bar_lines = D.bt[::4]
        assert float(np.min(np.abs(bar_lines - t))) < 1e-3
        assert c["start"] <= 3 * k and c["start"] + c["len"] <= 3 * (k + 1) + 1e-6
    lead = 3 * 1 - cl[1]["start"]                     # each starts just ahead of its line, clear of the hit
    assert 0.0 < lead * 0.5 <= G.LEAD + 1e-6


def test_waltz_drums_under_four_four_fill_the_fourth_beat_with_their_second():
    M = song("m", 120.0, 4, 8)
    D = song("d", 124.0, 3, 30)
    proj, desc = G.build_project({"m": M, "d": D}, [dict(song="m", stems=["other"]), dict(song="d", stems=["drums"])])
    assert next(x for x in desc["layers"] if x["song"] == "d")["beats"] == [1, 2, 3, 2]
    cl = clips_of(proj, "d", "drums")
    for k in range(8):
        for beat, want in ((0, 0), (1, 1), (2, 2), (3, 1)):          # master beat -> which beat of the waltz bar
            u = 4 * k + beat
            c = next(c for c in cl if c["start"] <= u + 1e-6 < c["start"] + c["len"])
            i = int(np.argmin(np.abs(D.bt - src_time(D, proj, c, u))))
            assert abs(D.bt[i] - src_time(D, proj, c, u)) < 1e-3 and i % 3 == want


def test_a_layer_that_drifts_follows_the_master_by_stretching_and_its_audio_is_never_cut():
    M = song("m", 120.0, 4, 32)
    D = song("d", 120.0, 4, 60, drift=0.0006)        # 0.6 ms a beat: 15 ms after about 6 bars
    proj, desc = G.build_project({"m": M, "d": D}, [dict(song="m", stems=["other"]), dict(song="d", stems=["drums"])])
    cl = clips_of(proj, "d", "drums")
    assert len(cl) == 1 and len(cl[0]["warps"]) == 1          # one clip, one speed for the whole song
    w = cl[0]["warps"][0]
    assert abs(w["r"] - (0.5006 / 0.5)) < 1e-4 and w["kp"] is False     # 0.12 % faster: plain varispeed
    for k in range(32):
        assert on_a_beat(D, src_time(D, proj, cl[0], 4 * k), 2e-3)
    lay = next(x for x in desc["layers"] if x["song"] == "d")
    assert lay["cuts"] == 0 and lay["follows"] == 1 and desc["follow"] == "stretch"


def test_a_layer_whose_tempo_wanders_gets_as_few_stretches_as_keep_it_within_reach():
    M = song("m", 120.0, 4, 48)
    n = 60 * 4 + 1
    per = 0.5 * (1.0 + 0.012 * np.sin(np.arange(n - 1) / 40.0))       # a drummer: +-1.2 %, slowly
    t = 0.5 + np.r_[0.0, np.cumsum(per)]
    D = G.Src("d", "d", t, [1 if i % 4 == 0 else 0 for i in range(n)], t, 120.0, bpb=4, content=(0.5, float(t[-1])))
    proj, desc = G.build_project({"m": M, "d": D}, [dict(song="m", stems=["other"]), dict(song="d", stems=["drums"])])
    cl = clips_of(proj, "d", "drums")
    assert len(cl) == 1 and 2 <= len(cl[0]["warps"]) <= 16
    assert all(G.RATE_SPAN[0] < w["r"] < G.RATE_SPAN[1] and w["kp"] is False for w in cl[0]["warps"])
    for k in range(48):                              # every master bar line plays one of the drummer's bar lines
        assert float(np.min(np.abs(D.bar_t - src_time(D, proj, cl[0], 4 * k)))) <= G.DRIFT + 2e-3


def test_a_pitched_layer_keeps_its_pitch_when_the_speed_change_is_more_than_a_few_cents():
    M = song("m", 120.0, 4, 32)
    B = song("b", 120.0, 4, 60, drift=0.006)         # 1.2 % slower than its stored tempo says
    proj, _ = G.build_project({"m": M, "b": B}, [dict(song="m", stems=["drums"]), dict(song="b", stems=["bass"])], master="m")
    w = clips_of(proj, "b", "bass")[0]["warps"]
    assert len(w) == 1 and w[0]["kp"] is True and abs(w[0]["r"] - 1.012) < 1e-3


def test_follow_cut_re_anchors_on_bar_lines_and_stretches_nothing():
    M = song("m", 120.0, 4, 32)
    D = song("d", 120.0, 4, 60, drift=0.0006)
    proj, desc = G.build_project({"m": M, "d": D}, [dict(song="m", stems=["other"]), dict(song="d", stems=["drums"])],
                                 opts=dict(follow="cut"))
    cl = clips_of(proj, "d", "drums")
    assert 3 <= len(cl) <= 8 and not any(c.get("warps") for c in cl) and desc["follow"] == "cut"
    for k in range(32):
        c = next(c for c in cl if c["start"] <= 4 * k + 1e-6 < c["start"] + c["len"])
        assert on_a_beat(D, src_time(D, proj, c, 4 * k), G.DRIFT + 1e-3)
    assert next(x for x in desc["layers"] if x["song"] == "d")["cuts"] == len(cl) - 1


def test_a_stretch_too_far_to_follow_is_cut_instead():
    M = song("m", 120.0, 4, 16)
    n = 40 * 4 + 1
    per = np.full(n - 1, 0.5)
    per[32:64] = 0.33                                # eight bars at 182 BPM in a 120 BPM song
    t = 0.5 + np.r_[0.0, np.cumsum(per)]
    D = G.Src("d", "d", t, [1 if i % 4 == 0 else 0 for i in range(n)], t, 120.0, bpb=4, content=(0.5, float(t[-1])))
    proj, desc = G.build_project({"m": M, "d": D}, [dict(song="m", stems=["other"]), dict(song="d", stems=["drums"])])
    cl = clips_of(proj, "d", "drums")
    assert len(cl) > 1 and not any(w["r"] > G.RATE_SPAN[1] for c in cl for w in c.get("warps") or [])
    for k in range(16):                              # ... and every bar still starts on one of its bar lines
        c = next(c for c in cl if c["start"] <= 4 * k + 1e-6 < c["start"] + c["len"])
        assert float(np.min(np.abs(D.bar_t - src_time(D, proj, c, 4 * k)))) <= G.DRIFT + 2e-3


def test_fit_points_keeps_only_the_corners_it_needs():
    U = np.arange(0.0, 41.0, 4.0)
    assert G.fit_points(U, 0.5 * U, 0.015) == [0, 10]                     # a straight line: its two ends
    T = np.where(U <= 20, 0.5 * U, 10.0 + 0.52 * (U - 20))                 # one change of speed, at the 6th point
    assert G.fit_points(U, T, 0.005) == [0, 5, 10]


def test_a_layer_that_is_the_backing_plays_its_bars_in_its_own_order_start_to_end():
    secs = [dict(t0=0.5 + 16 * k * 0.5, t1=0.5 + 16 * (k + 1) * 0.5, name=nm, energy=e, stems={})
            for k, (nm, e) in enumerate([("intro", 0.2), ("chorus", 0.9), ("verse", 0.4), ("chorus", 0.9)])]
    M = song("m", 120.0, 4, 16, sections=secs, on=dict(drums=np.zeros(65, bool)))      # an acapella: no drums of its own
    lsecs = [dict(t0=0.5, t1=0.5 + 32 * 0.5, name="chorus", energy=0.9, stems={}),
             dict(t0=0.5 + 32 * 0.5, t1=0.5 + 160 * 0.5, name="verse", energy=0.3, stems={})]
    quiet = np.ones(161, bool)
    quiet[8:16] = False                              # the band stops for two bars
    L = song("l", 120.0, 4, 40, sections=lsecs, on=dict(drums=quiet, bass=quiet, other=quiet))
    proj, _ = G.build_project({"m": M, "l": L}, [dict(song="m", stems=["vocals"]), dict(song="l", stems=["drums", "bass", "other"])],
                              master="m")
    for stem in ("drums", "bass", "other"):
        cl = clips_of(proj, "l", stem)
        covered = lambda u: next((c for c in cl if c["start"] <= u + 1e-6 < c["start"] + c["len"]), None)
        for k in (0, 1, 4, 9, 15):                   # master bar k plays the band's bar k: no jumps at the sections,
            assert abs(src_time(L, proj, covered(4 * k), 4 * k) - (0.5 + 4 * k * 0.5)) < 1e-3
        assert covered(8.0) is None and covered(12.0) is None           # ... and its own stop stays a stop


def test_drums_under_the_masters_music_still_jump_to_a_section_of_matching_energy():
    secs = [dict(t0=0.5, t1=0.5 + 32 * 0.5, name="verse", energy=0.2, stems={}),
            dict(t0=0.5 + 32 * 0.5, t1=0.5 + 64 * 0.5, name="chorus", energy=0.9, stems={})]
    M = song("m", 120.0, 4, 16, sections=secs)
    dsecs = [dict(t0=0.5, t1=0.5 + 64 * 0.5, name="verse", energy=0.2, stems={}),
             dict(t0=0.5 + 64 * 0.5, t1=0.5 + 160 * 0.5, name="chorus", energy=0.9, stems={})]
    D = song("d", 120.0, 4, 40, sections=dsecs)
    proj, _ = G.build_project({"m": M, "d": D}, [dict(song="m", stems=["bass", "other"]), dict(song="d", stems=["drums"])])
    cl = clips_of(proj, "d", "drums")
    c = next(c for c in cl if c["start"] <= 32 + 1e-6 < c["start"] + c["len"])
    assert abs(src_time(D, proj, c, 32.0) - (0.5 + 64 * 0.5)) < 1e-3      # the master's chorus plays the drums' chorus


def test_a_long_vocal_line_follows_the_masters_tempo():
    n = 24 * 4 + 1
    per = 0.5 * (1.0 + 0.02 * np.sin(np.arange(n - 1) / 25.0))            # the master wanders +-2 %
    t = 0.5 + np.r_[0.0, np.cumsum(per)]
    M = G.Src("m", "m", t, [1 if i % 4 == 0 else 0 for i in range(n)], t, 120.0, bpb=4, content=(0.5, float(t[-1])))
    V = song("v", 120.0, 4, 40, phrases=[(0.5, 0.5 + 60 * 0.5, "fifteen bars without a breath")])
    proj, desc = G.build_project({"m": M, "v": V}, [dict(song="m", stems=["drums", "other"]), dict(song="v", stems=["vocals"])])
    c = clips_of(proj, "v", "vocals")[0]
    assert c.get("warps") and all(G.VOX_SPAN[0] < w["r"] < G.VOX_SPAN[1] for w in c["warps"])
    grid = G.Grid(M, 1.0)
    for k in range(1, 15):                           # each of the singer's bar lines on the master's, where it really is
        u = float(grid.G[k])
        assert abs(src_time(V, proj, c, u) - (0.5 + 4 * k * 0.5)) <= G.VOX_TOL + 2e-3
    plain, _ = G.build_project({"m": M, "v": V}, [dict(song="m", stems=["drums", "other"]), dict(song="v", stems=["vocals"])],
                               opts=dict(follow="cut"))
    assert not clips_of(plain, "v", "vocals")[0].get("warps")


def test_a_line_the_master_does_not_pull_away_from_is_left_exactly_as_recorded():
    M = song("m", 120.0, 4, 24)
    V = song("v", 120.0, 4, 40, phrases=[(0.5, 0.5 + 40 * 0.5, "ten bars")])
    proj, _ = G.build_project({"m": M, "v": V}, [dict(song="m", stems=["drums", "other"]), dict(song="v", stems=["vocals"])])
    assert not clips_of(proj, "v", "vocals")[0].get("warps")


def _two_singers():
    bar = 4 * 0.5
    secs = [dict(t0=0.5 + 8 * k * bar, t1=0.5 + 8 * (k + 1) * bar, name=nm, energy=e, stems={})
            for k, (nm, e) in enumerate([("verse", 0.4), ("chorus", 0.9), ("verse", 0.4), ("chorus", 0.9)])]
    M = song("m", 120.0, 4, 32, sections=secs)
    line = lambda b0, bars, words, rep=1: (0.5 + b0 * bar, 0.5 + (b0 + bars) * bar - 0.3, words, rep)
    # A: a verse (bars 0-5), then its hook twice (bars 8-13), then another verse (bars 16-21)
    A = song("a", 120.0, 4, 40, phrases=[line(0, 2, "a verse one"), line(2, 2, "a verse two"), line(4, 2, "a verse three"),
                                         line(8, 2, "hold on tight", 4), line(10, 2, "hold on tight", 4), line(12, 2, "hold on tight", 4),
                                         line(16, 2, "a later verse"), line(18, 2, "a later verse two"), line(20, 2, "a later verse three")])
    # B: three verses, no hook
    B = song("b", 120.0, 4, 60, phrases=[line(0, 2, "b one"), line(2, 2, "b two"), line(4, 2, "b three"),
                                         line(10, 2, "b four"), line(12, 2, "b five"), line(14, 2, "b six"),
                                         line(20, 2, "b seven"), line(22, 2, "b eight")])
    return M, A, B


def test_two_singers_take_turns_one_gets_the_choruses_the_other_the_verses():
    M, A, B = _two_singers()
    picks = [dict(song="m", stems=["drums", "bass", "other"]), dict(song="a", stems=["vocals"]), dict(song="b", stems=["vocals"])]
    proj, desc = G.build_project({"m": M, "a": A, "b": B}, picks)
    part = {x["song"]: x.get("part") for x in desc["layers"]}
    assert part["a"] == "chorus" and part["b"] == "verses"
    sec_of = lambda c: int((c["start"] + 0.5 * c["len"]) // 32)           # 8-bar sections of 32 beats
    a, b = clips_of(proj, "a", "vocals"), clips_of(proj, "b", "vocals")
    assert a and b
    assert {sec_of(c) for c in a} == {1, 3}                               # the hook, in both choruses
    assert all("hold on tight" in c["line"] for c in a)
    assert {sec_of(c) for c in b} == {0, 2}                               # the other singer's verses, in order
    assert [c["line"] for c in b][:3] == ["b one", "b two", "b three"] and b[3]["line"] == "b four"
    allv = sorted(a + b, key=lambda c: c["start"])
    assert all(y["start"] >= x["start"] + x["len"] - 1e-6 for x, y in zip(allv, allv[1:]))


def test_turns_off_deals_the_blocks_out_back_to_back():
    M, A, B = _two_singers()
    picks = [dict(song="m", stems=["drums", "bass", "other"]), dict(song="a", stems=["vocals"]), dict(song="b", stems=["vocals"])]
    proj, desc = G.build_project({"m": M, "a": A, "b": B}, picks, opts=dict(turns=False))
    assert not any("part" in x for x in desc["layers"])
    first = sorted((c for c in proj["clips"] if c["stem"] == "vocals"), key=lambda c: c["start"])[:4]
    assert [c["song"] for c in first] == ["a", "a", "a", "b"]             # a's first block, then b's


def test_a_drum_layer_plays_only_where_the_masters_own_drums_did():
    n = 16 * 4 + 1
    drums = np.ones(n, bool)
    drums[16:32] = False                             # the master's breakdown: bars 5-8
    M = song("m", 120.0, 4, 16, on=dict(drums=drums))
    D = song("d", 120.0, 4, 40)
    proj, _ = G.build_project({"m": M, "d": D}, [dict(song="m", stems=["bass", "other"]), dict(song="d", stems=["drums"])])
    cl = clips_of(proj, "d", "drums")
    covered = lambda u: any(c["start"] <= u < c["start"] + c["len"] for c in cl)
    assert covered(2.0) and covered(40.0) and not covered(18.0) and not covered(30.0)


def test_a_line_keeps_its_pickup_and_lands_with_its_bar_line_on_a_master_bar_line():
    M = song("m", 120.0, 4, 24)
    beat = 60.0 / 100.0
    V = song("v", 100.0, 4, 40, phrases=[(0.5 + 15 * beat, 0.5 + 22 * beat, "one"),      # a 1-beat pickup into bar 5
                                         (0.5 + 24 * beat, 0.5 + 30 * beat, "two")])     # then one starting on a bar line
    proj, desc = G.build_project({"m": M, "v": V}, [dict(song="m", stems=["drums", "bass", "other"]), dict(song="v", stems=["vocals"])])
    a, b = clips_of(proj, "v", "vocals")
    for c, line_t in ((a, 0.5 + 16 * beat), (b, 0.5 + 24 * beat)):
        u = c["start"] + (line_t - float(P._beat_to_time(c["src"], V.ib))) / ((60.0 / 100.0) / proj["songs"]["v"]["mult"])
        assert abs(u / 4 - round(u / 4)) < 1e-6      # its bar line sits on one of the master's
    assert abs((b["start"] - a["start"]) - 9.0) < 1e-6    # ... and the two keep their own spacing (2 bars + the pickup)
    assert a["len"] == round(7 * 1.0, 9) and next(x for x in desc["layers"] if x["song"] == "v")["lines"] == 2


def test_a_line_that_would_run_past_the_end_is_left_out_whole():
    M = song("m", 120.0, 4, 6)
    beat = 0.5
    V = song("v", 120.0, 4, 40, phrases=[(0.5, 0.5 + 12 * beat, "fits"), (0.5 + 16 * beat, 0.5 + 40 * beat, "too long")])
    proj, desc = G.build_project({"m": M, "v": V}, [dict(song="m", stems=["other"]), dict(song="v", stems=["vocals"])])
    cl = clips_of(proj, "v", "vocals")
    lay = next(x for x in desc["layers"] if x["song"] == "v")
    assert len(cl) == 1 and cl[0]["len"] == 12.0 and lay["left_out"] == 1


def test_in_another_meter_every_line_is_put_on_a_bar_line_and_none_overlap():
    M = song("m", 120.0, 4, 40)
    beat = 60.0 / 126.0
    lines = [(0.5 + (6 * k) * beat, 0.5 + (6 * k + 5) * beat, f"l{k}") for k in range(6)]     # 5 beats long, every 2 waltz bars
    V = song("v", 126.0, 3, 40, phrases=lines)
    proj, _ = G.build_project({"m": M, "v": V}, [dict(song="m", stems=["drums", "other"]), dict(song="v", stems=["vocals"])])
    cl = clips_of(proj, "v", "vocals")
    assert len(cl) == 6
    for c in cl:
        assert abs(c["start"] / 4 - round(c["start"] / 4)) < 1e-6
    assert all(b["start"] >= a["start"] + a["len"] for a, b in zip(cl, cl[1:]))


def test_a_first_line_with_a_pickup_before_bar_one_moves_the_whole_mix_a_bar_later():
    M = song("m", 120.0, 4, 16)
    V = song("v", 120.0, 4, 20, start=2.5, phrases=[(2.0, 5.0, "pickup")])        # starts a beat before its bar line
    proj, _ = G.build_project({"m": M, "v": V}, [dict(song="m", stems=["other"]), dict(song="v", stems=["vocals"])])
    v = clips_of(proj, "v", "vocals")[0]
    m = clips_of(proj, "m", "other")[0]
    assert m["start"] == 4.0 and abs(v["start"] - 3.0) < 1e-6 and min(c["start"] for c in proj["clips"]) >= 0


def test_guest_vocals_stay_out_of_the_bars_the_masters_own_singer_is_in():
    n = 16 * 4 + 1
    vox = np.zeros(n, bool)
    vox[0:32] = True                                 # the master sings bars 1-8
    M = song("m", 120.0, 4, 16, on=dict(vocals=vox))
    V = song("v", 120.0, 4, 30, phrases=[(0.5, 0.5 + 7 * 0.5, "a")])
    proj, _ = G.build_project({"m": M, "v": V}, [dict(song="m", stems=["other", "vocals"]), dict(song="v", stems=["vocals"])])
    assert clips_of(proj, "v", "vocals")[0]["start"] >= 32.0


def test_marks_counted_at_half_the_stored_tempo_still_play_at_the_songs_own_speed():
    M = song("m", 100.0, 4, 16)
    D = song("d", 50.0, 4, 12, info_bpm=100.0)       # the user's marks: every other stored beat
    assert D.ratio == 0.5
    proj, desc = G.build_project({"m": M, "d": D}, [dict(song="m", stems=["other"]), dict(song="d", stems=["drums"])])
    lay = next(x for x in desc["layers"] if x["song"] == "d")
    assert abs(lay["stretch"]) < 0.1 and abs(proj["songs"]["d"]["mult"] - 1.0) < 1e-9     # half-time marks, x2 = as recorded


def test_the_song_bringing_bass_and_chords_is_the_master_unless_told_otherwise():
    songs = {"a": song("a", 120.0, 4, 8), "b": song("b", 90.0, 4, 8)}
    picks = [dict(song="a", stems=["drums"]), dict(song="b", stems=["bass", "other", "vocals"])]
    assert G.build_project(songs, picks)[1]["master"] == "b"
    proj, desc = G.build_project(songs, picks, master="a")
    assert desc["master"] == "a" and proj["bpm"] == 120.0


def test_the_result_is_an_ordinary_project_with_a_lane_per_stem():
    M = song("m", 120.0, 4, 8)
    D = song("d", 120.0, 4, 20)
    proj, desc = G.build_project({"m": M, "d": D}, [dict(song="m", stems=["drums", "bass", "other"]), dict(song="d", stems=["drums"])],
                                 gains={("d", "drums"): -3.2})
    ids = {t["id"] for t in proj["tracks"]}
    assert all(c["track"] in ids and c["len"] > 0 and c["start"] >= 0 for c in proj["clips"])
    names = [t["name"] for t in proj["tracks"]]
    assert "Drums · m" in names and "Drums · d" in names and len(names) == 5      # the four usual lanes + the second drums
    assert next(t for t in proj["tracks"] if t["name"] == "Drums · d")["gain_db"] == -3.2
    assert set(proj["songs"]) == {"m", "d"} and desc["kind"] == "grid" and proj["mashup"] is desc


def test_hits_are_measured_against_marks_and_a_song_with_no_clear_hits_says_nothing():
    sr = 44100
    marks = 0.5 + np.arange(64) * 0.5
    y = np.zeros(int(34 * sr), np.float32)
    for t in marks:                                  # every hit 10 ms after its mark
        i = int((t + 0.010) * sr)
        y[i:i + 400] += np.hanning(800)[400:] * 0.8
    assert abs(G.hits_vs_marks(y, sr, marks) - 0.010) < 0.0015
    rng = np.random.default_rng(1)
    assert G.hits_vs_marks(rng.standard_normal(len(y)).astype(np.float32) * 0.1, sr, marks) is None


def test_songs_are_lined_up_hit_on_hit_not_mark_on_mark():
    M = song("m", 120.0, 4, 16)
    D = song("d", 120.0, 4, 40)
    M.hit, D.hit = -0.009, 0.013                     # this song's drums fall 22 ms later after its marks than the master's
    proj, desc = G.build_project({"m": M, "d": D}, [dict(song="m", stems=["other"]), dict(song="d", stems=["drums"])])
    c = clips_of(proj, "d", "drums")[0]
    t = src_time(D, proj, c, 8.0)                    # the master's 3rd bar line plays the layer 22 ms past its own mark,
    assert abs((t - float(D.bt[np.argmin(np.abs(D.bt - t))])) - 0.022) < 1e-3     # so its hit lands where the master's does
    assert next(x for x in desc["layers"] if x["song"] == "d")["nudge_ms"] == -22.0


def test_lines_that_run_straight_on_stay_one_clip_even_in_another_meter():
    M = song("m", 120.0, 4, 40)
    beat = 60.0 / 126.0
    V = song("v", 126.0, 3, 60, phrases=[(0.5, 0.5 + 20 * beat, "a long stretch"), (0.5 + 20 * beat + 0.04, 0.5 + 41 * beat, "that carries on"),
                                         (0.5 + 50 * beat, 0.5 + 55 * beat, "then a new line")])
    assert [x[2] for x in G.lines_of(V)] == ["a long stretch that carries on", "then a new line"]
    assert len(G.lines_of(V)[0][3]) == 2                      # ... and remembers where it was joined
    proj, _ = G.build_project({"m": M, "v": V}, [dict(song="m", stems=["drums", "other"]), dict(song="v", stems=["vocals"])])
    cl = clips_of(proj, "v", "vocals")
    assert len(cl) == 2 and abs(cl[0]["len"] - 41 * beat / (60.0 / 126.0 / proj["songs"]["v"]["mult"])) < 1e-6


def _train(times, dur, step=0.004):
    e = np.zeros(int(dur / step))
    for t in times:
        if 0 <= t < dur - step:
            e[int(round(t / step))] += 1.0
    return e


def test_music_on_one_tempo_is_told_from_music_that_changes_tempo():
    dur = 240.0
    steady = np.arange(0.3, dur, 60.0 / 120.03)
    bpm, share, cover, snr = G.steady_fit(_train(steady, dur), 0.004, 120.6)       # the grid's own guess is half a BPM off
    assert abs(bpm - 120.03) < 0.01 and share > 0.9 and cover > 0.9 and snr > G.STEADY_SNR
    two = np.r_[np.arange(0.3, 120.0, 0.5), np.arange(120.0, dur, 60.0 / 123.0)]      # 120 BPM, then 123
    _, _, cover2, _ = G.steady_fit(_train(two, dur), 0.004, 120.0, main=(0.0, 120.0))
    assert cover2 < 0.2                                                      # the other half has no pulse at 120
    slip = np.r_[np.arange(0.3, 120.0, 0.5), np.arange(120.0 + 0.19, dur, 0.5)]       # same tempo, but it slips 190 ms
    _, share3, _, _ = G.steady_fit(_train(slip, dur), 0.004, 120.0)
    assert share3 < G.STEADY_SHARE
    gap = np.r_[np.arange(0.3, 80.0, 0.5), np.arange(160.3, dur, 0.5)]      # a breakdown with no pulse at all: still steady
    _, share4, cover4, _ = G.steady_fit(_train(gap, dur), 0.004, 120.0)
    assert share4 > 0.9 and cover4 > 0.9


def test_a_steady_grid_runs_through_the_stretch_that_is_counted_right_and_keeps_its_bar_lines():
    good = 20.0 + np.arange(200) * 0.5                                       # 20 s .. 119.5 s, counted right
    beats = np.r_[np.arange(0.4, 20.0, 0.73), good, 120.2 + np.arange(40) * 0.61]
    a = int(np.flatnonzero(beats == 20.0)[0])
    downs = good[2::4]                                                       # bar lines on 21.0, 23.0, ...
    marks, agree = G.steady_marks(beats, downs, 4, a, a + 199, 120.0, 0.0, 150.0)
    t = np.array(marks["t"])
    assert agree < 1e-6 and np.allclose(np.diff(t), 0.5) and t[0] < 0.5 and t[-1] > 149.4
    assert all(abs((x - 21.0) / 2.0 - round((x - 21.0) / 2.0)) < 1e-6 for x in t[np.flatnonzero(marks["d"])])


def test_stretches_counted_in_half_time_are_re_counted_on_the_pulse_of_the_stretch_next_to_them():
    half = np.r_[0.5 + np.arange(80) * 0.5, 40.5 + np.arange(40) * 1.0, 80.5 + np.arange(80) * 0.5]
    segs, main = G.stretches(half)
    assert [x[3] for x in segs] == [("count", 1.0), ("count", 0.5), ("count", 1.0)]
    new = G.tidy(half)
    assert np.allclose(np.diff(new), 0.5, atol=1e-6) and len(new) == 80 + 80 + 80
    assert G.needs_tidy(half) == (1, 0)
    # a song counted in half time all the way through has nothing to re-count: its own count stands
    assert G.needs_tidy(0.5 + np.arange(200) * 1.0) == (0, 0)


def test_a_real_tempo_change_is_kept_and_a_stretch_with_no_usable_count_gets_even_beats_at_its_neighbours_tempo():
    # 120 BPM for 100 s, 4 against 3 for 15 s (the detector lost it), then the song really speeds up to 138
    lost = 100.5 + np.arange(1, 41) * 0.375
    fast = lost[-1] + np.arange(1, 121) * (60.0 / 138.0)
    beats = np.r_[0.5 + np.arange(201) * 0.5, lost, fast]
    segs, main = G.stretches(beats)
    assert [x[3][0] for x in segs] == ["count", "even", "keep"] and main == 0
    assert G.needs_tidy(beats) == (0, 1)
    new = G.tidy(beats)
    mid = np.diff(new[(new > 101.0) & (new < 115.0)])
    assert np.allclose(mid, 0.5, atol=0.01)                                   # plays on at 120 through the lost stretch
    assert np.allclose(np.diff(new[new > 117.0]), 60.0 / 138.0, atol=1e-6)    # and the faster part is still its own tempo
    assert np.allclose(np.diff(new[new < 100.0]), 0.5, atol=1e-6)


def test_a_dropped_beat_does_not_shift_the_bar_lines_after_it():
    t = 0.5 + np.arange(160) * 0.5
    downs = np.r_[t[0:80:4], t[81:160:4]]             # after 40 s the old bar lines sit one beat later
    d = np.array(G.bar_flags(t, downs, 4, (0.5, 40.0), trusted=[(40.5, 80.0)]))
    assert list(np.flatnonzero(d)[:3]) == [0, 4, 8] and all(i % 4 == 1 for i in np.flatnonzero(d) if i > 81)


def test_calm_ignores_a_stray_mark_but_keeps_a_slide_and_a_jump():
    ramp = np.arange(20) * 0.004
    assert np.allclose(G.calm(ramp), ramp)
    stray = ramp.copy()
    stray[9] += 0.08
    assert abs(G.calm(stray)[9] - ramp[9]) < 0.005
    jump = np.r_[np.zeros(10), np.full(10, 0.3)]
    assert np.allclose(G.calm(jump), jump)


def test_rough_grids_are_followed_loosely_so_the_layer_does_not_wobble():
    rng = np.random.default_rng(7)
    def rough(sid, bpm, bars, sd):
        n = bars * 4 + 1
        t = 0.5 + np.arange(n) * 60.0 / bpm + rng.normal(0, sd, n)          # every mark a little off, each differently
        return G.Src(sid, sid, t, [1 if i % 4 == 0 else 0 for i in range(n)], 0.5 + np.arange(n) * 60.0 / bpm, bpm, bpb=4,
                     content=(0.5, 0.5 + (n - 1) * 60.0 / bpm))
    M, D = rough("m", 120.0, 48, 0.012), rough("d", 120.0, 60, 0.012)
    assert 0.005 < M.noise < 0.03
    proj, desc = G.build_project({"m": M, "d": D}, [dict(song="m", stems=["other"]), dict(song="d", stems=["drums"])])
    lay = next(x for x in desc["layers"] if x["song"] == "d")
    cl = clips_of(proj, "d", "drums")
    assert lay["reach_ms"] > G.DRIFT * 1000 and lay["cuts"] == 0
    w = [x for c in cl for x in c.get("warps") or []]
    assert len(w) <= 8 and all(abs(x["r"] - 1.0) < 0.006 and x["kp"] is False for x in w)     # both are steady: nothing to chase


def test_a_voice_is_never_made_to_lurch_to_follow_a_master_grid_that_jumps_about():
    n = 24 * 4 + 1
    per = np.full(n - 1, 0.5)
    per[16:20] = 0.6                                 # one bar of the master's grid is 20 % long (a bad stretch of marks)
    t = 0.5 + np.r_[0.0, np.cumsum(per)]
    M = G.Src("m", "m", t, [1 if i % 4 == 0 else 0 for i in range(n)], t, 120.0, bpb=4, content=(0.5, float(t[-1])))
    V = song("v", 120.0, 4, 40, phrases=[(0.5, 0.5 + 60 * 0.5, "fifteen bars without a breath")])
    proj, _ = G.build_project({"m": M, "v": V}, [dict(song="m", stems=["drums", "other"]), dict(song="v", stems=["vocals"])])
    c = clips_of(proj, "v", "vocals")[0]
    assert all(G.VOX_SPAN[0] <= w["r"] <= G.VOX_SPAN[1] for w in c.get("warps") or [])


def test_every_stem_a_song_brings_is_stretched_the_same_way_at_the_same_moments():
    M = song("m", 120.0, 4, 48)
    n = 60 * 4 + 1
    per = 0.5 * (1.0 + 0.015 * np.sin(np.arange(n - 1) / 30.0))
    t = 0.5 + np.r_[0.0, np.cumsum(per)]
    bass_on = np.ones(n, bool)
    bass_on[40:72] = False                           # the bass sits out bars 11-18
    L = G.Src("l", "l", t, [1 if i % 4 == 0 else 0 for i in range(n)], t, 120.0, bpb=4, content=(0.5, float(t[-1])),
              on=dict(bass=bass_on))
    proj, _ = G.build_project({"m": M, "l": L}, [dict(song="m", stems=["vocals"]), dict(song="l", stems=["drums", "bass", "other"])],
                              master="m")
    at = lambda stem, u: next((src_time(L, proj, c, u) for c in clips_of(proj, "l", stem) if c["start"] <= u < c["start"] + c["len"]), None)
    assert at("bass", 50.0) is None and len(clips_of(proj, "l", "bass")) == 2
    for u in (3.0, 20.5, 37.9, 80.0, 121.25, 170.0):
        ref = at("drums", u)
        for stem in ("bass", "other"):
            assert abs(at(stem, u) - ref) < 2e-4, (stem, u)


def test_a_verse_too_long_for_its_section_carries_on_at_that_singers_next_turn():
    M, A, _ = _two_singers()
    bar = 2.0
    B = song("b", 120.0, 4, 60, phrases=[(0.5 + 2 * k * bar, 0.5 + (2 * k + 2) * bar - 0.3, f"b{k + 1}", 1) for k in range(6)])   # 12 bars, no break
    picks = [dict(song="m", stems=["drums", "bass", "other"]), dict(song="a", stems=["vocals"]), dict(song="b", stems=["vocals"])]
    proj, desc = G.build_project({"m": M, "a": A, "b": B}, picks)
    b = clips_of(proj, "b", "vocals")
    sec_of = lambda c: int((c["start"] + 0.5 * c["len"]) // 32)
    assert [c["line"] for c in b if sec_of(c) == 0] == ["b1", "b2", "b3", "b4"]       # what fits the first verse
    assert [c["line"] for c in b if sec_of(c) == 2] == ["b5", "b6"]                   # the rest, in the next one
    assert next(x for x in desc["layers"] if x["song"] == "b")["left_out"] == 0
    assert {sec_of(c) for c in clips_of(proj, "a", "vocals")} == {1, 3}


def test_sections_of_one_kind_in_a_row_are_one_stretch_for_the_singers():
    bar = 2.0
    names = ["verse", "verse", "chorus", "chorus", "verse", "chorus"]              # 4-bar sections
    secs = [dict(t0=0.5 + 4 * k * bar, t1=0.5 + 4 * (k + 1) * bar, name=nm, energy=0.5, stems={}) for k, nm in enumerate(names)]
    M = song("m", 120.0, 4, 24, sections=secs)
    line = lambda b0, bars, words, rep=1: (0.5 + b0 * bar, 0.5 + (b0 + bars) * bar - 0.3, words, rep)
    A = song("a", 120.0, 4, 40, phrases=[line(0, 2, "hook", 3), line(2, 2, "hook", 3), line(4, 2, "hook", 3)])     # a 6-bar hook
    B = song("b", 120.0, 4, 40, phrases=[line(0, 2, "b1"), line(2, 2, "b2"), line(4, 2, "b3")])                    # a 6-bar verse
    picks = [dict(song="m", stems=["drums", "bass", "other"]), dict(song="a", stems=["vocals"]), dict(song="b", stems=["vocals"])]
    proj, _ = G.build_project({"m": M, "a": A, "b": B}, picks)
    a, b = clips_of(proj, "a", "vocals"), clips_of(proj, "b", "vocals")
    assert [c["line"] for c in b] == ["b1", "b2", "b3"] and b[-1]["start"] + b[-1]["len"] <= 32.0      # all of it, across both verse sections
    assert len(a) >= 3 and all(32.0 <= c["start"] < 64.0 for c in a[:3])                               # the hook, across both chorus sections


def _alternating(sid, bars, long=1.12):
    """A song whose every second bar is 12 % long (an odd meter the grid counts in 4)."""
    per = np.concatenate([np.full(4, 0.5 * (long if k % 2 else 1.0)) for k in range(bars)])
    t = 0.5 + np.r_[0.0, np.cumsum(per)]
    return G.Src(sid, sid, t, [1 if i % 4 == 0 else 0 for i in range(len(t))], t, 120.0, bpb=4, content=(0.5, float(t[-1])))


def test_a_layer_whose_bars_alternate_long_and_short_keeps_an_even_speed_instead_of_lurching_bar_by_bar():
    M = song("m", 120.0, 4, 40)
    D = _alternating("d", 60)
    proj, desc = G.build_project({"m": M, "d": D}, [dict(song="m", stems=["other"]), dict(song="d", stems=["drums"])])
    cl = clips_of(proj, "d", "drums")
    w = [x for c in cl for x in c.get("warps") or []]
    assert len(cl) == 1 and len(w) <= 3 and all(abs(x["r"] - 1.06) < 0.02 for x in w)      # one speed: the average
    for k in range(0, 40, 2):                        # and every second master bar line meets one of its bar lines
        assert float(np.min(np.abs(D.bar_t - src_time(D, proj, cl[0], 4 * k)))) <= 0.03
    assert next(x for x in desc["layers"] if x["song"] == "d")["cuts"] == 0


def test_a_lasting_change_of_tempo_is_still_followed_closely():
    M = song("m", 120.0, 4, 48)
    n = 60 * 4 + 1
    per = np.full(n - 1, 0.5)
    per[96:] = 0.56                                  # from bar 25 the song is 12 % slower, for good
    t = 0.5 + np.r_[0.0, np.cumsum(per)]
    D = G.Src("d", "d", t, [1 if i % 4 == 0 else 0 for i in range(n)], t, 120.0, bpb=4, content=(0.5, float(t[-1])))
    proj, _ = G.build_project({"m": M, "d": D}, [dict(song="m", stems=["other"]), dict(song="d", stems=["drums"])])
    cl = clips_of(proj, "d", "drums")
    assert len(cl) == 1
    for k in range(48):
        assert float(np.min(np.abs(D.bar_t - src_time(D, proj, cl[0], 4 * k)))) <= G.DRIFT + 2e-3


def test_only_the_bars_that_cannot_be_stretched_are_cut_the_rest_of_the_run_still_follows():
    M = song("m", 120.0, 4, 40)
    n = 60 * 4 + 1
    per = 0.5 * (1.0 + 0.012 * np.sin(np.arange(n - 1) / 40.0))
    per[80:84] = 0.5 * 0.55                           # one bar at nearly double speed (a bad stretch of marks)
    t = 0.5 + np.r_[0.0, np.cumsum(per)]
    D = G.Src("d", "d", t, [1 if i % 4 == 0 else 0 for i in range(n)], t, 120.0, bpb=4, content=(0.5, float(t[-1])))
    proj, desc = G.build_project({"m": M, "d": D}, [dict(song="m", stems=["other"]), dict(song="d", stems=["drums"])])
    cl = clips_of(proj, "d", "drums")
    assert 2 <= len(cl) <= 4 and sum(1 for c in cl if c.get("warps")) >= 1
    assert all(G.RATE_SPAN[0] <= x["r"] <= G.RATE_SPAN[1] for c in cl for x in c.get("warps") or [])


def test_a_long_joined_stretch_of_singing_stops_at_one_of_its_own_joins_when_the_section_ends():
    M, A, _ = _two_singers()                          # verse / chorus / verse / chorus, 8 bars each
    bar = 2.0
    # B: twelve bars of singing with no real breath - but it eases off after bar 6 (where the planner had split it)
    B = song("b", 120.0, 4, 60, phrases=[(0.5, 0.5 + 6 * bar - 0.1, "first half", 1), (0.5 + 6 * bar, 0.5 + 12 * bar - 0.3, "second half", 1)])
    picks = [dict(song="m", stems=["drums", "bass", "other"]), dict(song="a", stems=["vocals"]), dict(song="b", stems=["vocals"])]
    proj, desc = G.build_project({"m": M, "a": A, "b": B}, picks)
    b = clips_of(proj, "b", "vocals")
    assert len(b) == 2 and next(x for x in desc["layers"] if x["song"] == "b")["part"] == "verses"
    assert b[0]["start"] < 1.0 and abs(src_time(B, proj, b[0], b[0]["start"] + b[0]["len"] - 0.01) - (0.5 + 6 * bar - 0.1)) < 0.3   # stops at the join
    assert 64.0 <= b[1]["start"] < 68.0 and abs(src_time(B, proj, b[1], b[1]["start"] + 0.01) - (0.5 + 6 * bar)) < 0.3            # and carries on in verse 2
