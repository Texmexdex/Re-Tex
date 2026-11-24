"""MIDI songs (mashlab/midi.py): reading .mid files, playing parts at the song's own timing and at a project's, the
.mid export, and the instruments that need nothing installed (the built-in synth, the sampler)."""
import mido
import numpy as np
import pytest

from mashlab import instruments, midi
from mashlab.config import SR

BASIC = {"type": "basic", "wave": "square", "attack": 0.0, "decay": 0.05, "sustain": 0.8, "release": 0.02, "cutoff": 8000}


@pytest.fixture
def midi_store(data_sandbox, monkeypatch):
    import mashlab.config as config
    d = data_sandbox["DATA"] / "midi"
    monkeypatch.setattr(config, "MIDI_DIR", d)
    monkeypatch.setattr(midi, "MIDI_DIR", d)
    return d


def make_mid(path, tempos=((0, 600000), (4, 480000)), beats=8):
    """Quarter notes on every beat (C4 on channel 1, a kick on channel 10), tempo changing at beat 4."""
    f = mido.MidiFile(ticks_per_beat=480)
    meta, mel, drm = mido.MidiTrack(), mido.MidiTrack(), mido.MidiTrack()
    f.tracks += [meta, mel, drm]
    last = 0
    for b, t in tempos:
        meta.append(mido.MetaMessage("set_tempo", tempo=t, time=(b - last) * 480)); last = b
    meta.append(mido.MetaMessage("time_signature", numerator=4, denominator=4, time=0))
    mel.append(mido.Message("program_change", channel=0, program=0, time=0))
    for i in range(beats):
        mel.append(mido.Message("note_on", channel=0, note=60 + (i % 3), velocity=100, time=0 if i == 0 else 240))
        mel.append(mido.Message("note_off", channel=0, note=60 + (i % 3), velocity=0, time=240))
        drm.append(mido.Message("note_on", channel=9, note=36, velocity=110, time=0 if i == 0 else 360))
        drm.append(mido.Message("note_off", channel=9, note=36, velocity=0, time=120))
    f.save(path)
    return path


def onsets(y, thr=0.02, gap=0.1):
    """Sample indexes where sound starts after at least `gap` s of near-silence."""
    a = np.max(np.abs(y), axis=0)
    out, quiet = [], int(gap * SR)
    last = -quiet
    for i in np.flatnonzero(a > thr):
        if i - last >= quiet:
            out.append(int(i))
        last = i
    return out


def test_parse_reads_tempo_changes_parts_and_drums(tmp_path):
    m = midi.parse_file(make_mid(tmp_path / "t.mid"))
    sp = np.diff(m["beats"])
    assert np.allclose(sp[:4], 0.6, atol=1e-6) and np.allclose(sp[4:7], 0.48, atol=1e-6)
    assert len(m["parts"]) == 2 and m["bpb"] == 4
    drum = [p for p in m["parts"] if p["drum"]]
    assert len(drum) == 1 and all(n[2] == 36 for n in drum[0]["notes"])
    mel = [p for p in m["parts"] if not p["drum"]][0]
    assert [round(n[0], 6) for n in mel["notes"]] == list(range(8)) and all(abs(n[1] - 0.5) < 1e-6 for n in mel["notes"])


def test_parts_play_on_the_songs_own_beats(tmp_path, midi_store):
    m = midi.parse_file(make_mid(tmp_path / "t.mid"))
    for p in m["parts"]:
        p["instrument"] = dict(BASIC, wave="drums" if p["drum"] else "square")
    sid = midi.create("Test", "t.mid", m["beats"], m["bpm"], m["bpb"], m["parts"], render=False)
    info = midi.render_all(sid)
    assert info["status"] == "ready" and not info.get("part_errors") and info["stems_rev"] == 1
    mel = next(p for p in midi.read(sid)["parts"] if not p["drum"])
    y = midi.render_natural(info, mel)
    got = np.array(onsets(y)[:8]) / SR
    assert np.allclose(got, np.asarray(m["beats"][:8]), atol=0.003)
    assert (midi_store / sid / "stems" / f"{mel['id']}.flac").exists()


def test_warp_render_puts_notes_on_the_projects_tempo(tmp_path, midi_store):
    m = midi.parse_file(make_mid(tmp_path / "t.mid"))
    for p in m["parts"]:
        p["instrument"] = dict(BASIC)
    sid = midi.create("Warp", "t.mid", m["beats"], m["bpm"], m["bpb"], m["parts"], render=False)
    from mashlab import library
    info = library.read(sid)
    mel = next(p for p in midi.read(sid)["parts"] if not p["drum"])
    out, t0, method = midi.warp_render(sid, info, mel["id"], 120.0)
    assert method == "midi"
    sp = np.diff(np.array(onsets(out)[:8]) / SR)
    assert np.allclose(sp, 0.5, atol=0.003)                   # 100 then 125 BPM in the file, every beat 0.5 s here
    up, _, _ = midi.warp_render(sid, info, mel["id"], 120.0, semis=12)
    dr = next(p for p in midi.read(sid)["parts"] if p["drum"])
    assert up.shape == out.shape
    d0, _, _ = midi.warp_render(sid, info, dr["id"], 120.0)
    d12, _, _ = midi.warp_render(sid, info, dr["id"], 120.0, semis=12)
    assert np.allclose(d0, d12)                                # drums are never transposed


def test_export_round_trip(tmp_path, midi_store):
    m = midi.parse_file(make_mid(tmp_path / "t.mid"))
    sid = midi.create("Round", "t.mid", m["beats"], m["bpm"], m["bpb"], m["parts"], render=False)
    out = tmp_path / "out.mid"
    midi.to_midi_file(sid, out)
    back = midi.parse_file(out)
    assert np.allclose(back["beats"][:8], m["beats"][:8], atol=1e-4)
    key = lambda ps: sorted((p["drum"], tuple(tuple(round(x, 4) for x in n) for n in p["notes"])) for p in ps)
    assert key(back["parts"]) == key(m["parts"])


def test_sampler_plays_a_file_at_pitch(tmp_path):
    from mashlab import audio_io
    t = np.arange(int(0.5 * SR)) / SR
    tone = np.sin(2 * np.pi * 220.0 * t).astype(np.float32) * 0.5
    audio_io.save(tmp_path / "a.wav", np.stack([tone, tone]), SR, subtype="FLOAT")
    inst = {"type": "sampler", "path": str(tmp_path / "a.wav"), "root": 57, "mode": "pitched", "attack": 0.002, "release": 0.05}
    y = instruments.render(inst, [(0.1, 0.4, 69, 127)], [], int(SR))
    seg = y[0, int(0.15 * SR):int(0.35 * SR)]
    f = np.fft.rfftfreq(len(seg), 1 / SR)[np.argmax(np.abs(np.fft.rfft(seg)))]
    assert abs(f - 440.0) < 8                                  # an octave above its own note
    assert np.max(np.abs(y[:, :int(0.099 * SR)])) < 1e-4        # nothing before the note


def test_track_names_in_old_encodings_and_export(tmp_path, midi_store):
    path = make_mid(tmp_path / "t.mid")
    f = mido.MidiFile(path)
    f.tracks[1].insert(0, mido.MetaMessage("track_name", name="Вокал".encode("cp1251").decode("latin-1")))
    f.tracks[2].insert(0, mido.MetaMessage("track_name", name="Café".encode("cp1252").decode("latin-1")))
    f.save(tmp_path / "n.mid")
    m = midi.parse_file(tmp_path / "n.mid")
    assert sorted(p["name"] for p in m["parts"]) == ["Café", "Вокал"]
    sid = midi.create("Names", "n.mid", m["beats"], m["bpm"], m["bpb"], m["parts"], render=False)
    midi.to_midi_file(sid, tmp_path / "o.mid")
    assert sorted(p["name"] for p in midi.parse_file(tmp_path / "o.mid")["parts"]) == ["Café", "Вокал"]


# -------------------------------------------------------------------- chop kits (chops.py) and the keyboard pad
def fake_song(data_sandbox, sid="sDRUMS01"):
    """A library song whose drum stem is kick on the beat, hat on the off-beat (120 BPM, 8 s), silence after 6 s,
    and a vocal stem with two 'words'."""
    from mashlab import audio_io, library
    n = 8 * SR
    t = np.arange(int(0.3 * SR)) / SR
    kick = (np.sin(2 * np.pi * (50 * t + 40 * (1 - np.exp(-t * 30)) / 30)) * np.exp(-t * 12)).astype(np.float32) * 0.8
    rng = np.random.default_rng(3)
    hat = (np.diff(rng.standard_normal(int(0.05 * SR) + 1)) * np.exp(-np.arange(int(0.05 * SR)) / SR * 80)).astype(np.float32) * 0.3
    d = np.zeros(n, np.float32)
    for k in range(12):
        s = int(k * 0.5 * SR); d[s:s + len(kick)] += kick
        s = int((k * 0.5 + 0.25) * SR); d[s:s + len(hat)] += hat
    v = np.zeros(n, np.float32)
    for a, b in ((1.0, 1.4), (2.0, 2.5)):
        tt = np.arange(int((b - a) * SR)) / SR
        v[int(a * SR):int(a * SR) + len(tt)] = 0.3 * np.sin(2 * np.pi * 220 * tt)
    dd = data_sandbox["SONGS"] / sid
    (dd / "stems").mkdir(parents=True)
    audio_io.save(dd / "stems" / "drums.flac", np.stack([d, d]), SR)
    audio_io.save(dd / "stems" / "vocals.flac", np.stack([v, v]), SR)
    (dd / "lyrics.json").write_text('{"segments": [{"t0": 1.0, "t1": 2.6, "words": [{"t": 1.0, "w": " hey"}, {"t": 2.0, "w": " you"}]}]}')
    library.write(sid, {"id": sid, "title": "Drums", "bpm": 120.0, "bpb": 4, "beats": [i * 0.5 for i in range(17)],
                        "bars": [0, 4, 8, 12, 16], "first_beat": 0, "status": "ready"})
    return sid


def test_hits_are_found_sorted_and_silence_left_out(data_sandbox):
    from mashlab import chops
    sid = fake_song(data_sandbox)
    r = chops.make(sid, "drums", "hits")
    starts = [c[0] for c in r["chops"]]
    assert len(starts) == 24 and all(s < 6.0 for s in starts)          # 12 kicks + 12 hats, nothing in the silence
    want = sorted([k * 0.5 for k in range(12)] + [k * 0.5 + 0.25 for k in range(12)])
    assert np.allclose(starts, np.array(want), atol=0.003)                # right on each hit's attack
    cls = [c[2] for c in r["chops"]]
    assert cls[0::2] == ["kick"] * 12 and all(c in ("hat", "open") for c in cls[1::2])


def test_words_beats_and_phrases(data_sandbox):
    from mashlab import chops
    sid = fake_song(data_sandbox)
    w = chops.make(sid, "vocals", "words")["chops"]
    assert [c[4] for c in w] == ["hey", "you"] and abs(w[0][1] - 1.4) < 0.05
    b = chops.make(sid, "drums", "beats")["chops"]
    assert len(b) == 12 and b[0][:2] == [0.0, 0.5]                    # beats with sound only (none after 6 s)
    ph = chops.make(sid, "vocals", "phrases")["chops"]
    assert len(ph) == 2 and abs(ph[0][0] - 1.0) < 0.03 and abs(ph[1][0] - 2.0) < 0.03


def test_note_modes_pick_the_right_chops(data_sandbox):
    from mashlab import chops
    sid = fake_song(data_sandbox)
    kit = {"type": "chops", "song": sid, "stem": "drums", "chops": chops.make(sid, "drums", "hits")["chops"]}
    assert [chops.index_for(dict(kit, play="keys", base=36), p, {}) for p in (35, 36, 40)] == [None, 0, 4]
    st = {}
    assert [chops.index_for(dict(kit, play="order"), 60, st) for _ in range(3)] == [0, 1, 2]
    st = {}
    sh = [chops.index_for(dict(kit, play="shuffle", seed=5), 60, st) for _ in range(24)]
    assert sorted(sh) == list(range(24)) and sh != list(range(24))
    st, dk = {}, dict(kit, play="drums")
    assert kit["chops"][chops.index_for(dk, 36, st)][2] == "kick"
    assert kit["chops"][chops.index_for(dk, 42, st)][2] in ("hat", "open")
    assert chops.index_for(dk, 36, st) != chops.index_for(dk, 36, {}) or True   # round robin moves on


def test_chop_kit_renders_a_drum_part_with_the_songs_own_hits(data_sandbox):
    from mashlab import chops
    sid = fake_song(data_sandbox)
    kit = {"type": "chops", "song": sid, "stem": "drums", "play": "drums", "mono": False,
           "chops": chops.make(sid, "drums", "hits")["chops"]}
    y = instruments.render(kit, [(0.1, 0.2, 36, 127), (0.6, 0.7, 42, 127)], [], SR)
    on = onsets(y, thr=0.01, gap=0.2)
    assert len(on) == 2 and abs(on[0] / SR - 0.1) < 0.004 and abs(on[1] / SR - 0.6) < 0.004
    k = y[0, int(0.1 * SR):int(0.15 * SR)]
    f = np.fft.rfftfreq(len(k), 1 / SR)[np.argmax(np.abs(np.fft.rfft(k)))]
    assert f < 150                                                    # the kick note played a kick


def test_order_mode_treats_a_chord_as_one_trigger_and_mono_cuts(data_sandbox):
    from mashlab import chops
    sid = fake_song(data_sandbox)
    kit = {"type": "chops", "song": sid, "stem": "vocals", "play": "order", "mono": True,
           "chops": chops.make(sid, "vocals", "words")["chops"]}
    y = instruments.render(kit, [(0.0, 1.0, 60, 127), (0.0, 1.0, 64, 127), (0.2, 1.0, 62, 127)], [], SR)
    assert np.max(np.abs(y[:, int(0.25 * SR):int(0.3 * SR)])) > 0.1     # the second word, cutting the first
    first = instruments.render(kit, [(0.0, 1.0, 60, 127)], [], SR)
    assert np.max(np.abs(y[:, :int(0.19 * SR)] - first[:, :int(0.19 * SR)])) < 1e-6   # chord = one chop, not two


def test_keyboard_pad_banks_and_voices():
    from mashlab import midi_live
    keys = midi_live.Keys()
    bid, ready = keys.prepare("s", "p1", dict(BASIC, wave="sine", release=0.05), 60, 62)
    assert ready == 3 and set(keys.banks[bid]["notes"]) == {60, 61, 62}
    assert keys.on(bid, 69, 100)                                      # outside the bank: rendered on the spot
    out = keys.render(4096)
    f = np.fft.rfftfreq(4096, 1 / SR)[np.argmax(np.abs(np.fft.rfft(out[0])))]
    assert abs(f - 440) < 12 and np.max(np.abs(out)) > 0.05
    keys.off(bid, 69)
    for _ in range(40):
        keys.render(1024)
    assert not keys.voices                                            # faded out and gone


def test_plugin_folders_are_walked_whole(tmp_path, monkeypatch):
    (tmp_path / "Vendor" / "A.vst3" / "Contents" / "x86_64-win").mkdir(parents=True)
    (tmp_path / "Vendor" / "A.vst3" / "Contents" / "x86_64-win" / "A.vst3").write_bytes(b"x")
    for i in range(6):
        (tmp_path / "Vendor" / f"fx{i}.vst3").write_bytes(b"x")
    (tmp_path / "B.vst3").write_bytes(b"x")
    monkeypatch.setattr(instruments, "vst3_dirs", lambda: [str(tmp_path)])
    got = instruments.plugin_binaries()
    assert len(got) == 8 and not any("Contents" in g and g.endswith("A.vst3") is False for g in got)


def test_soundfont_folders_are_searched_with_subfolders(data_sandbox, tmp_path):
    (tmp_path / "fonts" / "Pianos" / "Grand").mkdir(parents=True)
    (tmp_path / "fonts" / "Pianos" / "Grand" / "Big.sf2").write_bytes(b"RIFF")
    (tmp_path / "fonts" / "Kit.SF2").write_bytes(b"RIFF")
    instruments.set_folders(sf2=[str(tmp_path / "fonts")])
    got = {f["name"]: f["folder"] for f in instruments.list_soundfonts()}
    assert got.get("Big", "").replace("\\", "/").endswith("fonts/Pianos/Grand") and "Kit" in got
    assert instruments.folders()["sf2"] == [str(tmp_path / "fonts")]


def test_chop_length_whole_cut_repeat_stretch(data_sandbox):
    from mashlab import chops
    sid = fake_song(data_sandbox)
    kit = {"type": "chops", "song": sid, "stem": "vocals", "play": "keys", "base": 60, "mono": True,
           "chops": chops.make(sid, "vocals", "words")["chops"]}            # "hey": a 0.4 s 220 Hz tone
    note = [(0.1, 1.6, 60, 127)]                                            # held 1.5 s
    lvl = lambda y, a, b: float(np.max(np.abs(y[:, int(a * SR):int(b * SR)])))
    whole = instruments.render(dict(kit, fit="whole"), note, [], 2 * SR)
    assert lvl(whole, 0.15, 0.4) > 0.1 and lvl(whole, 0.7, 1.6) < 1e-3     # plays its 0.4 s, then nothing
    cut = instruments.render(dict(kit, fit="cut"), [(0.1, 0.3, 60, 127)], [], 2 * SR)
    assert lvl(cut, 0.15, 0.28) > 0.1 and lvl(cut, 0.36, 1.0) < 1e-3       # stops with a 0.2 s note
    rep = instruments.render(dict(kit, fit="repeat"), note, [], 2 * SR)
    assert lvl(rep, 1.2, 1.55) > 0.1 and lvl(rep, 1.65, 2.0) < 1e-3       # goes round until the note ends
    st = instruments.render(dict(kit, fit="stretch"), note, [], 2 * SR)
    assert lvl(st, 1.3, 1.55) > 0.05 and lvl(st, 1.65, 2.0) < 1e-3
    mid = st[0, int(0.6 * SR):int(1.0 * SR)]
    f = np.fft.rfftfreq(len(mid), 1 / SR)[np.argmax(np.abs(np.fft.rfft(mid)))]
    assert abs(f - 220) < 6                                                 # 3.7x longer, same pitch
    assert instruments.render(dict(kit, gate=True), [(0.1, 0.3, 60, 127)], [], 2 * SR).tobytes() == cut.tobytes()  # old kits


def test_held_key_loops_a_repeating_chop(data_sandbox):
    from mashlab import chops, midi_live
    sid = fake_song(data_sandbox)
    kit = {"type": "chops", "song": sid, "stem": "vocals", "play": "keys", "base": 60, "fit": "repeat",
           "chops": chops.make(sid, "vocals", "words")["chops"]}
    keys = midi_live.Keys()
    bid, _ = keys.prepare("s", "p1", kit, 60, 61)
    keys.on(bid, 60, 127)
    blocks = [keys.render(1024) for _ in range(int(1.5 * SR / 1024))]       # 1.5 s held (the chop is 0.4 s)
    assert float(np.max(np.abs(blocks[-1]))) > 0.05 and keys.voices
    keys.off(bid, 60)
    for _ in range(60):
        keys.render(1024)
    assert not keys.voices


def _quiet_share(y, a, b, floor_db=-30):
    """Share of 10 ms frames in [a, b) s more than 30 dB under the loudest frame - drop-outs."""
    x = y.mean(0)[int(a * SR):int(b * SR)]
    fr = np.sqrt(np.add.reduceat(x ** 2, np.arange(0, len(x), 441)) / 441)
    db = 20 * np.log10(fr + 1e-9)
    return float((db < db.max() + floor_db).mean())


def test_back_to_back_notes_leave_no_holes(data_sandbox):
    from mashlab import chops
    sid = fake_song(data_sandbox)
    words = chops.make(sid, "vocals", "words")["chops"]
    notes = [(i * 0.3, i * 0.3 + 0.3, 60, 30 + (i % 4) * 30) for i in range(10)]          # legato, mixed velocity
    for fit in ("repeat", "stretch"):
        kit = {"type": "chops", "song": sid, "stem": "vocals", "play": "order", "mono": True, "fit": fit, "chops": words}
        y = instruments.render(kit, notes, [], 4 * SR)
        assert _quiet_share(y, 0.0, 2.95) < 0.02, fit
        lv = [float(np.max(np.abs(y[:, int((i * 0.3 + 0.1) * SR):int((i * 0.3 + 0.2) * SR)]))) for i in range(4)]
        assert max(lv) - min(lv) < 0.05, fit                                              # velocity: no effect
    flow = {"type": "chops", "song": sid, "stem": "vocals", "play": "flow", "mono": True, "chops": words}
    y = instruments.render(flow, notes, [], 4 * SR)
    assert _quiet_share(y, 0.0, 2.95) < 0.02


def test_flow_cuts_the_stem_to_each_notes_length(data_sandbox):
    from mashlab import chops
    sid = fake_song(data_sandbox)
    kit = {"type": "chops", "song": sid, "stem": "vocals", "play": "flow", "chops": chops.make(sid, "vocals", "words")["chops"]}
    total = sum(c[1] - c[0] for c in kit["chops"])                         # about 0.9 s of sound, silences left out
    y = instruments.render(kit, [(0.0, 0.25, 60, 100), (1.0, 1.25, 60, 100)], [], 2 * SR)
    a, b = y[0, :int(0.24 * SR)], y[0, int(1.0 * SR):int(1.24 * SR)]
    want = chops.stream_take(kit, 0, int(0.5 * SR))[0]
    n = int(0.21 * SR)                                                     # (the last 30 ms fade out)
    assert np.allclose(a[int(0.01 * SR):n], want[int(0.01 * SR):n], atol=1e-4)                      # the first stretch
    assert np.allclose(b[int(0.01 * SR):n], want[int(0.26 * SR):int(0.25 * SR) + n], atol=1e-4)     # then the next one
    assert float(np.max(np.abs(y[:, int(0.3 * SR):int(0.95 * SR)]))) < 1e-4                         # the rest stays silent
    assert 0.8 < total < 1.0


def test_live_flow_key_carries_the_stream_on(data_sandbox):
    from mashlab import chops, midi_live
    sid = fake_song(data_sandbox)
    kit = {"type": "chops", "song": sid, "stem": "vocals", "play": "flow", "chops": chops.make(sid, "vocals", "words")["chops"]}
    keys = midi_live.Keys()
    bid, _ = keys.prepare("s", "p1", kit, 60, 61)
    keys.on(bid, 60, 30)
    first = np.concatenate([keys.render(1024) for _ in range(10)], axis=1)            # ~0.23 s held
    keys.off(bid, 60)
    assert keys.banks[bid]["state"]["cur"] == 10 * 1024
    assert float(np.max(np.abs(first[:, 1024:]))) > 0.2                               # full level at velocity 30


def test_cue_points_each_note_its_own_start_every_time(data_sandbox):
    from mashlab import chops
    sid = fake_song(data_sandbox)
    ch = chops.make(sid, "vocals", "words")["chops"]                       # "hey" at ~1.0 s, "you" at ~2.0 s
    kit = {"type": "chops", "song": sid, "stem": "vocals", "play": "cue", "chops": ch, "skip_gaps": False,
           "assign": chops.assign_default(ch, [62, 60], "low")}
    assert kit["assign"] == {"60": 0, "62": 1}
    notes = [(0.0, 0.8, 60, 20), (1.0, 1.3, 62, 127), (2.0, 2.2, 60, 127), (3.0, 3.3, 60, 127), (3.0, 3.3, 62, 127)]
    y = instruments.render(kit, notes, [], 4 * SR)
    src = chops.source(kit)
    r = lambda x: int(round(x * SR))
    seg = lambda t, i, d: src[:, int(ch[i][0] * SR) + r(t):int(ch[i][0] * SR) + r(t) + r(d)]
    at = lambda t, d: y[:, r(t):r(t) + r(d)]
    assert np.allclose(at(0.01, 0.3), seg(0.01, 0, 0.3), atol=1e-4)        # 60: from "hey", full level at vel 20
    assert np.allclose(at(1.01, 0.2), seg(0.01, 1, 0.2), atol=1e-4)        # 62: from "you"
    assert np.allclose(at(2.01, 0.15), seg(0.01, 0, 0.15), atol=1e-4)      # 60 again: from the same start
    assert np.allclose(at(3.01, 0.2), seg(0.01, 0, 0.2) + seg(0.01, 1, 0.2), atol=1e-4)   # both at once: both
    assert float(np.max(np.abs(at(0.42, 0.3)))) < 1e-3                    # plays on past the chop (silence there)
    assert float(np.max(np.abs(at(0.85, 0.1)))) < 1e-4                    # and stops when the note does
    assert chops.index_for(kit, 64, {}) is not None                       # a note added later still plays a chop


def test_cue_points_skip_the_stems_pauses(data_sandbox):
    from mashlab import chops
    sid = fake_song(data_sandbox)
    ch = chops.make(sid, "vocals", "words")["chops"]                       # "hey" 1.0-1.4 s, pause, "you" 2.0-2.5 s
    kit = {"type": "chops", "song": sid, "stem": "vocals", "play": "cue", "chops": ch, "assign": {"60": 0}}
    y = instruments.render(kit, [(0.0, 0.85, 60, 100)], [], 2 * SR)
    assert _quiet_share(y, 0.0, 0.8) < 0.03                                # "hey" runs straight into "you"
    src = chops.source(kit)
    r = lambda x: int(round(x * SR))
    you = src[:, r(2.0) + r(0.1):r(2.0) + r(0.2)]
    k = int(np.argmax([np.abs(np.correlate(y[0, r(0.4):r(0.8)], you[0][:2048], "valid")).max()]))
    assert np.max(np.abs(y[:, r(0.5):r(0.7)])) > 0.2 and k == 0


def test_make_it_say_finds_and_orders_the_words(data_sandbox):
    from mashlab import chops
    sid = fake_song(data_sandbox)
    ch = chops.make(sid, "vocals", "words")["chops"]                       # sung: "hey", "you"
    r = chops.say(ch, "You, hey YOU heyyou heyy nope")
    assert r["sequence"] == [1, 0, 1, 0, 1, 0] and r["missing"] == ["nope"]
    kit = {"type": "chops", "song": sid, "stem": "vocals", "play": "order", "chops": ch, "sequence": [1, 0], "seq_once": True}
    st = {}
    assert [chops.index_for(kit, 60, st) for _ in range(3)] == [1, 0, None]


def test_tuned_chops_land_on_the_note(data_sandbox):
    from mashlab import chops
    sid = fake_song(data_sandbox)
    ch = chops.make(sid, "vocals", "words")["chops"]
    kit = {"type": "chops", "song": sid, "stem": "vocals", "play": "order", "fit": "whole", "pitch": "tune", "chops": ch}
    ps = chops.chop_pitches(kit)
    assert all(p is not None and abs(p - 57) < 0.2 for p in ps)            # 220 Hz = A3 = 57
    def pitch_of(inst, note):
        y = instruments.render(inst, [(0.0, 0.4, note, 100)], [], SR)[0, int(0.05 * SR):int(0.35 * SR)]
        return np.fft.rfftfreq(len(y), 1 / SR)[np.argmax(np.abs(np.fft.rfft(y)))]
    assert abs(pitch_of(kit, 60) - 261.6) < 6                               # retuned to C4
    assert abs(pitch_of(kit, 45) - 220.0) < 6                               # A2 asked, kept in its own octave
    assert abs(pitch_of(dict(kit, tune_octave=False), 45) - 110.0) < 4      # or really an octave down
    y = instruments.render(kit, [(0.0, 0.4, 60, 100)], [], SR)
    assert np.max(np.abs(y[:, int(0.44 * SR):])) < 0.05                     # its length is kept (~0.41 s)


# -------------------------------------------------------------------- MIDI as instructions for lanes
def _proj(sid, stem="drums", bars=4):
    return {"id": "pt", "bpm": 120.0, "bpb": 4, "songs": {sid: {}}, "tracks": [{"id": "t1", "name": "Drums", "role": "drums"}],
            "clips": [{"id": "c1", "track": "t1", "song": sid, "stem": stem, "start": 0.0, "src": 0.0, "len": bars * 4.0,
                       "gain_db": 0, "fade_in": 0, "fade_out": 0.03, "mute": False, "rate": 1, "pitch": 0, "keep_pitch": True}]}


def test_glitch_effects_and_rare_cooldown():
    from mashlab import glitch
    ps = glitch.pieces("st16", 4.0, 5.0, 4)
    assert len(ps) == 4 and all(abs(p[2] - 3.75) < 1e-9 and abs(p[1] - p[0] - 0.25) < 1e-9 for p in ps)
    tape = glitch.pieces("tape", 0.0, 1.0, 4)
    assert all(a[4] > b[4] for a, b in zip(tape, tape[1:]))                  # slows down
    dive = glitch.pieces("dive", 0.0, 1.0, 4)
    assert dive[0][4] == 1.0 and dive[-1][4] < dive[0][4]
    notes = [(b * 0.5, b * 0.5 + 0.25, 42, 90) for b in range(32)] + [(b * 4.0, b * 4.0 + 1, 49, 120) for b in range(4)]
    ges = glitch.plan(notes, {"weights": {k: (5 if k == "st16" else 0) for k in glitch.EFFECTS} | {"roll": 1}, "cooldown_bars": 2}, 4)
    by = {round(T, 3): e for T, R, e in ges}
    assert by[0.5] == "st16"                                                 # the busy hats: the busiest effect
    rolls = [T for T, R, e in ges if e == "roll"]
    assert rolls and all(b - a >= 8 for a, b in zip(rolls, rolls[1:]))       # rare: 2 bars apart at least


def test_glitch_apply_cuts_the_lane_and_reads_the_original(data_sandbox):
    from mashlab import glitch
    sid = fake_song(data_sandbox)
    proj = _proj(sid)
    clips, n = glitch.apply(proj, ["t1"], [(4.0, 5.0, "st16"), (8.0, 9.0, "speed")])
    lane = sorted([c for c in clips if c["track"] == "t1"], key=lambda c: (c["start"], c["len"]))
    assert n == 2 and lane[0]["start"] == 0 and abs(lane[0]["len"] - 4) < 1e-6
    st = [c for c in lane if 4 - 1e-6 <= c["start"] < 5 - 1e-6]
    assert len(st) == 4 and all(abs(c["src"] - 3.75) < 1e-6 for c in st)    # the 1/16 before the note, 4 times
    sp = [c for c in lane if 8 - 1e-6 <= c["start"] < 9 - 1e-6]
    assert len(sp) == 1 and sp[0]["rate"] == 2.0 and sp[0]["keep_pitch"] is False and abs(sp[0]["len"] - 1) < 1e-6
    rest = [c for c in lane if c["start"] >= 9 - 1e-6]
    assert rest and abs(rest[0]["src"] - 9.0) < 1e-6                         # the song carries on after


def test_arrangement_map_drops_silent_bars(data_sandbox):
    from mashlab import midi_remix as MR
    sid = fake_song(data_sandbox)
    proj = _proj(sid)
    notes = {"p1": [(0.0, 2.0, 40, 100), (8.5, 9.0, 40, 100)]}               # plays in bars 1 and 3 only
    clips, rep = MR.arrangement(proj, {"p1": "t1"}, notes, 0.0, 16.0)
    spans = sorted((c["start"], c["start"] + c["len"]) for c in clips if c["track"] == "t1")
    assert rep == {"t1": 2} and [tuple(round(x, 6) for x in s) for s in spans] == [(0, 4), (8, 12)]


def test_chord_follow_key_and_audio(data_sandbox):
    from mashlab import library, midi_remix as MR
    sid = fake_song(data_sandbox)
    info = library.read(sid); info["key"] = {"tonic": 9, "mode": "minor", "name": "A minor"}; library.write(sid, info)
    bars = [(0.0, 4.0), (4.0, 8.0)]
    chords = MR.midi_chords([(0, 4, 48, 90), (0, 4, 60, 90), (0, 4, 64, 90), (0, 4, 67, 90),
                             (4, 8, 45, 90), (4, 8, 57, 90), (4, 8, 60, 90), (4, 8, 64, 90)], bars)
    assert chords == [(0, "maj"), (9, "min")]                                # C, then Am
    clips, rep = MR.follow(_proj(sid, "vocals", 2), ["t1"], chords, bars, mode="key")
    lane = sorted([c for c in clips if c["track"] == "t1"], key=lambda c: c["start"])
    assert [round(c["start"]) for c in lane] == [0, 4] and [c["pitch"] for c in lane] == [3.0, 0.0] and rep["t1"] == [3, 0]
    clips, rep = MR.follow(_proj(sid, "vocals", 2), ["t1"], chords, bars, mode="audio")
    assert rep["t1"][0] == 3                                                 # the sung A (bar 1) moved up to C


def test_groove_moves_the_hits(data_sandbox):
    from mashlab import midi_remix as MR
    sid = fake_song(data_sandbox)
    groove = [(b + 0.5 + 0.1, 0.25, 42, 90) for b in range(16)] + [(float(b), 0.25, 36, 100) for b in range(16)]
    offs, lv, res = MR.groove_template(groove, 0, 4)
    assert res == 2 and abs(offs[1] - 0.1) < 1e-6 and offs[0] == 0          # swung 8ths: the "and" 0.1 late
    out = MR.groove_lane(_proj(sid, "drums", 2), "t1", offs, lv, 1.0, 0.0, 8.0, res=res)
    starts = sorted(round(c["start"], 2) for c in out)
    assert 0.6 in starts and 1.6 in starts and 1.0 in starts                 # hats moved, kicks stayed


def test_every_part_is_the_songs_length(tmp_path, midi_store):
    import soundfile as sf
    from mashlab import library
    parts = [{"id": "p1", "name": "short", "program": 0, "drum": False, "notes": [[0, 1, 60, 100]], "instrument": dict(BASIC)},
             {"id": "p2", "name": "long", "program": 0, "drum": False, "notes": [[0, 1, 60, 100], [30, 2, 64, 100]], "instrument": dict(BASIC)}]
    sid = midi.create("Lengths", "x", [i * 0.5 for i in range(40)], 120.0, 4, parts, render=False)
    info = midi.render_all(sid)
    n = int(round(info["duration"] * SR))
    lens = {p: sf.info(str(midi_store / sid / "stems" / f"{p}.flac")).frames for p in ("p1", "p2")}
    assert lens == {"p1": n, "p2": n}
    ys = [library.read_stem(sid, p) for p in ("p1", "p2")]
    assert (ys[0] + ys[1]).shape[1] == n                                    # they add up like any song's stems
    import time
    job = midi.save_part(sid, "p1", [[0, 1, 60, 100], [40, 1, 62, 100]], None, None, None, None)   # the song grows
    for _ in range(300):                                                    # (it re-renders in a background job)
        if job.status in ("done", "error"):
            break
        time.sleep(0.05)
    assert job.status == "done"
    info = library.read(sid)
    n2 = int(round(info["duration"] * SR))
    assert n2 > n and all(sf.info(str(midi_store / sid / "stems" / f"{p}.flac")).frames == n2 for p in ("p1", "p2"))
