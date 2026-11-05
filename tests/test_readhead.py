"""ReadHead bridge: edit -> clip conversion, and the Node engine end to end on synthetic audio."""
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from mashlab import readhead as R


def test_segments_cover_range_and_follow_events():
    ev = [{"t": 0, "off": 0, "gain": 0}, {"t": 4, "off": 2, "gain": 1}, {"t": 8, "off": 0, "gain": 0}]
    seg = R._segments(ev, 0, 12)
    assert [(a, b, off, g) for a, b, off, g, _ in seg] == [(0, 4, 0, 0), (4, 8, 2, 1), (8, 12, 0, 0)]
    # a window starting mid-segment takes the state in force there
    assert R._segments(ev, 6, 10)[0][:4] == (6, 8, 2.0, 1.0)


def test_landing_inverse_round_trip():
    beats = np.cumsum(np.full(64, 0.5)) + 0.1
    W = 3 + (beats - beats[3]) / 0.5
    for src in (0.0, 5.25, 17.5, 40.75):
        assert abs(R._src_of(W, R._w_of(W, src)) - src) < 1e-9
    assert R._src_of(None, 7.5) == 7.5


def test_vocal_clips_from_lane():
    # vocal source axis q (project beats); a 1-beat repeat at beat 8, then back in place
    ev = [{"t": 0, "off": 0, "gain": 0}, {"t": 4, "off": 0, "gain": 1}, {"t": 8, "off": 1, "gain": 1},
          {"t": 9, "off": 0, "gain": 1}, {"t": 12, "off": 0, "gain": 0}]
    clips = R._vocal_clips(ev, "song", "vocals", None, 1.0, 100.0, 0, 16)
    assert [(c["start"], c["src"], c["len"]) for c in clips] == [(4, 4, 4), (8, 7, 1), (9, 9, 3)]
    assert all(c["rate"] == 1 and c["keep_pitch"] for c in clips)


def test_vocal_clips_with_mult_and_echo_gain():
    ev = [{"t": 0, "off": -2, "gain": 0.5}]
    c = R._vocal_clips(ev, "s", "vocals", None, 2.0, 100.0, 0, 4)[0]
    assert c["src"] == 1.0 and c["len"] == 4 and abs(c["gain_db"] + 6.02) < 0.01


def test_timeline_clips_map_back_through_lane_clips(monkeypatch):
    monkeypatch.setattr(R, "_landing", lambda project, sid: None)
    project = {"songs": {"a": {"mult": 1}}, "bpm": 120}
    lane = [{"id": "c1", "song": "a", "stem": "drums", "start": 0, "src": 10, "len": 8, "gain_db": -3}]
    ev = [{"t": 2, "off": 1, "gain": 1}, {"t": 3, "off": 0, "gain": 1}]
    out = R._timeline_clips(ev, lane, project, 0, 8)
    assert [(c["start"], c["src"], c["len"], c["gain_db"]) for c in out] == [(0, 10, 2, -3), (2, 11, 1, -3), (3, 13, 5, -3)]


@pytest.mark.skipif(R.node_exe() is None, reason="Node.js not installed")
def test_engine_runs_on_synthetic_audio(tmp_path):
    sr, bpm = 22050, 120.0
    spb = 60 / bpm
    n = int(sr * spb * 4 * 24)
    t = np.arange(n) / sr
    drums = np.zeros(n, np.float32)
    for k in range(int(n / (sr * spb))):
        i = int(k * spb * sr)
        drums[i:i + 2000] += np.exp(-np.arange(min(2000, n - i)) / 300) * (0.9 if k % 2 == 0 else 0.5)
    pad = (0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    voc = np.zeros(n, np.float32)
    for k in range(16, int(n / (sr * spb / 2)) - 2):
        if k % 8 in (6, 7):          # a breath every bar
            continue
        i = int(k * spb / 2 * sr)
        L = int(0.22 * sr)
        voc[i:i + L] += (0.4 * np.sin(2 * np.pi * (330 + 40 * (k % 5)) * np.arange(L) / sr) * np.minimum(1, np.arange(L) / 300)).astype(np.float32)
    paths = {}
    for name, arr in (("d", drums), ("o", pad), ("v", voc)):
        p = tmp_path / f"{name}.f32"
        arr.tofile(p)
        paths[name] = str(p)
    req = {"bpm": bpm, "bpb": 4, "lengthBeats": 96, "mode": "arrange", "opts": {"seed": 3, "chop": 0.5},
           "backing": [{"role": "drums", "path": paths["d"], "sr": sr}, {"role": "other", "path": paths["o"], "sr": sr}],
           "source": {"role": "vocal", "path": paths["v"], "sr": sr}, "sections": []}
    rq = tmp_path / "req.json"
    rq.write_text(json.dumps(req))
    p = subprocess.run([R.node_exe(), str(R.JS_DIR / "cli.mjs"), str(rq)], capture_output=True, text=True, timeout=300)
    assert p.returncode == 0, p.stderr
    res = json.loads(p.stdout)
    assert res["ok"] and res["lane"] and res["phrases"]
    assert res["vocal"]["lines"] >= 4
    clips = R._vocal_clips(res["lane"], "s", "vocals", None, 1.0, n / (sr * spb), 0, 96)
    assert clips and all(0 <= c["start"] < 96 and c["len"] > 0 for c in clips)
