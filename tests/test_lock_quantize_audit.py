"""Zero-phase transient audit, R3 click-pilot slope clamping, verified grid & lock quantization tests."""
import json
import numpy as np
import pytest

from mashlab import audio_io, gridaudit, library, lock, warp
from mashlab import project as P
from mashlab.config import SR, STEMS


def _synth_drums(beat_times, dur, sr=SR, kick_hz=65.0, snare=True):
    """Synthesize sharp zero-delay sub-kick (65 Hz) on every beat and snare burst (800 Hz) on beats 2 & 4."""
    n = int(round(dur * sr))
    y = np.zeros(n, np.float32)
    t_k = np.arange(int(0.09 * sr), dtype=np.float32) / sr
    # Pitch-drop sub-kick purely inside 30-160 Hz (completely invisible to old 150-1500 Hz filter!)
    f_k = kick_hz + 55.0 * np.exp(-t_k / 0.015)
    kick = (np.sin(2.0 * np.pi * np.cumsum(f_k) / sr) * np.exp(-t_k / 0.035)).astype(np.float32)
    rng = np.random.default_rng(7)
    t_s = np.arange(int(0.07 * sr), dtype=np.float32) / sr
    sn = (rng.standard_normal(len(t_s)).astype(np.float32) * np.exp(-t_s / 0.02)) * 0.45
    for idx, bt in enumerate(beat_times):
        i = int(round(bt * sr))
        if 0 <= i < n - len(kick):
            y[i:i + len(kick)] += kick * (1.0 if idx % 4 == 0 else 0.75)
            if snare and idx % 2 == 1:
                y[i:i + len(sn)] += sn
    peak = float(np.abs(y).max())
    if peak > 0.9:
        y = y * (0.85 / peak)
    return y


def test_zero_phase_sub_kick_audit_and_phase_recovery():
    """Sub-65 Hz kicks (invisible to old 150-1500 Hz causal filter) must audit at <= 4 ms and recover a +105 ms offset."""
    p = 0.5  # 120 BPM
    true_beats = 0.4 + np.arange(64) * p
    dur = float(true_beats[-1] + 1.0)
    drums = _synth_drums(true_beats, dur, kick_hz=62.0, snare=False)

    body, _, _, fps, src = gridaudit.onset_curves(drums, sr=SR)
    assert src == "drums" and float(body.max()) > 0.1

    aud_true = gridaudit.audit_beats(true_beats, body, fps)
    assert aud_true["status"] == "ok" and aud_true["median_offset_ms"] <= 4.0

    # Shift grid 105 ms early (exact Reyin 1/4-beat off-beat failure mode!)
    bad_beats = true_beats - 0.105
    aud_bad = gridaudit.audit_beats(bad_beats, body, fps)
    assert aud_bad["status"] == "off_beat" and abs(aud_bad["signed_median_ms"] - 105.0) < 12.0

    res = gridaudit.verify_and_fix(bad_beats, dur, drums, sr=SR, bpb=4)
    assert res["mode"] == "steady_locked"
    assert res["post_audit"]["status"] == "ok"
    assert res["post_audit"]["median_offset_ms"] <= 4.0


def test_recount_slips_fixes_isolated_2x_glitch_and_preserves_tempo_shift():
    """Isolated 2x Beat This! glitch beats and half-time gaps are repaired while a real tempo change stays intact."""
    p1, p2 = 0.5, 0.4  # 120 BPM -> 150 BPM
    sec1 = np.arange(32) * p1
    # Inject a 2x split glitch (0.24 + 0.26 = 0.50) and an isolated stray sub-beat
    g1 = np.insert(sec1, 10, sec1[9] + 0.24)
    # Inject a missed-beat half-time gap (delete beat 22)
    g1 = np.delete(g1, 22)
    sec2 = g1[-1] + p2 * np.arange(1, 33)
    raw = np.concatenate([g1, sec2])

    fixed = gridaudit.recount_slips(raw)
    iv = np.diff(fixed)
    assert np.all(iv > 0.30) and np.all(iv < 0.60)
    assert abs(float(np.median(iv[:25])) - p1) < 0.01
    assert abs(float(np.median(iv[-25:])) - p2) < 0.01


def test_smooth_pilot_map_clamps_pitch_wobble_while_preserving_drift():
    """Noisy +-20 ms click-pilot jitter must be smoothed and slope-clamped <= +-0.8% (+-13.8 cents max)."""
    n = 120
    per = 0.45 * SR
    dst = np.arange(n, dtype=np.float64) * per + 0.5 * SR
    # True slow R3 cumulative drift (+35 ms linear ramp) + violent +-20 ms single-beat click jitter + 2 outliers
    rng = np.random.default_rng(42)
    drift = np.linspace(0.0, 0.035 * SR, n)
    jitter = rng.uniform(-0.020 * SR, 0.020 * SR, n)
    meas = dst + drift + jitter
    meas[30] += 0.12 * SR   # outlier
    meas[75] -= 0.11 * SR   # outlier
    good = np.ones(n, dtype=bool)

    clean = warp._smooth_pilot_map(meas, good, dst, max_slope=0.008)
    assert clean is not None and len(clean) == n
    # Local varispeed ratio between consecutive beats must never exceed +-0.8% (+-13.8 cents)
    slopes = np.abs(np.diff(clean) / np.diff(dst) - 1.0)
    assert float(slopes.max()) <= 0.0081
    # Cumulative drift is tracked cleanly across the song without single-beat jitter
    err_clean = (clean - dst) / SR * 1000.0
    drift_ms = drift / SR * 1000.0
    assert float(np.median(np.abs(err_clean - drift_ms))) < 5.0
    assert abs(err_clean[-1] - 35.0) < 10.0 and abs(err_clean[0]) < 10.0


def test_lock_verify_and_make_quantizes_drifting_song_to_sub_5ms(data_sandbox, monkeypatch):
    """End-to-end lock on a drifting song with off-beat user_marks verifies and locks drums within <= 5 ms."""
    monkeypatch.setattr(P, "PROJECTS", data_sandbox["PROJECTS"])
    # Synthesize a song that genuinely drifts from 120 BPM -> 124 BPM (~3.3% drift)
    n_beats = 48
    pers = np.linspace(0.500, 0.484, n_beats - 1)
    true_beats = np.r_[0.35, 0.35 + np.cumsum(pers)]
    dur = float(true_beats[-1] + 0.8)
    drums_m = _synth_drums(true_beats, dur, kick_hz=68.0, snare=True)
    # Pitched steady tone on bass/other/vocals so we can verify no crash & clean multi-stem render
    t = np.arange(len(drums_m), dtype=np.float32) / SR
    tone = (0.15 * np.sin(2.0 * np.pi * 110.0 * t)).astype(np.float32)

    sid = "drift_test"
    sd = library.song_dir(sid)
    (sd / "stems").mkdir(parents=True, exist_ok=True)
    for s in STEMS:
        y = np.stack([drums_m, drums_m]) if s == "drums" else np.stack([tone, tone])
        audio_io.save(sd / "stems" / f"{s}.flac", y, SR, subtype="PCM_24")

    # Deliberately give song.json BOTH off-beat user_marks (+90 ms early) and slightly biased detector beats
    bad_um = (true_beats - 0.090).tolist()
    info = dict(
        id=sid, title="Drift Test", status="ready", duration=dur, bpm=122.0, bpb=4,
        beats=(true_beats - 0.025).tolist(), bars=list(range(0, n_beats, 4)),
        first_beat=0, last_beat=n_beats - 1, stems_rev=0, stem_scale=1.0,
        user_marks=dict(t=[round(x, 5) for x in bad_um], d=[1 if i % 4 == 0 else 0 for i in range(n_beats)],
                        s=[1] * n_beats, bpb=4),
    )
    library.write(sid, info)

    # 1. lock.check must honestly catch the +90 ms off-beat user_marks!
    chk = lock.check(sid)
    assert chk["off_beat"] is True
    assert chk["audit"]["median_offset_ms"] > 70.0

    # 2. Run verify_and_make synchronously (avoid spawning child process pool in unit test)
    from mashlab import jobs

    class _SyncPool:
        def submit(self, fn, *args, priority=0, **kwargs):
            val = fn(*args, **kwargs)
            return type("F", (), {"done": lambda self: True, "result": lambda self: val})()

    monkeypatch.setattr(warp, "_get_pool", lambda: _SyncPool())
    monkeypatch.setattr(jobs, "submit", lambda kind, title, work, **k: type("J", (), {
        "to_dict": lambda self, r=work(type("Job", (), {"update": lambda *a, **kw: None})()): {"status": "done", "result": r}
    })())

    lock.verify_and_make(sid, kind="marks", force=True)
    updated = library.read(sid)
    lk = updated["lock"]
    assert lk["rev"].startswith("L")
    assert lk["audit"]["status"] == "ok"
    assert lk["audit"]["median_offset_ms"] <= 5.0
    assert lk["marks_t"] and len(lk["marks_t"]) == len(updated["user_marks"]["t"])
    # Verify steady grid spacing in marks_t
    assert np.allclose(np.diff(lk["marks_t"]), lk["period"], atol=2e-4)
    # Independent audit on saved stems_locked/drums.flac confirms real drum hits land on tl!
    ext_aud = lock.audit_existing(sid)
    assert ext_aud["status"] == "ok" and ext_aud["median_offset_ms"] <= 5.0


def test_grid_audit_and_verify_server_endpoints(data_sandbox, monkeypatch):
    """POST /api/songs/{sid}/grid/audit and /grid/verify work cleanly over HTTP without mutating song.json."""
    from fastapi.testclient import TestClient
    from mashlab import server

    p = 0.5
    true_beats = 0.35 + np.arange(48) * p
    dur = float(true_beats[-1] + 0.8)
    drums_m = _synth_drums(true_beats, dur, kick_hz=64.0, snare=True)

    sid = "api_audit_song"
    sd = library.song_dir(sid)
    (sd / "stems").mkdir(parents=True, exist_ok=True)
    for s in STEMS:
        y = np.stack([drums_m, drums_m]) if s == "drums" else np.zeros((2, len(drums_m)), np.float32)
        audio_io.save(sd / "stems" / f"{s}.flac", y, SR, subtype="PCM_24")

    bad_beats = (true_beats - 0.085).tolist()
    info = dict(
        id=sid, title="API Audit", status="ready", duration=dur, bpm=120.0, bpb=4,
        beats=bad_beats, bars=list(range(0, 48, 4)), first_beat=0, last_beat=47,
        stems_rev=0, stem_scale=1.0,
    )
    library.write(sid, info)

    client = TestClient(server.app)
    r_aud = client.post(f"/api/songs/{sid}/grid/audit", json={"t": bad_beats})
    assert r_aud.status_code == 200
    d_aud = r_aud.json()
    assert d_aud["status"] == "off_beat" and abs(d_aud["signed_median_ms"] - 85.0) < 12.0

    r_ver = client.post(f"/api/songs/{sid}/grid/verify")
    assert r_ver.status_code == 200
    d_ver = r_ver.json()
    assert d_ver["report"]["post_audit"]["status"] == "ok"
    assert d_ver["report"]["post_audit"]["median_offset_ms"] <= 5.0
    # Proposal only: song.json must NOT be mutated until Save
    assert library.read(sid).get("user_marks") is None

