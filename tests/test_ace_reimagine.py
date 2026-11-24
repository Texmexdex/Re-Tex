"""Tests for ACE Studio Reimagine integration (acelink, acereimagine, components, and /api/ace/* routes).
All live tests are strictly 0-credit (status, models, cloud history read-only, validation, and mocked safety checks).
"""
import json
from pathlib import Path
from unittest import mock

import pytest
from fastapi.testclient import TestClient

from mashlab import acelink, acereimagine, components, server


def test_component_registered_and_found():
    info = components.info("ace_studio")
    assert info["key"] == "ace_studio"
    if info["ok"]:
        assert info["where"].lower().endswith("acestudio-cli.exe")


def test_unsaved_project_guard_blocks_overwrite():
    fake_dirty = dict(
        installed=True, cli="acestudio-cli.exe", running=True, bridge=True,
        awaiting_user="", project_name="MyRealSong", dirty=True, is_temp=False, error="",
    )
    with mock.patch.object(acelink, "ensure_running", return_value=fake_dirty):
        with pytest.raises(acelink.AceError) as exc:
            acelink.ensure_clean_stage(bpm=128.0)
        assert exc.value.code == "UNSAVED_CHANGES"
        assert "MyRealSong" in str(exc.value)


def test_vault_register_writes_v2_schema_and_dedupes(tmp_path):
    with mock.patch.object(acelink, "_cfg", return_value=(acelink.DEFAULT_CLI, str(tmp_path))):
        e1 = acelink.vault_register(
            kind="cover", source="music-enhancer", title="Test Cover",
            prompt="synthwave, heavy drums", tags=["synthwave", "drums"],
            bpm=124, key="Am", duration_sec=62.5, result_id="task1:9001",
            local_path=str(tmp_path / "songs" / "test.flac"), credit_cost=10,
        )
        assert e1 is not None
        # Re-register same result_id updates in place instead of duplicating
        e2 = acelink.vault_register(
            kind="cover", source="music-enhancer", title="Test Cover Updated",
            prompt="synthwave, heavy drums", tags=["synthwave"],
            bpm=124, key="Am", duration_sec=62.5, result_id="task1:9001",
            local_path=str(tmp_path / "songs" / "test.flac"), credit_cost=10,
        )
        cat = json.loads((tmp_path / "catalog.json").read_text(encoding="utf-8"))
        assert cat["version"] == 2
        assert len(cat["entries"]) == 1
        assert cat["entries"][0]["title"] == "Test Cover Updated"
        assert cat["stats"]["totalEntries"] == 1
        assert cat["stats"]["totalCreditsLogged"] == 10.0


def test_project_key_and_metrics_parsing():
    proj = {
        "bpm": 128, "bpb": 4, "key": "F# minor",
        "clips": [
            {"stem": "drums", "start": 0, "len": 32, "mute": False},
            {"stem": "vocals", "start": 8, "len": 24, "mute": False},
            {"stem": "bass", "start": 0, "len": 64, "mute": True},
        ],
    }
    assert acereimagine._project_key(proj, []) == ("F#", "Minor")
    bpm, bpb, end_beat, total_bars, secs, stems = acereimagine._project_metrics(proj)
    assert bpm == 128.0
    assert bpb == 4
    assert end_beat == 32.0
    assert total_bars == 8
    assert stems == ["drums", "vocals"]
    assert secs == 15.0


def test_enhancer_duration_guard_rejects_over_240s(tmp_path):
    prep_id = "aceprep-pytest-dur"
    pdir = acereimagine.ACE_DIR / prep_id
    pdir.mkdir(parents=True, exist_ok=True)
    try:
        (pdir / "prep.json").write_text(json.dumps({
            "prep_id": prep_id, "title": "Long Mix", "bpm": 120.0, "bpb": 4,
            "total_bars": 150, "mix_seconds": 300.0, "songs": [], "stems_present": ["drums"],
        }), encoding="utf-8")
        (pdir / "project.json").write_text(json.dumps({"bpm": 120.0, "bpb": 4, "clips": []}), encoding="utf-8")
        with pytest.raises(ValueError) as exc:
            acereimagine.perform(prep_id, mode="enhancer", style="synthwave")
        assert "240s" in str(exc.value)
        assert "bars" in str(exc.value).lower()
    finally:
        import shutil
        shutil.rmtree(pdir, ignore_errors=True)


def test_wait_settled_result_ids_waits_past_ephemeral_res_id():
    # Simulates ACE returning ephemeral 'res-5' first, then durable 'uuid:1'/'uuid:2' on the next poll
    calls = [
        {"results": [{"id": "res-5", "state": "streaming"}]},
        {"results": [{"id": "res-5", "state": "settled"}]},
        {"results": [{"id": "49fe6401-425b-46bf-92ba-c1e7024bfe8f:1", "state": "settled"},
                     {"id": "49fe6401-425b-46bf-92ba-c1e7024bfe8f:2", "state": "settled"}]},
    ]
    with mock.patch.object(acelink, "job_results", side_effect=calls), \
         mock.patch("mashlab.acelink.time.sleep", return_value=None):
        ids = acelink.wait_settled_result_ids("job-4", kit="music-enhancer", title="Test", timeout_s=10.0)
        assert ids == [
            "49fe6401-425b-46bf-92ba-c1e7024bfe8f:1",
            "49fe6401-425b-46bf-92ba-c1e7024bfe8f:2",
        ]


def test_instrumental_mode_strips_vocals_and_sets_marker():
    from mashlab import reimagine as RI
    raw = "glitch-pop, chopped glitchy female vocals, shoegaze, reverb-drenched vocals, clean electric guitar"
    cleaned = RI._no_vocals(raw)
    assert "vocal" not in cleaned.lower()
    assert "glitch-pop" in cleaned and "shoegaze" in cleaned and "clean electric guitar" in cleaned


def test_api_ace_readonly_endpoints_and_validation():
    client = TestClient(server.app)

    r_stat = client.get("/api/ace/status")
    assert r_stat.status_code == 200
    st = r_stat.json()
    assert "installed" in st and "bridge" in st

    if st["bridge"]:
        r_mod = client.get("/api/ace/models")
        assert r_mod.status_code == 200
        mod = r_mod.json()
        assert len(mod["models"]) >= 10
        assert "drums" in mod["sound_types"] and "custom" in mod["sound_types"]
        kinds = {m["kind"] for m in mod["models"]}
        assert "voice" in kinds and "instrument" in kinds

        r_hist = client.get("/api/ace/history?kit=all&refresh=false")
        assert r_hist.status_code == 200
        assert isinstance(r_hist.json().get("items"), list)

    r_ren = client.get("/api/ace/renders")
    assert r_ren.status_code == 200
    assert isinstance(r_ren.json(), list)

    # Validation guards (must reject before launching anything)
    assert client.post("/api/ace/prepare", json={}).status_code == 400
    assert client.post("/api/ace/perform", json={"prep_id": "nope"}).status_code == 400

    # Create a temporary prep to test perform parameter validation
    tmp_prep_dir = acereimagine.ACE_DIR / "aceprep-pytest-val"
    tmp_prep_dir.mkdir(parents=True, exist_ok=True)
    try:
        (tmp_prep_dir / "prep.json").write_text(json.dumps({"prep_id": "aceprep-pytest-val"}), encoding="utf-8")
        assert client.post("/api/ace/perform", json={"prep_id": "aceprep-pytest-val", "mode": "bad"}).status_code == 400
        assert client.post("/api/ace/perform", json={"prep_id": "aceprep-pytest-val", "mode": "enhancer", "influence": 1.5}).status_code == 400
        assert client.post("/api/ace/perform", json={"prep_id": "aceprep-pytest-val", "mode": "voice_swap", "model_ids": []}).status_code == 400
        assert client.post("/api/ace/transcribe", json={"prep_id": "aceprep-pytest-val", "engine": "bogus"}).status_code == 400
        assert client.post("/api/ace/transcribe", json={"prep_id": "aceprep-pytest-val", "engine": "ace_v2m", "language": "klingon"}).status_code == 400
    finally:
        import shutil
        shutil.rmtree(tmp_prep_dir, ignore_errors=True)

    # STT metadata endpoint + missing prep validation
    r_stt = client.get("/api/ace/stt")
    assert r_stt.status_code == 200
    stt = r_stt.json()
    eng_ids = [e["id"] for e in stt["engines"]]
    assert eng_ids == ["whisper_vad", "whisper_novad", "ace_v2m", "parakeet_tdt"]
    assert "english" in stt["languages"]
    assert client.post("/api/ace/transcribe", json={"prep_id": "nope", "engine": "whisper_novad"}).status_code == 400
    assert client.post("/api/ace/prepare", json={"project": {"clips": [{"stem": "vocals"}]}, "stt_engine": "bogus"}).status_code == 400


def test_default_faster_whisper_route_untouched():
    from mashlab.backends import asr
    assert asr.FW_MODEL == "large-v3"
    assert asr.FW_OPTS["vad_filter"] is True
    assert isinstance(asr.local_asr(), asr.FasterWhisperLarge)
    assert type(asr.local_asr()) is asr.FasterWhisperLarge


def test_parakeet_component_registered():
    info = components.info("parakeet_tdt")
    assert info["key"] == "parakeet_tdt"
    assert info["download"] is True


def test_ace_lrc_parser_cleans_syllables_and_melisma_dups():
    sample_lrc = "\n".join([
        "[01:10.70]every move",
        "[01:11.90]move I dictate#1",
        "[01:14.09]dictate#2 No no",
        "[01:15.53]clean lines lines all",
        "[01:18.27]misplaced misplaced",
        "[01:24.04]",
    ])
    lines = acelink.parse_lrc(sample_lrc)
    assert [l["text"] for l in lines] == [
        "every move",
        "I dictate",
        "No no",
        "clean lines all",
        "misplaced",
    ]
    assert lines[0]["t0"] == 70.7 and lines[0]["t1"] == 71.9
    assert lines[-1]["t0"] == 78.27 and lines[-1]["t1"] == 84.04


def test_active_spans_and_parakeet_token_merging():
    import numpy as np
    from mashlab.backends import asr

    # 3s dead silence + 2s soft singing (-26 dB) + 3s dead silence
    sr = 16000
    t = np.arange(sr * 2, dtype=np.float32) / sr
    soft_vox = 0.05 * np.sin(2 * np.pi * 220.0 * t).astype(np.float32)
    audio = np.concatenate([np.zeros(sr * 3, np.float32), soft_vox, np.zeros(sr * 3, np.float32)])
    spans = asr._active_spans(audio, min_silence_s=1.5, db_thresh=-52.0, pad_s=0.25)
    assert len(spans) == 1
    assert 2.5 <= spans[0][0] / sr <= 3.1
    assert 4.9 <= spans[0][1] / sr <= 5.5

    # Subword merging + punctuation + pause splitting
    tokens = [" Oh", " the", " f", "lo", "or", ".", " Sin", "ce", " when"]
    times = [1.0, 1.3, 1.6, 1.75, 1.9, 2.0, 3.2, 3.35, 3.7]
    lps = [-0.2] * len(tokens)
    segs = asr.ParakeetTdtAsr._tokens_to_segments(tokens, times, lps, offset=10.0, chunk_dur=5.0)
    assert len(segs) == 2
    assert segs[0]["text"] == "Oh the floor."
    assert segs[0]["t0"] == 11.0
    assert segs[1]["text"] == "Since when"
    assert segs[1]["t0"] == 13.2
