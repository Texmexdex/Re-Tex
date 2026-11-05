"""Tests for mashlab.appcfg (data/config.json load/save + lyrics_calib accessor)."""
import json

import mashlab.appcfg as appcfg


def test_defaults_are_local_only_cloud_off(data_sandbox):
    cfg = appcfg.load_config()
    assert cfg["backends"]["asr"] == "local"
    assert cfg["backends"]["concept"] == "local"
    assert cfg["backends"]["separation"] == "local"
    assert cfg["backends"]["generative"] == "null"
    # Cloud is DEFAULT OFF and carries no inlined secret, only an (empty) env-var name.
    assert cfg["cloud"]["llm"]["enabled"] is False
    assert cfg["cloud"]["asr"]["enabled"] is False
    assert cfg["cloud"]["llm"]["api_key_env"] == ""
    assert cfg["lyrics_calib"] is None


def test_missing_config_returns_defaults(data_sandbox):
    assert not appcfg.CONFIG_JSON.exists()
    cfg = appcfg.load_config()
    assert cfg == appcfg.defaults()


def test_save_load_roundtrips_atomically(data_sandbox):
    cfg = appcfg.defaults()
    cfg["backends"]["asr"] = "cloud"
    cfg["cloud"]["llm"] = {"enabled": True, "base_url": "https://openrouter.ai/api/v1",
                           "api_key_env": "OPENROUTER_API_KEY", "model": "meta-llama/llama-3.1-8b-instruct:free"}
    appcfg.save_config(cfg)

    assert appcfg.CONFIG_JSON.exists()
    # No stray temp file left behind by write-then-rename.
    assert list(appcfg.CONFIG_JSON.parent.glob("config.*.tmp")) == []

    loaded = appcfg.load_config()
    assert loaded["backends"]["asr"] == "cloud"
    assert loaded["cloud"]["llm"]["base_url"] == "https://openrouter.ai/api/v1"
    assert loaded["cloud"]["llm"]["api_key_env"] == "OPENROUTER_API_KEY"


def test_partial_config_merges_over_defaults(data_sandbox):
    # A config written by an older build with only one key still exposes every expected key after load.
    appcfg.CONFIG_JSON.write_text(json.dumps({"backends": {"concept": "cloud"}}), encoding="utf-8")
    cfg = appcfg.load_config()
    assert cfg["backends"]["concept"] == "cloud"
    assert cfg["backends"]["asr"] == "local"          # untouched key keeps its default
    assert cfg["cloud"]["llm"]["enabled"] is False     # whole cloud block still present


def test_corrupt_config_falls_back_to_defaults(data_sandbox):
    appcfg.CONFIG_JSON.write_text("{ not valid json", encoding="utf-8")
    assert appcfg.load_config() == appcfg.defaults()


def test_lyrics_calib_accessor(data_sandbox):
    # Absent -> None.
    assert appcfg.lyrics_calib() is None
    cfg = appcfg.load_config()
    assert appcfg.lyrics_calib(cfg) is None
    # Present -> returned verbatim.
    cfg["lyrics_calib"] = {"lo": 0.14, "hi": 0.63, "n_pairs": 180, "date": "2026-02-01"}
    appcfg.save_config(cfg)
    calib = appcfg.lyrics_calib()
    assert calib["n_pairs"] == 180
    assert calib["lo"] == 0.14 and calib["hi"] == 0.63
