"""Where the library lives: env > settings file > ROOT/data, and song sources stored so both the dev code and a
frozen copy (different ROOT, same library) resolve them."""
import json
import os
import subprocess
import sys
from pathlib import Path

from mashlab import config


def _paths(tmp_path, **env):
    e = {k: v for k, v in os.environ.items() if not k.startswith("MASHLAB_")}
    e.update(env)
    out = subprocess.run([sys.executable, "-c", "from mashlab import config as c; print(c.DATA); print(c.EXPORTS)"],
                         cwd=str(config.ROOT), env=e, capture_output=True, text=True, check=True).stdout.split("\n")
    return Path(out[0]), Path(out[1])


def test_settings_file_then_env(tmp_path):
    s = tmp_path / "settings.json"
    lib, exp = tmp_path / "lib", tmp_path / "exp"
    s.write_text(json.dumps({"data": str(lib)}), encoding="utf-8")
    assert _paths(tmp_path, MASHLAB_SETTINGS=str(s)) == (lib, lib / "exports")
    s.write_text(json.dumps({"data": str(lib), "exports": str(exp)}), encoding="utf-8")
    assert _paths(tmp_path, MASHLAB_SETTINGS=str(s)) == (lib, exp)
    other = tmp_path / "other"
    assert _paths(tmp_path, MASHLAB_SETTINGS=str(s), MASHLAB_DATA=str(other))[0] == other
    assert (lib / "songs").is_dir() and (lib / "projects").is_dir()      # a new folder becomes a library


def test_no_settings_is_root_data(tmp_path):
    assert _paths(tmp_path, MASHLAB_SETTINGS=str(tmp_path / "missing.json"))[0] == config.ROOT / "data"


def test_write_settings_merges(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "SETTINGS_JSON", tmp_path / "s" / "settings.json")
    config.write_settings({"data": "X:/a", "exports": "X:/b"})
    config.write_settings({"exports": None})
    assert config.read_settings() == {"data": "X:/a"}


def test_sources_portable_across_code_folders(tmp_path, monkeypatch):
    lib = tmp_path / "library"
    (lib / "uploads").mkdir(parents=True)
    f = lib / "uploads" / "song.flac"
    f.write_bytes(b"x")
    monkeypatch.setattr(config, "DATA", lib)
    monkeypatch.setattr(config, "UPLOADS", lib / "uploads")
    assert config.rel_source(f) == "data/uploads/song.flac"           # same string the dev library always stored
    assert config.resolve_source("data/uploads/song.flac") == f
    assert config.resolve_source("C:/gone/elsewhere/song.flac") == f   # self-heals by name


def test_a_listening_session_removed_keeps_its_notes(tmp_path, monkeypatch):
    import json
    from mashlab import listening
    monkeypatch.setattr(listening, "DIR", tmp_path)
    d = tmp_path / "s1"
    d.mkdir()
    (d / "session.json").write_text(json.dumps({"id": "s1", "items": [{"id": "i1", "file": "a.flac"}]}), encoding="utf-8")
    (d / "a.flac").write_bytes(b"x" * 10)
    listening.set_feedback("s1", "i1", "right", "nice")
    listening.remove("s1")
    assert not d.exists()
    kept = tmp_path / "_archive" / "s1"
    assert json.loads((kept / "feedback.json").read_text(encoding="utf-8"))["items"]["i1"]["notes"] == "nice"
    assert not (kept / "a.flac").exists() and listening.sessions() == []
