"""Shared pytest fixtures for the Mashup Lab Phase-2 suite.

There were ZERO pre-existing tests; this establishes the harness. It prepends the copy root to sys.path (so
`import mashlab` works regardless of the invoking cwd), gives a tmp data/ sandbox that monkeypatches the
config paths, and provides synthetic lyrics/concept/song dicts, a minimal empty project, and a stub
AsrBackend returning fixed segments. CPU-only: no model loads, no GPU, no network.
"""
import importlib
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


_EXIT_STATUS = {"code": 0}


def pytest_sessionfinish(session, exitstatus):
    """Record the TRUE session status; the hard-exit happens later in pytest_unconfigure (after the terminal
    summary is printed), so the pass/fail count is never truncated."""
    _EXIT_STATUS["code"] = int(exitstatus)


def pytest_unconfigure(config):
    """Hard-exit with the recorded session status as the very last step.

    Importing mashlab.server and exercising it through a FastAPI TestClient (FEAT-004) leaves native library
    state (PortAudio via sounddevice / the anyio test portal) whose atexit teardown HANGS the interpreter on
    this Windows box — the tests all pass and the summary prints, then the process never exits, so the runner
    reports a spurious non-zero code. pytest_unconfigure runs AFTER the terminal summary, so the result is
    fully reported before we exit; the recorded status is propagated verbatim, so a real failure is never
    hidden. (In the no-server case this is just an explicit clean exit.)"""
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(_EXIT_STATUS["code"])


@pytest.fixture
def data_sandbox(tmp_path, monkeypatch):
    """Redirect mashlab.config.* paths (and the already-imported appcfg.CONFIG_JSON) at a throwaway tmp data/
    tree, so load/save_config and any model-dir access touch only the sandbox — never the real data/."""
    import mashlab.config as config
    import mashlab.appcfg as appcfg

    data = tmp_path / "data"
    paths = dict(DATA=data, SONGS=data / "songs", CACHE=data / "cache", PROJECTS=data / "projects",
                 EXPORTS=data / "exports", MODELS=data / "models", CONFIG_JSON=data / "config.json")
    for name, p in paths.items():
        monkeypatch.setattr(config, name, p, raising=True)
        (p.parent if name == "CONFIG_JSON" else p).mkdir(parents=True, exist_ok=True)
    # appcfg captured CONFIG_JSON at import time via `from .config import CONFIG_JSON`; repoint that too.
    monkeypatch.setattr(appcfg, "CONFIG_JSON", paths["CONFIG_JSON"], raising=True)
    # Several modules captured SONGS/CACHE at import time (`from .config import SONGS`), so patching config
    # alone leaves their module-level copies pointing at the REAL data/ — a test that calls e.g.
    # library.write would then write into the real data/songs/ and pollute the live library. Repoint every
    # such module-level name at the sandbox so tests can NEVER touch real data/ (mirrors test_backfill's
    # song_store, generalized once here so each test doesn't have to remember to do it).
    for modname, names in (("library", ("SONGS",)), ("analyze", ("SONGS",)), ("lock", ("SONGS",)),
                           ("variant", ("SONGS",)), ("draft", ("SONGS", "CACHE")), ("warp", ("SONGS", "CACHE")),
                           ("clipfx", ("CACHE",)), ("transitions", ("CACHE",))):
        try:
            mod = importlib.import_module(f"mashlab.{modname}")
        except Exception:
            continue
        for n in names:
            if hasattr(mod, n):
                monkeypatch.setattr(mod, n, paths["SONGS"] if n == "SONGS" else paths["CACHE"], raising=False)
    return paths


@pytest.fixture
def lyrics_dict():
    """A synthetic lyrics.json-shaped dict (what an AsrBackend.transcribe returns + a quality block)."""
    return {
        "version": 1,
        "backend": "faster-whisper-large-v3",
        "lang": "en",
        "segments": [
            {"t0": 0.50, "t1": 3.20, "text": "we ride through fire on the open road tonight",
             "avg_logprob": -0.22, "no_speech_prob": 0.03,
             "words": [{"t": 0.50, "w": "we", "p": 0.91}, {"t": 0.80, "w": "ride", "p": 0.88}]},
            {"t0": 3.40, "t1": 6.10, "text": "nothing left to lose we never fold",
             "avg_logprob": -0.31, "no_speech_prob": 0.05, "words": []},
        ],
        "quality": {"mean_logprob": -0.26, "voiced_coverage": 0.62, "asr_conf": 0.71, "usable": True},
    }


@pytest.fixture
def concept_dict():
    """A synthetic concept.json-shaped dict (v1 zero-shot default: no LLM tags)."""
    return {
        "version": 1, "embed_model": "all-MiniLM-L6-v2",
        "concept_vec": [0.0] * 384,
        "valence": 0.31, "arousal": 0.77,
        "tags": {"theme": None, "mood": "defiant", "person": None, "tense": None,
                 "imagery": ["fire", "road", "night"]},
        "tags_backend": "zeroshot-minilm", "has_tags": False, "judgeable": 0.72,
    }


@pytest.fixture
def song_dict():
    """A synthetic song.json-shaped dict with a beat grid (enough for grid-dependent helpers)."""
    beats = [round(0.5 * i, 4) for i in range(0, 33)]
    return {
        "id": "sTEST0001", "name": "Test Track", "bpm": 120.0, "bpb": 4,
        "beats": beats, "first_beat": 0, "last_beat": len(beats) - 1,
        "beat_content": [50] * len(beats), "beat_conf": [80] * len(beats),
        "duration": beats[-1], "tuning_st": 0.0,
    }


@pytest.fixture
def empty_project():
    """A minimal project dict from project.empty() (the shared manual/autonomous arrangement model)."""
    import mashlab.project as project
    return project.empty(name="pytest project", bpm=120.0, bpb=4)


@pytest.fixture
def stub_asr(lyrics_dict):
    """A stub AsrBackend returning fixed segments — no model, no GPU, no network. Honors the ABC + unload()."""
    from mashlab.backends.asr import AsrBackend

    class StubAsr(AsrBackend):
        name = "stub-asr"

        def __init__(self, result):
            self.result = result
            self.unloaded = False

        def transcribe(self, vocal_audio, sr):
            return dict(self.result)

        def unload(self):
            self.unloaded = True

    return StubAsr(lyrics_dict)
