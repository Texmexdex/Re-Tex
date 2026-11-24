"""No quiet fallbacks (Nov 16): a missing model / library / program stops with what it is, the error, the fix and a
link (mashlab/components.py); the few places that carry on say so."""
import pytest

from mashlab import components as C


def _fake(monkeypatch, key, problem):
    monkeypatch.setitem(C.C, key, dict(C.C[key], check=lambda: problem))


def test_every_component_has_what_the_dialog_needs():
    for k, c in C.C.items():
        assert c["name"] and c["used_for"] and c["fix"] and callable(c["check"]), k
        d = C.info(k, "x")
        assert set(d) >= {"key", "name", "ok", "problem", "used_for", "fix", "url", "where", "download"}


def test_missing_carries_its_mark_through_plain_text(monkeypatch):
    _fake(monkeypatch, "sep_sw", "its checkpoint isn't there")
    with pytest.raises(C.Missing) as e:
        C.require("sep_sw")
    text = f"WorkerError: {e.value}"                    # as it comes back from a worker process
    d = C.from_text(text)
    assert d["key"] == "sep_sw" and d["problem"] == "its checkpoint isn't there" and d["download"]
    assert "[component:" not in C.strip(text)
    assert C.from_text("an ordinary error") is None


def test_require_passes_when_it_is_there(monkeypatch):
    _fake(monkeypatch, "node", None)
    C.require("node")


def test_fallback_is_listed_once(monkeypatch):
    monkeypatch.setattr(C, "_warned", {})
    C.fallback("pedalboard", "pitch moves")
    C.fallback("pedalboard", "pitch moves again")
    C.notice("style_prompt", "Style prompt", "plain tags", problem="bad answer")
    f = {x["key"]: x for x in C.fallbacks()}
    assert f["pedalboard"]["instead"] == "pitch moves" and f["style_prompt"]["problem"] == "bad answer"


def test_separation_never_drops_to_demucs_quietly(monkeypatch):
    from mashlab import library
    assert library.default_quality() == "blend"
    _fake(monkeypatch, "sep_sw", "BS-Rofo-SW-Fixed.ckpt isn't there")
    with pytest.raises(C.Missing):
        library.run_separation(None, "blend", lambda *a: None)
    _fake(monkeypatch, "sep_ultra", "not there")
    with pytest.raises(C.Missing):
        library.run_separation(None, "ultra", lambda *a: None)


def test_a_failed_job_says_what_is_missing(monkeypatch):
    import threading
    from mashlab import jobs
    _fake(monkeypatch, "ffmpeg", "ffmpeg isn't on PATH")
    done = threading.Event()

    def work(job):
        try:
            C.require("ffmpeg")
        finally:
            done.set()
    j = jobs.submit("test", "test", work)
    assert done.wait(10)
    for _ in range(100):
        if j.status == "error":
            break
        threading.Event().wait(0.02)
    assert j.status == "error" and j.missing and j.missing["key"] == "ffmpeg"
    assert "[component:" not in j.error


def test_server_answers_424_with_the_details(monkeypatch):
    from fastapi.testclient import TestClient
    from mashlab import server
    c = TestClient(server.app)
    r = c.get("/api/components")
    assert r.status_code == 200 and {i["key"] for i in r.json()["items"]} == set(C.C)
    assert c.post("/api/components/nope/download").status_code == 404
    _fake(monkeypatch, "flamingo", "not set up")
    monkeypatch.setattr("mashlab.listen.available", lambda: (False, "not set up"))
    r = c.post("/api/library/listen", json={})
    assert r.status_code == 424 and r.json()["missing"]["key"] == "flamingo"


def test_cpu_is_never_quiet(monkeypatch):
    monkeypatch.setattr(C, "_warned", {})
    monkeypatch.setattr(C, "_chk_torch_cuda", lambda: "torch 2.8.0+cpu is the CPU-only build")
    assert C.device("Separation") == "cpu"
    f = {x["key"]: x for x in C.fallbacks()}
    assert "processor" in f["torch_cuda"]["instead"] and "CPU-only" in f["torch_cuda"]["problem"]
    monkeypatch.setattr(C, "_chk_torch_cuda", lambda: None)
    assert C.device("Separation") == "cuda"


def test_download_wont_pull_in_cpu_torch(monkeypatch):
    monkeypatch.setattr(C, "_chk_torch_cuda", lambda: "torch isn't installed")
    called = []
    monkeypatch.setattr(C, "_pip", lambda *a, **k: called.append(a))
    with pytest.raises(C.Missing) as e:
        C.download("demucs")
    assert e.value.key == "torch_cuda" and not called


def test_startup_names_the_missing_library(capsys):
    from mashlab import __main__ as M
    with pytest.raises(SystemExit) as e:
        M._cant_start(ModuleNotFoundError("No module named 'multipart'", name="multipart"))
    out = capsys.readouterr().out
    assert e.value.code == 2 and "pip install python-multipart" in out and "requirements.txt" in out
