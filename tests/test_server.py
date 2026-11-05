"""FastAPI route tests for the FEAT-004 server endpoints.

Verifies that the new routes are registered, that GET /api/capabilities returns the backends detect() block,
and that lyrics_mode is accepted on the rate/table/generate request bodies. No GPU, no model loads: the heavy
job functions are never executed here — we assert registration + the cheap capabilities probe, and that the
request handlers accept lyrics_mode without rejecting it.

The TestClient is built WITHOUT its context manager so the app lifespan/startup (library.resume_interrupted)
never runs against the real data dir; the data_sandbox fixture additionally repoints config paths at a tmp
tree for the handlers that do touch disk.
"""
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(data_sandbox):
    """A bare (un-entered) TestClient: the startup lifespan (library.resume_interrupted + render pool) is NOT
    run, and no context-manager teardown is invoked — route + handler tests only. data_sandbox repoints the
    config paths at a tmp tree so any handler disk access touches only the sandbox. The client is explicitly
    closed in teardown so its anyio portal / httpx transport threads do not linger (the interpreter's native
    atexit teardown is additionally pinned down by conftest's pytest_unconfigure hard-exit; see the note
    there)."""
    import mashlab.server as server
    c = TestClient(server.app)
    try:
        yield c
    finally:
        c.close()


@pytest.fixture
def sync_jobs(monkeypatch):
    """Run jobs.submit's fn(job) SYNCHRONOUSLY on the request thread instead of the background _gpu/_misc
    thread pools. Keeps handler tests deterministic and, crucially, leaves no non-daemon worker thread with a
    pending task at interpreter teardown (which otherwise hangs process exit). Returns a job-like dict."""
    from mashlab import jobs

    class _FakeJob:
        def __init__(self, kind, title, ref):
            self.id, self.kind, self.title, self.ref = 1, kind, title, ref
            self.status, self.progress, self.message, self.error = "done", 1.0, "done", None

        def update(self, *a, **k):
            pass

        def sub(self, lo, hi, prefix=""):
            return lambda f, m="": None

        def to_dict(self):
            return dict(id=self.id, kind=self.kind, title=self.title, ref=self.ref,
                        progress=self.progress, message=self.message, status=self.status, error=self.error)

    def fake_submit(kind, title, fn, ref=None, gpu=False):
        job = _FakeJob(kind, title, ref)
        try:
            fn(job)
        except Exception as e:          # the handler's own validation (e.g. HTTPException) still surfaces via result
            job.status, job.error = "error", str(e)
        return job

    monkeypatch.setattr(jobs, "submit", fake_submit)
    return jobs


def _route_paths(app):
    return {getattr(r, "path", None) for r in app.routes}


# ------------------------------------------------------------------ route registration

def test_new_routes_registered():
    import mashlab.server as server
    paths = _route_paths(server.app)
    assert "/api/songs/{sid}/lyrics" in paths
    assert "/api/mashup/auto" in paths
    assert "/api/songs/{sid}/backfill_beat_content" in paths


def test_lyrics_route_is_post():
    import mashlab.server as server
    methods = {}
    for r in server.app.routes:
        methods.setdefault(getattr(r, "path", None), set()).update(getattr(r, "methods", set()) or set())
    assert "POST" in methods["/api/songs/{sid}/lyrics"]
    assert "POST" in methods["/api/mashup/auto"]
    assert "POST" in methods["/api/songs/{sid}/backfill_beat_content"]


# ------------------------------------------------------------------ capabilities detect() block

def test_capabilities_returns_detect_block(client, data_sandbox):
    r = client.get("/api/capabilities")
    assert r.status_code == 200
    body = r.json()
    assert "backends" in body
    be = body["backends"]
    # detect() reports every heavy component with the pinned sub-keys.
    for comp in ("asr", "concept", "separation", "generative"):
        assert comp in be
        assert set(["available", "impl", "device", "vram_free", "reason"]).issubset(be[comp].keys())
    # the pre-existing capabilities fields are still present (no regression).
    assert "default_quality" in body and "ultra" in body and "sw" in body


# ------------------------------------------------------------------ lyrics_mode accepted on bodies

def test_rate_body_accepts_and_forwards_lyrics_mode(client, data_sandbox, sync_jobs, monkeypatch):
    # stub the heavy worker so no process pool spawns; assert the handler accepts the body AND forwards
    # lyrics_mode through to rate_all (run synchronously via sync_jobs).
    from mashlab import pairs
    seen = {}

    def fake_rate_all(ids, density=0.5, vocal_ids=None, bed_ids=None, progress=None,
                      cancelled=None, involving=None, lyrics_mode="off"):
        seen["lyrics_mode"] = lyrics_mode
        return {"pairs": 0, "rated": 0}

    monkeypatch.setattr(pairs, "rate_all", fake_rate_all)
    r = client.post("/api/mashup/rate", json={"songs": ["sA", "sB"], "lyrics_mode": "resonance"})
    assert r.status_code == 200
    assert r.json().get("kind") == "rate"
    assert seen.get("lyrics_mode") == "resonance"


def test_table_body_accepts_and_forwards_lyrics_mode(client, data_sandbox, monkeypatch):
    from mashlab import pairs
    seen = {}

    def fake_table(song_ids, density=0.5, vocal_ids=None, bed_ids=None, lyrics_mode="off"):
        seen["lyrics_mode"] = lyrics_mode
        return {"rows": [], "skipped": {}}

    monkeypatch.setattr(pairs, "table", fake_table)
    r = client.post("/api/mashup/table", json={"songs": ["sA"], "lyrics_mode": "contrast"})
    assert r.status_code == 200
    assert isinstance(r.json(), dict)
    assert seen.get("lyrics_mode") == "contrast"


def test_generate_body_accepts_and_forwards_lyrics_mode(client, data_sandbox, sync_jobs, monkeypatch):
    from mashlab import automash
    seen = {}

    def fake_generate_roles(roles, max_variants=500, density=0.5, progress=None, lyrics_mode="off"):
        seen["lyrics_mode"] = lyrics_mode
        return []

    monkeypatch.setattr(automash, "generate_roles", fake_generate_roles)
    r = client.post("/api/mashup/generate",
                    json={"roles": {"music": ["sX"], "vocals": ["sY"]}, "lyrics_mode": "off"})
    assert r.status_code == 200
    assert r.json().get("kind") == "mashup"
    assert seen.get("lyrics_mode") == "off"


# ------------------------------------------------------------------ auto endpoint input validation

def test_auto_requires_two_inputs(client, data_sandbox):
    r = client.post("/api/mashup/auto", json={})
    assert r.status_code == 400


def test_auto_accepts_paths_and_prompt(client, data_sandbox, sync_jobs, monkeypatch):
    # two paths + a style prompt: the endpoint accepts the body and submits the orchestrator job. Stub the
    # heavy run (via sync_jobs) so no separation/GPU work spawns during the test.
    from mashlab import orchestrate
    seen = {}

    def fake_run(paths, prompt=None, density=None, job=None, progress=None, **kw):
        seen["paths"] = list(paths)
        seen["prompt"] = prompt
        return {"id": "pFAKE", "mashup": {"style": orchestrate.resolve_style(prompt)}}

    monkeypatch.setattr(orchestrate, "run", fake_run)
    r = client.post("/api/mashup/auto",
                    json={"paths": ["a.wav", "b.wav"], "prompt": "chopped and screwed"})
    assert r.status_code == 200
    assert r.json().get("kind") == "auto"
    assert seen.get("paths") == ["a.wav", "b.wav"]
    assert seen.get("prompt") == "chopped and screwed"
