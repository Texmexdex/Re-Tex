"""Tests for mashlab.backends (detect(), unload() no-ops, cloud adapter against a LOCAL STUB server).

CPU-only: no heavy model loads. detect() runs cheap import/file probes; the cloud adapter is exercised
against an in-process stub HTTP server (never a real provider) and is OFF with no config.
"""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

import mashlab.backends as backends
from mashlab.backends.asr import FasterWhisperLarge, WhisperXBackend, local_asr
from mashlab.backends.concept import MiniLmZeroShot, local_concept
from mashlab.backends.separation import LocalSeparation, local_separation
from mashlab.backends.generative import NullGenerative, local_generative
from mashlab.backends.cloud import CloudLlmBackend, CloudAsrBackend, CloudDisabled


# ------------------------------------------------------------------ detect()

def test_detect_reports_all_local_components():
    rep = backends.detect()
    for comp in ("asr", "concept", "separation", "generative"):
        assert comp in rep
        c = rep[comp]
        assert set(c) >= {"available", "impl", "device", "vram_free", "reason"}

    # On this Phase-2 install the local ASR/concept/separation paths are present.
    assert rep["asr"]["available"] is True
    assert rep["asr"]["impl"] == "FasterWhisperLarge"
    assert rep["concept"]["available"] is True
    assert rep["concept"]["impl"] in ("MiniLmZeroShot", "MiniLmLlamaCpp")
    assert rep["separation"]["available"] is True
    # Generative is the Null default (optional tier, never required by the local path).
    assert rep["generative"]["impl"] == "NullGenerative"


def test_detect_vram_free_consistent_across_components():
    rep = backends.detect()
    vfs = {rep[c]["vram_free"] for c in ("asr", "concept", "separation", "generative")}
    assert len(vfs) == 1  # single GPU -> one shared free-VRAM figure


# ------------------------------------------------------------------ unload() is a safe no-op callable

@pytest.mark.parametrize("factory", [
    local_asr, local_concept, local_separation, local_generative,
    lambda: FasterWhisperLarge(), lambda: WhisperXBackend(), lambda: MiniLmZeroShot(),
    lambda: LocalSeparation(), lambda: NullGenerative(),
])
def test_unload_is_safe_noop(factory):
    b = factory()
    assert callable(b.unload)
    # Unload with nothing loaded must not raise and must be idempotent.
    assert b.unload() is None
    assert b.unload() is None


def test_stub_asr_unload(stub_asr, lyrics_dict):
    out = stub_asr.transcribe(vocal_audio=[0.0] * 16000, sr=16000)
    assert out["segments"] == lyrics_dict["segments"]
    assert stub_asr.unloaded is False
    stub_asr.unload()
    assert stub_asr.unloaded is True


# ------------------------------------------------------------------ cloud adapter: OFF by default

def test_cloud_llm_off_by_default(data_sandbox):
    import mashlab.appcfg as appcfg
    cfg = appcfg.load_config()
    client = CloudLlmBackend(cfg["cloud"]["llm"])
    assert client.available() is False
    with pytest.raises(CloudDisabled):
        client.chat([{"role": "user", "content": "hi"}])


def test_cloud_asr_off_by_default(data_sandbox):
    import mashlab.appcfg as appcfg
    cfg = appcfg.load_config()
    client = CloudAsrBackend(cfg["cloud"]["asr"])
    assert client.available() is False
    with pytest.raises(CloudDisabled):
        client.transcribe_file(b"RIFFfake", filename="x.wav")


# ------------------------------------------------------------------ cloud adapter: round-trip vs LOCAL STUB

class _StubOpenAIHandler(BaseHTTPRequestHandler):
    """Minimal OpenAI-compatible stub: /chat/completions and /audio/transcriptions."""

    def log_message(self, *_):  # silence
        pass

    def _send(self, obj):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length)
        auth = self.headers.get("Authorization", "")
        if self.path.endswith("/chat/completions"):
            payload = json.loads(raw)
            echo = payload["messages"][-1]["content"]
            self._send({"choices": [{"message": {"role": "assistant",
                                                 "content": f"stub-reply:{echo}:auth={auth}"}}]})
        elif self.path.endswith("/audio/transcriptions"):
            # Multipart body; we only need to prove the request shape round-trips.
            got_file = b'name="file"' in raw
            self._send({"text": "stub transcription", "language": "en",
                        "segments": [{"start": 0.0, "end": 1.0, "text": "stub transcription",
                                      "avg_logprob": -0.2, "no_speech_prob": 0.01}],
                        "_got_file": got_file})
        else:
            self.send_response(404)
            self.end_headers()


@pytest.fixture
def stub_server():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _StubOpenAIHandler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    host, port = srv.server_address
    yield f"http://{host}:{port}"
    srv.shutdown()
    srv.server_close()


def test_cloud_llm_roundtrips_against_stub(stub_server, monkeypatch):
    monkeypatch.setenv("STUB_KEY", "sekret-token")
    client = CloudLlmBackend({"enabled": True, "base_url": stub_server,
                              "api_key_env": "STUB_KEY", "model": "stub-model"})
    assert client.available() is True
    reply = client.chat([{"role": "user", "content": "ping"}], timeout=10)
    # The adapter resolved the key from the env var NAME and sent it as a Bearer token (never inlined).
    assert reply == "stub-reply:ping:auth=Bearer sekret-token"


def test_cloud_asr_roundtrips_against_stub(stub_server):
    client = CloudAsrBackend({"enabled": True, "base_url": stub_server,
                              "api_key_env": "", "model": "stub-whisper"})
    assert client.available() is True
    resp = client.transcribe_file(b"RIFFfakewavdata", filename="vocal.wav",
                                   response_format="verbose_json", timeout=10)
    assert resp["text"] == "stub transcription"
    assert resp["_got_file"] is True
    assert resp["segments"][0]["text"] == "stub transcription"
