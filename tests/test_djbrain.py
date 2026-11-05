"""DJ brain plumbing: the swappable model client (llm.py) against a local stub server, reply parsing, and the
brief/schema helpers. CPU-only: no model weights, no audio, no network beyond 127.0.0.1.
"""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from mashlab import djbrain, llm

PLAN = {"title": "t", "parts": [{"why": "open", "drums": "A1", "bass": "A1", "instruments": "A1", "vocal": "none",
                                 "vocal2": "none", "measures": 4, "transition": "cut"}]}


class _Handler(BaseHTTPRequestHandler):
    seen = []
    reject_schema = False

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        _Handler.seen.append(dict(body, _auth=self.headers.get("Authorization")))
        if _Handler.reject_schema and "response_format" in body:
            self.send_response(400)
            self.end_headers()
            return
        reply = "Here is the plan:\n```json\n" + json.dumps(PLAN) + "\n```"
        out = json.dumps({"choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": reply}}]})
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(out.encode())

    def log_message(self, *a):
        pass


@pytest.fixture
def server():
    _Handler.seen, _Handler.reject_schema = [], False
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}/v1"
    srv.shutdown()


def test_openai_client_asks_for_the_schema_and_parses_the_reply(server, monkeypatch):
    monkeypatch.setenv("DJ_TEST_KEY", "sekrit")
    c = llm.OpenAiLlm(server, "some-model", "DJ_TEST_KEY")
    schema = djbrain.schema(["A1", "B1"])
    assert c.chat_json([{"role": "user", "content": "plan"}], schema, seed=3) == PLAN
    sent = _Handler.seen[-1]
    assert sent["response_format"]["json_schema"]["schema"] == schema
    assert sent["model"] == "some-model" and sent["seed"] == 3
    assert sent["_auth"] == "Bearer sekrit"                     # key read from the env var by name


def test_openai_client_falls_back_when_the_server_has_no_structured_output(server):
    _Handler.reject_schema = True
    c = llm.OpenAiLlm(server, "")
    assert c.chat_json([{"role": "user", "content": "plan"}], djbrain.schema(["A1"])) == PLAN
    assert "response_format" not in _Handler.seen[-1] and "model" not in _Handler.seen[-1]


class _Run:
    """Stands in for subprocess.run: records the command, answers like `claude -p --output-format json`."""
    def __init__(self, out):
        self.out, self.calls = out, []

    def __call__(self, cmd, **kw):
        import subprocess
        self.calls.append((cmd, kw))
        return subprocess.CompletedProcess(cmd, 0, json.dumps(self.out), "")


def test_claude_code_client_runs_headless_with_the_schema_and_no_tools(monkeypatch):
    import subprocess
    run = _Run({"type": "result", "is_error": False, "result": "", "structured_output": PLAN})
    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(llm, "claude_exe", lambda: "claude.exe")
    c = llm.ClaudeCodeLlm(model="opus", effort="medium")
    got = c.chat_json([{"role": "system", "content": "be a DJ"}, {"role": "user", "content": "the brief"}],
                      djbrain.schema(["A1"]))
    assert got == PLAN
    cmd, kw = run.calls[0]
    assert cmd[:2] == ["claude.exe", "-p"] and kw["input"] == "the brief"
    flag = lambda f: cmd[cmd.index(f) + 1]
    assert json.loads(flag("--json-schema")) == djbrain.schema(["A1"]) and flag("--system-prompt") == "be a DJ"
    assert flag("--tools") == "" and "--strict-mcp-config" in cmd and "--no-session-persistence" in cmd
    assert flag("--model") == "opus" and flag("--effort") == "medium"
    assert "--bare" not in cmd                                   # --bare ignores the user's login (API key only)


def test_claude_code_not_signed_in_is_a_clear_error(monkeypatch):
    import subprocess
    monkeypatch.setattr(subprocess, "run", _Run({"is_error": True, "result": "Failed to authenticate: OAuth session expired"}))
    monkeypatch.setattr(llm, "claude_exe", lambda: "claude.exe")
    with pytest.raises(llm.LlmError, match="claude auth login"):
        llm.ClaudeCodeLlm().chat_json([{"role": "user", "content": "x"}], {})
    monkeypatch.setattr(subprocess, "run", _Run({"loggedIn": False}))
    ok, why = llm.claude_status()
    assert not ok and "claude auth login" in why
    monkeypatch.setattr(llm, "claude_exe", lambda: None)
    assert llm.claude_status() == (False, "Claude Code isn't installed")


def test_unreachable_server_is_a_clear_error():
    with pytest.raises(llm.LlmError, match="can't reach"):
        llm.OpenAiLlm("http://127.0.0.1:9/v1", "m").chat_json([], {})


def test_parse_tolerates_prose_and_rejects_truncation():
    assert llm._parse('sure! {"a": 1} done') == {"a": 1}
    with pytest.raises(llm.LlmError):
        llm._parse('{"parts": [{"why": "cut off mid')


def test_schema_only_allows_real_sections():
    s = djbrain.schema(["A1", "A2", "B1"])
    part = s["properties"]["parts"]["items"]["properties"]
    for stem in ("drums", "bass", "instruments", "vocal", "vocal2"):    # every stem: any real section, or none
        assert part[stem]["enum"] == ["A1", "A2", "B1", "none"]
    assert part["measures"]["enum"] == djbrain.MEASURE_CHOICES
    assert list(part)[0] == "why"                                # reason first, then the choices


def test_snap_prefers_the_later_measure_on_a_tie():
    M = [0, 2, 4, 6]                                            # measure-start bars
    assert djbrain._snap(M, 3) == 2                             # bar 3 is between 2 and 4 -> 4 (index 2)
    assert djbrain._snap(M, 2) == 1 and djbrain._snap(M, 5.4) == 3
