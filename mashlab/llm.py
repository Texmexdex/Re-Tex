"""The DJ brain's language model, swappable.

  local   - a GGUF instruction model run in-process by llama.cpp (default: the first *.gguf under data/models).
  openai  - any OpenAI-compatible chat server: LM Studio's local server (pick any model in LM Studio), Ollama,
            or a cloud endpoint (OpenRouter, Anthropic's OpenAI-compatible API, ...).

Configured by data/config.json "dj": {"backend": "local" | "openai", "gguf": "<path>", "base_url": "...",
"model": "...", "api_key_env": "<ENV VAR NAME>"}. Keys are read from the environment by name, never stored.
Every call asks for JSON matching a schema (llama.cpp: grammar-constrained; servers: response_format), so the
reply always parses into the shape the caller expects - the caller still validates what it means.
"""
import gc
import json
import os
import re
import urllib.error
import urllib.request

from . import appcfg


class LlmError(RuntimeError):
    pass


def _cfg():
    return (appcfg.load_config() or {}).get("dj") or {}


def client(planner=None):
    """The configured model: LocalLlm or OpenAiLlm. MASHLAB_GGUF (set by a frozen copy's launcher) pins the local
    model and wins over data/config.json. planner="claude" (the DJ Mix dialog's choice): ClaudeCodeLlm - Claude on
    the user's own Claude login; data/config.json "dj" "claude_model" / "claude_effort" tune it."""
    c = _cfg()
    if planner == "claude":
        return ClaudeCodeLlm(model=c.get("claude_model") or "", effort=c.get("claude_effort") or "")
    if os.environ.get("MASHLAB_GGUF"):
        c = {"backend": "local", "gguf": os.environ["MASHLAB_GGUF"]}
    if c.get("backend") == "openai":
        return OpenAiLlm(c.get("base_url") or "http://127.0.0.1:1234/v1", c.get("model") or "",
                         c.get("api_key_env") or "")
    path = c.get("gguf")
    if not path:
        from .backends import find_gguf
        p = find_gguf()
        path = str(p) if p else None
    if not path or not os.path.exists(path):
        raise LlmError("no language model: put a GGUF instruction model in data/models, or point data/config.json "
                       "\"dj\" at an OpenAI-compatible server (e.g. LM Studio's)")
    return LocalLlm(path)


def _parse(text):
    """JSON object from a model reply (tolerates code fences / prose around it)."""
    text = (text or "").strip()
    try:
        return json.loads(text)
    except ValueError:
        pass
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except ValueError:
            pass
    raise LlmError("the model's reply wasn't valid JSON (it may have run out of tokens)")


class LocalLlm:
    def __init__(self, path, n_ctx=8192):
        self.path, self.n_ctx, self._llm = path, n_ctx, None
        self.name = "llama.cpp:" + os.path.basename(path)

    def _load(self):
        if self._llm is None:
            from llama_cpp import Llama
            self._llm = Llama(model_path=self.path, n_ctx=self.n_ctx, n_gpu_layers=-1, verbose=False)
        return self._llm

    def chat_json(self, messages, schema, max_tokens=3000, temperature=0.5, seed=None):
        out = self._load().create_chat_completion(
            messages=messages, response_format={"type": "json_object", "schema": schema},
            max_tokens=max_tokens, temperature=temperature, seed=seed if seed is not None else -1)
        ch = out["choices"][0]
        if ch.get("finish_reason") == "length":
            raise LlmError("the model ran out of tokens before finishing its plan")
        return _parse(ch["message"]["content"])

    def unload(self):
        self._llm = None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


class OpenAiLlm:
    def __init__(self, base_url, model, api_key_env=""):
        self.base_url, self.model, self.api_key_env = base_url.rstrip("/"), model, api_key_env
        self.name = f"{model or 'server default'} @ {self.base_url}"

    def _post(self, body, timeout=300):
        headers = {"Content-Type": "application/json"}
        key = os.environ.get(self.api_key_env, "") if self.api_key_env else ""
        if key:
            headers["Authorization"] = f"Bearer {key}"
        req = urllib.request.Request(self.base_url + "/chat/completions", data=json.dumps(body).encode("utf-8"),
                                     headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))

    def chat_json(self, messages, schema, max_tokens=3000, temperature=0.5, seed=None):
        body = dict(messages=messages, max_tokens=max_tokens, temperature=temperature)
        if self.model:
            body["model"] = self.model
        if seed is not None:
            body["seed"] = int(seed)
        try:
            out = self._post(dict(body, response_format={"type": "json_schema", "json_schema": {
                "name": "dj_plan", "schema": schema, "strict": True}}))
        except urllib.error.HTTPError as e:
            if e.code != 400:
                raise LlmError(f"model server error {e.code}") from e
            out = self._post(body)                     # server without structured output: plain JSON request
        except urllib.error.URLError as e:
            raise LlmError(f"can't reach the model server at {self.base_url} ({e.reason})") from e
        ch = out["choices"][0]
        if ch.get("finish_reason") == "length":
            raise LlmError("the model ran out of tokens before finishing its plan")
        return _parse(ch["message"]["content"])

    def unload(self):
        pass


def claude_exe():
    """The Claude Code command-line program, or None."""
    import shutil
    p = shutil.which("claude")
    if p:
        return p
    p = os.path.join(os.path.expanduser("~"), ".local", "bin", "claude.exe" if os.name == "nt" else "claude")
    return p if os.path.isfile(p) else None


def claude_status():
    """(ok, why) for planning with Claude through Claude Code on the user's own Claude login."""
    import subprocess
    exe = claude_exe()
    if not exe:
        return False, "Claude Code isn't installed"
    try:
        r = subprocess.run([exe, "auth", "status", "--json"], capture_output=True, text=True, encoding="utf-8",
                           errors="replace", timeout=30, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        st = json.loads(r.stdout)
    except Exception as e:
        return False, f"couldn't ask Claude Code ({type(e).__name__})"
    if not st.get("loggedIn"):
        return False, "Claude Code isn't signed in - in a terminal, run: claude auth login"
    return True, ""


class ClaudeCodeLlm:
    """Claude through Claude Code's headless mode (`claude -p`) on the user's own Claude login - no API key, it uses
    their plan's usage. Tools off, no MCP servers, no saved session, run from an empty folder so no project files or
    instructions reach it; the reply is held to the JSON schema (--json-schema), like llama.cpp's grammar."""

    def __init__(self, model="", effort="", timeout=900):
        self.exe, self.model, self.effort, self.timeout = claude_exe(), model, effort, timeout
        self.name = "Claude" + (f" ({model})" if model else "") + " via Claude Code"

    def chat_json(self, messages, schema, max_tokens=3000, temperature=0.5, seed=None):
        import subprocess
        import tempfile
        if not self.exe:
            raise LlmError("Claude Code isn't installed")
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        user = "\n\n".join(m["content"] for m in messages if m["role"] != "system")
        cmd = [self.exe, "-p", "--output-format", "json", "--json-schema", json.dumps(schema),
               "--tools", "", "--strict-mcp-config", "--no-session-persistence"]
        if system:
            cmd += ["--system-prompt", system]
        if self.model:
            cmd += ["--model", self.model]
        if self.effort:
            cmd += ["--effort", self.effort]
        with tempfile.TemporaryDirectory(prefix="retex-claude-") as d:
            try:
                r = subprocess.run(cmd, input=user, capture_output=True, text=True, encoding="utf-8", errors="replace",
                                   timeout=self.timeout, cwd=d, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            except subprocess.TimeoutExpired as e:
                raise LlmError(f"Claude took longer than {self.timeout // 60} min") from e
        try:
            out = json.loads(r.stdout)
        except ValueError:
            raise LlmError("Claude Code gave no answer: " + ((r.stderr or r.stdout).strip()[-200:] or f"exit {r.returncode}"))
        if out.get("is_error"):
            msg = str(out.get("result") or out.get("terminal_reason") or "error")
            if any(w in msg.lower() for w in ("auth", "login", "log in", "oauth")):
                raise LlmError("Claude Code isn't signed in - in a terminal, run: claude auth login")
            raise LlmError(f"Claude Code: {msg[:300]}")
        plan = out.get("structured_output")
        if plan is None:
            plan = _parse(out.get("result"))
        if not isinstance(plan, dict):
            raise LlmError("Claude's answer wasn't in the expected shape")
        return plan

    def unload(self):
        pass
