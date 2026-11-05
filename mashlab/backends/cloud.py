"""Generic OpenAI-compatible cloud entry point (design E.1, OD-2).

A single thin adapter over the OpenAI chat/transcription API shape, configured by data/config.json's `cloud`
block: {"cloud": {"llm": {base_url, api_key_env, model}, "asr": {...}}}. OpenRouter and Groq are OpenAI-compat
and so are just a different base_url + model; Anthropic (Claude) is the premium tier via its OpenAI-compat
endpoint behind this same interface. Gemini is NOT wired in v1.

Security (design E.3 / safety): API keys are referenced ONLY by env-var NAME in config and read from
os.environ at call time — never inlined into config, never logged. Cloud is DEFAULT OFF: with no config
(enabled False) these backends raise a clear "disabled" error instead of making a network call.

The plumbing is proven against a LOCAL STUB server (tests), so the adapter ships built-and-tested but off.
"""
import json
import os
import urllib.request


class CloudDisabled(RuntimeError):
    """Raised when a cloud call is attempted while the cloud block is disabled or unconfigured."""


def _api_key(api_key_env):
    """Resolve the API key from the named env var. Returns '' when the name is empty or unset (never logs)."""
    if not api_key_env:
        return ""
    return os.environ.get(api_key_env, "")


def _post_json(url, headers, payload, timeout=60):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


class CloudLlmBackend:
    """OpenAI-compatible chat-completions client. Used for the premium concept-tag path (CloudConcept) and any
    other LLM need. `cfg` is the data/config.json `cloud.llm` block."""

    def __init__(self, cfg):
        self.enabled = bool((cfg or {}).get("enabled"))
        self.base_url = ((cfg or {}).get("base_url") or "").rstrip("/")
        self.api_key_env = (cfg or {}).get("api_key_env") or ""
        self.model = (cfg or {}).get("model") or ""

    def available(self):
        return bool(self.enabled and self.base_url and self.model)

    def chat(self, messages, timeout=60, **params):
        """POST /chat/completions with an OpenAI-shaped body. Returns the assistant message text.
        messages: [{"role","content"}, ...]."""
        if not self.enabled:
            raise CloudDisabled("cloud LLM is disabled (data/config.json cloud.llm.enabled is false)")
        if not (self.base_url and self.model):
            raise CloudDisabled("cloud LLM is enabled but base_url/model are not configured")
        headers = {"Content-Type": "application/json"}
        key = _api_key(self.api_key_env)
        if key:
            headers["Authorization"] = f"Bearer {key}"
        payload = {"model": self.model, "messages": list(messages)}
        payload.update(params)
        resp = _post_json(f"{self.base_url}/chat/completions", headers, payload, timeout=timeout)
        return resp["choices"][0]["message"]["content"]

    def unload(self):
        """No GPU/model residency to release (pure network client). Safe no-op."""
        return None


class CloudAsrBackend:
    """OpenAI-compatible audio-transcription client (POST /audio/transcriptions). `cfg` is the
    data/config.json `cloud.asr` block. Returns the raw provider JSON (callers map it to lyrics.json shape)."""

    def __init__(self, cfg):
        self.enabled = bool((cfg or {}).get("enabled"))
        self.base_url = ((cfg or {}).get("base_url") or "").rstrip("/")
        self.api_key_env = (cfg or {}).get("api_key_env") or ""
        self.model = (cfg or {}).get("model") or ""

    def available(self):
        return bool(self.enabled and self.base_url and self.model)

    def transcribe_file(self, audio_bytes, filename="audio.wav", timeout=120, **params):
        """POST a multipart audio-transcription request (OpenAI /audio/transcriptions shape). Returns provider
        JSON (expected to carry at least `text`). Uses a minimal multipart encoder (no extra deps)."""
        if not self.enabled:
            raise CloudDisabled("cloud ASR is disabled (data/config.json cloud.asr.enabled is false)")
        if not (self.base_url and self.model):
            raise CloudDisabled("cloud ASR is enabled but base_url/model are not configured")
        boundary = "----mashlabCloudAsr"
        fields = {"model": self.model}
        fields.update({k: str(v) for k, v in params.items()})
        body = bytearray()
        for name, value in fields.items():
            body += (f"--{boundary}\r\n"
                     f'Content-Disposition: form-data; name="{name}"\r\n\r\n'
                     f"{value}\r\n").encode("utf-8")
        body += (f"--{boundary}\r\n"
                 f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
                 f"Content-Type: application/octet-stream\r\n\r\n").encode("utf-8")
        body += bytes(audio_bytes) + b"\r\n"
        body += f"--{boundary}--\r\n".encode("utf-8")
        headers = {"Content-Type": f"multipart/form-data; boundary={boundary}"}
        key = _api_key(self.api_key_env)
        if key:
            headers["Authorization"] = f"Bearer {key}"
        req = urllib.request.Request(f"{self.base_url}/audio/transcriptions", data=bytes(body),
                                     headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def unload(self):
        """No GPU/model residency to release (pure network client). Safe no-op."""
        return None
