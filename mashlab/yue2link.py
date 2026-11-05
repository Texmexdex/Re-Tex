"""Link to the user's YuE2 Studio: a separate install (its own Python environments and models) that Mashup Lab
drives over its local HTTP API and never imports.

Configured by data/config.json "yue2": {"studio": "<the YuE2 folder>", "url": "http://127.0.0.1:8791"}. When the
Studio isn't running it is started from that folder (its own .venv, --no-browser) and left running, as if the
user had opened it. Its jobs and renders live in its own runs/ folder; finished takes are copied into Mashup Lab.
"""
import json
import re
import os
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from . import appcfg

DEFAULT_URL = "http://127.0.0.1:8791"


class StudioError(RuntimeError):
    pass


def _cfg():
    """(url, studio folder). MASHLAB_YUE2_URL / MASHLAB_YUE2_STUDIO (set by a frozen copy's launcher, which brings
    its own Studio) win over data/config.json."""
    c = (appcfg.load_config() or {}).get("yue2") or {}
    url = os.environ.get("MASHLAB_YUE2_URL") or c.get("url") or DEFAULT_URL
    return url.rstrip("/"), os.environ.get("MASHLAB_YUE2_STUDIO") or c.get("studio") or ""


def _req(path, body=None, raw=None, ctype="application/json", timeout=300):
    url, _ = _cfg()
    data = raw if raw is not None else (json.dumps(body).encode("utf-8") if body is not None else None)
    req = urllib.request.Request(url + path, data=data, method="POST" if data is not None else "GET",
                                 headers={"Content-Type": ctype} if data is not None else {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read()
    except urllib.error.HTTPError as e:
        try:
            msg = json.loads(e.read().decode("utf-8")).get("error") or str(e)
        except Exception:
            msg = str(e)
        raise StudioError(f"YuE2 Studio: {msg}") from e
    except urllib.error.URLError as e:
        raise StudioError(f"can't reach YuE2 Studio at {url} ({e.reason})") from e


def _json(path, body=None, **kw):
    return json.loads(_req(path, body, **kw).decode("utf-8"))


def status():
    """The Studio's /api/status, or None when it isn't running."""
    try:
        return _json("/api/status", timeout=3)
    except StudioError:
        return None


def ensure_running(wait_s=90):
    """Start the Studio from its folder if it isn't running (it then stays up, like the user had opened it)."""
    if status():
        return
    url, studio = _cfg()
    server = Path(studio, "ui", "server.py") if studio else None
    py = Path(studio, ".venv", "Scripts", "python.exe") if studio else None
    if not server or not server.exists() or not py.exists():
        raise StudioError('YuE2 Studio isn\'t running, and data/config.json "yue2" -> "studio" doesn\'t point at '
                          "its folder, so it can't be started")
    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(subprocess, "DETACHED_PROCESS", 0)
    subprocess.Popen([str(py), "server.py", "--no-browser", "--port", str(urllib.parse.urlparse(url).port or 8791)],
                     cwd=str(server.parent), env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                     creationflags=flags)
    t = time.time()
    while time.time() - t < wait_s:
        if status():
            return
        time.sleep(1)
    raise StudioError("YuE2 Studio didn't start")


def upload(path):
    p = Path(path)
    return _json(f"/api/m2m/upload?filename={urllib.parse.quote(p.name)}", raw=p.read_bytes(),
                 ctype="application/octet-stream")["upload_id"]


def analyze(upload_id, task="full", kind="score"):
    """kind "score": SheetSage2 melody (+ chords with task "full"); kind "realaudio": the real-audio tokenizer's codes
    (what the re-render / continuation modes perform from)."""
    body = dict(upload_id=upload_id, kind=kind)
    if kind == "score":
        body["task"] = task
    return _json("/api/m2m/analyze", body)["job_id"]


_STAGE = (("gen", re.compile(r"Generating song: (\d+) tokens")),
          ("synth", re.compile(r"Synthesizing audio: (\d+)/(\d+)")),
          ("decode", re.compile(r"Decoding audio: (\d+)/(\d+)")))


class _Progress:
    """Reads the Studio's progress line ("[YuE2] Running Synthesizing audio: 9/32 steps (28%) | elapsed 80s") into a
    fraction of the whole render: a take is generated (song tokens, ~25 per second of audio), synthesised (flow
    steps) and decoded, takes one after another - a new take starts when the stages start over."""

    def __init__(self, takes=1, expect_tokens=None):
        self.takes, self.expect, self.k, self.last = max(1, int(takes)), expect_tokens, 0, None
        self.prev = (0.0, None)

    def read(self, text):
        f, msg = self._read(text)
        if msg is None and isinstance(text, str):             # not a status line (the Studio passes on a tool's raw
            return self.prev                                  # stderr - warnings, tracebacks): keep the last one
        self.prev = (f, msg)
        return f, msg

    def _read(self, text):
        if not isinstance(text, str):
            return (float(text) if isinstance(text, (int, float)) else 0.0), None
        if not text.strip().startswith("[YuE2]") and text.strip() not in ("running", "queued"):
            return 0.0, None
        line = re.sub(r"^\[YuE2\]\s*(Running|Starting|Completed)?\s*", "", text.strip()).split(" | ")[0]
        stage, f = None, None
        for name, rx in _STAGE:
            m = rx.search(text)
            if m:
                stage = name
                if name == "gen":
                    f = 0.5 * min(1.0, int(m.group(1)) / self.expect) if self.expect else 0.25
                else:
                    a, b = int(m.group(1)), max(1, int(m.group(2)))
                    f = (0.5 + 0.45 * a / b) if name == "synth" else (0.95 + 0.05 * a / b)
                break
        if stage and self.last in ("synth", "decode") and stage == "gen" or self.last == "decode" and stage == "synth":
            self.k = min(self.takes - 1, self.k + 1)          # the stages started over: the next take
        if stage:
            self.last = stage
        if "Finished" in text and self.k >= self.takes - 1:
            return 1.0, "ranking the takes"
        if f is None:
            return (self.k / self.takes), line
        take = f"take {self.k + 1} of {self.takes} · " if self.takes > 1 else ""
        return (self.k + f) / self.takes, take + line


def lyrics_fit(analysis_id, lyrics, voices="both", abc=None):
    """The Studio's lyric fit against an analysis' score - or against `abc` (a score of our own, e.g. a vocal guide)."""
    body = dict(lyrics=lyrics, voices=voices, **({"abc": abc} if abc else {"analysis_id": analysis_id}))
    return _json("/api/m2m/lyrics-fit", body)


def render(**body):
    return _json("/api/m2m/render", body)["job_id"]


def wait(job_id, progress=lambda f, m="": None, lo=0.0, hi=1.0, cancelled=lambda: False, takes=1, expect_tokens=None):
    """Follow a Studio job to the end -> its result. Raises StudioError when it fails or is cancelled. takes /
    expect_tokens (song tokens per take) let a render's progress be read as a real fraction."""
    reader = _Progress(takes, expect_tokens)
    while True:
        if cancelled():
            try:
                _json(f"/api/job/{urllib.parse.quote(job_id)}/cancel", {})
            finally:
                raise StudioError("cancelled")
        j = _json(f"/api/job/{urllib.parse.quote(job_id)}", timeout=30)
        st = j.get("status")
        if st == "done":                                   # the Studio marks a job done a moment before it
            for _ in range(40):                            # attaches the result: wait for that, too
                if j.get("result"):
                    return j["result"]
                time.sleep(1.5)
                j = _json(f"/api/job/{urllib.parse.quote(job_id)}", timeout=30)
            raise StudioError("YuE2 Studio finished the job but returned no result")
        if st in ("error", "cancelled", "unknown"):
            raise StudioError(f"YuE2 Studio job {st}: {j.get('error') or ''}".strip())
        f, msg = reader.read(j.get("progress"))
        f = f / 100 if f > 1 else f
        progress(lo + (hi - lo) * min(1.0, max(0.0, f)), "YuE2: " + (msg or j.get("stage") or "working"))
        time.sleep(3)


def read_text(rel):
    """A Studio run file as text (e.g. its analysis' score.abc)."""
    return _req("/api/file/" + urllib.parse.quote(rel), timeout=60).decode("utf-8")


def fetch(rel, dst):
    """Copy a Studio run file (e.g. "render-.../cand-00/audio.flac") to dst."""
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    Path(dst).write_bytes(_req("/api/audio/" + urllib.parse.quote(rel), timeout=600))
    return dst
