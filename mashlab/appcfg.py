"""Phase-2 application config (data/config.json).

A single small JSON file that holds the backend-tier selection (which ASR/concept impl to use, plus the
optional OpenAI-compatible cloud block) and the `lyrics_calib` artifact the lyrical-compatibility gate reads
(design A.3 / E.3). It is read by the backend tiers (backends/) and by concept.lyrics_params() (FEAT-002).

Written with the project's write-then-rename discipline (mirrors mashfeat.compute / project.save) so a
partial or interrupted write never presents as a complete config. Cloud is DEFAULT OFF: a fresh install has
no config.json, load_config() returns the local-only defaults, and nothing cloud runs until the user opts in.
"""
import json
import os

from .config import CONFIG_JSON


def defaults():
    """The local-only default config. Cloud is OFF (enabled False) and carries no inlined secrets — only
    env-var NAMES. This is what load_config() returns when data/config.json is absent."""
    return {
        "version": 1,
        "backends": {
            # Per-component tier selection. "local" is the standalone default and the verification target.
            "asr": "local",          # local -> FasterWhisperLarge ; "cloud" -> CloudAsrBackend (opt-in)
            "concept": "local",      # local -> MiniLmEmbed + ZeroShotMood (LlamaCppTags auto if a GGUF is found)
            "separation": "local",   # local -> separate.py (Ultra/SW/blend)
            "generative": "null",    # "null" -> NullGenerative ; "yue2" -> YuE2Glue (opt-in, never required)
        },
        "cloud": {
            # OpenAI-compatible entry point (OD-2). DEFAULT OFF. api_key_env is an ENV-VAR NAME, never a key.
            "llm": {"enabled": False, "base_url": "", "api_key_env": "", "model": ""},
            "asr": {"enabled": False, "base_url": "", "api_key_env": "", "model": ""},
        },
        # Lyrical-compatibility calibration (design A.3). Absent by default -> the ±5% safe gate.
        "lyrics_calib": None,
    }


def _merge(base, over):
    """Shallow-recursive merge of a loaded config over the defaults, so a config written by an older build (or
    a user who only set one key) still exposes every expected key."""
    out = dict(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config():
    """Read data/config.json merged over defaults(). Missing/corrupt file -> defaults (local-only, cloud OFF)."""
    try:
        raw = json.loads(CONFIG_JSON.read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError):
        return defaults()
    if not isinstance(raw, dict):
        return defaults()
    return _merge(defaults(), raw)


def save_config(cfg):
    """Atomically persist cfg to data/config.json (write-then-rename, mirrors mashfeat/project.save). Returns
    the path. Never logs the config body (it may name secret env vars)."""
    CONFIG_JSON.parent.mkdir(parents=True, exist_ok=True)
    tmp = CONFIG_JSON.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(cfg, indent=1), encoding="utf-8")
    os.replace(tmp, CONFIG_JSON)
    return CONFIG_JSON


def lyrics_calib(cfg=None):
    """The lyrics_calib artifact ({lo,hi,n_pairs,date,...}) or None when absent. concept.lyrics_params()
    (FEAT-002) consumes this to decide the safe (±5%) vs strong (±12%) lyrical factor (design A.3)."""
    if cfg is None:
        cfg = load_config()
    return cfg.get("lyrics_calib")
