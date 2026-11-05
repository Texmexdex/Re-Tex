"""Quality-tier backend abstraction (design E).

One interface per heavy component (ASR, concept, separation, generative). Each has a local-default impl and
an optional premium/cloud impl, selected by data/config.json (appcfg) + a health check. Every backend exposes
an explicit `unload()` so the orchestrator controls GPU residency per stage (design E.2 / D.1a) rather than
relying on long-lived singletons.

`detect()` reports, per component, whether the local path is available on this install, which impl would be
chosen, the device, free VRAM, and a human reason. It is surfaced at GET /api/capabilities (FEAT later). The
local path must pass detect() and run standalone with ZERO cloud config.
"""
from .asr import AsrBackend, FasterWhisperLarge, WhisperXBackend, CloudAsrBackend, local_asr
from .concept import (ConceptBackend, MiniLmZeroShot, MiniLmLlamaCpp, CloudConcept, local_concept)
from .separation import SeparationBackend, LocalSeparation, local_separation
from .generative import GenerativeBackend, NullGenerative, YuE2Glue, local_generative
from .cloud import CloudLlmBackend, CloudAsrBackend as CloudAsrAdapter

__all__ = [
    "AsrBackend", "FasterWhisperLarge", "WhisperXBackend", "CloudAsrBackend", "local_asr",
    "ConceptBackend", "MiniLmZeroShot", "MiniLmLlamaCpp", "CloudConcept", "local_concept",
    "SeparationBackend", "LocalSeparation", "local_separation",
    "GenerativeBackend", "NullGenerative", "YuE2Glue", "local_generative",
    "CloudLlmBackend", "CloudAsrAdapter",
    "detect", "importable", "find_gguf", "vram_free",
]


def importable(module):
    """True if `module` can be imported, without actually keeping it loaded for use. Cheap import probe used by
    detect()."""
    import importlib.util
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def vram_free():
    """Free CUDA VRAM in bytes (torch.cuda.mem_get_info on device 0), or None when CUDA/torch is unavailable.
    Used both by detect() and by the orchestrator's per-stage free>=required assertion (D.1a)."""
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        free, _total = torch.cuda.mem_get_info(0)
        return int(free)
    except Exception:
        return None


def find_gguf():
    """Path to a GGUF instruction model on the box, or None. Enables the optional LLM concept-tag path (design
    A.2.2 / E.1). Looks under data/models/ for any *.gguf; recursive so a HF cache layout is also found."""
    from ..config import MODELS
    try:
        for p in MODELS.rglob("*.gguf"):
            if p.is_file():
                return p
    except Exception:
        pass
    return None


def _probe_asr():
    """Local ASR availability: faster-whisper + ctranslate2 importable AND ctranslate2 sees a CUDA device
    (the A12 gate). Falls back to CPU int8 only as an explicit, recorded degradation, so detect() reports it."""
    if not (importable("faster_whisper") and importable("ctranslate2")):
        return dict(available=False, impl=None, reason="faster-whisper/ctranslate2 not installed")
    device, reason = "cpu", "faster-whisper present; CUDA not visible to ctranslate2 (CPU int8 fallback)"
    try:
        import ctranslate2
        if ctranslate2.get_cuda_device_count() >= 1:
            device, reason = "cuda", "faster-whisper large-v3 (CUDA float16)"
    except Exception as e:  # pragma: no cover - defensive
        reason = f"ctranslate2 import/cuda probe failed: {e}"
    return dict(available=True, impl="FasterWhisperLarge", device=device, reason=reason)


def _probe_concept():
    """Local concept availability: sentence-transformers importable gives the mandatory embedding + zero-shot
    mood layer. A detected GGUF + llama-cpp-python upgrades the impl to the optional LLM-tag path."""
    if not importable("sentence_transformers"):
        return dict(available=False, impl=None, reason="sentence-transformers not installed")
    gguf = find_gguf()
    if gguf is not None and importable("llama_cpp"):
        return dict(available=True, impl="MiniLmLlamaCpp", device="cpu",
                    reason=f"MiniLM embed + llama.cpp tags ({gguf.name})")
    reason = "MiniLM embed + zero-shot mood (no LLM)"
    if gguf is not None:
        reason += "; GGUF found but llama-cpp-python not installed"
    return dict(available=True, impl="MiniLmZeroShot", device="cpu", reason=reason)


def _probe_separation():
    """Local separation availability: the existing separate.py path. Demucs always; BS-RoFormer Ultra/SW when
    the checkpoints are present (separate.ultra_available / SW_CKPT)."""
    if not importable("demucs"):
        return dict(available=False, impl=None, reason="demucs not installed")
    try:
        from . import separation as _sep
        modes = _sep.available_modes()
        return dict(available=True, impl="LocalSeparation", device="cuda" if _torch_cuda() else "cpu",
                    reason="separate.py modes: " + ", ".join(modes))
    except Exception as e:  # pragma: no cover - defensive
        return dict(available=True, impl="LocalSeparation", device="cpu", reason=f"demucs present ({e})")


def _probe_generative():
    """Generative is OPTIONAL and defaults to NullGenerative (always available). YuE2Glue is reported only
    when the YuE2 source tree is importable; it is never required by the local path."""
    if importable("yue2"):
        return dict(available=True, impl="YuE2Glue", device="cuda" if _torch_cuda() else "cpu",
                    reason="YuE2 generative glue available (optional)")
    return dict(available=True, impl="NullGenerative", device="cpu",
                reason="NullGenerative default (source fills + DSP sweeps only)")


def _torch_cuda():
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def detect():
    """Per-component capability/health: {component: {available, impl, device, vram_free, reason}} (design E.2).

    device/vram_free are informational; `vram_free` is the current free-VRAM floor (bytes) shared across every
    component since there is a single GPU. No model is loaded by detect() beyond cheap import/file probes and a
    ctranslate2 CUDA-count check."""
    vf = vram_free()
    out = {
        "asr": _probe_asr(),
        "concept": _probe_concept(),
        "separation": _probe_separation(),
        "generative": _probe_generative(),
    }
    for comp in out.values():
        comp.setdefault("device", None)
        comp.setdefault("impl", None)
        comp["vram_free"] = vf
    return out
