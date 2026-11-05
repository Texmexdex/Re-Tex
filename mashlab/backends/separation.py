"""Separation backend (design E.1).

SeparationBackend wraps the existing separate.py paths (Demucs + BS-RoFormer Ultra/SW/blend) as the local
default; the premium slot is reserved (e.g. a cloud separator) but the four current modes already satisfy
quality, so there is no premium impl in v1. unload() is a safe no-op here because separate.py manages its own
module-level model handles and already empties the CUDA cache after each run.
"""
from abc import ABC, abstractmethod


def available_modes():
    """Which separation modes are usable on this install. Demucs is always present (local default); the
    BS-RoFormer Ultra/SW/blend modes require their checkpoints (separate.ultra_available / SW_CKPT)."""
    from .. import separate
    modes = ["demucs"]
    try:
        if separate.ultra_available():
            modes.append("ultra")
    except Exception:
        pass
    try:
        from ..config import SEP_MODELS
        if (SEP_MODELS / separate.SW_CKPT).exists():
            modes.append("sw")
            if "ultra" in modes:
                modes.append("blend")
    except Exception:
        pass
    return modes


class SeparationBackend(ABC):
    """Separate a stereo mix into {drums,bass,other,vocals} stems."""

    name = "abstract-separation"

    @abstractmethod
    def separate(self, y, mode="best", progress=lambda f: None):
        raise NotImplementedError

    def unload(self):
        return None


class LocalSeparation(SeparationBackend):
    """Local default: dispatches to separate.py. mode: 'fast'/'best' (Demucs), 'ultra', 'sw', 'blend'."""

    name = "local-separate"

    def separate(self, y, mode="best", progress=lambda f: None):
        from .. import separate
        if mode == "ultra":
            return separate.separate_ultra(y, progress)
        if mode == "sw":
            return separate.separate_sw(y, progress)
        if mode == "blend":
            return separate.separate_blend(y, progress)
        return separate.separate(y, quality=mode, progress=progress)

    def unload(self):
        """separate.py owns its model handles and empties the CUDA cache after each run; nothing to drop here."""
        return None


def local_separation():
    return LocalSeparation()
