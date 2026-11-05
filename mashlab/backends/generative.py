"""Generative glue backend (design E.1 / G, optional tier).

GenerativeBackend produces optional re-sung hooks / transition risers. The v1 default is NullGenerative,
which generates nothing (the pipeline uses source fills + DSP sweeps only). YuE2Glue is the optional quality
tier backed by the existing YuE2 install; it is never required by the local path and defaults off. Both expose
unload().
"""
from abc import ABC, abstractmethod


def _empty_cuda_cache():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


class GenerativeBackend(ABC):
    """Optionally synthesize audio (risers, re-sung hooks). available() tells the orchestrator whether to use
    it; the local path must run with available() False."""

    name = "abstract-generative"

    def available(self):
        return False

    @abstractmethod
    def generate(self, spec):
        raise NotImplementedError

    def unload(self):
        _empty_cuda_cache()


class NullGenerative(GenerativeBackend):
    """v1 default: generates nothing. The orchestrator falls back to source fills + DSP sweeps."""

    name = "null"

    def available(self):
        return False

    def generate(self, spec):
        return None

    def unload(self):
        return None


class YuE2Glue(GenerativeBackend):
    """Optional quality tier: a YuE2 install (<YuE2 folder>/src/yue2). Available only when `yue2` is
    importable. Default off; the generation call is wired in a later FEAT. unload() releases the model."""

    name = "yue2"

    def __init__(self):
        self._model = None

    def available(self):
        from . import importable
        return importable("yue2")

    def generate(self, spec):  # pragma: no cover - optional tier, wired later
        raise NotImplementedError("YuE2Glue generation is wired in the generative FEAT")

    def unload(self):
        self._model = None
        _empty_cuda_cache()


def local_generative():
    """The default generative backend: NullGenerative (never required)."""
    return NullGenerative()
