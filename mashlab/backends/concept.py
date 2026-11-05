"""Concept-profiling backends (design A.2 / E.1).

ConceptBackend.profile(lyrics) -> concept.json-shaped dict. The v1 LOCAL DEFAULT is MiniLM embedding +
zero-shot mood (no LLM, zero downloads). If detect() finds a GGUF instruction model and llama-cpp-python is
importable, MiniLmLlamaCpp adds the richer theme/person/tense/imagery tags automatically. The premium tier is
CloudConcept (the OpenAI-compatible LLM adapter) for tags, optionally with a larger embedding model.

Each backend exposes unload(). Heavy models are imported lazily inside load/profile so importing
mashlab.backends never pulls sentence-transformers or torch.
"""
from abc import ABC, abstractmethod

from .cloud import CloudLlmBackend as _CloudLlmClient


EMBED_MODEL = "all-MiniLM-L6-v2"            # local default sentence-transformers embedder (384-d)
EMBED_MODEL_PREMIUM = "all-mpnet-base-v2"   # optional larger embedder (premium quality tier)

# The 11 fixed moods (design A.2.2). Zero-shot cosine of concept_vec vs these prototype sentences.
MOODS = ("aggressive", "melancholy", "euphoric", "romantic", "dark", "playful",
         "reflective", "defiant", "sorrowful", "triumphant", "numb")
MOOD_PROTOTYPES = {
    "aggressive": "angry, hostile, confrontational lyrics full of rage and attack",
    "melancholy": "sad, wistful, gently sorrowful and bittersweet lyrics",
    "euphoric": "ecstatic, joyful, euphoric and celebratory lyrics",
    "romantic": "tender, loving, romantic lyrics about desire and affection",
    "dark": "ominous, bleak, menacing and shadowy lyrics",
    "playful": "lighthearted, fun, cheeky and playful lyrics",
    "reflective": "thoughtful, introspective, contemplative lyrics",
    "defiant": "rebellious, defiant, unbowed and resistant lyrics",
    "sorrowful": "grieving, heartbroken, deeply mournful lyrics",
    "triumphant": "victorious, powerful, triumphant and uplifting lyrics",
    "numb": "detached, empty, emotionally numb and hollow lyrics",
}


def _empty_cuda_cache():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _lyrics_text(lyrics):
    """Concatenate lyrics.json segment texts into one string."""
    if not lyrics:
        return ""
    return " ".join((s.get("text") or "").strip() for s in lyrics.get("segments", [])).strip()


class ConceptBackend(ABC):
    """Profile a lyrics dict into a concept dict (embedding + mood tags)."""

    name = "abstract-concept"

    @abstractmethod
    def profile(self, lyrics):
        raise NotImplementedError

    def unload(self):
        _empty_cuda_cache()


class _MiniLmEmbedMixin:
    """Shared MiniLM embedding + zero-shot mood. Model loaded lazily; reused across profile() calls."""

    embed_model = EMBED_MODEL

    def __init__(self):
        self._embed = None

    def _embedder(self):
        if self._embed is None:
            from sentence_transformers import SentenceTransformer
            self._embed = SentenceTransformer(self.embed_model)
        return self._embed

    def _embed_vec(self, text):
        import numpy as np
        model = self._embedder()
        vec = model.encode([text or ""], normalize_embeddings=True)[0]
        return np.asarray(vec, dtype=np.float32)

    def _zero_shot_mood(self, concept_vec):
        """Cosine of concept_vec against the fixed mood prototype embeddings; returns the top mood."""
        import numpy as np
        model = self._embedder()
        protos = model.encode([MOOD_PROTOTYPES[m] for m in MOODS], normalize_embeddings=True)
        sims = np.asarray(protos, dtype=np.float32) @ np.asarray(concept_vec, dtype=np.float32)
        return MOODS[int(np.argmax(sims))]

    def unload(self):
        self._embed = None
        _empty_cuda_cache()


class MiniLmZeroShot(_MiniLmEmbedMixin, ConceptBackend):
    """v1 LOCAL DEFAULT: MiniLM embedding + zero-shot mood. No LLM, zero extra downloads. theme/person/tense
    are null; mood is zero-shot; imagery is left to the lyrics layer (TF-IDF, FEAT-002)."""

    name = "zeroshot-minilm"

    def profile(self, lyrics):
        text = _lyrics_text(lyrics)
        vec = self._embed_vec(text)
        mood = self._zero_shot_mood(vec) if text else None
        return dict(version=1, embed_model=self.embed_model, concept_vec=vec.tolist(),
                    tags=dict(theme=None, mood=mood, person=None, tense=None, imagery=[]),
                    tags_backend=self.name, has_tags=False)


class MiniLmLlamaCpp(_MiniLmEmbedMixin, ConceptBackend):
    """Optional enhancement (auto when a GGUF is detected): MiniLM embedding + llama.cpp structured tags
    (theme/person/tense/imagery, mood override). On parse failure it falls back to zero-shot mood. The GGUF
    path + prompt/schema are implemented in FEAT-002's concept.py; here the embedding + mood baseline is
    produced and `has_tags` flags whether the LLM filled the richer tags."""

    name = "llama.cpp"

    def __init__(self, gguf_path=None):
        _MiniLmEmbedMixin.__init__(self)
        self.gguf_path = gguf_path
        self._llm = None

    def profile(self, lyrics):
        text = _lyrics_text(lyrics)
        vec = self._embed_vec(text)
        mood = self._zero_shot_mood(vec) if text else None
        tag_backend = self.name + (f":{self.gguf_path.name}" if self.gguf_path else "")
        return dict(version=1, embed_model=self.embed_model, concept_vec=vec.tolist(),
                    tags=dict(theme=None, mood=mood, person=None, tense=None, imagery=[]),
                    tags_backend=tag_backend, has_tags=False)

    def unload(self):
        self._llm = None
        _MiniLmEmbedMixin.unload(self)


class CloudConcept(_MiniLmEmbedMixin, ConceptBackend):
    """Premium tier: local MiniLM embedding + cloud OpenAI-compatible LLM tags (CloudLlmTags). DEFAULT OFF
    (the cloud client raises unless enabled). `cfg` is data/config.json cloud.llm block."""

    name = "cloud-llm"

    def __init__(self, cfg):
        _MiniLmEmbedMixin.__init__(self)
        self.client = _CloudLlmClient(cfg)

    def available(self):
        return self.client.available()

    def profile(self, lyrics):
        # Embedding + zero-shot baseline always; the LLM tag enrichment is wired in FEAT-002's concept.py
        # against this client (CloudLlmBackend.chat). Kept here as the baseline so the premium backend is a
        # drop-in replacement with the same output shape.
        text = _lyrics_text(lyrics)
        vec = self._embed_vec(text)
        mood = self._zero_shot_mood(vec) if text else None
        return dict(version=1, embed_model=self.embed_model, concept_vec=vec.tolist(),
                    tags=dict(theme=None, mood=mood, person=None, tense=None, imagery=[]),
                    tags_backend=self.name, has_tags=False)

    def unload(self):
        self.client.unload()
        _MiniLmEmbedMixin.unload(self)


def local_concept():
    """The local-default concept backend: MiniLmLlamaCpp when a GGUF is present (and llama-cpp-python is
    importable), else MiniLmZeroShot (no LLM). Mirrors backends.detect()'s concept impl choice."""
    from . import find_gguf, importable
    gguf = find_gguf()
    if gguf is not None and importable("llama_cpp"):
        return MiniLmLlamaCpp(gguf_path=gguf)
    return MiniLmZeroShot()
