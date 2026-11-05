"""ASR (lyrics) backends (design A.1 / E.1).

AsrBackend.transcribe(vocal_audio, sr) -> lyrics.json-shaped dict. The local default is faster-whisper
large-v3 on CUDA float16 with the pinned decode options; premium tiers are WhisperX (word alignment) and the
generic OpenAI-compatible cloud adapter (CloudAsrBackend). Every backend exposes an explicit unload() that
drops model refs and calls torch.cuda.empty_cache() (design E.2) — a safe no-op when nothing is loaded.

This module defines the interface + lazy-loading impls; the heavy models are imported only inside transcribe/
load so importing mashlab.backends never pulls torch/faster-whisper.
"""
from abc import ABC, abstractmethod

from .cloud import CloudAsrBackend as _CloudAsrClient


# faster-whisper large-v3 pinned decode options (design A.1). Lyrics repeat, so condition_on_previous_text is
# False to avoid loops; word timestamps + temperature fallback are on.
# VAD tuned for ISOLATED VOCAL STEMS (not full mixes): a stem has long silent gaps between phrases (no
# backing to fill them), which the default VAD over-clips/mis-segments, producing garbled low-confidence
# fragments. A lower speech-probability threshold + a longer allowed in-phrase silence keeps sung phrases
# intact. This (plus the peak-normalization in transcribe()) is the main lever on transcription quality.
FW_MODEL = "large-v3"
FW_OPTS = dict(vad_filter=True,
               vad_parameters=dict(threshold=0.2, min_silence_duration_ms=2000, speech_pad_ms=400),
               word_timestamps=True, beam_size=5,
               temperature=[0.0, 0.2, 0.4, 0.6], condition_on_previous_text=False)


def _empty_cuda_cache():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


class AsrBackend(ABC):
    """Transcribe an isolated vocal stem into a lyrics.json-shaped dict."""

    name = "abstract-asr"

    @abstractmethod
    def transcribe(self, vocal_audio, sr):
        """vocal_audio: mono float32 numpy at `sr`. Returns a dict with at least `segments` (list of
        {t0,t1,text,...}) and `backend`. Confidence (`quality.asr_conf`) is computed by the lyrics layer
        (FEAT-002), not here — this returns the raw transcript."""
        raise NotImplementedError

    def unload(self):
        """Drop model references + empty the CUDA cache. Base impl is a safe no-op."""
        _empty_cuda_cache()


class FasterWhisperLarge(AsrBackend):
    """Local default: faster-whisper (CTranslate2) large-v3, CUDA float16 (A12 gate), pinned FW_OPTS."""

    name = "faster-whisper-large-v3"

    def __init__(self, model_size=FW_MODEL, device=None, compute_type=None):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self._model = None

    def _load(self):
        if self._model is None:
            from faster_whisper import WhisperModel
            device = self.device
            compute_type = self.compute_type
            if device is None:
                try:
                    import ctranslate2
                    if ctranslate2.get_cuda_device_count() >= 1:
                        device, compute_type = "cuda", compute_type or "float16"
                    else:
                        device, compute_type = "cpu", compute_type or "int8"
                except Exception:
                    device, compute_type = "cpu", compute_type or "int8"
            self.device, self.compute_type = device, compute_type
            self._model = WhisperModel(self.model_size, device=device, compute_type=compute_type)
        return self._model

    def transcribe(self, vocal_audio, sr):
        import numpy as np
        model = self._load()
        audio = np.asarray(vocal_audio, dtype=np.float32)
        if audio.ndim > 1:
            audio = audio.mean(axis=0)
        if sr != 16000:
            import soxr
            audio = soxr.resample(audio, sr, 16000, quality="HQ").astype(np.float32)
        # Peak-normalize the isolated vocal stem before transcription. The stem is stored at its true
        # (often low / heavily-processed) level, and Whisper transcribes quiet audio poorly; scaling the
        # peak to ~0.95 full-scale materially lifts transcription quality and asr_conf. Guard against a
        # silent/empty stem (don't divide by ~0).
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        if peak > 1e-4:
            audio = (audio * (0.95 / peak)).astype(np.float32)
        segments, info = model.transcribe(audio, **FW_OPTS)
        segs = []
        for s in segments:
            words = [dict(t=float(w.start), w=w.word, p=float(w.probability))
                     for w in (s.words or [])] if getattr(s, "words", None) else []
            segs.append(dict(t0=float(s.start), t1=float(s.end), text=s.text,
                             avg_logprob=float(s.avg_logprob), no_speech_prob=float(s.no_speech_prob),
                             words=words))
        return dict(version=1, backend=self.name, lang=getattr(info, "language", None), segments=segs)

    def unload(self):
        self._model = None
        _empty_cuda_cache()


class WhisperXBackend(AsrBackend):
    """Premium tier: WhisperX for sharper word alignment. Loaded lazily; unload() releases it."""

    name = "whisperx"

    def __init__(self, model_size=FW_MODEL, device=None):
        self.model_size = model_size
        self.device = device
        self._model = None

    def _load(self):
        if self._model is None:
            import whisperx
            import torch
            device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
            self.device = device
            compute = "float16" if device == "cuda" else "int8"
            self._model = whisperx.load_model(self.model_size, device, compute_type=compute)
        return self._model

    def transcribe(self, vocal_audio, sr):
        import numpy as np
        model = self._load()
        audio = np.asarray(vocal_audio, dtype=np.float32)
        if audio.ndim > 1:
            audio = audio.mean(axis=0)
        if sr != 16000:
            import soxr
            audio = soxr.resample(audio, sr, 16000, quality="HQ").astype(np.float32)
        result = model.transcribe(audio)
        segs = []
        for s in result.get("segments", []):
            segs.append(dict(t0=float(s.get("start", 0.0)), t1=float(s.get("end", 0.0)),
                             text=s.get("text", ""), avg_logprob=float(s.get("avg_logprob", 0.0)),
                             no_speech_prob=float(s.get("no_speech_prob", 0.0)), words=[]))
        return dict(version=1, backend=self.name, lang=result.get("language"), segments=segs)

    def unload(self):
        self._model = None
        _empty_cuda_cache()


class CloudAsrBackend(AsrBackend):
    """Premium cloud tier: the generic OpenAI-compatible transcription adapter (design E.1, OD-2). DEFAULT OFF
    — raises CloudDisabled unless data/config.json cloud.asr.enabled is true. `cfg` is the cloud.asr block."""

    name = "cloud-asr"

    def __init__(self, cfg):
        self.client = _CloudAsrClient(cfg)

    def available(self):
        return self.client.available()

    def transcribe(self, vocal_audio, sr):
        import io
        import numpy as np
        import soundfile as sf
        audio = np.asarray(vocal_audio, dtype=np.float32)
        if audio.ndim > 1:
            audio = audio.mean(axis=0)
        buf = io.BytesIO()
        sf.write(buf, audio, int(sr), format="WAV")
        resp = self.client.transcribe_file(buf.getvalue(), filename="vocal.wav",
                                            response_format="verbose_json")
        segs = []
        for s in resp.get("segments", []) or []:
            segs.append(dict(t0=float(s.get("start", 0.0)), t1=float(s.get("end", 0.0)),
                             text=s.get("text", ""), avg_logprob=float(s.get("avg_logprob", 0.0)),
                             no_speech_prob=float(s.get("no_speech_prob", 0.0)), words=[]))
        if not segs and resp.get("text"):
            segs = [dict(t0=0.0, t1=0.0, text=resp["text"], avg_logprob=0.0, no_speech_prob=0.0, words=[])]
        return dict(version=1, backend=self.name, lang=resp.get("language"), segments=segs)

    def unload(self):
        self.client.unload()


def local_asr():
    """The local-default ASR backend (faster-whisper large-v3)."""
    return FasterWhisperLarge()
