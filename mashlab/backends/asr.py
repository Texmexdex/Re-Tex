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

# Opt-in singing-friendly decode options: disables Silero VAD (which often drops soft/breathy/reverb-drenched
# singing phrases) and relies instead on strict dead-silence island splitting + per-island peak normalization.
FW_OPTS_NOVAD = dict(vad_filter=False, word_timestamps=True, beam_size=5,
                     temperature=[0.0, 0.2, 0.4, 0.6], condition_on_previous_text=False,
                     no_speech_threshold=0.85, compression_ratio_threshold=2.2)

_SILENCE_HALLUCINATIONS = {
    "thank you", "thank you.", "thanks for watching", "thanks for watching.",
    "thank you for watching", "thank you for watching.", "please subscribe",
    "subscribe", "you", "bye", "bye.", "...",
    "продолжение следует", "продолжение следует...", "субтитры",
    "sous-titres", "subtitles by", "amara.org",
}
_ALWAYS_DROP_SUBTITLES = (
    "продолжение следует",
    "thanks for watching",
    "thank you for watching",
    "subtitles by",
    "amara.org",
    "субтитры сделал",
)


def _to_mono_16k(vocal_audio, sr):
    import numpy as np
    audio = np.asarray(vocal_audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=0)
    if int(sr) != 16000:
        import soxr
        audio = soxr.resample(audio, int(sr), 16000, quality="HQ").astype(np.float32)
    return audio


def _active_spans(audio_16k, min_silence_s=2.0, db_thresh=-52.0, pad_s=0.45, max_chunk_s=None):
    """Find non-silent vocal spans [(s0, s1), ...] in 16 kHz mono audio so VAD-off / chunked decoders skip dead
    instrumental silence (preventing hallucinations) without dropping soft breathy singing. When `max_chunk_s` is
    set, long spans are split at the quietest RMS point inside [0.55*max, max] so words aren't chopped mid-syllable."""
    import numpy as np
    n = len(audio_16k)
    if n == 0:
        return []
    peak = float(np.max(np.abs(audio_16k)))
    if peak <= 1e-4:
        return []
    hop = 800  # 50 ms at 16 kHz
    n_frames = max(1, (n + hop - 1) // hop)
    padded = np.zeros(n_frames * hop, dtype=np.float32)
    padded[:n] = audio_16k
    rms = np.sqrt(np.mean(padded.reshape(n_frames, hop) ** 2, axis=1))
    # Adaptive floor: max of absolute dead-silence threshold and -58 dB below track peak
    thresh = max(10.0 ** (db_thresh / 20.0), peak * (10.0 ** (-58.0 / 20.0)))
    active = rms >= thresh
    if not np.any(active):
        spans = [(0, n)]
    else:
        min_gap_f = max(1, int(round(min_silence_s * 16000 / hop)))
        pad_f = max(0, int(round(pad_s * 16000 / hop)))
        raw = []
        i = 0
        while i < n_frames:
            if not active[i]:
                i += 1
                continue
            j = i + 1
            while j < n_frames and active[j]:
                j += 1
            if raw and (i - raw[-1][1]) < min_gap_f:
                raw[-1] = (raw[-1][0], j)
            else:
                raw.append((i, j))
            i = j
        spans = []
        for a, b in raw:
            s0 = max(0, (a - pad_f) * hop)
            s1 = min(n, (b + pad_f) * hop)
            if spans and s0 <= spans[-1][1]:
                spans[-1] = (spans[-1][0], max(spans[-1][1], s1))
            elif s1 - s0 >= 1600:  # >= 0.1 s
                spans.append((s0, s1))
        if not spans:
            spans = [(0, n)]

    if not max_chunk_s:
        return spans

    max_samp = int(round(max_chunk_s * 16000))
    min_split = int(round(max_chunk_s * 0.55 * 16000))
    chunked = []
    for s0, s1 in spans:
        cur = s0
        while s1 - cur > max_samp:
            lo_f = min(n_frames - 1, (cur + min_split) // hop)
            hi_f = min(n_frames, (cur + max_samp) // hop)
            if hi_f > lo_f:
                best_f = lo_f + int(np.argmin(rms[lo_f:hi_f]))
                cut = max(cur + min_split, min(s1, best_f * hop))
            else:
                cut = cur + max_samp
            chunked.append((cur, cut))
            cur = cut
        if s1 - cur >= 1600:
            chunked.append((cur, s1))
    return chunked


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
            if self.model_size == FW_MODEL:
                from .. import components
                components.require("whisper")   # its 3 GB model downloads only when you press Download
            from faster_whisper import WhisperModel
            device = self.device
            compute_type = self.compute_type
            if device is None:
                from .. import components
                problem = components.C["ct2_cuda"]["check"]()
                if problem is None:
                    device, compute_type = "cuda", compute_type or "float16"
                else:                         # runs, slower and a little less accurate - and says so
                    device, compute_type = "cpu", compute_type or "int8"
                    components.fallback("ct2_cuda", "Whisper runs on the processor (int8) instead of the graphics "
                                                    "card - much slower, slightly less accurate", problem=problem)
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


class FasterWhisperNoVad(FasterWhisperLarge):
    """Opt-in singing variant of faster-whisper large-v3: disables Silero VAD (which drops soft, breathy, or
    reverb-drenched vocal phrases) and instead splits only on true dead silence (<-52 dB) with per-island peak
    normalization so quiet verses aren't crushed by loud choruses. Never alters the default FasterWhisperLarge."""

    name = "faster-whisper-large-v3-novad"

    def transcribe(self, vocal_audio, sr):
        import numpy as np
        model = self._load()
        audio = _to_mono_16k(vocal_audio, sr)
        spans = _active_spans(audio, min_silence_s=2.0, db_thresh=-52.0, pad_s=0.45)
        segs = []
        lang = None
        for s0, s1 in spans:
            chunk = audio[s0:s1]
            pk = float(np.max(np.abs(chunk)))
            if pk <= 1e-4:
                continue
            chunk = (chunk * (0.95 / pk)).astype(np.float32)
            offset = s0 / 16000.0
            segments, info = model.transcribe(chunk, **FW_OPTS_NOVAD)
            if lang is None:
                lang = getattr(info, "language", None)
            for s in segments:
                txt = (s.text or "").strip()
                if not txt:
                    continue
                nosp = float(getattr(s, "no_speech_prob", 0.0))
                logp = float(getattr(s, "avg_logprob", 0.0))
                low = txt.lower().strip()
                if nosp > 0.92:
                    continue
                if any(h in low for h in _ALWAYS_DROP_SUBTITLES):
                    continue
                if low in _SILENCE_HALLUCINATIONS and (nosp > 0.45 or logp < -0.9):
                    continue
                words = [dict(t=round(offset + float(w.start), 3), w=w.word, p=float(w.probability))
                         for w in (s.words or [])] if getattr(s, "words", None) else []
                segs.append(dict(
                    t0=round(offset + float(s.start), 3),
                    t1=round(offset + float(s.end), 3),
                    text=txt, avg_logprob=logp, no_speech_prob=nosp, words=words,
                ))
        return dict(version=1, backend=self.name, lang=lang, segments=segs)


class ParakeetTdtAsr(AsrBackend):
    """Opt-in local ONNX ASR: NVIDIA Parakeet-TDT 0.6B v3 (int8 via onnx-asr + onnxruntime). Runs on CPU with
    zero PyTorch/CUDA dependency conflict and produces word-level timestamps."""

    name = "parakeet-tdt-0.6b-v3"
    MODEL_NAME = "nemo-parakeet-tdt-0.6b-v3"

    def __init__(self, quantization="int8"):
        self.quantization = quantization
        self._model = None

    def _load(self):
        if self._model is None:
            from .. import components
            components.require("parakeet_tdt")
            import onnx_asr
            self._model = onnx_asr.load_model(self.MODEL_NAME, quantization=self.quantization).with_timestamps()
        return self._model

    @staticmethod
    def _tokens_to_segments(tokens, timestamps, logprobs, offset, chunk_dur):
        """Merge SentencePiece tokens (space or '▁' word starts) into words, then group words into natural lyric
        lines split by pauses >= 0.75 s, sentence punctuation, or line length."""
        import math
        words = []
        for idx, tok in enumerate(tokens or []):
            t = float(timestamps[idx]) if timestamps and idx < len(timestamps) else 0.0
            lp = float(logprobs[idx]) if logprobs and idx < len(logprobs) else -0.3
            if not tok:
                continue
            starts_word = tok.startswith(" ") or tok.startswith("\u2581")
            piece = tok.lstrip(" \u2581")
            if not piece:
                continue
            if starts_word or not words:
                words.append(dict(t0=offset + t, t1=offset + t, w=piece, lps=[lp]))
            else:
                words[-1]["w"] += piece
                words[-1]["t1"] = offset + t
                words[-1]["lps"].append(lp)

        if not words:
            return []
        for i, w in enumerate(words):
            nxt_t = words[i + 1]["t0"] if i + 1 < len(words) else (offset + chunk_dur)
            w["t1"] = max(w["t1"] + 0.08, min(nxt_t, w["t0"] + 1.2))
            w["lp"] = sum(w["lps"]) / len(w["lps"])
            w["p"] = round(float(min(1.0, max(0.0, math.exp(w["lp"])))), 4)

        lines, cur = [], []
        for w in words:
            if cur:
                gap = w["t0"] - cur[-1]["t1"]
                prev_end = cur[-1]["w"].endswith((".", "?", "!"))
                if gap >= 0.75 or prev_end or len(cur) >= 12:
                    lines.append(cur)
                    cur = []
            cur.append(w)
        if cur:
            lines.append(cur)

        segs = []
        for group in lines:
            text = " ".join(x["w"] for x in group).strip()
            if not text:
                continue
            avg_lp = sum(x["lp"] for x in group) / len(group)
            segs.append(dict(
                t0=round(group[0]["t0"], 3),
                t1=round(group[-1]["t1"], 3),
                text=text,
                avg_logprob=round(float(avg_lp), 4),
                no_speech_prob=0.0,
                words=[dict(t=round(x["t0"], 3), w=" " + x["w"], p=x["p"]) for x in group],
            ))
        return segs

    def transcribe(self, vocal_audio, sr):
        import numpy as np
        model = self._load()
        audio = _to_mono_16k(vocal_audio, sr)
        spans = _active_spans(audio, min_silence_s=1.2, db_thresh=-50.0, pad_s=0.35, max_chunk_s=22.0)
        segs = []
        for s0, s1 in spans:
            chunk = audio[s0:s1]
            pk = float(np.max(np.abs(chunk)))
            if pk <= 1e-4:
                continue
            chunk = (chunk * (0.95 / pk)).astype(np.float32)
            offset = s0 / 16000.0
            chunk_dur = len(chunk) / 16000.0
            res = model.recognize(chunk, sample_rate=16000)
            if not (getattr(res, "text", "") or "").strip():
                continue
            if getattr(res, "tokens", None):
                segs.extend(self._tokens_to_segments(res.tokens, res.timestamps, res.logprobs, offset, chunk_dur))
            else:
                segs.append(dict(
                    t0=round(offset, 3), t1=round(offset + chunk_dur, 3),
                    text=res.text.strip(), avg_logprob=-0.3, no_speech_prob=0.0, words=[],
                ))
        return dict(version=1, backend=self.name, lang=None, segments=segs)

    def unload(self):
        self._model = None


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
            from .. import components
            device = self.device or components.device("WhisperX")
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
