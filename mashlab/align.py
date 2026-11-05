"""Lyrics alignment ("forced alignment" - what YouTube does with an uploaded transcript): the words are known (yours,
typed), only WHEN each is sung is found, by matching the sounds those letters make against the vocal audio with a
character-level speech model (wav2vec2, trained on read English; torchaudio's WAV2VEC2_ASR_BASE_960H, cached locally
in the torch hub folder) and CTC Viterbi alignment (torchaudio.functional.forced_align).

Whisper guesses the words and mishears; this never changes a word. Singing and rap are harder than speech (stretched
vowels, slang, abbreviations the model can't spell out), so each word gets a confidence, and a word that would start
in silence is moved to where the voice actually comes in.

align(y, sr, text) -> [{t, t1, w, conf, line}]   lines(words, text) -> [{t0, t1, text, words}]
"""
import re
import threading

import numpy as np

SR16 = 16000
WIN_S = 30                       # seconds of audio per model pass (emissions are joined)
_model = None
_lock = threading.Lock()

NUMBERS = {"0": "zero", "1": "one", "2": "two", "3": "three", "4": "four", "5": "five", "6": "six", "7": "seven",
           "8": "eight", "9": "nine", "10": "ten", "12": "twelve", "16": "sixteen", "20": "twenty", "100": "hundred"}


def _load():
    global _model
    if _model is None:
        import torch
        import torchaudio
        B = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        _model = (B.get_model().to(dev).eval(), {c: i for i, c in enumerate(B.get_labels())}, dev)
    return _model


def unload():
    global _model
    with _lock:
        _model = None
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass


def text_lines(text):
    """The sung lines of a lyrics text ([Section] tags and blank lines dropped)."""
    return [l.strip() for l in str(text).splitlines() if l.strip() and not re.fullmatch(r"\[[^\]]*\]", l.strip())]


def _spell(word):
    """A word as the model's letters (A-Z and '): digits read out, everything else dropped."""
    w = NUMBERS.get(word.strip(".,!?;:"), word)
    return re.sub(r"[^A-Z']", "", w.upper())


def align(y, sr, text):
    """Word timings for `text` sung in `y` (mono vocals). -> [{t, t1, w, conf, line}] (seconds)."""
    import torch
    import torchaudio
    import soxr
    lines = text_lines(text)
    words = [(li, w, _spell(w)) for li, l in enumerate(lines) for w in l.split()]
    words = [x for x in words if x[2]]
    if not words:
        return []
    x = soxr.resample(np.asarray(y, np.float32), sr, SR16) if sr != SR16 else np.asarray(y, np.float32)
    peak = float(np.max(np.abs(x))) or 1.0
    x = x / peak * 0.9
    with _lock:
        model, idx, dev = _load()
        em = []
        with torch.inference_mode():
            for a in range(0, len(x), WIN_S * SR16):
                e, _ = model(torch.from_numpy(x[a:a + WIN_S * SR16]).to(dev)[None])
                em.append(torch.log_softmax(e, -1)[0].float().cpu())
        em = torch.cat(em)
    fps = em.shape[0] / (len(x) / SR16)
    tokens = [idx[c] for k, (_, _, cw) in enumerate(words) for c in (("|" if k else "") + cw)]
    if len(tokens) > em.shape[0]:
        raise ValueError("more letters than audio frames - is this the right audio for these lyrics?")
    ali, sc = torchaudio.functional.forced_align(em[None], torch.tensor([tokens], dtype=torch.int32), blank=0)
    ali, sc = ali[0].numpy(), sc[0].exp().numpy()
    spans, prev = [], 0                                  # one span per target token, in order
    for f, tok in enumerate(ali):
        if tok != 0 and (tok != prev or prev == 0):
            spans.append([f, f])
        elif tok != 0 and spans:
            spans[-1][1] = f
        prev = tok

    voiced = _voiced(x)                                  # 10 ms frames
    out, ti = [], 0
    for k, (li, w, cw) in enumerate(words):
        if k:
            ti += 1                                      # the word separator
        seg = spans[ti:ti + len(cw)]
        ti += len(cw)
        if not seg:
            continue
        t0, t1 = seg[0][0] / fps, (seg[-1][1] + 1) / fps
        conf = float(np.mean([sc[a:b + 1].mean() for a, b in seg]))
        # a word can't start in silence: move its start to where the voice comes in inside its span
        f0, f1 = int(t0 * 100), int(t1 * 100)
        if f1 > f0 and not voiced[min(f0, len(voiced) - 1)]:
            on = np.flatnonzero(voiced[f0:f1])
            if len(on):
                t0 = (f0 + on[0]) / 100
        out.append(dict(t=round(t0, 3), t1=round(max(t1, t0 + 0.05), 3), w=w, conf=round(conf, 3), line=li))
    return settle(out, voiced)


def settle(words, voiced, hop=0.01):
    """A word whose whole span is silent was parked there by the aligner (CTC puts letters it can't place in the
    silence - e.g. the first word at 0:00 of a vocal that comes in at 0:40): move it to where the voice next comes
    in, before the following word. Last word first, so each move sees where the next word finally is."""
    n = len(voiced)
    for k in range(len(words) - 1, -1, -1):
        w = words[k]
        f0 = min(int(w["t"] / hop), n)
        f1 = min(max(int(w["t1"] / hop), f0 + 1), n)
        if voiced[f0:f1].any():
            continue
        lim = min(int(words[k + 1]["t"] / hop), n) if k + 1 < len(words) else n
        on = np.flatnonzero(voiced[f1:lim])
        if len(on):
            t = (f1 + on[0]) * hop
            w["t1"] = round(min(t + (w["t1"] - w["t"]), lim * hop), 3)
            w["t"] = round(t, 3)
    return words


def _voiced(x):
    """Is the vocal sounding, per 10 ms (within 30 dB of its loud parts)."""
    hop = SR16 // 100
    n = len(x) // hop
    rms = np.sqrt(np.mean(x[: n * hop].reshape(n, hop) ** 2, axis=1) + 1e-12)
    ref = np.quantile(rms, 0.95) or 1e-6
    v = rms > ref * 10 ** (-30 / 20)
    v = np.convolve(v.astype(float), np.ones(5), "same") > 0       # close 40 ms gaps (consonants)
    return v


def lines(words, text):
    """Group aligned words back into the text's lines. -> [{t0, t1, text, words, conf, line (its index in the text)}]"""
    tl = text_lines(text)
    out = []
    for li, l in enumerate(tl):
        ws = [w for w in words if w["line"] == li]
        if ws:
            out.append(dict(t0=ws[0]["t"], t1=ws[-1]["t1"], text=l, conf=round(float(np.mean([w["conf"] for w in ws])), 3),
                            words=[dict(t=w["t"], w=w["w"]) for w in ws], line=li))
    return out


# ---------------------------------------------------------------- run in a process of its own
# Whisper's CTranslate2 and torch each bring a cuDNN; once Whisper's is loaded in a process, torch's convolutions there
# fail ("Could not load symbol cudnnGetLibConfig") and take the process down. Reimagine runs Whisper first, so the
# aligner always runs in a fresh process (~2-3 s to load the model).

def align_file(wav, text, timeout=600):
    """align() on an audio file, in a separate process. -> words"""
    import json
    import subprocess
    import sys
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory(prefix="mashlab-align-") as d:
        tf, of = Path(d) / "lyrics.txt", Path(d) / "words.json"
        tf.write_text(str(text), encoding="utf-8")
        r = subprocess.run([sys.executable, "-m", "mashlab.align", str(wav), str(tf), str(of)], capture_output=True,
                           text=True, encoding="utf-8", errors="replace", timeout=timeout,
                           cwd=str(Path(__file__).resolve().parent.parent))
        if r.returncode != 0 or not of.exists():
            tail = (r.stderr or r.stdout or "").strip().splitlines()[-3:]
            raise RuntimeError("lyrics alignment failed: " + " | ".join(tail))
        return json.loads(of.read_text(encoding="utf-8"))


if __name__ == "__main__":
    import json
    import sys
    from pathlib import Path
    import torch  # noqa: F401  - torch (and its cuDNN) first in this process
    from .audio_io import load
    wav, txt, out = sys.argv[1:4]
    y = load(wav, SR16).mean(0).astype(np.float32)
    words = align(y, SR16, Path(txt).read_text(encoding="utf-8"))
    Path(out).write_text(json.dumps(words), encoding="utf-8")
