"""Concept layer (design A.2 / A.3): turn a lyrics.json transcript into a concept.json profile and score the
lyrical compatibility between two songs.

Two responsibilities:

  * profiling (A.2) — concatenate the transcript texts, embed with MiniLM (L2-normalized 384-d concept_vec,
    via the ConceptBackend from FEAT-001), pick a zero-shot `mood`, approximate `imagery` with top-k TF-IDF
    content words, read the pinned NRC-VAD lexicon (under the MEDIUM-3 integrity gate) for a coarse
    valence/arousal, and compute the pinned, library-independent `judgeable`.

  * scoring (A.3) — `resonance(V, I)` / `contrast(V, I)` fold the base cosine (calibrated by the A.3
    endpoints), a fixed mood-antonym bonus, and a small VAD nudge into `Lraw ∈ [0,1]`; `lyrics_params()` is
    the single gate that chooses the safe (±5%) vs strong (±12%) factor from data/config.json's lyrics_calib.

Every formula/constant below is pinned VERBATIM from design A.2/A.3 (MEDIUM-1/2/3, NIT-3). The lexicon is a
quality enhancer, never a hard dependency: if it is absent, `valence/arousal` are null and the VAD term is 0.
"""
import hashlib
import os
import re

import numpy as np

from . import config

VERSION = 1

# The 11 fixed moods (design A.2.2). Mirrors backends.concept.MOODS.
MOODS = ("aggressive", "melancholy", "euphoric", "romantic", "dark", "playful",
         "reflective", "defiant", "sorrowful", "triumphant", "numb")

# Fixed antonym graph (design A.3, NIT-3). Every mood is a key; `defiant -> numb` is the one asymmetric entry,
# but moods_opposite() checks BOTH orderings so defiant<->numb scores -0.10 either way.
MOOD_ANTONYMS = {
    "aggressive": "reflective", "reflective": "aggressive",
    "melancholy": "euphoric", "euphoric": "melancholy",
    "sorrowful": "triumphant", "triumphant": "sorrowful",
    "dark": "playful", "playful": "dark",
    "numb": "romantic", "romantic": "numb",
    "defiant": "numb",   # defiant's opposite is resignation/numbness (asymmetric; see moods_opposite)
}

# Safe (pre-calibration) affine endpoints + factor strength (design A.3 gate).
_SAFE_LO, _SAFE_HI = 0.15, 0.65
_SAFE_ALPHA, _SAFE_CLAMP = 0.06, (0.95, 1.05)
_STRONG_ALPHA, _STRONG_CLAMP = 0.12, (0.90, 1.12)
_MIN_PAIRS = 50
_MIN_SPAN = 0.15

_MOOD_EQUAL_BONUS = 0.15
_MOOD_OPPOSITE_BONUS = -0.10
_VAD_WEIGHT = -0.08

_JUDGE_TOKEN_SAT = 40.0        # judgeable saturates at 40 distinct words

# content-word (imagery) stopwords — small, fixed, library-independent
_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "of", "to", "in", "on", "at", "for", "with", "as", "by",
    "is", "it", "its", "be", "am", "are", "was", "were", "been", "being", "i", "you", "he", "she", "we",
    "they", "me", "him", "her", "us", "them", "my", "your", "his", "our", "their", "this", "that", "these",
    "those", "do", "does", "did", "done", "have", "has", "had", "will", "would", "can", "could", "shall",
    "should", "may", "might", "must", "not", "no", "so", "up", "down", "out", "oh", "yeah", "na", "la",
    "all", "just", "like", "got", "get", "gonna", "wanna", "cause", "em", "ya", "uh", "ooh",
}

_WORD_RE = re.compile(r"[a-z0-9']+")


# ----------------------------------------------------------------- text helpers

def lyrics_text(lyrics):
    """Concatenate lyrics.json segment texts into one string."""
    if not lyrics:
        return ""
    return " ".join((s.get("text") or "").strip() for s in lyrics.get("segments", [])).strip()


def _tokens(text):
    """Lowercased, punctuation-stripped word tokens (design A.2 judgeable tokenization)."""
    return _WORD_RE.findall((text or "").lower())


# ----------------------------------------------------------------- judgeable (design A.2, pinned)

def judgeable(lyrics, asr_conf=None):
    """Pinned, library-independent transcript-usefulness in [0,1] (design A.2):

        judgeable = clip(asr_conf,0,1) * clip(n_unique_tokens/40,0,1) * (1 - clip(repeat_ratio,0,1))

    `asr_conf` defaults to lyrics.quality.asr_conf so judgeable and the A.5 guard read the same authority. A
    near-empty / garbled transcript (low asr_conf or few unique tokens) and an all-"na-na-na" hook (high
    repeat_ratio) both collapse judgeable toward 0."""
    if asr_conf is None:
        asr_conf = ((lyrics or {}).get("quality", {}) or {}).get("asr_conf", 0.0)
    toks = _tokens(lyrics_text(lyrics))
    n_tokens = len(toks)
    n_unique = len(set(toks))
    repeat_ratio = 1.0 - n_unique / max(1, n_tokens)
    return float(np.clip(asr_conf, 0, 1)
                 * np.clip(n_unique / _JUDGE_TOKEN_SAT, 0, 1)
                 * (1.0 - np.clip(repeat_ratio, 0, 1)))


def imagery(lyrics, k=5):
    """Top-k content words by TF-IDF-style weight within the single transcript. With one document TF-IDF
    reduces to frequency over content (non-stopword) tokens; returns up to k most frequent content words."""
    toks = [t for t in _tokens(lyrics_text(lyrics)) if t not in _STOPWORDS and len(t) > 2]
    if not toks:
        return []
    freq = {}
    for t in toks:
        freq[t] = freq.get(t, 0) + 1
    ranked = sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))
    return [w for w, _ in ranked[:k]]


# ----------------------------------------------------------------- mood graph (design A.3, pinned)

def moods_equal(a, b):
    return a is not None and b is not None and a == b


def moods_opposite(a, b):
    """True iff MOOD_ANTONYMS.get(a) == b OR MOOD_ANTONYMS.get(b) == a (BOTH orderings, NIT-3), so the one
    asymmetric entry (defiant->numb) scores -0.10 in effect either way."""
    if a is None or b is None:
        return False
    return MOOD_ANTONYMS.get(a) == b or MOOD_ANTONYMS.get(b) == a


def mood_bonus(a, b):
    """+0.15 equal / -0.10 opposite / 0 otherwise (design A.3)."""
    if moods_equal(a, b):
        return _MOOD_EQUAL_BONUS
    if moods_opposite(a, b):
        return _MOOD_OPPOSITE_BONUS
    return 0.0


# ----------------------------------------------------------------- NRC-VAD lexicon (design A.2.1 / MEDIUM-3)

_vad_cache = None   # (table_or_None); None sentinel means "not loaded yet"


def _lexicon_path():
    return config.MODELS / "nrc_vad.tsv"


def _parse_vad_tsv(path):
    """Parse a tab-separated NRC-VAD/Warriner-style table into {word: (valence, arousal)} under the integrity
    gate: a header `Word\\tValence\\tArousal\\tDominance`, EXACTLY NRC_VAD_ROWS data rows (the v1-vs-v2
    discriminator), and every V/A/D a float in [0,1]. On ANY failure raise ValueError (-> treated as absent).
    When NRC_VAD_SHA256 is a real constant (not None) the file hash must also match."""
    raw = path.read_bytes()
    expected_hash = getattr(config, "NRC_VAD_SHA256", None)
    if expected_hash:
        if hashlib.sha256(raw).hexdigest().lower() != str(expected_hash).lower():
            raise ValueError("nrc_vad.tsv sha256 mismatch")
    lines = raw.decode("utf-8").splitlines()
    if not lines:
        raise ValueError("empty lexicon")
    header = lines[0].split("\t")
    if [h.strip().lower() for h in header[:4]] != ["word", "valence", "arousal", "dominance"]:
        raise ValueError("unexpected lexicon header")
    rows = lines[1:]
    expected_rows = getattr(config, "NRC_VAD_ROWS", None)
    if expected_rows is not None and len(rows) != int(expected_rows):
        raise ValueError(f"row count {len(rows)} != expected {expected_rows}")
    table = {}
    for ln in rows:
        parts = ln.split("\t")
        if len(parts) < 4:
            raise ValueError("short lexicon row")
        word = parts[0].strip().lower()
        v, a, d = float(parts[1]), float(parts[2]), float(parts[3])
        for x in (v, a, d):
            if not (0.0 <= x <= 1.0):
                raise ValueError("VAD value out of [0,1]")
        table[word] = (v, a)
    return table


def load_vad_lexicon(force=False):
    """Return {word:(valence,arousal)} or None when the lexicon is absent / fails its integrity gate (design
    A.2.1 / MEDIUM-3). On ANY failure the lexicon is treated as absent and the reason is logged once; there is
    no half-loaded state. Result is memoized for the process."""
    global _vad_cache
    if _vad_cache is not None and not force:
        return _vad_cache if _vad_cache is not False else None
    path = _lexicon_path()
    if not path.exists():
        _vad_cache = False
        return None
    try:
        table = _parse_vad_tsv(path)
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning("NRC-VAD lexicon at %s treated as absent: %s", path, e)
        _vad_cache = False
        return None
    _vad_cache = table
    return table


def _reset_vad_cache():
    """Test hook: forget any memoized lexicon so a sandbox swap is picked up."""
    global _vad_cache
    _vad_cache = None


def mean_vad(lyrics):
    """Mean (valence, arousal) over the transcript's content words from the lexicon, or (None, None) when the
    lexicon is absent or no word is covered (design A.2.1: valence/arousal stored null, VAD term drops to 0)."""
    table = load_vad_lexicon()
    if table is None:
        return None, None
    toks = _tokens(lyrics_text(lyrics))
    vals = [table[t] for t in toks if t in table]
    if not vals:
        return None, None
    arr = np.asarray(vals, dtype=np.float64)
    return float(arr[:, 0].mean()), float(arr[:, 1].mean())


# ----------------------------------------------------------------- profiling (design A.2)

def profile(lyrics, backend=None):
    """Build a concept.json-shaped dict from a lyrics.json dict (design A.2). The MiniLM embedding + zero-shot
    mood come from the ConceptBackend (FEAT-001 local default); imagery (TF-IDF), valence/arousal (NRC-VAD),
    and `judgeable` are added here. Falls back to a zero concept_vec if sentence-transformers is unavailable
    (keeps the module importable + unit-testable on CPU without model downloads)."""
    if backend is None:
        from .backends.concept import local_concept
        backend = local_concept()
    try:
        base = backend.profile(lyrics)
    finally:
        try:
            backend.unload()
        except Exception:
            pass
    tags = dict(base.get("tags") or {})
    tags["imagery"] = imagery(lyrics)
    val, aro = mean_vad(lyrics)
    return dict(version=VERSION,
                embed_model=base.get("embed_model"),
                concept_vec=base.get("concept_vec"),
                valence=val, arousal=aro,
                tags=dict(theme=tags.get("theme"), mood=tags.get("mood"), person=tags.get("person"),
                          tense=tags.get("tense"), imagery=tags.get("imagery", [])),
                tags_backend=base.get("tags_backend"),
                has_tags=bool(base.get("has_tags", False)),
                judgeable=round(judgeable(lyrics), 4))


# ----------------------------------------------------------------- A.3 calibration gate (pinned)

def lyrics_params(cfg=None):
    """The single place alpha/clamp/endpoints are chosen (design A.3). Reads data/config.json lyrics_calib:

        lyrics_calib present AND n_pairs>=50 AND (hi-lo)>=0.15  -> alpha=0.12, clamp=(0.90,1.12), (lo,hi)=measured
        otherwise                                               -> alpha=0.06, clamp=(0.95,1.05), (lo,hi)=(0.15,0.65)

    Returns (alpha, (lo_clamp, hi_clamp), (lo, hi)). The ±12% strong path is unreachable without a real,
    wide-enough lyrics_calib artifact (a narrow span stays safe — span guard, MEDIUM-6)."""
    if cfg is None:
        from . import appcfg
        calib = appcfg.lyrics_calib()
    elif isinstance(cfg, dict) and ("lo" in cfg or "hi" in cfg or "n_pairs" in cfg):
        calib = cfg                      # a lyrics_calib dict was passed directly
    else:
        calib = (cfg or {}).get("lyrics_calib")
    if isinstance(calib, dict):
        lo = calib.get("lo")
        hi = calib.get("hi")
        n_pairs = calib.get("n_pairs", 0) or 0
        if lo is not None and hi is not None and n_pairs >= _MIN_PAIRS and (hi - lo) >= _MIN_SPAN:
            return _STRONG_ALPHA, _STRONG_CLAMP, (float(lo), float(hi))
    return _SAFE_ALPHA, _SAFE_CLAMP, (_SAFE_LO, _SAFE_HI)


# ----------------------------------------------------------------- A.3 lyrical-compatibility score (pinned)

def _cos(vv, iv):
    a = np.asarray(vv, dtype=np.float64)
    b = np.asarray(iv, dtype=np.float64)
    if a.size == 0 or b.size == 0:
        return 0.0
    # both are already L2-normalized MiniLM vectors; renormalize defensively for synthetic inputs
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _res_base(cos, lo, hi):
    span = hi - lo
    if span <= 0:
        return 0.0
    return float(np.clip((cos - lo) / span, 0.0, 1.0))


def _vad_term(v_concept, i_concept):
    """vad = -0.08 * clip((|dV| + |dA|)/2, 0, 1); 0 when either song's lexicon valence/arousal is null
    (lexicon absent) — the nudge is dropped, not faked (design A.3 / MEDIUM-3)."""
    vV, aV = (v_concept or {}).get("valence"), (v_concept or {}).get("arousal")
    vI, aI = (i_concept or {}).get("valence"), (i_concept or {}).get("arousal")
    if None in (vV, aV, vI, aI):
        return 0.0
    d = (abs(vV - vI) + abs(aV - aI)) / 2.0
    return _VAD_WEIGHT * float(np.clip(d, 0.0, 1.0))


def _mood_of(concept):
    return ((concept or {}).get("tags") or {}).get("mood")


def resonance(v_concept, i_concept, lo=_SAFE_LO, hi=_SAFE_HI):
    """Lraw for the resonance mode (design A.3): Lraw = clip(res_base + mood_bonus + vad, 0, 1)."""
    cos = _cos(v_concept.get("concept_vec"), i_concept.get("concept_vec"))
    rb = _res_base(cos, lo, hi)
    mb = mood_bonus(_mood_of(v_concept), _mood_of(i_concept))
    vad = _vad_term(v_concept, i_concept)
    return float(np.clip(rb + mb + vad, 0.0, 1.0))


def contrast(v_concept, i_concept, lo=_SAFE_LO, hi=_SAFE_HI):
    """Lraw for the contrast mode (design A.3): invert base + mood/vad so opposite material scores high.

        Lraw = clip((1 - res_base) + mood_bonus_inv + (-vad), 0, 1)
        mood_bonus_inv: +0.15 if moods OPPOSITE, -0.10 if EQUAL."""
    cos = _cos(v_concept.get("concept_vec"), i_concept.get("concept_vec"))
    rb = _res_base(cos, lo, hi)
    a, b = _mood_of(v_concept), _mood_of(i_concept)
    if moods_opposite(a, b):
        mb_inv = _MOOD_EQUAL_BONUS       # +0.15 when opposite
    elif moods_equal(a, b):
        mb_inv = _MOOD_OPPOSITE_BONUS    # -0.10 when equal
    else:
        mb_inv = 0.0
    vad = _vad_term(v_concept, i_concept)
    return float(np.clip((1.0 - rb) + mb_inv + (-vad), 0.0, 1.0))


def lraw(v_concept, i_concept, mode, lo=_SAFE_LO, hi=_SAFE_HI):
    """Lraw for the given lyrics_mode. `off` returns None (no lyrical signal); direction is baked into Lraw so
    A.4 uses mode_sign=+1 for both resonance and contrast."""
    if mode == "resonance":
        return resonance(v_concept, i_concept, lo, hi)
    if mode == "contrast":
        return contrast(v_concept, i_concept, lo, hi)
    return None


# ----------------------------------------------------------------- concept.json cache (design A.2 artifact)

def _path(sid):
    from . import library
    return library.song_dir(sid) / "concept.json"


def load(sid):
    """Read concept.json for `sid`, or None when missing / wrong version."""
    import json
    p = _path(sid)
    if not p.exists():
        return None
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return None
    if d.get("version") != VERSION:
        return None
    return d


def write(sid, data):
    """Atomic write-then-rename (design F4)."""
    import json
    p = _path(sid)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, p)
    return p
