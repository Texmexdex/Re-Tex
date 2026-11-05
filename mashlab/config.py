"""Paths and global constants.

The code (ROOT) and the library (DATA: songs, stems, projects, uploads, the YuE2 takes) are separate, so a frozen
read-only copy of the code can run on a library that keeps growing. Where the library and the exports go comes from
(first that's set): the MASHLAB_DATA / MASHLAB_EXPORTS environment variables, the settings file (⚙ Settings in the
app writes it; MASHLAB_SETTINGS names it, else ROOT/settings.json), or ROOT/data and its exports/ folder.
MASHLAB_MODELS points the models folder somewhere else (a frozen copy keeps its own models)."""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SETTINGS_JSON = Path(os.environ.get("MASHLAB_SETTINGS") or ROOT / "settings.json")


def read_settings():
    """The settings file as a dict ({} when there isn't one or it can't be read)."""
    try:
        s = json.loads(SETTINGS_JSON.read_text(encoding="utf-8"))
        return s if isinstance(s, dict) else {}
    except (OSError, ValueError):
        return {}


def write_settings(changes):
    """Merge `changes` into the settings file (None removes a key). Takes effect the next time the app starts."""
    s = read_settings()
    for k, v in changes.items():
        if v is None:
            s.pop(k, None)
        else:
            s[k] = v
    SETTINGS_JSON.parent.mkdir(parents=True, exist_ok=True)
    tmp = SETTINGS_JSON.with_suffix(".tmp")
    tmp.write_text(json.dumps(s, indent=1), encoding="utf-8")
    os.replace(tmp, SETTINGS_JSON)
    return s


_settings = read_settings()
DATA = Path(os.environ.get("MASHLAB_DATA") or _settings.get("data") or ROOT / "data")
SONGS = DATA / "songs"          # data/songs/<id>/{song.json, stems/*.flac, peaks.bin}
SAMPLES = DATA / "samples"      # data/samples/smp_<id>/: clips baked in the sample editor (sampler.py) - song-shaped,
                                # kept apart so the library (and the frozen releases, which scan songs/) never list them


def song_path(sid):
    """A song's folder - or a baked sample's (ids "smp_..."). Everything that opens a song's files goes through here."""
    sid = str(sid)
    return (SAMPLES if sid.startswith("smp_") else SONGS) / sid
CACHE = DATA / "cache"          # warped stem renders, keyed by song/stem/tempo/pitch
PROJECTS = DATA / "projects"
EXPORTS = Path(os.environ.get("MASHLAB_EXPORTS") or _settings.get("exports") or DATA / "exports")
MODELS = Path(os.environ.get("MASHLAB_MODELS") or DATA / "models")   # nrc_vad.tsv, GGUFs (Phase 2)
SEP_MODELS = Path(os.environ.get("MASHLAB_SEP_MODELS") or ROOT / "models")  # BS-Roformer checkpoints + yamls
WEB = ROOT / "web"

CONFIG_JSON = DATA / "config.json"   # Phase-2 app config: backend tier selection + lyrics_calib (appcfg.py)

UPLOADS = DATA / "uploads"      # data/uploads/ : original dropped audio files

SR = 44100                      # internal sample rate (Demucs native)
STEMS = ("drums", "bass", "other", "vocals")
PITCHED = ("bass", "other", "vocals")   # drums are never key-shifted
PEAKS_PER_SEC = 100             # waveform resolution sent to the UI
PEAKS_HI_PER_SEC = 2205        # = 44100 / 20 exactly; close-up waveform (16-bit, made on first zoom-in): peaks/<stem>.hi.bin


# --------------------------------------------------------------- portable source paths (project-relative)
# A song record's `source` must never be a frozen absolute path tied to one machine/folder, or moving/
# cloning the project (e.g. to another drive or PC) wedges every job that can't find the
# file. Rule: a source inside the library (DATA) is stored as "data/<path inside DATA>" - the same string the
# library has always used when DATA was ROOT/data, so records read the same from the dev code and from a frozen
# copy; one inside the code folder is stored RELATIVE to ROOT; anything else keeps its absolute path (a
# genuinely-external upload). On READ, resolve_source() is library-first and self-healing.
def rel_source(path):
    """Normalize a source path for storage: "data/..." inside the library, ROOT-relative inside the code folder,
    else absolute (forward slashes for the relative forms)."""
    p = Path(path)
    try:
        ap = p.resolve()
    except OSError:
        ap = p
    for base, prefix in ((DATA, "data/"), (ROOT, "")):
        try:
            return prefix + ap.relative_to(base.resolve()).as_posix()
        except (ValueError, OSError):
            pass
    return str(ap)

def resolve_source(stored):
    """Resolve a stored `source` to a real file on THIS machine, keeping the project SELF-CONTAINED.
    Order (library-first so the clone never depends on the Desktop original existing):
      (1) "data/..." inside the CURRENT library, then a relative path joined to the CURRENT ROOT, if it exists;
      (2) the SAME FILENAME under the library's own uploads/, if it exists (preferred over any foreign
          absolute path, so an inherited Desktop path resolves to the LOCAL copy);
      (3) the stored absolute path itself, if it still exists (a legitimately-external upload);
      (4) best-guess local uploads path (may not exist -> caller reports a clean 'source missing')."""
    if not stored:
        return UPLOADS
    p = Path(stored)
    if not p.is_absolute():
        if p.parts and p.parts[0] == "data":
            cand = DATA.joinpath(*p.parts[1:])
            if cand.exists():
                return cand
        cand = ROOT / p
        if cand.exists():
            return cand
    byname = UPLOADS / p.name
    if byname.exists():
        return byname
    if p.is_absolute() and p.exists():
        return p
    return byname

# --------------------------------------------------------------- Phase-2 word-emotion (VAD) lexicon integrity
# NRC-VAD Lexicon v1 (Mohammad 2018): a tab-separated table of EXACTLY 19,971 English words, each with
# valence/arousal/dominance in [0,1] (we read valence + arousal). Downloaded on first run under an integrity
# gate (design A.2.1 / MEDIUM-3), never committed (not OSS-redistributable). Stored at data/models/nrc_vad.tsv.
NRC_VAD_URL = "https://saifmohammad.com/WebDocs/Lexicons/NRC-VAD-Lexicon.zip"
NRC_VAD_ROWS = 19971            # published v1 data-row count; also the v1-vs-v2 (~55k) discriminator (MEDIUM-7)
# PLACEHOLDER until cross-checked at implementation time (A.2.1/MEDIUM-6): the real anchor must be the SHA-256
# of the v1 NRC-VAD-Lexicon.txt confirmed by >=2 INDEPENDENT sources (e.g. the textdata R package + a public
# mirror) AND a 19,971-row file. Until that cross-check is recorded here, the loader must treat a NULL hash as
# "no hash to check against" and fall back to the row-count + float-range gate (never auto-seed from one fetch).
NRC_VAD_SHA256 = None

# Fallback lexicon if NRC-VAD terms are unacceptable: Warriner et al. (2013) valence/arousal norms
# (BRM-emot-submit.csv, ~13,915 words, CC-BY-NC-SA), same path + same integrity discipline (design A.2.1).
WARRINER_URL = "https://static-content.springer.com/esm/art%3A10.3758%2Fs13428-012-0314-x/MediaObjects/13428_2012_314_MOESM1_ESM.zip"
WARRINER_ROWS = 13915           # published Warriner et al. (2013) word count
WARRINER_SHA256 = None          # PLACEHOLDER, same cross-check discipline as NRC_VAD_SHA256

# --------------------------------------------------------------- Phase-2 optional LLM tag models (GGUF ids ONLY)
# ConceptBackend's structured-tag path is OPTIONAL and auto-enabled only if a GGUF instruction model is found
# on the box (backends.detect()). These are the recommended ids ONLY (OD-1) — nothing is downloaded here.
GGUF_PRIMARY = "google/gemma-3-12b-it-qat-q4_0-gguf"   # Gemma 4 12B Q4_K_M-class primary (OD-1)
GGUF_ALT = "Qwen/Qwen3-8B-GGUF:Q4_K_M"                 # Qwen3-8B Q4_K_M alternate (OD-1)

for d in (SONGS, CACHE, PROJECTS, EXPORTS, UPLOADS, MODELS):
    d.mkdir(parents=True, exist_ok=True)
