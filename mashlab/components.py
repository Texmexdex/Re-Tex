"""What Re-Tex needs from outside its own code - models, libraries, programs - and what to do when one isn't there.

The user's rule (Nov 16): no quiet fallbacks. When something is missing or broken, Re-Tex says so: what isn't
working, the error, what it's needed for, how to fix it (a link, and a Download button where it can be fetched),
and nothing downloads until the user presses it. Where carrying on still makes sense (a MIDI part on the built-in
synth while FluidSynth is missing), it carries on AND says plainly that it's a fallback and why.

require(key)            raises Missing(key) when the component isn't usable (the server answers 424 with the details;
                        a job that hits it carries them in its error) - the page shows the dialog.
fallback(key, instead)  for the few places that carry on: publishes "component_fallback" (once per session per
                        component) so the page shows the same dialog, worded as a fallback.
status()                every component, checked (GET /api/components - the Components list in Settings).
download(key, job)      fetches it (POST /api/components/{key}/download, a background job): files from a verified
                        source (sha256 / git object checked before it's used), a Hugging Face snapshot, the
                        library's own loader (torch hub), or pip into Re-Tex's own Python (.venv-phase2).
"""
import hashlib
import importlib.util
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from .config import MODELS, ROOT, SEP_MODELS


START = int(time.time())          # this run of Re-Tex (the page shows each fallback once per run)
_last_pub = {}                    # key -> when a component_missing for it was last sent (at most once a minute)
MARK = re.compile(r"\[component:([a-z0-9_]+)\]")


class Missing(RuntimeError):
    """A component Re-Tex needs isn't usable. .info: what the page shows (see info()). The message ends with a
    [component:key] mark, so the page still finds it after the error has been passed along as plain text (a worker
    process, an HTTP 400, a job's error)."""

    def __init__(self, key, problem=None):
        self.key = key
        self.info = info(key, problem)
        super().__init__(f"{self.info['name']} isn't working: {self.info['problem']}. To fix it: {self.info['fix']} "
                         f"[component:{key}]")
        if time.time() - _last_pub.get(key, 0) > 60:
            _last_pub[key] = time.time()
            _publish("component_missing", self.info)


def from_text(text):
    """The component named by a [component:key] mark in an error message, checked now (or None)."""
    m = MARK.search(str(text or ""))
    if not m or m.group(1) not in C:
        return None
    problem = str(text).split(" isn't working: ", 1)[-1].split(". To fix it:", 1)[0] if " isn't working: " in str(text) else None
    return info(m.group(1), problem)


def strip(text):
    return MARK.sub("", str(text or "")).strip()


def _publish(kind, d):
    try:
        from .events import bus
        bus.publish(kind, info=d)
    except Exception:
        pass


def _importable(mod):
    try:
        return importlib.util.find_spec(mod) is not None
    except (ImportError, ValueError):
        return False


def _torch_ckpt(name):
    """A file in torch hub's checkpoints folder (where demucs, Beat This! and torchaudio keep their weights)."""
    try:
        import torch
        d = Path(torch.hub.get_dir()) / "checkpoints"
    except Exception:
        d = Path.home() / ".cache" / "torch" / "hub" / "checkpoints"
    return d / name


def _hf_has(repo):
    """The Hugging Face cache holds a downloaded snapshot of `repo`."""
    try:
        from huggingface_hub.constants import HF_HUB_CACHE
        base = Path(HF_HUB_CACHE)
    except Exception:
        base = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    snaps = base / ("models--" + repo.replace("/", "--")) / "snapshots"
    return snaps.is_dir() and any(any(s.iterdir()) for s in snaps.iterdir() if s.is_dir())


# ---------------------------------------------------------------- checks: () -> problem text, or None when fine

def _chk_torch_cuda():
    if not _importable("torch"):
        return "torch isn't installed"
    import torch
    if torch.version.cuda is None:
        return f"torch {torch.__version__} is the CPU-only build - it can't use the graphics card"
    if not torch.cuda.is_available():
        return (f"torch {torch.__version__} (CUDA {torch.version.cuda}) can't see the graphics card - the NVIDIA "
                f"driver may be missing or too old")
    return None


def device(what):
    """'cuda', or 'cpu' with a loud fallback notice: nothing runs on the processor quietly (Nov 16)."""
    problem = _chk_torch_cuda()
    if problem is None:
        return "cuda"
    fallback("torch_cuda", f"{what} runs on the processor instead of the graphics card - many times slower",
             problem=problem)
    return "cpu"


def _chk_ctranslate2_cuda():
    if not _importable("ctranslate2"):
        return "ctranslate2 isn't installed"
    import ctranslate2
    return None if ctranslate2.get_cuda_device_count() >= 1 else \
        f"ctranslate2 {ctranslate2.__version__} sees no CUDA graphics card (CPU-only build, or no NVIDIA driver)"


def _chk_sep_ultra():
    from .separate import roformer_files
    return None if roformer_files() else f"its checkpoint and config aren't in {SEP_MODELS}"


def _chk_sep_sw():
    from .separate import sw_files
    return None if sw_files() else f"BS-Rofo-SW-Fixed.ckpt / .yaml aren't in {SEP_MODELS}"


DEMUCS_SIGS = ("f7e0c4bc", "d12395a8", "92cfc3b6", "04573f0d")     # htdemucs_ft's four models (demucs/remote)


def _chk_demucs():
    if not _importable("demucs"):
        return "the demucs library isn't installed"
    d = _torch_ckpt("x").parent
    have = {p.name.split("-")[0] for p in d.glob("*.th")} if d.is_dir() else set()
    lost = [s for s in DEMUCS_SIGS if s not in have]
    return f"htdemucs_ft's weights aren't downloaded ({len(lost)} of 4 missing in {d})" if lost else None


def _chk_beat_this():
    if not _importable("beat_this"):
        return "the beat_this library isn't installed"
    return None if _torch_ckpt("beat_this-final0.ckpt").exists() else "its checkpoint (final0) isn't downloaded"


def _chk_whisper():
    if not (_importable("faster_whisper") and _importable("ctranslate2")):
        return "faster-whisper isn't installed"
    return None if _hf_has("Systran/faster-whisper-large-v3") else "the large-v3 model isn't downloaded"


def _chk_parakeet_tdt():
    if not _importable("onnx_asr"):
        return "the onnx-asr library isn't installed"
    if not _importable("onnxruntime"):
        return "onnxruntime isn't installed"
    try:
        from huggingface_hub.constants import HF_HUB_CACHE
        base = Path(HF_HUB_CACHE)
    except Exception:
        base = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    snaps = base / "models--istupakov--parakeet-tdt-0.6b-v3-onnx" / "snapshots"
    has_int8 = snaps.is_dir() and any((s / "encoder-model.int8.onnx").exists() for s in snaps.iterdir() if s.is_dir())
    return None if has_int8 else "the Parakeet-TDT 0.6B v3 int8 ONNX model isn't downloaded (~670 MB)"


def _chk_minilm():
    if not _importable("sentence_transformers"):
        return "sentence-transformers isn't installed"
    return None if _hf_has("sentence-transformers/all-MiniLM-L6-v2") else "the all-MiniLM-L6-v2 model isn't downloaded"


def _chk_wav2vec2():
    return None if _torch_ckpt("wav2vec2_fairseq_base_ls960_asr_ls960.pth").exists() else "its weights aren't downloaded"


def _chk_muscriptor():
    if not _importable("muscriptor"):
        return "the muscriptor library isn't installed"
    return None if _hf_has("MuScriptor/muscriptor-large") else "the large model isn't downloaded"


def _chk_pedalboard():
    return None if _importable("pedalboard") else "the pedalboard library isn't installed"


def _chk_fluidsynth():
    from .instruments import fluidsynth_exe
    return None if fluidsynth_exe() else "fluidsynth.exe isn't on this PC (not on PATH, not in mashup_lab/tools)"


def _chk_soundfont():
    from . import appcfg
    p = ((appcfg.load_config() or {}).get("midi") or {}).get("sf2")
    if p and not Path(p).exists():
        return f"the soundfont set in data/config.json is gone: {p}"
    d = ROOT / "models" / "soundfonts"
    have = d.is_dir() and any(list(d.glob("*.sf2")) + list(d.glob("*.sf3")))
    return None if (p or have) else f"no .sf2 in {d}"


def _chk_node():
    from .readhead import node_exe
    return None if node_exe() else "node.exe isn't on this PC"


def _chk_ffmpeg():
    return None if shutil.which("ffmpeg") else "ffmpeg isn't on PATH"


def _chk_gguf():
    from .backends import find_gguf
    p = find_gguf()
    if not p:
        return f"no .gguf model in {MODELS}"
    if p.stat().st_size == 0:
        return f"{p.name} is an empty file (0 bytes) - the download never finished"
    return None


def _chk_flamingo():
    from . import listen
    ok, why = listen.available()
    return None if ok else why


def _chk_yue2():
    from . import yue2link
    _, studio = yue2link._cfg()
    if studio and not Path(studio).exists():
        return f"its folder isn't there: {studio}"
    return None


def _chk_ace_studio():
    from . import acelink
    exe = acelink.cli_exe()
    if not exe:
        return f"acestudio-cli.exe not found (expected {acelink._cfg()[0]})"
    return None


def _where_ace_studio():
    from . import acelink
    return acelink.cli_exe() or ""


# ---------------------------------------------------------------- where each one is on this PC (the Components list)

def _where_ultra():
    from .separate import roformer_files
    f = roformer_files()
    return str(f[0]) if f else ""


def _where_sw():
    from .separate import sw_files
    f = sw_files()
    return str(f[0]) if f else ""


def _where_fluidsynth():
    from .instruments import fluidsynth_exe
    return fluidsynth_exe() or ""


def _where_soundfont():
    from .instruments import default_sf2
    return default_sf2(quiet=True) or ""


def _where_node():
    from .readhead import node_exe
    return node_exe() or ""


def _where_lib(mod):
    def f():
        sp = importlib.util.find_spec(mod)
        return str(Path(sp.origin).parent) if sp and sp.origin else ""
    return f


# ---------------------------------------------------------------- the list

TORCH_PIP = ["torch==2.8.0", "torchaudio==2.8.0", "--index-url", "https://download.pytorch.org/whl/cu128"]

C = {
    "torch_cuda": dict(
        where=_where_lib("torch"), name="PyTorch with CUDA (graphics card)", check=_chk_torch_cuda,
        used_for="separation, beat detection, lyrics alignment, MuScriptor - on the graphics card",
        fix="install the CUDA build of torch into Re-Tex's Python (about 3 GB; the plain 'pip install torch' "
            "is CPU-only on Windows)", url="https://pytorch.org/get-started/locally/", pip=TORCH_PIP),
    "ct2_cuda": dict(
        where=_where_lib("ctranslate2"), name="CTranslate2 with CUDA (Whisper on the graphics card)",
        check=_chk_ctranslate2_cuda, used_for="lyrics (faster-whisper) on the graphics card",
        fix="update the NVIDIA driver, or reinstall ctranslate2 (pip install --force-reinstall ctranslate2)",
        url="https://opennmt.net/CTranslate2/installation.html"),
    "sep_ultra": dict(
        where=_where_ultra, name="BS-Roformer vocal model (Ultra)", check=_chk_sep_ultra,
        used_for="separating new songs (the default SW + Ultra blend, and Ultra)",
        fix="download it (639 MB) into mashup_lab\\models (or install Ultimate Vocal Remover, which has it)",
        url="https://huggingface.co/Politrees/UVR_resources/tree/main/models/Roformer/BandSplit",
        files=[dict(url="https://huggingface.co/Politrees/UVR_resources/resolve/main/models/Roformer/BandSplit/"
                        "model_bs_roformer_ep_317_sdr_12.9755.ckpt",
                    dest="model_bs_roformer_ep_317_sdr_12.9755.ckpt",
                    sha256="5b84f37e8d444c8cb30c79d77f613a41c05868ff9c9ac6c7049c00aefae115aa"),
               dict(url="https://huggingface.co/Politrees/UVR_resources/resolve/main/models/Roformer/BandSplit/"
                        "config_bs_roformer_ep_317_sdr_12.9755.yaml",          # = UVR's copy (same git object)
                    dest="model_bs_roformer_ep_317_sdr_12.9755.yaml",
                    sha256="2bfdd16c656bd9519aba757cc4f8834b7ede675eb1e00ec4772d74ae1c41af7f")]),
    "sep_sw": dict(
        where=_where_sw, name="BS-Roformer SW 6-stem model", check=_chk_sep_sw,
        used_for="separating new songs (the default SW + Ultra blend, and SW)",
        fix="download it (699 MB) into mashup_lab\\models",
        url="https://huggingface.co/enerjazzer/BS-ROFO-SW-Fixed",
        files=[dict(url="https://huggingface.co/enerjazzer/BS-ROFO-SW-Fixed/resolve/main/BS-Rofo-SW-Fixed.ckpt",
                    dest="BS-Rofo-SW-Fixed.ckpt",
                    sha256="24e7d35ee9c64415673d3fd33e06a67cac2c103c5df6267ba1576459c775916e"),
               dict(url="https://huggingface.co/enerjazzer/BS-ROFO-SW-Fixed/resolve/main/BS-Rofo-SW-Fixed.yaml",
                    dest="BS-Rofo-SW-Fixed.yaml",
                    sha256="f9fada9f94e5ba2d2e4600196299459294bc5f532b314c209cc156ac63e4329b")]),
    "demucs": dict(
        where=_where_lib("demucs"), name="Demucs (htdemucs_ft)", check=_chk_demucs,
        used_for="the Ultra half of the default blend (drums / bass / other once the vocal is out), and plain Demucs separation",
        fix="download its weights (about 320 MB, by demucs itself)",
        url="https://github.com/facebookresearch/demucs", loader="demucs", pip="demucs"),
    "beat_this": dict(
        where=_where_lib("beat_this"), name="Beat This! (beat and downbeat detector)", check=_chk_beat_this,
        used_for="finding the beats and bars of every imported song, and the tempo map's bar 1",
        fix="download its checkpoint (81 MB, from cloud.cp.jku.at, by beat_this itself)",
        url="https://github.com/CPJKU/beat_this", loader="beat_this", pip="beat-this"),
    "whisper": dict(
        where=_where_lib("faster_whisper"), name="faster-whisper large-v3", check=_chk_whisper,
        used_for="lyrics (DJ Mix, Reimagine, word chops)",
        fix="download the model (about 3 GB)", url="https://huggingface.co/Systran/faster-whisper-large-v3",
        hf="Systran/faster-whisper-large-v3", pip="faster-whisper"),
    "parakeet_tdt": dict(
        where=_where_lib("onnx_asr"), name="NVIDIA Parakeet-TDT 0.6B v3 (ONNX)", check=_chk_parakeet_tdt,
        used_for="alternative local vocal transcription (ACE Reimagine STT engine selector)",
        fix="install onnx-asr and download the Parakeet-TDT 0.6B v3 int8 model (~670 MB)",
        url="https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx",
        loader="parakeet_tdt", pip="onnx-asr"),
    "minilm": dict(
        where=_where_lib("sentence_transformers"), name="MiniLM sentence embedder", check=_chk_minilm,
        used_for="what lyrics are about (pair ratings, DJ brief)",
        fix="download the model (about 90 MB)", url="https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2",
        hf="sentence-transformers/all-MiniLM-L6-v2", pip="sentence-transformers"),
    "wav2vec2": dict(
        name="wav2vec2 aligner", check=_chk_wav2vec2,
        used_for="lining typed lyrics up with a YuE2 take (Reimagine)",
        fix="download its weights (360 MB, by torchaudio itself)",
        url="https://pytorch.org/audio/stable/generated/torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H.html",
        loader="wav2vec2"),
    "muscriptor": dict(
        where=_where_lib("muscriptor"), name="MuScriptor", check=_chk_muscriptor,
        used_for="'Turn a stem into MIDI' and the MuScriptor second opinion on beat marks",
        fix="install it and download the large model (5 GB; you have to accept its conditions on Hugging Face "
            "once, and be signed in there)", url="https://huggingface.co/MuScriptor/muscriptor-large",
        hf="MuScriptor/muscriptor-large", pip="git+https://github.com/muscriptor/muscriptor.git"),
    "pedalboard": dict(
        where=_where_lib("pedalboard"), name="pedalboard (audio effects + time-stretch)", check=_chk_pedalboard,
        used_for="VST3 instruments, chop kits that stretch to the note, filter sweeps",
        fix="install it into Re-Tex's Python", url="https://github.com/spotify/pedalboard", pip="pedalboard"),
    "fluidsynth": dict(
        where=_where_fluidsynth, name="FluidSynth", check=_chk_fluidsynth,
        used_for="playing MIDI parts with soundfonts",
        fix="install FluidSynth (or put it in mashup_lab\\tools\\fluidsynth) and restart Re-Tex",
        url="https://github.com/FluidSynth/fluidsynth/releases"),
    "soundfont": dict(
        where=_where_soundfont, name="General MIDI soundfont", check=_chk_soundfont,
        used_for="the sound MIDI parts start with",
        fix="put a .sf2 (GeneralUser GS) in mashup_lab\\models\\soundfonts, or fix the path in data/config.json",
        url="https://schristiancollins.com/generaluser.php"),
    "node": dict(
        where=_where_node, name="Node.js", check=_chk_node, used_for="ReadHead (its engine runs on Node)",
        fix="install Node.js (LTS) and restart Re-Tex", url="https://nodejs.org/"),
    "ffmpeg": dict(
        where=lambda: shutil.which("ffmpeg") or "", name="FFmpeg", check=_chk_ffmpeg, used_for="reading MP3 / M4A / video files when importing",
        fix="install FFmpeg and add it to PATH, then restart Re-Tex", url="https://ffmpeg.org/download.html"),
    "gguf": dict(
        name="Local language model (GGUF)", check=_chk_gguf,
        used_for="DJ Mix planning without Claude, and lyric concept tags",
        fix="put a GGUF instruction model in data\\models (or plan with Claude in the DJ Mix dialog)",
        url="https://huggingface.co/models?library=gguf"),
    "flamingo": dict(
        name="Music Flamingo", check=_chk_flamingo, used_for="'Listen and describe' (genre and sound lines)",
        fix="check its Python in .venv-listen (data/config.json \"listen\")",
        url="https://huggingface.co/nvidia/music-flamingo-2601-hf"),
    "yue2": dict(
        name="YuE2 Studio", check=_chk_yue2, used_for="Reimagine (YuE2 engine)",
        fix="check data/config.json \"yue2\" points at your YuE2 folder, and start YuE2 Studio",
        url=""),
    "ace_studio": dict(
        where=_where_ace_studio, name="ACE Studio CLI", check=_chk_ace_studio,
        used_for="Reimagine (ACE Studio engine: Full Reimagine, Layer Swap, Free Voice/Timbre Swap)",
        fix="install ACE Studio (with acestudio-cli.exe) or set data/config.json \"ace\" -> \"cli\" to its path",
        url="https://www.acestudio.ai"),
}


def info(key, problem=None):
    c = C[key]
    if problem is None:
        try:
            problem = c["check"]()
        except Exception as e:                       # a check that can't even run is itself the problem
            problem = f"couldn't check it ({type(e).__name__}: {e})"
    where = ""
    if c.get("where"):
        try:
            where = c["where"]() or ""
        except Exception:
            where = ""
    return dict(key=key, name=c["name"], ok=problem is None, where=str(where), problem=problem or "", used_for=c["used_for"],
                fix=c["fix"], url=c["url"],
                download=bool(c.get("files") or c.get("hf") or c.get("loader") or c.get("pip")))


def status():
    return [info(k) for k in C]


def fallbacks():
    """The fallbacks used since Re-Tex started: [{...info, instead}]."""
    return list(_warned.values())


def require(key):
    """Raise Missing when the component isn't usable (nothing is downloaded or substituted here)."""
    problem = C[key]["check"]()
    if problem:
        raise Missing(key, problem)


_warned = {}          # key -> the fallback notice sent this session (also listed in GET /api/components)


def fallback(key, instead, problem=None):
    """Carry on without the component, and say so (once per session): the page shows it as a fallback.
    problem: the error that made it fall back, when there is one (else the component's own check says)."""
    if key in _warned:
        return
    d = info(key, problem)
    d["instead"], d["since"] = instead, START
    _send(d)


def notice(key, name, instead, problem="", fix="", url="", used_for=""):
    """A fallback that isn't about a missing component (a model answered badly, a step failed and Re-Tex went on):
    the same dialog, once per session per key."""
    if key in _warned:
        return
    _send(dict(key=key, name=name, ok=False, where="", problem=problem, used_for=used_for, fix=fix, url=url,
               download=False, instead=instead, since=START))


def _send(d):
    _warned[d["key"]] = d
    _publish("component_fallback", d)
    print(f"[fallback] {d['name']}: {d['instead']} ({d['problem']})", flush=True)


# ---------------------------------------------------------------- downloading (only when the user asks)

def _fetch(url, dest, sha256, job=None, frac=(0.0, 1.0)):
    """Stream url to dest (a .part file first), check its sha256, then put it in place. Never left half-written."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    h = hashlib.sha256()
    req = urllib.request.Request(url, headers={"User-Agent": "Re-Tex"})
    with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        got = 0
        while True:
            b = r.read(1 << 20)
            if not b:
                break
            f.write(b)
            h.update(b)
            got += len(b)
            if job and total:
                job.update(frac[0] + (frac[1] - frac[0]) * got / total, f"{dest.name}: {got >> 20} of {total >> 20} MB")
    if h.hexdigest() != sha256:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"{dest.name} didn't match its checksum - not used (the download was damaged, or the "
                           f"file at {url} isn't the one Re-Tex expects)")
    os.replace(tmp, dest)


NEEDS_TORCH = {"demucs", "beat_this", "muscriptor", "minilm", "wav2vec2"}


def _pip(spec, job=None):
    args = spec if isinstance(spec, list) else [spec]
    spec = " ".join(args)
    if job:
        job.update(0.05, f"pip install {spec}")
    r = subprocess.run([sys.executable, "-m", "pip", "install", *args], capture_output=True, text=True,
                       encoding="utf-8", errors="replace", creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if r.returncode != 0:
        raise RuntimeError(f"pip install {spec} failed: " + ((r.stderr or r.stdout).strip().splitlines() or ["?"])[-1])


def _load(which):
    if which == "demucs":
        from demucs.pretrained import get_model
        get_model("htdemucs_ft")
    elif which == "beat_this":
        from beat_this.inference import load_checkpoint
        load_checkpoint("final0")
    elif which == "wav2vec2":
        import torchaudio
        torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H.get_model()
    elif which == "parakeet_tdt":
        import onnx_asr
        onnx_asr.load_model("nemo-parakeet-tdt-0.6b-v3", quantization="int8")


def download(key, job=None):
    """Fetch the component (the user pressed Download). -> its status afterwards."""
    c = C[key]
    if key in NEEDS_TORCH and _chk_torch_cuda():
        # pip would pull in the CPU-only torch as a dependency: a silent slowdown later. Not doing that.
        raise Missing("torch_cuda", _chk_torch_cuda() + f" - install it first ({C[key]['name']} needs it)")
    lib_missing = c.get("pip") and (key == "torch_cuda" or "isn't installed" in (c["check"]() or ""))
    if lib_missing:
        _pip(c["pip"], job)
        importlib.invalidate_caches()
    if c.get("files"):
        n = len(c["files"])
        for i, f in enumerate(c["files"]):
            dest = SEP_MODELS / f["dest"]
            if dest.exists() and hashlib.sha256(dest.read_bytes()).hexdigest() == f["sha256"]:
                continue
            _fetch(f["url"], dest, f["sha256"], job, (i / n, (i + 1) / n))
    if c.get("hf"):
        if job:
            job.update(0.2, f"downloading {c['hf']} from Hugging Face")
        from huggingface_hub import snapshot_download
        try:
            snapshot_download(c["hf"])
        except Exception as e:
            raise RuntimeError(f"downloading {c['hf']} from Hugging Face failed ({type(e).__name__}: {e}). If it says "
                               f"gated / 401 / 403: open {c['url']}, accept its terms, and sign in on this PC with "
                               f"'huggingface-cli login'")
    if c.get("loader"):
        if job:
            job.update(0.2, f"downloading {c['name']}")
        try:
            _load(c["loader"])
        except Exception as e:
            raise RuntimeError(f"downloading {c['name']} failed ({type(e).__name__}: {e}) - see {c['url']}")
    d = info(key)
    if not d["ok"]:
        raise RuntimeError(f"{c['name']} still isn't working after the download: {d['problem']}")
    return d
