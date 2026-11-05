"""Download the models Re-Tex uses, each from where its authors publish it (see THIRD_PARTY_NOTICES.md).
Run by install.ps1 with the app's Python:  python installer/fetch_models.py core|muscriptor|gemma
Files with a known hash are checked after download and refused if they don't match. Safe to run again."""
import hashlib
import os
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SEP = ROOT / "models"                      # config.SEP_MODELS: BS-RoFormer checkpoints + yamls
DATA_MODELS = ROOT / "data" / "models"     # config.MODELS: GGUFs

# (file, url, sha256, size) - hashes checked against the hosts' own metadata on 2026-11-05
FILES = [
    ("model_bs_roformer_ep_317_sdr_12.9755.ckpt",
     "https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/model_bs_roformer_ep_317_sdr_12.9755.ckpt",
     "5b84f37e8d444c8cb30c79d77f613a41c05868ff9c9ac6c7049c00aefae115aa", 639331213),
    ("model_bs_roformer_ep_317_sdr_12.9755.yaml",
     "https://raw.githubusercontent.com/TRvlvr/application_data/main/mdx_model_data/mdx_c_configs/model_bs_roformer_ep_317_sdr_12.9755.yaml",
     "2bfdd16c656bd9519aba757cc4f8834b7ede675eb1e00ec4772d74ae1c41af7f", 2273),
    ("BS-Rofo-SW-Fixed.ckpt", "https://huggingface.co/enerjazzer/BS-ROFO-SW-Fixed/resolve/main/BS-Rofo-SW-Fixed.ckpt",
     "24e7d35ee9c64415673d3fd33e06a67cac2c103c5df6267ba1576459c775916e", 699412152),
    ("BS-Rofo-SW-Fixed.yaml", "https://huggingface.co/enerjazzer/BS-ROFO-SW-Fixed/resolve/main/BS-Rofo-SW-Fixed.yaml",
     "f9fada9f94e5ba2d2e4600196299459294bc5f532b314c209cc156ac63e4329b", 4613),
]


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def download(name, url, want, size):
    dest = SEP / name
    if dest.exists() and dest.stat().st_size == size and sha256(dest) == want:
        print(f"  ok   {name} (already here)")
        return True
    SEP.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"  get  {name} ({size / 1e6:.0f} MB)")
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Re-Tex-installer"})
            with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as f:
                got, t0 = 0, time.time()
                while True:
                    b = r.read(1 << 20)
                    if not b:
                        break
                    f.write(b)
                    got += len(b)
                    if time.time() - t0 > 2:
                        print(f"       {got / 1e6:.0f} / {size / 1e6:.0f} MB", end="\r", flush=True)
                        t0 = time.time()
            break
        except Exception as e:
            print(f"       download failed ({e}); retrying" if attempt < 2 else f"       download failed: {e}")
            time.sleep(3)
    else:
        return False
    got_sha = sha256(tmp)
    if got_sha != want:
        tmp.unlink(missing_ok=True)
        print(f"  FAIL {name}: the downloaded file isn't the expected one (sha256 {got_sha[:12]}..., expected "
              f"{want[:12]}...). Not used. The host may have changed it; see models/README.md.")
        return False
    os.replace(tmp, dest)
    print(f"  ok   {name} (hash checked)")
    return True


def step(label, fn):
    print(f"  get  {label}")
    try:
        fn()
        print(f"  ok   {label}")
        return True
    except Exception as e:
        print(f"  FAIL {label}: {type(e).__name__}: {e}")
        return False


def core():
    ok = all([download(*f) for f in FILES])

    def demucs():
        from demucs.pretrained import get_model
        get_model("htdemucs_ft")

    def beat_this():
        from beat_this.inference import Audio2Frames
        Audio2Frames(checkpoint_path="final0", device="cpu")

    def whisper():
        from faster_whisper import download_model
        download_model("large-v3")

    def wav2vec2():
        import torchaudio
        torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H.get_model()

    def minilm():
        from sentence_transformers import SentenceTransformer
        SentenceTransformer("all-MiniLM-L6-v2")

    for label, fn in (("Demucs htdemucs_ft", demucs), ("Beat This! final0", beat_this),
                      ("faster-whisper large-v3", whisper), ("wav2vec2 base 960h", wav2vec2),
                      ("all-MiniLM-L6-v2", minilm)):
        ok = step(label, fn) and ok
    return ok


def muscriptor():
    def get():
        from muscriptor import TranscriptionModel
        TranscriptionModel.load_model("large")
    ok = step("MuScriptor large", get)
    if not ok:
        print("       MuScriptor is gated: log in to Hugging Face (install.ps1 offers it) and accept its terms at\n"
              "       https://huggingface.co/MuScriptor/muscriptor-large , then run install.bat again.")
    return ok


def gemma():
    def get():
        from huggingface_hub import hf_hub_download
        DATA_MODELS.mkdir(parents=True, exist_ok=True)
        hf_hub_download("google/gemma-3-12b-it-qat-q4_0-gguf", "gemma-3-12b-it-q4_0.gguf", local_dir=DATA_MODELS)
    ok = step("Gemma 3 12B GGUF (8 GB)", get)
    if not ok:
        print("       Gemma is gated: log in to Hugging Face and accept Google's terms at\n"
              "       https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-gguf , then run install.bat again.")
    return ok


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "core"
    sys.exit(0 if {"core": core, "muscriptor": muscriptor, "gemma": gemma}[what]() else 1)
