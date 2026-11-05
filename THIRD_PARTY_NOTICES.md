# Third-party software, models and data

Re-Tex's own code is GPL-3.0 (see `LICENSE`). It uses the projects below. **None of the models, datasets or
programs listed here are stored in this repository**: `install.bat` downloads each one from where its authors
publish it, so you get it under its own terms. Python packages come from PyPI / the PyTorch index; their
licenses are as each package declares.

Checked on 2026-11-05. If you share or build on Re-Tex, keep this file with it.

## Copied into this repository

| What | Where | License |
|---|---|---|
| BS-RoFormer model code by Phil Wang (lucidrains/BS-RoFormer 0.3.10) | `mashlab/vendor/bs_roformer/` | MIT (its `LICENSE` is kept next to it) |

## Python packages (installed by `install.bat`)

| Package | License | Notes |
|---|---|---|
| pedalboard (Spotify) | GPL-3.0 | Includes the **Rubber Band Library** (GPL-2.0-or-later; commercial licence available from its authors) and FFTW (GPL-2.0-or-later). This is why Re-Tex is GPL-3.0. |
| torch, torchaudio (PyTorch) | BSD-3-Clause | The CUDA builds pull NVIDIA's CUDA libraries, which are under NVIDIA's own licence. |
| demucs | MIT | |
| beat-this (CPJKU) | MIT | |
| faster-whisper, ctranslate2 | MIT | |
| muscriptor (Kyutai / Mirelo) | MIT (code) | weights: see below |
| llama-cpp-python | MIT | |
| transformers, huggingface_hub | Apache-2.0 | |
| xgrammar, accelerate | Apache-2.0 | Music Flamingo environment only |
| sentence-transformers | Apache-2.0 | |
| numpy, scipy, numba, scikit-learn, soundfile, uvicorn, starlette, threadpoolctl | BSD | soundfile bundles libsndfile (LGPL-2.1) |
| librosa | ISC | |
| soxr (python-soxr) | LGPL-2.1-or-later | used as an unmodified, separately installed library |
| sounddevice | MIT | bundles PortAudio (MIT) |
| pyloudnorm, fastapi, einops, rotary-embedding-torch, PyYAML | MIT | |
| python-multipart | Apache-2.0 | |

## Programs (installed with winget)

| Program | License |
|---|---|
| Python | PSF License |
| FFmpeg (Gyan.dev build) | GPL-3.0 (this build); see https://ffmpeg.org/legal.html |
| Node.js | MIT |

## Models (downloaded on install or first use)

| Model | Used for | License / terms | Source |
|---|---|---|---|
| Demucs htdemucs_ft (Meta) | stems | MIT (Demucs repository) | https://github.com/facebookresearch/demucs |
| BS-RoFormer vocals, `model_bs_roformer_ep_317_sdr_12.9755` (viperx) | "Ultra" stems | **No license stated** by the original host | https://github.com/TRvlvr/model_repo (UVR's model repository) |
| BS-RoFormer SW 6-stem, `BS-Rofo-SW-Fixed` | "SW" and "SW + Ultra blend" stems | **License: unknown** (as listed on its page) | https://huggingface.co/enerjazzer/BS-ROFO-SW-Fixed |
| Beat This! final0 | beat grids | MIT (code and weights) | https://github.com/CPJKU/beat_this |
| Whisper large-v3 via faster-whisper (Systran conversion) | lyrics | MIT | https://huggingface.co/Systran/faster-whisper-large-v3 |
| wav2vec2 ASR base 960h (torchaudio) | word timing | MIT | https://docs.pytorch.org/audio/stable/generated/torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H.html |
| MuScriptor large (Kyutai / Mirelo) — Full install | beat-grid second opinion | **CC BY-NC 4.0**, gated (log in and accept). Use it only on music you have the rights to transcribe. | https://huggingface.co/MuScriptor/muscriptor-large |
| Gemma 3 12B it QAT Q4_0 GGUF (Google) — Full install | offline DJ planner / tags | **Gemma Terms of Use** and Prohibited Use Policy, gated | https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-gguf |
| Music Flamingo 2601 (NVIDIA) — Full install | song descriptions | **NVIDIA OneWay Noncommercial License**: non-commercial research use | https://huggingface.co/nvidia/music-flamingo-2601-hf |
| all-MiniLM-L6-v2 (via sentence-transformers) | mood tags | Apache-2.0 | https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2 |

The two BS-RoFormer weights are shared by their authors for free download (they're what Ultimate Vocal Remover
and similar tools use), but their pages state no licence, so no rights are formally granted. Fine for personal
use; don't use them for commercial work. Demucs alone (the "Best" separation option) has no such question.

## Data (downloaded on first use)

| Data | Used for | Terms |
|---|---|---|
| **NRC Valence, Arousal, and Dominance (VAD) Lexicon**, Saif M. Mohammad, National Research Council Canada | word emotion in lyrics | Free for non-commercial research and educational use. Do not redistribute it. Re-Tex downloads it from its home page on first run. |
| Warriner, Kuperman & Brysbaert (2013), *Norms of valence, arousal, and dominance for 13,915 English lemmas* (fallback) | the same, if NRC VAD can't be used | CC BY-NC-SA |

Acknowledgement as requested by the NRC VAD Lexicon's terms: this application uses the NRC VAD Lexicon by
Saif M. Mohammad, National Research Council Canada (https://saifmohammad.com/WebPages/nrc-vad.html).

## Non-commercial parts

Re-Tex itself may be used commercially under the GPL. These parts may not: MuScriptor, Music Flamingo, the NRC VAD
Lexicon and the Warriner norms (and, to be safe, the two BS-RoFormer models). A commercial setup would turn
those features off and use Demucs for separation.
