"""Stem separation with Demucs htdemucs_ft, run in-process so we can report real progress."""
import threading
import numpy as np

_model = None
_lock = threading.Lock()


def _get_model():
    global _model
    if _model is None:
        from demucs.pretrained import get_model
        _model = get_model("htdemucs_ft")
        _model.eval()
    return _model


class _ProgressTqdm:
    """Stand-in for the `tqdm` module inside demucs.apply: wraps each chunk loop and reports progress."""

    def __init__(self, report, total_passes):
        self.report, self.total, self.done = report, max(1, total_passes), 0

    def tqdm(self, iterable, **_):
        items = list(iterable)
        n = max(1, len(items))
        for i, it in enumerate(items):
            yield it
            self.report((self.done + (i + 1) / n) / self.total)
        self.done += 1


def separate(y, quality="best", progress=lambda f: None):
    """y: (2, n) float32 at 44.1 kHz -> dict stem -> (2, n) float32.

    quality: "fast" (1 shift) | "best" (4 test-time shifts + 0.5 overlap; cleaner stems, ~4x slower).
    """
    import torch
    import demucs.apply as dapply

    shifts, overlap = (4, 0.5) if quality == "best" else (1, 0.25)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    with _lock:
        model = _get_model()
        n_models = len(getattr(model, "models", [model]))
        wav = torch.from_numpy(np.ascontiguousarray(y))
        ref = wav.mean(0)
        mu, sd = ref.mean(), ref.std() + 1e-8
        wav = (wav - mu) / sd
        orig = dapply.tqdm
        dapply.tqdm = _ProgressTqdm(progress, n_models * shifts)
        try:
            with torch.no_grad():
                out = dapply.apply_model(model, wav[None], shifts=shifts, split=True, overlap=overlap,
                                         progress=True, device=dev)[0]
        finally:
            dapply.tqdm = orig
            if dev == "cuda":
                torch.cuda.empty_cache()
        out = out * sd + mu
    return {name: out[i].cpu().numpy().astype(np.float32) for i, name in enumerate(model.sources)}


def spread_leftover(mix, parts, catch_all="other", nperseg=2048):
    """Separators' outputs don't add up to their input exactly. Give the difference back to the parts, each
    getting the share of every time/frequency cell it already owns - so nothing is lost, and a part that came
    out too loud somewhere is turned down there (rather than some other part getting an inverted copy of it).
    STFT round-off goes to `catch_all`, so the parts add up to `mix` exactly."""
    import scipy.signal as sg
    n = min([mix.shape[1]] + [v.shape[1] for v in parts.values()])
    mix = np.asarray(mix[:, :n], np.float32)
    parts = {k: np.asarray(v[:, :n], np.float32) for k, v in parts.items()}
    left = mix - sum(parts.values())
    _, _, L = sg.stft(left, nperseg=nperseg)
    power = {k: np.abs(sg.stft(v, nperseg=nperseg)[2]) ** 2 for k, v in parts.items()}
    total = sum(power.values()) + 1e-12
    out = {}
    for k, v in parts.items():
        _, x = sg.istft(L * (power[k] / total), nperseg=nperseg)
        out[k] = (v + x[:, :n]).astype(np.float32)
    rest = [k for k in out if k != catch_all]
    out[catch_all] = (mix - sum(out[k] for k in rest)).astype(np.float32)
    return out


# ---------------------------------------------------------------- "ultra": BS-Roformer vocals + Demucs on the rest

ROFORMER_CKPT = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"
ROFORMER_YAML = "model_bs_roformer_ep_317_sdr_12.9755.yaml"
_rof = None


def roformer_files():
    """(ckpt, yaml) for the BS-Roformer vocal model: mashup_lab/models/ (config.SEP_MODELS) first, then a UVR
    install."""
    import os
    from pathlib import Path
    from .config import SEP_MODELS
    cands = [SEP_MODELS]
    local = os.environ.get("LOCALAPPDATA", "")
    if local:
        cands.append(Path(local) / "Programs" / "Ultimate Vocal Remover" / "models" / "MDX_Net_Models")
    cands.append(Path("C:/Program Files/Ultimate Vocal Remover/models/MDX_Net_Models"))
    for d in cands:
        ck = d / ROFORMER_CKPT
        if not ck.exists():
            continue
        for y in (d / ROFORMER_YAML, d / "model_data" / "mdx_c_configs" / ROFORMER_YAML):
            if y.exists():
                return ck, y
    return None


def ultra_available():
    return roformer_files() is not None


def _get_roformer():
    global _rof
    if _rof is None:
        import torch
        import yaml
        from .vendor.bs_roformer import BSRoformer
        ck, yml = roformer_files()
        cfg = yaml.load(open(yml, encoding="utf-8"), Loader=yaml.FullLoader)
        m = cfg["model"]
        keys = ["dim", "depth", "stereo", "num_stems", "time_transformer_depth", "freq_transformer_depth",
                "freqs_per_bands", "dim_head", "heads", "attn_dropout", "ff_dropout", "flash_attn", "dim_freqs_in",
                "stft_n_fft", "stft_hop_length", "stft_win_length", "stft_normalized", "mask_estimator_depth"]
        model = BSRoformer(**{k: m[k] for k in keys if k in m})
        sd = torch.load(ck, map_location="cpu", weights_only=False)
        model.load_state_dict(sd.get("state_dict", sd), strict=True)
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        _rof = (model.to(dev).eval(), dev, int(cfg["audio"]["chunk_size"]), int(cfg["inference"].get("num_overlap", 4)))
    return _rof


def _chunked(model, dev, C, overlap, y, progress=lambda f: None, batch=4):
    """Run a Roformer over y (2, n) in overlapped chunks, cross-faded (as the models were trained) -> (stems, 2, n)."""
    import torch
    n = y.shape[1]
    step = C // overlap
    fade = C // 10
    win = np.ones(C, np.float32)
    win[:fade] = np.linspace(0, 1, fade)
    win[-fade:] = np.linspace(1, 0, fade)
    border = C - step
    x = np.pad(y, ((0, 0), (border, border + C)), mode="reflect" if n > border + C else "constant").astype(np.float32)
    starts = list(range(0, x.shape[1] - C + 1, step))
    out = None
    acc = np.zeros(x.shape[1], np.float32)
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(dev == "cuda")):
        for i in range(0, len(starts), batch):
            idx = starts[i:i + batch]
            chunk = torch.from_numpy(np.stack([x[:, s:s + C] for s in idx])).to(dev)
            pred = model(chunk).float().cpu().numpy()          # (b, 2, C) or (b, stems, 2, C)
            if pred.ndim == 3:
                pred = pred[:, None]
            if out is None:
                out = np.zeros((pred.shape[1],) + x.shape, np.float32)
            for j, s in enumerate(idx):
                w = win.copy()
                if s == 0:
                    w[:fade] = 1                                   # no fade-in at the very start
                if s == starts[-1]:
                    w[-fade:] = 1
                out[:, :, s:s + C] += pred[j] * w
                acc[s:s + C] += w
            progress(min(1.0, (i + batch) / len(starts)))
    out /= np.maximum(acc, 1e-6)
    if dev == "cuda":
        torch.cuda.empty_cache()
    return out[:, :, border:border + n]


def roformer_vocals(y, progress=lambda f: None, batch=4):
    """y: (2, n) at 44.1 kHz -> vocals (2, n)."""
    model, dev, C, overlap = _get_roformer()
    return _chunked(model, dev, C, overlap, y, progress, batch)[0]


def separate_ultra(y, progress=lambda f: None):
    """Cleanest stems: BS-Roformer takes the vocals out of the mix; Demucs (best) then splits the vocal-free
    remainder into drums / bass / other - it no longer has to untangle the voice from the instruments."""
    with _lock:
        vocals = roformer_vocals(y, lambda f: progress(0.3 * f))
    inst = (y - vocals).astype(np.float32)
    rest = separate(inst, "best", lambda f: progress(0.3 + 0.7 * f))
    # Demucs also returns a "vocals" part of the vocal-free remainder (whatever sounds voice-like to it: synth
    # leads, pads, chops). Dropping it deleted real parts of songs (e.g. "Easy": up to ~70 % of the mids in some
    # sections) - it belongs to "other". Its small leftover is shared out too, so the stems add up to the original.
    n = min(inst.shape[1], *(v.shape[1] for v in rest.values()))
    parts = {"drums": rest["drums"][:, :n], "bass": rest["bass"][:, :n],
             "other": rest["other"][:, :n] + rest["vocals"][:, :n]}
    parts = spread_leftover(inst[:, :n], parts)
    return {**parts, "vocals": vocals[:, :n].astype(np.float32)}


# ---------------------------------------------------------------- "sw": BS-Roformer SW, one model for all the stems

SW_CKPT = "BS-Rofo-SW-Fixed.ckpt"          # huggingface.co/enerjazzer/BS-ROFO-SW-Fixed (sha256 24e7d35e...916e)
SW_YAML = "BS-Rofo-SW-Fixed.yaml"
SW_OVERLAP = 4                              # chunks overlap 4x (its config says 2): every moment is the average of 4 runs
_sw = None


def sw_files():
    """(ckpt, yaml) for the 6-stem BS-Roformer SW model, in mashup_lab/models/ (config.SEP_MODELS)."""
    from .config import SEP_MODELS
    ck, yml = SEP_MODELS / SW_CKPT, SEP_MODELS / SW_YAML
    return (ck, yml) if ck.exists() and yml.exists() else None


def sw_available():
    return sw_files() is not None


def _get_sw():
    global _sw
    if _sw is None:
        import torch
        import yaml
        from .vendor.bs_roformer import BSRoformer
        ck, yml = sw_files()
        cfg = yaml.load(open(yml, encoding="utf-8"), Loader=yaml.FullLoader)
        m = cfg["model"]
        keys = ["dim", "depth", "stereo", "num_stems", "time_transformer_depth", "freq_transformer_depth",
                "freqs_per_bands", "dim_head", "heads", "attn_dropout", "ff_dropout", "flash_attn", "dim_freqs_in",
                "stft_n_fft", "stft_hop_length", "stft_win_length", "stft_normalized", "mask_estimator_depth"]
        model = BSRoformer(**{k: m[k] for k in keys if k in m})
        sd = torch.load(ck, map_location="cpu", weights_only=True)    # plain weights only (no code in the file runs)
        model.load_state_dict(sd.get("state_dict", sd), strict=True)
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        _sw = (model.to(dev).eval(), dev, int(cfg["audio"]["chunk_size"]), list(cfg["training"]["instruments"]))
    return _sw


def separate_sw(y, progress=lambda f: None):
    """All four stems from BS-Roformer SW. It splits six ways (bass, drums, other, vocals, guitar, piano); guitar and
    piano go back into "other". What the model left in no stem is shared out as with the other separators, so the
    stems add up to the song."""
    with _lock:
        model, dev, C, names = _get_sw()
        out = _chunked(model, dev, C, SW_OVERLAP, y, progress, batch=2)
    st = dict(zip(names, out))
    n = out.shape[2]
    parts = {"drums": st["drums"], "bass": st["bass"], "vocals": st["vocals"],
             "other": st["other"] + st.get("guitar", 0) + st.get("piano", 0)}
    return spread_leftover(y[:, :n], parts)


def separate_blend(y, progress=lambda f: None):
    """BS-Roformer SW and Ultra (BS-Roformer vocals + Demucs), averaged stem by stem. Their artifacts differ, so they
    partly cancel: in a blind listening test (Agostina, latitudes) the blend was the best vocals and as good as the
    best drums / bass. Both add up to the song, so the average does too."""
    a = separate_sw(y, lambda f: progress(0.2 * f))
    b = separate_ultra(y, lambda f: progress(0.2 + 0.8 * f))
    n = min(min(v.shape[1] for v in a.values()), min(v.shape[1] for v in b.values()))
    return {k: (0.5 * (a[k][:, :n] + b[k][:, :n])).astype(np.float32) for k in a}
