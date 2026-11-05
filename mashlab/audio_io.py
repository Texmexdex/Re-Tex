"""Audio loading/saving. Anything soundfile can't read (mp3/m4a/mp4/...) goes through ffmpeg."""
import shutil
import subprocess
import numpy as np
import soundfile as sf
import soxr

from .config import SR


def save_npy_atomic(out_path, arr):
    """Write arr to out_path via a temp file + rename. Windows can't replace a file another process has open
    (the player memory-maps renders), so if the file is already there - a duplicate render of the same slice
    finished first - the new copy is simply dropped. Returns out_path."""
    import os
    import time
    out_path = str(out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmp = out_path[:-4] + f".{os.getpid()}.tmp.npy"
    np.save(tmp, arr)
    for attempt in range(5):
        try:
            os.replace(tmp, out_path)
            return out_path
        except PermissionError:
            if os.path.exists(out_path):
                break
            time.sleep(0.1 * (attempt + 1))
    try:
        os.remove(tmp)
    except OSError:
        pass
    if not os.path.exists(out_path):
        raise PermissionError(f"couldn't write {out_path}")
    return out_path


def load(path, sr=SR, mono=False):
    """Return float32 array shaped (channels, n) at `sr` (always 2 channels unless mono=True)."""
    path = str(path)
    try:
        y, s = sf.read(path, dtype="float32", always_2d=True)
        y = y.T
    except Exception:
        y, s = _ffmpeg_decode(path, sr)
    if y.shape[0] == 1 and not mono:
        y = np.repeat(y, 2, axis=0)
    elif y.shape[0] > 2:
        y = y[:2]
    if s != sr:
        y = soxr.resample(y.T, s, sr, quality="VHQ").T.astype(np.float32)
    if mono:
        y = y.mean(0, keepdims=True)
    return np.ascontiguousarray(y, dtype=np.float32)


def _ffmpeg_decode(path, sr):
    exe = shutil.which("ffmpeg")
    if not exe:
        raise RuntimeError("ffmpeg not found on PATH - needed to read this file type")
    cmd = [exe, "-v", "error", "-i", path, "-f", "f32le", "-acodec", "pcm_f32le", "-ac", "2", "-ar", str(sr), "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    y = np.frombuffer(raw, dtype=np.float32).reshape(-1, 2).T.copy()
    return y, sr


def save(path, y, sr=SR, subtype="PCM_24"):
    """y: (channels, n) float."""
    sf.write(str(path), np.asarray(y, dtype=np.float32).T, sr, subtype=subtype)


def save_mp3(path_wav, path_mp3, bitrate="320k"):
    exe = shutil.which("ffmpeg")
    if not exe:
        raise RuntimeError("ffmpeg not found - can't write mp3")
    subprocess.run([exe, "-y", "-v", "error", "-i", str(path_wav), "-b:a", bitrate, str(path_mp3)], check=True)


def peaks(y, sr=SR, per_sec=100):
    """Min/max waveform envelope as int8 pairs, for drawing. y: (ch, n) or (n,)."""
    m = y.mean(0) if y.ndim == 2 else y
    hop = max(1, int(sr // per_sec))
    nb = int(np.ceil(len(m) / hop))
    pad = np.zeros(nb * hop, np.float32)
    pad[: len(m)] = m
    f = pad.reshape(nb, hop)
    lo, hi = f.min(1), f.max(1)
    out = np.empty(nb * 2, np.int8)
    out[0::2] = np.clip(lo * 127, -127, 127).astype(np.int8)
    out[1::2] = np.clip(hi * 127, -127, 127).astype(np.int8)
    return out


def peaks16(y, sr=SR, per_sec=2205):
    """Like peaks() but 16-bit, for close-up waveforms: quiet onsets (-40 dB and below) still show."""
    m = y.mean(0) if y.ndim == 2 else y
    hop = max(1, int(round(sr / per_sec)))
    assert abs(sr / hop - per_sec) < 1e-6, "per_sec must divide the sample rate"
    nb = int(np.ceil(len(m) / hop))
    pad = np.zeros(nb * hop, np.float32)
    pad[: len(m)] = m
    f = pad.reshape(nb, hop)
    out = np.empty(nb * 2, np.int16)
    out[0::2] = np.clip(f.min(1) * 32767, -32767, 32767).astype(np.int16)
    out[1::2] = np.clip(f.max(1) * 32767, -32767, 32767).astype(np.int16)
    return out
