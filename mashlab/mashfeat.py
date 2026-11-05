"""Mashup features that need the audio itself (computed once per song, stored in mash.npz).

  melody   - the vocal's actual sung pitch (pYIN on the vocal stem), folded to pitch classes per beat.
             CQT chroma of a vocal is smeared by its overtones (every note also lights up its fifth and
             third); the tracked fundamental says which note is really being sung.
  stability- how much of the voiced vocal holds steady notes (singing) vs gliding / unpitched (rap,
             screams). Key only matters for the steady part.
  groove   - onset strength of drums / vocals sampled at 12 points per beat (straight 16ths and triplets
             both land on that grid), for swing / feel comparisons.
"""
import numpy as np
import librosa
import soxr

from .config import SR

VERSION = 1
FSR = 16000          # pitch tracking rate
HOP = 160            # 10 ms frames
GSR = 22050          # onset analysis rate
GHOP = 256
SUB = 12             # groove samples per beat


def _pc_hist(midi, w, tuning):
    """Soft pitch-class histogram (12,) of frame pitches `midi` with weights `w`."""
    out = np.zeros(12)
    if len(midi) == 0:
        return out
    p = (midi - tuning) % 12.0
    lo = np.floor(p).astype(int) % 12
    fr = p - np.floor(p)
    np.add.at(out, lo, w * (1 - fr))
    np.add.at(out, (lo + 1) % 12, w * fr)
    return out


def _beat_frames(beats, fps, n):
    return np.clip(np.round(np.asarray(beats) * fps).astype(int), 0, n)


def melody(vocal_mono_sr, beats, tuning, progress=lambda f: None):
    """-> dict of per-beat melody features + frame-level f0 (MIDI, NaN = unvoiced)."""
    x = soxr.resample(vocal_mono_sr, SR, FSR, quality="HQ").astype(np.float32)
    n_fr = 1 + len(x) // HOP
    # pYIN in blocks so progress can be reported (blocks overlap by a second to keep the HMM settled)
    f0 = np.full(n_fr, np.nan)
    vp = np.zeros(n_fr)
    blk = 60 * FSR
    pad = FSR
    starts = list(range(0, len(x), blk))
    for i, s in enumerate(starts):
        a, b = max(0, s - pad), min(len(x), s + blk + pad)
        seg = x[a:b]
        if np.sqrt(np.mean(seg ** 2)) < 1e-5:
            progress((i + 1) / len(starts))
            continue
        f, vf, pr = librosa.pyin(seg, fmin=65.0, fmax=1100.0, sr=FSR, frame_length=1024, hop_length=HOP,
                                 center=True)
        f = np.where(vf, f, np.nan)
        o = (s - a) // HOP
        k0, k1 = s // HOP, min(n_fr, (s + blk) // HOP)
        take = min(k1 - k0, len(f) - o)
        f0[k0:k0 + take] = f[o:o + take]
        vp[k0:k0 + take] = pr[o:o + take]
        progress((i + 1) / len(starts))
    rms = librosa.feature.rms(y=x, frame_length=1024, hop_length=HOP, center=True)[0]
    rms = np.pad(rms, (0, max(0, n_fr - len(rms))))[:n_fr]
    midi = librosa.hz_to_midi(f0)
    voiced = np.isfinite(midi)
    # steady notes: pitch moves < 0.35 st across ~70 ms (vibrato-tolerant after a light median)
    from scipy.ndimage import median_filter
    mf = np.where(voiced, midi, 0.0)
    mf = median_filter(mf, size=5)
    d = np.abs(mf[7:] - mf[:-7])
    steady = np.zeros(n_fr, bool)
    steady[3:-4] = (d < 0.35) & voiced[:-7] & voiced[7:]
    steady &= voiced
    loud = np.clip(rms / (np.quantile(rms[voiced], 0.9) + 1e-9) if voiced.any() else rms * 0, 0, 1.5)

    fps = FSR / HOP
    bf = _beat_frames(beats, fps, n_fr)
    nb = len(beats)
    mel = np.zeros((nb, 12))
    mel_steady = np.zeros((nb, 12))
    vfrac = np.zeros(nb)
    sfrac = np.zeros(nb)
    for j in range(nb):
        a = bf[j]
        b = bf[j + 1] if j + 1 < nb else min(n_fr, a + (a - bf[j - 1] if j else 50))
        if b <= a:
            continue
        v = voiced[a:b]
        vfrac[j] = v.mean()
        sfrac[j] = steady[a:b].mean()
        if v.any():
            w = (vp[a:b] * loud[a:b])[v]
            mel[j] = _pc_hist(midi[a:b][v], w, tuning)
            st = steady[a:b]
            if st.any():
                mel_steady[j] = _pc_hist(midi[a:b][st], (vp[a:b] * loud[a:b])[st], tuning)
    vm = midi[voiced]
    stats = dict(
        voiced=float(voiced.mean()),
        stability=float(steady.sum() / max(1, voiced.sum())),
        p05=float(np.quantile(vm, 0.05)) if len(vm) else 0.0,
        p50=float(np.median(vm)) if len(vm) else 0.0,
        p95=float(np.quantile(vm, 0.95)) if len(vm) else 0.0,
    )
    f0_ds = midi.astype(np.float32)       # 100 fps
    return dict(mel=mel.astype(np.float32), mel_steady=mel_steady.astype(np.float32),
                vfrac=vfrac.astype(np.float32), sfrac=sfrac.astype(np.float32), f0=f0_ds), stats


def groove(y_mono_sr, beats):
    """(n_beats, SUB) onset strength sampled on a 12-per-beat grid, normalized per song."""
    x = soxr.resample(y_mono_sr, SR, GSR, quality="HQ").astype(np.float32)
    env = librosa.onset.onset_strength(y=x, sr=GSR, hop_length=GHOP)
    t_env = np.arange(len(env)) * GHOP / GSR
    bt = np.asarray(beats, float)
    nb = len(bt)
    ibi = np.diff(bt)
    ibi = np.r_[ibi, ibi[-1:] if len(ibi) else [0.5]]
    pos = bt[:, None] + ibi[:, None] * (np.arange(SUB)[None, :] / SUB)
    g = np.interp(pos.ravel(), t_env, env, left=0, right=0).reshape(nb, SUB)
    top = np.quantile(g, 0.98) + 1e-9
    return (g / top).astype(np.float32)


def compute(song_dir, info, load_stem, progress=lambda f, m="": None):
    """load_stem(stem) -> (2, n) float32 at SR. Writes song_dir/mash.npz and returns the stats dict."""
    beats = np.array(info["beats"])
    tuning = float(info.get("tuning", 0.0))
    voc = load_stem("vocals").mean(0)
    progress(0.0, "tracking the vocal melody")
    mel, stats = melody(voc, beats, tuning, lambda f: progress(0.85 * f, "tracking the vocal melody"))
    progress(0.86, "groove")
    drums = load_stem("drums").mean(0)
    g_d = groove(drums, beats)
    g_v = groove(voc, beats)
    stats["version"] = VERSION
    import os
    tmp = song_dir / f"mash.{os.getpid()}.tmp.npz"          # write-then-rename: never a half-written file
    np.savez_compressed(tmp, **mel, groove_drums=g_d, groove_vocals=g_v,
                        stats=np.frombuffer(__import__("json").dumps(stats).encode(), np.uint8))
    os.replace(tmp, song_dir / "mash.npz")
    progress(1.0, "done")
    return stats


def load(song_dir):
    """-> dict or None if missing / outdated."""
    p = song_dir / "mash.npz"
    if not p.exists():
        return None
    import json
    with np.load(p) as z:
        d = {k: z[k] for k in z.files}
    d["stats"] = json.loads(bytes(d["stats"]).decode())
    if d["stats"].get("version") != VERSION:
        return None
    return d
