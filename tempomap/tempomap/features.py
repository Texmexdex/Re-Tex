"""What the music does, frame by frame (about 200 frames a second), from the song's stems.

  onsets   how sharply each part gets louder (spectral flux): kick, snare, cymbals (from the drum stem), bass, other
  chroma   which pitch classes sound (bass + other), about 50 frames a second, for where the chords change

Everything is plain numpy: no librosa, so the tool runs with numpy + scipy + soundfile only.
"""
import numpy as np

FPS = 200.0                       # onset frames a second (about: the hop is a whole number of samples)
DRUM_BANDS = {"kick": (30, 150), "snare": (200, 4000), "cymbal": (5000, 16000)}


def _frames(y, n_fft, hop, chunk=4096):
    """|STFT| in chunks of frames: yields (first frame, magnitudes [frames, bins]). Frame k is centred on sample
    k * hop, so frame times are k * hop / sr."""
    win = np.hanning(n_fft).astype(np.float32)
    y = np.pad(np.asarray(y, np.float32), (n_fft // 2, n_fft // 2))
    n = 1 + (len(y) - n_fft) // hop
    offs = np.arange(n_fft)
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        idx = np.arange(s, e)[:, None] * hop + offs[None, :]
        yield s, np.abs(np.fft.rfft(y[idx] * win, axis=1)).astype(np.float32)


def _nfft(sr):
    """About 12 ms of audio, a power of two: short enough to place a hit, long enough for a kick's body."""
    return 1 << int(round(np.log2(0.012 * sr)))


def band_flux(y, sr, bands, n_fft=None):
    """Spectral flux (rise of log magnitude, summed over the band) per band -> ([frames, len(bands)], fps)."""
    n_fft = n_fft or _nfft(sr)
    hop = int(round(sr / FPS))
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    sel = [(freqs >= lo) & (freqs < hi) for lo, hi in bands]
    out, prev = [], None
    for _, m in _frames(y, n_fft, hop):
        lg = np.log1p(10.0 * m)
        if prev is None:
            prev = lg[:1]
        d = np.maximum(0.0, np.diff(np.vstack([prev, lg]), axis=0))
        prev = lg[-1:]
        out.append(np.stack([d[:, s].sum(axis=1) for s in sel], axis=1))
    return np.vstack(out), sr / hop


def chroma(y, sr, n_fft=4096):
    """Pitch-class energy, 12 per frame, each frame scaled to sum 1 (silence stays 0) -> ([frames, 12], fps)."""
    hop = int(round(sr / FPS)) * 4
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    ok = (freqs >= 55.0) & (freqs <= 2000.0)
    pc = np.zeros(len(freqs), int)
    pc[ok] = np.round(12 * np.log2(freqs[ok] / 440.0)).astype(int) % 12
    proj = np.zeros((len(freqs), 12), np.float32)
    proj[np.flatnonzero(ok), pc[ok]] = 1.0
    out = []
    for _, m in _frames(y, n_fft, hop):
        c = np.log1p(10.0 * m) @ proj
        out.append(c)
    c = np.vstack(out)
    s = c.sum(axis=1, keepdims=True)
    return np.where(s > 1e-6, c / np.maximum(s, 1e-9), 0.0), sr / hop


_LAG = {}


def flux_lag(sr):
    """How much later than a hit's start its flux peaks (s), measured once per sample rate on clicks of known
    timing (a sharp kick and a noise burst), so every onset time can be moved back onto the hit's start."""
    if sr not in _LAG:
        rng = np.random.default_rng(0)
        n = int(4 * sr)
        y = np.zeros(n, np.float32)
        times = np.arange(0.5, 3.6, 0.5)
        for k, t in enumerate(times):
            i = int(t * sr)
            m = int(0.15 * sr)
            e = np.exp(-np.arange(m) / (0.04 * sr))
            if k % 2:
                y[i:i + m] += (rng.standard_normal(m) * e).astype(np.float32) * 0.5
            else:
                f = 50 + 70 * np.exp(-np.arange(m) / (0.03 * sr))
                y[i:i + m] += (np.sin(2 * np.pi * np.cumsum(f) / sr) * e).astype(np.float32)
        fl, fps = band_flux(y, sr, [(30, 16000)])
        fl = fl[:, 0]
        lags = []
        for t in times:
            i0, i1 = int((t - 0.03) * fps), int((t + 0.06) * fps)
            lags.append((i0 + int(np.argmax(fl[i0:i1]))) / fps - t)
        _LAG[sr] = float(np.median(lags))
    return _LAG[sr]


def _shift(x, frames):
    """x moved earlier by `frames` frames (the end padded with zeros)."""
    if frames <= 0:
        return x
    return np.concatenate([x[frames:], np.zeros(frames, x.dtype)])


def _norm(x):
    p = np.percentile(x, 99.5) if len(x) else 0.0
    return x / p if p > 1e-9 else np.zeros_like(x)


class Features:
    """The onset curves and chroma of a song. parts: {name: curve at fps}; env: the combined curve the pulse is
    measured on; t(i): time of frame i."""

    def __init__(self, stems, sr):
        self.sr = sr
        self.parts, fps = {}, None
        if "drums" in stems:
            f, fps = band_flux(stems["drums"], sr, list(DRUM_BANDS.values()))
            for k, name in enumerate(DRUM_BANDS):
                self.parts[name] = _norm(f[:, k])
        for name, band in (("bass", (30, 400)), ("other", (100, 8000))):
            if name in stems:
                f, fps = band_flux(stems[name], sr, [band])
                self.parts[name] = _norm(f[:, 0])
        n = min(len(v) for v in self.parts.values())
        self.lag = flux_lag(sr)                           # every curve moved back onto the hits' starts
        lagf = int(round(self.lag * fps))
        self.parts = {k: _shift(v[:n], lagf) for k, v in self.parts.items()}
        self.fps = fps
        self.n = n
        z = np.zeros(n)
        g = self.parts.get
        self.env = (g("kick", z) + g("snare", z) + 0.5 * g("cymbal", z) + 0.6 * g("bass", z) + 0.6 * g("other", z))
        self.hits = g("kick", z) + g("snare", z)          # what a beat sits on (the drums' body)
        if not self.hits.any():
            self.hits = self.env
        tonal = [stems[k] for k in ("bass", "other") if k in stems]
        if tonal:
            m = min(len(x) for x in tonal)
            self.chroma, self.cfps = chroma(sum(x[:m] for x in tonal), sr)
        else:
            self.chroma, self.cfps = np.zeros((1, 12)), 1.0

    def t(self, i):
        return np.asarray(i) / self.fps

    def idx(self, t):
        return np.clip(np.round(np.asarray(t) * self.fps).astype(int), 0, self.n - 1)
