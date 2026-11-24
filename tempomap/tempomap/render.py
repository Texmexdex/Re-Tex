"""The song with a click on a grid, as an MP3 to listen to: bar 1 high, the other beats lower, the subdivisions
a soft tick (so an odd bar can be heard counting 2 + 2 + 3)."""
import numpy as np
import soundfile as sf


def _click(sr, freq, ms, amp):
    n = int(sr * ms / 1000.0)
    k = np.arange(n)
    return (amp * np.sin(2 * np.pi * freq * k / sr) * np.exp(-k / (0.25 * n))).astype(np.float32)


def mix_of(stems):
    names = [k for k in ("drums", "bass", "other", "vocals") if k in stems]
    n = min(len(stems[k]) for k in names)
    y = sum(stems[k][:n] for k in names)
    pk = float(np.abs(y).max()) or 1.0
    return (y / pk * 0.8).astype(np.float32)


def with_clicks(mix, sr, bars=(), beats=(), ticks=(), level=0.35):
    """mix + clicks: bars (bar lines, s), beats (other beats), ticks (subdivisions)."""
    y = mix.copy()
    ref = float(np.percentile(np.abs(mix), 99.5)) or 0.3
    sounds = ((ticks, _click(sr, 2600, 6, 0.22 * level * ref / 0.35)),
              (beats, _click(sr, 1250, 18, 0.6 * level * ref / 0.35)),
              (bars, _click(sr, 1900, 30, 1.0 * level * ref / 0.35)))
    taken = set()
    for times, snd in reversed(sounds):                 # bar lines first; a beat on a bar line isn't clicked twice
        for t in times:
            i = int(round(t * sr))
            key = int(round(t * 100))
            if key in taken or i < 0 or i >= len(y):
                continue
            taken.add(key)
            m = min(len(snd), len(y) - i)
            y[i:i + m] += snd[:m]
    pk = float(np.abs(y).max())
    return y / pk * 0.95 if pk > 0.95 else y


def write_mp3(path, y, sr, start=None, end=None):
    a = int((start or 0) * sr)
    b = int(end * sr) if end else len(y)
    seg = np.array(y[a:b], copy=True)
    shape = (-1,) + (1,) * (seg.ndim - 1)
    if start:
        f = min(len(seg), int(0.3 * sr))
        seg[:f] *= np.linspace(0, 1, f).reshape(shape)
    if end:
        f = min(len(seg), int(0.5 * sr))
        seg[-f:] *= np.linspace(1, 0, f).reshape(shape)
    sf.write(str(path), seg, sr, format="MP3")
