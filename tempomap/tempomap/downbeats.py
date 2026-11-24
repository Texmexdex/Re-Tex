"""Beat This!'s downbeat probabilities for a song, from Re-Tex's own copy of the model (imported read-only from
the Re-Tex code folder, run on the sum of the song's stems), cached next to the maps. If Re-Tex's Python and
model aren't available, there are none and the bars are found from the onsets alone - and `last_error` says why
(Re-Tex shows it as a fallback, with the fix: no quiet fallbacks)."""
from pathlib import Path

import numpy as np

last_error = ""          # why the last logits() call had none ("" when it did)


def logits(stems, sr, cache, code_root=None):
    """-> downbeat logits per 20 ms frame (np.ndarray) or None (then last_error says why). `cache`: an .npz path."""
    global last_error
    last_error = ""
    cache = Path(cache)
    if cache.exists():
        try:
            return np.load(cache)["down_logits"].astype(np.float64)
        except (OSError, KeyError, ValueError):
            pass
    import sys
    here = Path(__file__).resolve().parents[2]          # TempoMap lives inside Re-Tex's folder
    root = Path(code_root) if code_root else here
    if not (root / "mashlab" / "beats.py").exists():
        last_error = f"Re-Tex's code isn't at {root}"
        return None
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        from mashlab import beats as bt                # needs Re-Tex's Python (torch, beat_this)
    except Exception as e:  # noqa: BLE001 - any import problem: go on without, and say why
        last_error = f"couldn't load Re-Tex's Beat This! ({type(e).__name__}: {e})"
        return None
    n = max(len(v) for v in stems.values())
    y = np.zeros(n, np.float32)
    for v in stems.values():
        y[:len(v)] += v
    try:
        b, d, dl, bl = bt.detect_full(y, sr)
    except Exception as e:  # noqa: BLE001
        last_error = f"Beat This! failed on this song ({type(e).__name__}: {e})"
        return None
    dl = np.asarray(dl, np.float32).ravel()
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, down_logits=dl.astype(np.float16), beats=b, downbeats=d)
    return dl.astype(np.float64)
