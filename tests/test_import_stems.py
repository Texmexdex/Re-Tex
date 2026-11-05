"""Importing already-split stems: several files on one stem are mixed together, a stem with no file is silent, shorter
files are padded - and nothing is separated again (the stems come back exactly as given)."""
import numpy as np
import soundfile as sf

from mashlab import library
from mashlab.config import SR


def test_stems_from_files_mixes_pads_and_fills_silence(tmp_path):
    rng = np.random.default_rng(0)
    kick, hats = rng.uniform(-.3, .3, (SR, 2)), rng.uniform(-.3, .3, (SR // 2, 2))
    acap = rng.uniform(-.3, .3, (SR, 1))                                  # mono: comes back as two channels
    for name, y in (("kick.wav", kick), ("hats.wav", hats), ("acapella.wav", acap)):
        sf.write(tmp_path / name, y, SR, subtype="FLOAT")
    st = library.stems_from_files({"drums": [tmp_path / "kick.wav", tmp_path / "hats.wav"], "vocals": tmp_path / "acapella.wav"})
    assert set(st) == {"drums", "bass", "other", "vocals"} and all(v.shape == (2, SR) for v in st.values())
    want = kick.T.copy()
    want[:, :SR // 2] += hats.T
    assert np.abs(st["drums"] - want).max() < 1e-6                       # mixed, the shorter file padded at the end
    assert np.abs(st["vocals"] - np.repeat(acap.T, 2, 0)).max() < 1e-6
    assert not st["bass"].any() and not st["other"].any()                # no file: silence
