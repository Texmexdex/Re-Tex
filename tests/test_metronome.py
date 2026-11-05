"""The main mix's metronome (player._add_clicks): a click on every beat of the project's grid, accented on each bar's 1
and each 4-bar phrase's 1, the same whatever size the audio blocks come in (a click can straddle two)."""
import numpy as np

from mashlab.config import SR
from mashlab.player import CLICKS, Player


def test_clicks_land_on_the_grid_with_accents_and_dont_depend_on_block_size():
    p = Player()
    p._spb, p._bpb, p.click_gain = 60.0 / 130.0 * SR, 4, 1.0        # 130 BPM, 4/4 (the reference mix)
    n = int(18 * p._spb)                                             # 4.5 bars
    whole = np.zeros((2, n), np.float32)
    p._add_clicks(whole, 0, 0, n)
    pieces = np.zeros((2, n), np.float32)
    pos = 0
    for size in [1024, 333, 4096, 17, 1024] * 400:                  # odd block sizes, clicks straddling them
        k = min(size, n - pos)
        if k <= 0:
            break
        p._add_clicks(pieces, pos, pos, k)
        pos += k
    assert np.allclose(whole, pieces, atol=1e-7)
    onsets = [int(round(b * p._spb)) for b in range(18)]
    for b, s in enumerate(onsets):                                   # each click starts exactly on its beat
        kind = 2 if b % 16 == 0 else 1 if b % 4 == 0 else 0
        assert np.allclose(whole[0, s:s + 200], CLICKS[kind][:200], atol=1e-6)
        assert not whole[:, s - 5:s].any() or b == 0                 # silence just before it
    assert np.abs(CLICKS[2]).max() > np.abs(CLICKS[1]).max() > np.abs(CLICKS[0]).max()


def test_metronome_switch_and_level_are_in_the_transport_state():
    p = Player()
    p.set_click(on=True, gain=0.25)
    st = p.state()
    assert st["click"] is True and st["click_gain"] == 0.25
    p.set_click(on=False)
    assert p.state()["click"] is False and p.state()["click_gain"] == 0.25
