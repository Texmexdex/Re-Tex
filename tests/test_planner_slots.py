"""Planner: a vocal line sung through a 1-bar bed slot (an odd-length bed section) must continue from where it
actually got to, not skip a bar. Before the fix every slot was assumed to be 2 bars, so the line "continued" 8
beats on after a 4-beat slot and the cut cost was read a bar too late (Easy over Hard Refresh: bars skipped at
1:04.9 and 1:22.7, words chopped there). CPU-only: a stub grid, no harmfit model, no audio.
"""
import numpy as np

from mashlab import planner

BPB = 4


class _Model:
    meta = {"fit_random": 0.5, "plan_fit_random": 0.5}


class _Song:
    def __init__(self, n_bars, sections, has_vocals=False):
        self.bpb = BPB
        self.bars = [b * BPB for b in range(n_bars)]
        self.nb = n_bars * BPB
        self.sections = sections
        self.has_real_vocals = has_vocals
        self.beat_energy = np.full(self.nb, 0.5)

    def bar_energy(self):
        return np.full(len(self.bars), 0.5)

    def bar_presence(self, stem):
        return np.zeros(len(self.bars))

    def harm_on(self):
        return np.ones(self.nb)

    def hook_starts(self):
        return set()

    def cut_cost(self):
        # one long sung line from beat 0: entering anywhere but its start, or leaving before its end, chops it
        c_in = np.ones(self.nb)
        c_in[0] = 0.0
        c_out = np.ones(self.nb)
        c_out[-1] = 0.0
        return c_in, c_out


class _Grid:
    def __init__(self, V, I, slots, starts, entry):
        self.V, self.I, self.rv, self.slots = V, I, 1.0, slots
        self.starts, self.entry = starts, np.array(entry)
        self.K, self.S = len(slots), len(starts)
        self.nb_slot = np.array([n * BPB for _, n, _ in slots])
        self.vocal = np.repeat(self.nb_slot[:, None], self.S, 1).astype(float)
        self.model = _Model()

    def fit_matrix(self, shift):
        return np.full((self.K, self.S), 0.5), np.zeros((self.K, self.S))

    def info_sums(self, shift):
        return np.zeros((self.K, self.S)), np.zeros((self.K, self.S))


def _grid():
    # bed: a 3-bar section then a 4-bar one -> slots of 2, 1, 2, 2 bars
    a, b = dict(start=0, end=3), dict(start=3, end=7)
    I = _Song(7, [a, b])
    V = _Song(16, [dict(start=0, end=16)])
    slots = [(0, 2, a), (2, 1, a), (3, 2, b), (5, 2, b)]
    starts = list(range(0, 57, BPB))                      # every bar; lines open only on 2-bar phrase starts
    return _Grid(V, I, slots, starts, [s % 8 == 0 for s in starts])


def test_line_continues_through_one_bar_slot_without_skipping():
    p = planner.plan_vocals(_grid(), 0, density=1.0)
    assert p["path"] == [0, 8, 12, 20]                    # was [0, 8, 16, 24]: beats 12..16 skipped
    assert p["n_entries"] == 1 and p["cut_bad"] == 1      # one line in, one (forced) way out at the very end


def test_first_slot_pays_for_opening_mid_line():
    # the old plan opened its first slot for free, so Easy over Hard Refresh began mid-lyric at 0:00
    g = _grid()
    c_in, c_out = g.V.cut_cost()
    c_in[:] = 1.0
    c_in[8] = 0.0                                         # the only clean line start is beat 8
    g.V.cut_cost = lambda: (c_in, c_out)
    p = planner.plan_vocals(g, 0, density=1.0)
    assert p["path"][0] == 8


def test_lines_open_only_on_bed_measure_starts():
    # bed measures start on even bars, but its section detector put a section start on bar 3 (Hard Refresh's 11-bar
    # 'chorus'): a line must not open there - it would sit half a measure off the bed - but on bar 4
    a, b = dict(start=0, end=3), dict(start=3, end=6)
    g = _grid()
    g.I = _Song(6, [a, b])
    g.I.measure_starts = lambda: np.arange(6) % 2 == 0
    g.slots = [(0, 2, a), (2, 1, a), (3, 1, b), (4, 2, b)]
    g.K, g.nb_slot = 4, np.array([8, 4, 4, 8])
    g.vocal = np.repeat(g.nb_slot[:, None], g.S, 1).astype(float)
    g.vocal[:2] = 0.0                                     # nothing to sing before bar 3
    g.fit_matrix = lambda shift: (np.full((g.K, g.S), 0.5), g.vocal.copy())   # singing clearly pays
    p = planner.plan_vocals(g, 0, density=1.0)
    assert p["path"][2] is None and p["path"][3] is not None


def test_continuation_only_starts_never_open_a_line():
    g = _grid()
    g.entry[:] = [s in (4, 12) for s in g.starts]        # only off-lattice starts... and they may not open lines
    p = planner.plan_vocals(g, 0, density=1.0)
    assert p["path"][0] is None or g.entry[g.starts.index(p["path"][0])]
