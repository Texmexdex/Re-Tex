"""Vocal arrangement planner: which phrase of V's vocal goes on which part of bed song I.

The bed keeps its own structure. Its content is cut into 2-bar slots (aligned to its sections); for every
slot the planner picks either silence or a stretch of V's vocal that starts on one of V's phrase starts
(section starts / every 2 bars) - and prefers to simply CONTINUE the line that is already playing.

Scores (all per slot, summed by a beam search with memory of what was used):
  harmony   - the learned note-against-chords fit (harmfit) at the exact alignment, per beat, weighted by
              how clearly the vocal sings a note there. Better than a random key adds, worse subtracts.
  vocal     - amount of vocal actually placed (the density slider scales how much that is worth)
  energy    - V's phrase intensity vs the bed's (both relative to their own songs); hooks / choruses of
              V on the bed's biggest sections
  cuts      - jumping away from / into the middle of a sung line costs; phrase ends and gaps are free
  structure - enter on section starts, keep V's lyric order (repeating a chorus is fine, a verse isn't)
"""
import math
import numpy as np

from . import harmfit, harmony as H


def _norm01(x, lo_q=0.1, hi_q=0.9):
    x = np.asarray(x, float)
    lo, hi = np.quantile(x, lo_q), np.quantile(x, hi_q)
    return np.clip((x - lo) / max(1e-3, hi - lo), 0, 1)


def bed_slots(I, n_max=2):
    """[(bar index, n bars, section)] over I's content, at the main tempo. 2-bar slots sit on I's measures
    (I.measure_starts); a section that starts or ends mid-measure gets a 1-bar slot there."""
    c0, c1 = I.info["content_bars"]
    ms = I.measure_starts() if n_max == 2 and hasattr(I, "measure_starts") else None
    slots = []
    for sec in I.sections:
        a, b = max(sec["start"], c0), min(sec["end"], c1)
        i = a
        while i < b:
            if ms is None:
                n = min(n_max, b - i)
            else:
                n = 2 if ms[i] and i + 1 < b and not ms[i + 1] else 1
            if all(I.bar_range_ok(i + j) for j in range(n)):
                slots.append((i, n, sec))
            i += n
    return slots


def _bar_beats(S, bi):
    a = S.bars[bi]
    b = S.bars[bi + 1] if bi + 1 < len(S.bars) else min(S.nb, a + S.bpb)
    return a, b


class Grid:
    """Every (bed slot, vocal start) alignment with its per-beat harmony scores, computed once per pair
    and tempo ratio; any transposition is then read off the same scores."""

    def __init__(self, V, I, rv, slots=None):
        self.V, self.I, self.rv = V, I, rv            # rv: V source beats per bed beat
        self.slots = slots or bed_slots(I)
        if not self.slots:
            raise ValueError("bed song has no usable bars")
        bpb = I.bpb
        span = bpb * 2 * rv                             # V beats covered by a 2-bar slot
        # a 2-bar bed slot covers 2*rv of V's bars: offer starts at that spacing so a line can always continue
        step = max(1, int(round(min(2.0, 2 * rv))))
        usable = lambda vb: vb + span <= V.info["last_beat"] + 1 and V.tempo_ok[vb:int(math.ceil(vb + span))].all()
        entries = [vb for vb in V.phrase_starts(step) if usable(vb)]
        if not entries:
            raise ValueError(f"{V.title} has no usable vocal phrases at its main tempo")
        # A bed section with an odd number of bars ends in a 1-bar slot; a line sung through it continues one bar
        # off V's 2-bar phrase lattice. Offer those positions as continuations only (a new line still enters on
        # a phrase start) - without them, "continuing" past a 1-bar slot silently skipped a bar of the vocal.
        starts = set(entries)
        one = bpb * rv                                  # V beats under a 1-bar slot
        if any(n % 2 for _, n, _ in self.slots) and abs(one - round(one)) < 1e-6:
            todo = list(entries)
            while todo:
                vb = todo.pop() + int(round(one))
                if vb not in starts and vb < V.nb and usable(vb):
                    starts.add(vb)
                    todo.append(vb)
        self.starts = sorted(starts)
        ent = set(entries)
        self.entry = np.array([vb in ent for vb in self.starts])   # False: reachable only by continuing a line
        vw = V.vocal_weight()                           # (nb,) how clearly a note is sung, 0 = silent
        hon = I.harm_on()                               # (nb,) bed has harmony there
        # enumerate (slot, start, beat) cells: bed beat i, V beat(s) j
        K, S = len(self.slots), len(self.starts)
        cells_i, cells_j, cells_w, cells_ks = [], [], [], []
        nb_slot = np.zeros(K, int)
        for k, (bi, n, _) in enumerate(self.slots):
            a, _ = _bar_beats(I, bi)
            _, b = _bar_beats(I, bi + n - 1)
            ib = np.arange(a, b)
            nb_slot[k] = len(ib)
            t = np.arange(len(ib))
            for s, st in enumerate(self.starts):
                if rv <= 1.0001:
                    j = np.floor(st + t * rv + 1e-6).astype(int)
                    jj, ii, ww = j, ib, np.ones(len(ib))
                else:
                    r = int(round(rv))
                    j = (np.floor(st + t[:, None] * rv + 1e-6).astype(int) + np.arange(r)[None, :]).ravel()
                    jj, ii, ww = j, np.repeat(ib, r), np.full(len(j), 1.0 / r)
                ok = jj < V.nb
                cells_i.append(ii[ok]); cells_j.append(jj[ok]); cells_w.append(ww[ok])
                cells_ks.append(np.full(ok.sum(), k * S + s))
        ci, cj = np.concatenate(cells_i), np.concatenate(cells_j)
        cw = np.concatenate(cells_w) * vw[cj] * (0.35 + 0.65 * hon[ci])
        cks = np.concatenate(cells_ks)
        live = cw > 1e-3
        self.ci, self.cj, self.cw, self.cks = ci[live], cj[live], cw[live], cks[live]
        self.K, self.S, self.nb_slot = K, S, nb_slot
        # harmony scores for the distinct (bed beat, vocal beat) pairs
        pair = self.ci.astype(np.int64) * (V.nb + 1) + self.cj
        uniq, inv = np.unique(pair, return_inverse=True)
        ui, uj = uniq // (V.nb + 1), uniq % (V.nb + 1)
        model = harmfit.get()
        bass, oth, ctx = I.backing()
        X = harmfit.features([V.melody_rep(r)[uj] for r in model.reps], bass[ui], oth[ui], ctx[ui])
        self.scores = model.shift_scores(X)             # (n pairs, 12)
        # how much the key matters on this beat: spread of the scores over the 12 transpositions
        # (a held note over clear chords: large; a scream, a whisper or a drums-only bar: ~0)
        self.info = self.scores.std(1)[inv] / model.meta.get("u_ref", 1.0)
        self.inv = inv
        self.model = model
        # vocal amount per (slot, start), independent of key
        self.vocal = np.bincount(self.cks, weights=self.cw, minlength=K * S).reshape(K, S).astype(float)

    def info_sums(self, vocal_shift):
        """-> (K, S) sum of weight*info*fit, (K, S) sum of weight*info: harmony measured only where it
        can be judged (a clear sung note against clear harmony)."""
        fit = self.model.fit(self.model.regret(self.scores, vocal_shift))[self.inv]
        w = self.cw * np.minimum(self.info, 2.0)
        n = self.K * self.S
        return (np.bincount(self.cks, weights=w * fit, minlength=n).reshape(self.K, self.S).astype(float),
                np.bincount(self.cks, weights=w, minlength=n).reshape(self.K, self.S).astype(float))

    def fit_matrix(self, vocal_shift):
        """-> (K, S) weighted mean fit, (K, S) weighted fit-minus-random sum (in 'beats')."""
        fit = self.model.fit(self.model.regret(self.scores, vocal_shift))[self.inv]
        base = self.model.meta["fit_random"]
        num = np.bincount(self.cks, weights=self.cw * fit, minlength=self.K * self.S).reshape(self.K, self.S).astype(float)
        gain = np.bincount(self.cks, weights=self.cw * (fit - base), minlength=self.K * self.S).reshape(self.K, self.S).astype(float)
        mean = np.divide(num, self.vocal, out=np.full_like(num, base), where=self.vocal > 1e-6)
        return mean, gain


# weights, tuned by letting the planner rebuild songs from their own vocal + instrumental (see README)
# Held-out songs: 63 % of phrases land exactly where the original had them (hand-set weights: 38 %),
# vocal-vs-silence right on 77 % of slots. Weighting harmony harder per phrase made plans choppier (it
# chases noise at 2-bar resolution) - key choice, done per song, carries most of the harmony decision.
PARAMS = dict(harm=0.8, voc=0.6, own=0.9, energy=1.3, hook=0.3, sil=0.2, sil_low=0.5, dens=0.8,
              cont=0.78, jump=2.13, cut=0.8, back=1.34, sec=0.3, reuse=0.63, reuse_hook=0.15)


def plan_vocals(grid, vocal_shift, density=0.5, beam_w=48, P=None):
    """Beam search over the bed's slots. Returns the chosen path and its statistics."""
    P = dict(PARAMS, **(P or {}))
    V, I = grid.V, grid.I
    slots, starts = grid.slots, grid.starts
    K, S, NONE = grid.K, grid.S, grid.S
    rv = grid.rv
    bpb = I.bpb
    span = 2 * bpb * rv
    st = np.array(starts, float)
    entry = getattr(grid, "entry", np.ones(S, bool))
    # a line may only open on one of the bed's measure starts (V's starts are its own measure starts): entering
    # mid-measure puts the whole line half a measure off the bed
    ims = I.measure_starts() if hasattr(I, "measure_starts") else None
    slot_in = np.array([bool(ims[bi]) if ims is not None else True for bi, _, _ in slots])
    # where a line from each start has got to by the end of each slot (slots are 2 bars, or 1 at an odd section end)
    ends_k = st[None, :] + np.array([n for _, n, _ in slots], float)[:, None] * bpb * rv     # (K, S)
    fit_mean, gain = grid.fit_matrix(vocal_shift)
    vocal = grid.vocal
    model = grid.model
    nbs = np.maximum(grid.nb_slot.astype(float), 1)
    # energies relative to each song's own dynamics
    ien = _norm01([I.bar_energy()[bi:bi + n].mean() for bi, n, _ in slots])
    ven = _norm01([V.beat_energy[int(s):int(s + span)].mean() for s in st])
    hook = V.hook_starts()
    is_hook = np.array([s in hook for s in starts])
    top = ien >= np.quantile(ien, 0.6)
    own_vox = np.array([I.bar_presence("vocals")[bi:bi + n].mean() for bi, n, _ in slots])
    bed_has_vox = I.has_real_vocals

    # ---- emissions (K, S+1)
    cover = np.minimum(vocal / (nbs[:, None] * 0.5), 1.0)              # ~1 = well-filled slot
    e_match = 1 - np.abs(ien[:, None] - ven[None, :])
    emit = np.full((K, S + 1), -1e9)
    place = own_vox if bed_has_vox else ien                             # where the bed leaves room for a voice
    emit[:, :S] = (P["harm"] * gain / nbs[:, None] * 4                  # harmony, per ~4 beats of confident singing
                   + P["voc"] * cover
                   + P["own"] * (place[:, None] - 0.5)
                   + P["energy"] * (e_match - 0.7)
                   + P["hook"] * np.where(is_hook[None, :], np.where(top[:, None], 1.0, -0.5), 0.0))
    emit[:, :S][vocal < 0.5] = -1e9                                     # nothing sung there
    emit[:, NONE] = P["sil"] + P["sil_low"] * (1 - ien) + (0.5 - density) * P["dens"]

    # ---- transitions
    c_in, c_out = V.cut_cost()                          # per beat: 0 = clean gap / short pickup, 1 = mid-line
    def cut_at(c, b):
        return c[np.clip(np.round(b).astype(int), 0, len(c) - 1)]
    cut_start = cut_at(c_in, st)
    cut_end_k = cut_at(c_out, ends_k)                   # (K, S): leaving after slot k, from start s
    secs = V.sections
    sec_of = np.array([next((i for i, sc in enumerate(secs) if V.bars[sc["start"]] <= b < (V.bars[sc["end"]] if sc["end"] < len(V.bars) else V.nb)), -1) for b in starts])
    rep = np.array([secs[i].get("repeats", 1) > 1 if i >= 0 else False for i in sec_of])
    v_sec_start = np.array([any(V.bars[sc["start"]] == b for sc in secs if sc["start"] < len(V.bars)) for b in starts])

    def enter_at(k):
        """Cost of opening a line in slot k, per start (the first slot too: a plan may not open mid-line for free)."""
        if not slot_in[k]:
            return np.full(S, -1e18)
        bi, _, sec = slots[k]
        i_sec_start = bi == sec["start"]
        e = -P["cut"] * cut_start + P["sec"] * (v_sec_start & i_sec_start) - (0 if i_sec_start else 0.3)
        return np.where(entry, e, -1e18)                # continuation-only starts can't open a line

    T = [None]
    for k in range(1, K):
        pbi, pn, _ = slots[k - 1]
        bi, n, sec = slots[k]
        adjacent = pbi + pn == bi
        M = np.zeros((S + 1, S + 1))
        cont = (np.abs(ends_k[k - 1][:, None] - st[None, :]) < 0.5) & adjacent
        enter = enter_at(k)
        leave = -P["cut"] * cut_end_k[k - 1]
        M[NONE, :S] = enter
        M[:S, NONE] = leave
        back = (st[None, :] <= st[:, None]) & (~rep[None, :])
        jump = leave[:, None] + enter[None, :] - P["jump"] - P["back"] * back
        M[:S, :S] = np.where(cont, P["cont"], jump)
        T.append((M, cont))

    reuse_pen = np.where(rep | is_hook, P["reuse_hook"], P["reuse"])
    first = emit[0].copy()
    first[:S] += enter_at(0)
    beams = [(first[s], [int(s)], np.zeros(S)) for s in np.where(first > -1e8)[0]]
    beams = sorted(beams, key=lambda h: -h[0])[:beam_w]
    for k in range(1, K):
        M, cont = T[k]
        valid = emit[k] > -1e8
        cand = []
        for sc, path, use in beams:
            a = path[-1]
            row = sc + M[a] + emit[k]
            c = cont[a] if a != NONE else np.zeros(S, bool)
            row[:S] -= reuse_pen * use * (~c)
            row[~valid] = -1e18
            for b in np.argsort(-row)[:8]:
                if row[b] > -1e17:
                    cand.append((row[b], path, use, int(b)))
        cand.sort(key=lambda c: -c[0])
        new, per_last = [], {}
        for sc, path, use, b in cand:
            if per_last.get(b, 0) >= 3:
                continue
            per_last[b] = per_last.get(b, 0) + 1
            u = use.copy()
            if b != NONE:
                u[b] += 1
            new.append((sc, path + [b], u))
            if len(new) >= beam_w:
                break
        beams = new
    total, path = float(beams[0][0]), beams[0][1]

    # ---- statistics of the chosen plan
    used = [(k, s) for k, s in enumerate(path) if s != NONE]
    vw_sum = sum(vocal[k, s] for k, s in used)
    fit = (sum(fit_mean[k, s] * vocal[k, s] for k, s in used) / vw_sum) if vw_sum > 0 else model.meta["fit_random"]
    inum, iden = grid.info_sums(vocal_shift)
    i_n = sum(inum[k, s] for k, s in used)
    i_d = sum(iden[k, s] for k, s in used)
    fit_info = i_n / i_d if i_d > 1e-6 else model.meta.get("plan_fit_random", 0.5)
    relevance = i_d / vw_sum if vw_sum > 0 else 0.0            # ~1 = as judgeable as a typical sung vocal
    hon = I.harm_on()
    bed_harm = float(np.mean([hon[_bar_beats(I, bi)[0]:_bar_beats(I, bi + n - 1)[1]].mean() for bi, n, _ in slots]))
    bars = sum(slots[k][1] for k, _ in used)
    total_bars = sum(n for _, n, _ in slots)
    top_bars = sum(slots[k][1] for k in range(K) if top[k])
    top_voc = sum(slots[k][1] for k, _ in used if top[k])
    cuts = []
    entries = 0
    for k in range(K):
        a = path[k - 1] if k else NONE
        b = path[k]
        continuing = k > 0 and a != NONE and b != NONE and T[k][1][a, b]
        if b != NONE and not continuing:
            entries += 1
            cuts.append(float(cut_start[b]))
        if a != NONE and not continuing and k > 0:
            cuts.append(float(cut_end_k[k - 1][a]))
    if path and path[-1] != NONE:
        cuts.append(float(cut_end_k[K - 1][path[-1]]))
    starts_used = [s for _, s in used]
    return dict(path=[None if s == NONE else starts[s] for s in path], slots=slots, score=total / K,
                fit=float(fit), fit_info=float(fit_info), relevance=float(relevance), bed_harm=bed_harm,
                vocal_beats=float(vw_sum), bars=bars, total_bars=total_bars,
                hook_cover=(top_voc / top_bars) if top_bars else 0.0,
                cut_mean=float(np.mean(cuts)) if cuts else 0.0, cut_bad=int(sum(c > 0.5 for c in cuts)),
                n_entries=entries, repeats=len(starts_used) - len(set(starts_used)),
                own_vocal_bars=float((own_vox * np.array([n for _, n, _ in slots])).sum()),
                rv=rv, shift=int(vocal_shift))
