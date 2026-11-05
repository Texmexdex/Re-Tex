"""Automatic mashup generation.

Every arrangement has a BED song I: its music (the "other" stem) and its structure (sections, phrase
slots) are the backbone. Any of the other roles can come from another song as a LAYER:

    vocals  - phrases of V's vocal placed on I's phrase slots (or I's own vocal, or none)
    drums   - D's drum phrases, energy-matched, only where I had drums (breakdowns stay breakdowns)
    bass    - B's bass phrases, harmonically matched against I's "other" stem

Per layer (song L, role):
  1. tempo  - project BPM leans toward the bed; layers may run at half/double time. Only each song's
              main-tempo region is used (a 200 BPM breakdown never gets squashed to 120).
  2. key    - pitched layers are scored for every transpose by real bar-level compatibility against
              the bed: chroma agreement + how clearly in-key the *combined* bar sounds. The bed's own
              transpose is chosen so the layers move as little as possible (least squares).
  3. phrases- beam search over I's phrase slots choosing, per slot, a phrase of L (or silence).
              Scores: bar-level harmony, energy match, coverage; transitions reward continuing a line,
              entering at section starts, leaving at phrase ends; reusing a phrase costs (vocals most,
              drum loops hardly at all).
  4. polish - vocal pickups/tails, loudness matched to the stem being replaced, vocal pocket + ducking.

Ratings: harmony 100 % = the layer fits I as well as it fits its own original song; 0 % = a random key.
"""
import itertools
import math
import numpy as np

from . import analyze, harmony as H, library, project as P
from .config import STEMS
from .structure import presence

MULTS = (0.5, 1.0, 2.0)
SHIFTS = np.arange(-6, 6)
PITCHED_ROLES = ("vocals", "bass")
STRETCH_LIMIT = {"vocals": 0.2, "bass": 0.22, "drums": 0.28}
_PROF = np.vstack([np.roll((H.MAJ - H.MAJ.mean()) / H.MAJ.std(), t) for t in range(12)] +
                  [np.roll((H.MIN - H.MIN.mean()) / H.MIN.std(), t) for t in range(12)])     # (24, 12)


def _unit(X):
    return X / (np.linalg.norm(X, axis=-1, keepdims=True) + 1e-9)


def compat(Ic, Vc, shift):
    """Bar-level harmonic compatibility. Ic (..., 12) bed chroma, Vc (..., 12) layer chroma
    (broadcastable). Vc is transposed by `shift`. Returns (...) roughly in [0, 1]."""
    Vs = np.roll(Vc, shift, axis=-1)
    cos = np.sum(_unit(Ic) * _unit(Vs), -1)
    mix = Ic / (Ic.sum(-1, keepdims=True) + 1e-9) + Vs / (Vs.sum(-1, keepdims=True) + 1e-9)
    z = (mix - mix.mean(-1, keepdims=True)) / (mix.std(-1, keepdims=True) + 1e-9)
    clar = (z @ _PROF.T).max(-1) / 12
    return 0.5 * cos + 0.5 * clar


def _window_mean(X, w):
    """M[b] = mean of rows X[b:b+w] (per-beat -> per-window)."""
    X = np.asarray(X, float)
    if X.ndim == 1:
        X = X[:, None]
    cs = np.vstack([np.zeros((1, X.shape[1])), np.cumsum(X, 0)])
    return (cs[w:] - cs[:-w]) / w


# ------------------------------------------------------------------ song data

class SongData:
    def __init__(self, sid):
        self.id = sid
        self.info = library.read(sid)
        if not self.info or self.info.get("status") != "ready":
            raise ValueError(f"song {sid} is not analyzed yet")
        self.title = self.info["title"]
        self.f = analyze.load_features(library.song_dir(sid))
        self.bpm = float(self.info["bpm"])
        self.bpb = int(self.info["bpb"])
        self.bars = list(self.info["bars"])
        self.beats = np.array(self.info["beats"])
        self.nb = len(self.beats)
        pk = self.info["stem_peak_db"]
        self.on = {s: presence(self.f[f"rms_{s}"], pk[s], self.f["rms_mix"]) for s in STEMS}
        self.chroma = {
            "vocals": H.normalize_rows(self.f["chroma_vocals"]),
            "bass": H.normalize_rows(self.f["chroma_bass"]),
            "other": H.normalize_rows(self.f["chroma_other"]),
            "inst": H.normalize_rows(1.5 * H.normalize_rows(self.f["chroma_bass"]) +
                                     H.normalize_rows(self.f["chroma_other"])),
        }
        rm = self.f["rms_mix"]
        top = np.quantile(rm[self.info["first_beat"]:self.info["last_beat"] + 1], 0.95)
        self.beat_energy = np.clip((rm - (top - 30)) / 30, 0, 1)
        self.sections = self.info["sections"]
        # beats that belong to the main tempo: the LOCAL tempo (median beat length over ~2 bars) within
        # 10 % of the song tempo. Beat-to-beat jitter of live playing is fine (the warp maps every beat);
        # what must be excluded is a section at another tempo (half-time breakdown, rubato intro).
        from scipy.ndimage import median_filter
        ibi = np.diff(self.beats)
        loc = median_filter(ibi, size=9, mode="nearest") if len(ibi) else ibi
        ok = np.abs(loc * self.bpm / 60 - 1) < 0.10
        self.tempo_ok = np.r_[ok, ok[-1:]] if len(ok) else np.ones(self.nb, bool)
        self._bar = {}
        self._href = {}

    @property
    def vocal_fraction(self):
        a, b = self.info["first_beat"], self.info["last_beat"]
        return float(self.on["vocals"][a:b].mean())

    @property
    def has_real_vocals(self):
        if "has_vocals" in self.info:
            return bool(self.info["has_vocals"])
        L = self.info["loudness"]
        return L.get("vocals", -70) - L.get("mix", 0) > -13 and self.vocal_fraction > 0.1

    def has_stem(self, stem):
        """Is there enough of this stem to use it as a layer?"""
        if stem == "vocals":
            return self.has_real_vocals
        a, b = self.info["first_beat"], self.info["last_beat"]
        L = self.info["loudness"]
        return float(self.on[stem][a:b].mean()) > 0.15 and L.get(stem, -70) - L.get("mix", 0) > -20

    def bar_range_ok(self, bi):
        a = self.bars[bi]
        b = self.bars[bi + 1] if bi + 1 < len(self.bars) else min(self.nb, a + self.bpb)
        return bool(self.tempo_ok[a:b].all())

    def _per_bar(self, key, X):
        if key not in self._bar:
            out = []
            for i, a in enumerate(self.bars):
                b = self.bars[i + 1] if i + 1 < len(self.bars) else min(self.nb, a + self.bpb)
                out.append(X[a:max(b, a + 1)].mean(0))
            self._bar[key] = np.array(out)
        return self._bar[key]

    def bar_chroma(self, which):
        return self._per_bar("c_" + which, self.chroma[which])

    def bar_presence(self, stem):
        return self._per_bar("p_" + stem, self.on[stem].astype(float))

    def bar_harm_presence(self):
        return self._per_bar("p_harm", (self.on["bass"] | self.on["other"]).astype(float))

    def bar_energy(self):
        return self._per_bar("e", self.beat_energy)

    def harmony_reference(self, stem="vocals"):
        """(self, base): compatibility of this song's `stem` with its own backing at 0 st, and averaged
        over all transposes (what a random key would give)."""
        if stem not in self._href:
            tgt = self.bar_chroma("inst" if stem == "vocals" else "other")
            lay = self.bar_chroma(stem)
            other_on = self.bar_harm_presence() if stem == "vocals" else self.bar_presence("other")
            sel = (self.bar_presence(stem) > 0.5) & (other_on > 0.5)
            if sel.sum() < 4:
                self._href[stem] = (0.75, 0.55)
            else:
                allk = np.array([compat(tgt[sel], lay[sel], k).mean() for k in SHIFTS])
                self._href[stem] = (float(allk[6]), float(allk.mean()))
        return self._href[stem]

    # ---------------------------------------------------------- features for the learned planner
    @property
    def mash(self):
        """Audio-derived melody / groove features (mash.npz), or None if not computed yet."""
        if not hasattr(self, "_mash"):
            from . import mashfeat
            self._mash = mashfeat.load(library.song_dir(self.id))
        return self._mash

    @property
    def lyrics(self):
        """Transcript + pinned quality block (lyrics.json, design A.1), or None if not transcribed yet. Lazy
        disk-cached like `mash` — read from disk only; transcription (GPU) is an explicit batch step
        (lyrics.ensure), never triggered implicitly by a rating pass."""
        if not hasattr(self, "_lyrics"):
            from . import lyrics as _lyr
            self._lyrics = _lyr.load(self.id)
        return self._lyrics

    @property
    def concept(self):
        """Concept profile (concept.json, design A.2), or None if not profiled yet. Lazy disk-cached like
        `mash`; profiling is an explicit batch step (concept.write), never triggered by a rating pass."""
        if not hasattr(self, "_concept"):
            from . import concept as _con
            self._concept = _con.load(self.id)
        return self._concept

    @property
    def stability(self):
        """0..1: share of the voiced vocal that holds steady notes (singing) rather than rap / screams."""
        m = self.mash
        return float(m["stats"]["stability"]) if m is not None else 0.5

    def _pad(self, X):
        X = np.asarray(X)
        if len(X) >= self.nb:
            return X[:self.nb]
        return np.concatenate([X, np.zeros((self.nb - len(X),) + X.shape[1:], X.dtype)])

    def melody_rep(self, rep):
        """Per-beat vocal pitch content: 'mel' = tracked sung notes, 'cqt' = spectral chroma."""
        key = "rep_" + rep
        if key not in self._bar:
            if rep == "cqt" or self.mash is None:
                X = H.normalize_rows(self.f["chroma_vocals"])
            else:
                X = H.normalize_rows(self._pad(self.mash["mel"]))
            self._bar[key] = X.astype(np.float32)
        return self._bar[key]

    def vocal_weight(self):
        """(nb,) how clearly a note is sung on each beat (0 = no vocal)."""
        if "vw" not in self._bar:
            on = self.on["vocals"].astype(float)
            m = self.mash
            if m is None:
                w = on * 0.6
            else:
                w = on * self._pad(m["vfrac"]) * (0.25 + 0.75 * self._pad(m["sfrac"]))
            self._bar["vw"] = w
        return self._bar["vw"]

    def harm_on(self):
        return (self.on["bass"] | self.on["other"]).astype(float)

    def backing(self):
        """(bass, other, key context) per beat, each a pitch-class distribution."""
        if "backing" not in self._bar:
            from . import harmfit
            bass = H.normalize_rows(self.f["chroma_bass"])
            oth = H.normalize_rows(self.f["chroma_other"])
            ctx = harmfit.context(H.normalize_rows(1.3 * bass + oth), self.harm_on())
            self._bar["backing"] = (bass.astype(np.float32), oth.astype(np.float32), ctx.astype(np.float32))
        return self._bar["backing"]

    def hook_starts(self):
        """Phrase starts inside V's hooks: repeated, loud sections with vocals."""
        es = [s["energy"] for s in self.sections] or [0]
        hi = np.quantile(es, 0.6)
        ms = self.measure_starts()
        out = set()
        for sec in self.sections:
            if (sec.get("name") == "chorus" or (sec.get("repeats", 1) > 1 and sec["energy"] >= hi))                     and sec.get("stems", {}).get("vocals", 0) > 0.5:
                for bi in range(sec["start"], min(sec["end"], len(self.bars))):
                    if ms[bi]:
                        out.add(self.bars[bi])
        return out

    MEASURE_SWITCH = 3.0      # evidence (~3 clear vocal entries) it takes to move the measure phase mid-song

    def measure_starts(self):
        """(bars,) bool: bar i starts a 2-bar measure - what a listener counts in at these tempos (a 150 BPM track is
        felt at 75) and what phrases come in on. Section boundaries can't be trusted for it: Hard Refresh's 11-bar
        'chorus' put every later section half a measure off, and every vocal entered there landed mid-measure.

        Tracked from where the singer comes in (a line after a breath starts on, or picks up into, a measure start)
        and where the instrumental changes most (the 2 bars after vs before), with one phase all the way through
        strongly preferred: a 2-state Viterbi over the bars, MEASURE_SWITCH to change phase."""
        if "measure" in self._bar:
            return self._bar["measure"]
        nbar = len(self.bars)
        e = np.zeros(nbar)
        vm = self.voicing()
        if vm is not None:
            on = np.flatnonzero(np.diff(np.r_[0, vm.astype(np.int8)]) == 1)
            off = np.flatnonzero(np.diff(np.r_[vm.astype(np.int8), 0]) == -1)
            bix = np.arange(self.nb, dtype=float)
            for o in on:
                prev = off[off < o]
                gap = (o - prev[-1]) / 100 if len(prev) else 9.0
                if gap < 0.3:
                    continue                                   # not a new line, just a breath inside one
                p = float(np.interp(o / 100, self.beats, bix))
                bi = int(np.searchsorted(self.bars, p + 1.75)) - 1
                if 0 <= bi < nbar and -0.75 <= self.bars[bi] - p <= 1.75:
                    e[bi] += min(1.0, gap)                     # up to 1.75 beats of pickup, or a little late
        f = self.f
        feat = np.hstack([np.c_[f["rms_drums"], f["rms_bass"], f["rms_other"]] / 10.0,
                          H.normalize_rows(f["chroma_bass"]), H.normalize_rows(f["chroma_other"])])
        B = self._per_bar("measure_feat", feat)
        c0, c1 = self.info.get("content_bars", (0, nbar))
        if c1 - c0 >= 5:
            ch = np.zeros(nbar)
            for i in range(c0 + 2, c1 - 1):                    # not across the music's own start / end
                ch[i] = np.linalg.norm(B[i - 2:i].mean(0) - B[i:i + 2].mean(0))
            ch[1:-1] -= 0.5 * (ch[:-2] + ch[2:])               # only the contrast with the neighbouring bars
            ch[:c0 + 2] = ch[c1 - 1:] = 0.0
            e += ch / (ch.std() + 1e-9) * 0.5
        e[:c0] = e[c1:] = 0.0                                  # silence before / after the music says nothing
        score = np.array([e[0] + 0.5, 0.0])                    # mild prior: the music starts on a measure
        back = np.zeros((nbar, 2), int)
        for i in range(1, nbar):
            new = np.empty(2)
            for s in (0, 1):
                stay, sw = score[s], score[1 - s] - self.MEASURE_SWITCH
                back[i, s] = s if stay >= sw else 1 - s
                new[s] = max(stay, sw) + (e[i] if i % 2 == s else 0.0)
            score = new
        s = int(np.argmax(score))
        phase = np.empty(nbar, int)
        for i in range(nbar - 1, -1, -1):
            phase[i] = s
            s = back[i, s]
        self._bar["measure"] = (np.arange(nbar) % 2) == phase
        return self._bar["measure"]

    def voicing(self):
        """(frames,) bool at 100 fps: the vocal is sounding, or None without melody features. Voicing is extended a
        little (consonants, breathy onsets) and gaps under 150 ms are closed, so a gap is a real breath. The planner's
        cut costs and the clip edges both use this, so a cut the planner calls clean really lands in a gap."""
        if "voicing" not in self._bar:
            m = self.mash
            if m is None:
                self._bar["voicing"] = None
            else:
                from scipy.ndimage import binary_closing, binary_dilation
                v = np.isfinite(m["f0"])
                self._bar["voicing"] = binary_dilation(binary_closing(v, np.ones(15, bool)), np.ones(5, bool))
        return self._bar["voicing"]

    def cut_cost(self):
        """(start, end) per beat: 0 = the vocal can be cut here cleanly (gap, or only a short pickup /
        held word that the clip edges absorb), 1 = it would chop through a sung line."""
        if "cut" in self._bar:
            return self._bar["cut"]
        nb = self.nb
        v = self.voicing()
        if v is None:
            on = self.on["vocals"].astype(float)
            cs = np.clip(np.r_[0, on[:-1]] * on, 0, 1)
            self._bar["cut"] = (cs, cs)
            return self._bar["cut"]
        fps = 100.0
        n = len(v)
        # distance (s) back to the last gap / forward to the next gap, for every frame
        back = np.zeros(n)
        fwd = np.zeros(n)
        for i in range(1, n):
            back[i] = back[i - 1] + 1 / fps if v[i] else 0.0
        for i in range(n - 2, -1, -1):
            fwd[i] = fwd[i + 1] + 1 / fps if v[i] else 0.0
        t = np.clip(np.round(self.beats * fps).astype(int), 0, n - 1)
        bar = 60.0 / self.bpm * self.bpb
        c_in = np.clip((back[t] - 0.5 * bar) / (0.5 * bar), 0, 1)     # pickup longer than half a bar
        c_out = np.clip((fwd[t] - 0.25 * bar) / (0.5 * bar), 0, 1)    # word held past a quarter bar
        self._bar["cut"] = (c_in, c_out)
        return self._bar["cut"]

    def groove(self):
        """Swing of the drums (or, without drums, the vocal) - 8th- and 16th-note level, 0 = straight."""
        if "groove" in self._bar:
            return self._bar["groove"]
        m = self.mash
        out = None
        if m is not None:
            for key, on in (("groove_drums", self.on["drums"]), ("groove_vocals", self.on["vocals"])):
                g = self._pad(m[key])
                sel = on & self.tempo_ok
                if sel.sum() >= 32 and g[sel].mean() > 0.05:
                    T = g[sel].mean(0)
                    s8 = T[8] / (T[6] + T[8] + 1e-6)
                    s16 = (T[4] + T[10]) / (T[3] + T[9] + T[4] + T[10] + 1e-6)
                    out = dict(s8=float(s8), s16=float(s16), src=key.split("_")[1],
                               act8=float(T[6] + T[8]), act16=float(T[3] + T[9] + T[4] + T[10]))
                    break
        self._bar["groove"] = out
        return out

    def bleed(self):
        """0..1 how much other instruments leak into the vocal stem (0 = clean acapella)."""
        a, b = self.info["first_beat"], self.info["last_beat"] + 1
        rv = self.f["rms_vocals"][a:b]
        rm = self.f["rms_mix"][a:b]
        off = ~self.on["vocals"][a:b] & (rm > np.quantile(rm, 0.5))
        if off.sum() < 16:
            return 0.0
        floor = float(np.median(rv[off])) - float(self.info["stem_peak_db"]["vocals"])
        return float(np.clip((floor + 38) / 20, 0, 1))      # -38 dB under the vocal: clean, -18 dB: heavy

    def phrase_starts(self, step=2):
        """Source beats where a phrase may start, inside sections: the measure starts (step 2) or every bar (step 1)."""
        ms = self.measure_starts() if step == 2 else None
        out = []
        for s in self.sections:
            for bi in range(s["start"], min(s["end"], len(self.bars))):
                if (ms[bi] if ms is not None else (bi - s["start"]) % step == 0):
                    out.append(self.bars[bi])
        return sorted(set(out))


def _best_mult(src_bpm, bpm):
    best = min(MULTS, key=lambda m: abs(math.log(bpm / (src_bpm * m))))
    return best, bpm / (src_bpm * best) - 1.0          # stretch (+ = sped up)


def _pct(h, L, stem="vocals"):
    """Harmonic fit relative to the layer's own song (1.0) vs a random key (0.0)."""
    ref, base = L.harmony_reference(stem)
    return np.clip((np.asarray(h) - base) / max(0.03, ref - base), -0.5, 1.1)


def tonality(L, stem="vocals"):
    """0..1: how much key matters for this layer (sung melody ~1, rap/screams/noise ~0)."""
    ref, base = L.harmony_reference(stem)
    return float(np.clip((ref - base) / 0.08, 0, 1))


def harmony_score(h, L, stem="vocals"):
    """What the user sees: key fit weighted by how pitched the layer is (non-tonal parts rarely clash)."""
    t = tonality(L, stem)
    return float(t * np.clip(_pct(h, L, stem), 0, 1.0) + (1 - t) * 0.85)


def bass_harmfit(L, I, shift_rel):
    """Per bed bar, how well L's bass pitch content locks to I's actual chords at relative shift `shift_rel`
    (design C.1: run the bass through harmfit.shift_scores vs the bed's chords instead of chroma-only).

    harmfit scores all 12 relative transpositions per beat (column k = the backing moved up k semitones, i.e.
    the voice down k); we read the column for `-shift_rel` so a positive layer shift reads the matching key.
    The score is normalized by that beat's best-over-keys value so it is 0..1 (1 = this shift IS the best key
    for that beat, the harmfit neutral-credit reference). Returned per I bar (length len(I.bars)); bars where
    the bass is silent / harmfit has nothing to judge get the neutral 0.85 so the chroma fallback dominates.
    """
    from . import harmfit
    bass_pc = H.normalize_rows(L.chroma["bass"])                 # (nL, 12) per-beat bass chroma
    bass_on = L.on["bass"].astype(float)
    bed_bass, bed_other, bed_ctx = I.backing()                   # per I beat (12 each)
    nI = min(len(bed_bass), I.nb)
    # map each I beat to an L beat (both already on their own grids; align by fractional index ratio)
    if L.nb <= 1 or nI <= 0:
        return np.full(max(len(I.bars), 1), 0.85)
    idx = np.clip(np.round(np.linspace(0, L.nb - 1, nI)).astype(int), 0, L.nb - 1)
    voc = bass_pc[idx]                                           # the bass as the "note" voice (n, 12)
    zeros = np.zeros_like(voc)
    X = harmfit.features([voc, voc], bed_bass[:nI], bed_other[:nI], bed_ctx[:nI])   # (n, 5, 12)
    S = harmfit.get().shift_scores(X)                            # (n, 12)
    k = (-int(shift_rel)) % 12
    best = S.max(1)
    at = S[:, k]
    rel = np.where(best > 1e-6, at / best, 0.85)                 # 1 = this shift is the beat's best key
    w = bass_on[idx]                                             # down-weight silent bass beats
    rel = np.where(w > 0.15, rel, 0.85)
    # fold per-I-beat scores up to per-I-bar (plan_layer indexes `fit` by bed bar)
    out = np.full(len(I.bars), 0.85)
    for bi, a in enumerate(I.bars):
        b = I.bars[bi + 1] if bi + 1 < len(I.bars) else min(nI, a + I.bpb)
        a = min(a, nI)
        b = min(b, nI)
        if b > a:
            out[bi] = float(np.clip(rel[a:b].mean(), 0.0, 1.0))
    return out


def key_scores(L, I, stem="vocals", max_bars=64):
    """Mean compatibility for every relative shift, over (bed harmonic bar, layer bar) pairs."""
    tgt = I.bar_chroma("inst" if stem == "vocals" else "other")
    ion = I.bar_harm_presence() if stem == "vocals" else I.bar_presence("other")
    lay, lon = L.bar_chroma(stem), L.bar_presence(stem)
    ii, vv = np.where(ion > 0.5)[0], np.where(lon > 0.5)[0]
    if len(ii) == 0 or len(vv) == 0:
        return np.zeros(len(SHIFTS))
    rng = np.random.default_rng(0)
    if len(ii) > max_bars:
        ii = np.sort(rng.choice(ii, max_bars, replace=False))
    if len(vv) > max_bars:
        vv = np.sort(rng.choice(vv, max_bars, replace=False))
    A, B = tgt[ii][:, None, :], lay[vv][None, :, :]
    return np.array([compat(A, B, k).mean() for k in SHIFTS])


def ranked_shifts(L, I, stem="vocals"):
    ks = key_scores(L, I, stem)
    order = np.argsort(-(ks - 0.004 * np.abs(SHIFTS)))
    return [(int(SHIFTS[i]), float(ks[i])) for i in order]


# ------------------------------------------------------------------ phrase mapping (beam search)

def _slots(I, c0, c1):
    """I's phrase slots (bar index, n bars, section) in the content region, aligned to its sections,
    skipping bars that aren't at the main tempo."""
    slots = []
    for sec in I.sections:
        a, b = max(sec["start"], c0), min(sec["end"], c1)
        i = a
        while i < b:
            n = min(4, b - i)
            if all(I.bar_range_ok(i + j) for j in range(n)):
                slots.append((i, n, sec))
            i += n
    return slots


def plan_layer(L, I, stem, shift_rel, mL, density=0.5):
    """Choose, for each of I's phrase slots, a phrase of L's `stem` (or silence)."""
    bpb = I.bpb
    c0, c1 = I.info["content_bars"]
    slots = _slots(I, c0, c1)
    if not slots:
        raise ValueError("bed song has no usable bars")
    spb = bpb / mL                                   # L source beats per project bar
    w = max(1, int(round(spb)))
    starts = [vb for vb in L.phrase_starts()
              if vb + 4 * spb <= L.info["last_beat"] + 1 and L.tempo_ok[vb:int(vb + 4 * spb)].all()]
    if not starts:
        raise ValueError(f"{L.title} has no usable phrases at its main tempo")
    st = np.array(starts, float)
    S, K, NONE = len(starts), len(slots), len(starts)
    pitched = stem in PITCHED_ROLES

    Lp = _window_mean(L.on[stem].astype(float), w)[:, 0]
    Le = _window_mean(L.beat_energy, w)[:, 0]
    Lc = H.normalize_rows(_window_mean(L.chroma[stem], w)) if pitched else None
    nL = len(Lp)
    tgt = I.bar_chroma("inst" if stem == "vocals" else "other")
    ihp = I.bar_harm_presence() if stem == "vocals" else I.bar_presence("other")
    ien = I.bar_energy()
    iown = I.bar_presence(stem)                       # where the bed had this stem itself
    tone = tonality(L, stem) if pitched else 0.0
    # bass locks to the bed's ACTUAL chords via harmfit (C.1), with chroma compat as the neutral fallback
    # where harmfit relevance (tonality) is low. Per bed bar, 0..1 (0.85 = neutral).
    bass_hf = bass_harmfit(L, I, shift_rel) if stem == "bass" else None
    # energy relative to each song's own dynamics (loud electronic beds are "high" everywhere otherwise)
    sl_en = np.array([ien[bi:bi + n].mean() for bi, n, _ in slots])
    lo, hi = np.quantile(sl_en, 0.1), np.quantile(sl_en, 0.9)
    ien = np.clip((ien - lo) / max(1e-3, hi - lo), 0, 1)
    vlo, vhi = np.quantile(Le, 0.1), np.quantile(Le, 0.9)
    Le = np.clip((Le - vlo) / max(1e-3, vhi - vlo), 0, 1)

    emit = np.full((K, S + 1), -1e9)
    harm_cell = np.zeros((K, S))
    for k, (bi, n, _) in enumerate(slots):
        idx = np.clip(np.round(st[:, None] + np.arange(n)[None, :] * spb).astype(int), 0, nL - 1)   # (S, n)
        lp, le = Lp[idx], Le[idx]
        ib = [min(bi + j, len(tgt) - 1) for j in range(n)]
        cover = lp.mean(1)
        i_en, i_harm, i_own = ien[ib].mean(), ihp[ib].mean(), iown[ib].mean()
        e_match = 1 - np.abs(i_en - le.mean(1))
        if pitched:
            h = compat(tgt[ib][None, :, :], Lc[idx], shift_rel)                            # (S, n)
            harm = (h * lp).sum(1) / np.maximum(lp.sum(1), 1e-6)
            if bass_hf is not None:
                # harmfit vs the bed's ACTUAL chords where relevant (tone); chroma compat is the neutral
                # fallback where harmfit relevance is low (C.1).
                hf = float(np.clip(bass_hf[ib].mean(), 0.0, 1.0))
                chroma_fit = np.clip(_pct(harm, L, stem), 0, 1.0)
                fit = tone * hf + (1 - tone) * chroma_fit
            else:
                fit = tone * _pct(harm, L, stem) + (1 - tone) * 0.85
            harm_cell[k] = harm
        else:
            fit = np.full(S, 0.85)
        edge = 0.8 if k == 0 or k == K - 1 else 0.0
        if stem == "vocals":
            score = 1.0 * fit * (0.4 + 0.6 * i_harm) + 0.8 * cover + 0.5 * i_en - 0.6 * (1 - e_match)
            none = 1.0 + 1.0 * (1 - i_en) + edge + (0.5 - density) * 1.2
        else:
            # drums / bass: play where the bed had its own drums / bass, stay out where it didn't
            score = (1.0 * fit * (0.4 + 0.6 * i_harm) if pitched else 0.6) + 0.9 * cover + 0.8 * e_match
            none = 0.4 + 2.6 * (1 - i_own)
        score[cover < 0.15] = -1e9
        emit[k, :S] = score
        emit[k, NONE] = none

    # ---- transitions as matrices
    lsecs = {L.bars[s["start"]]: s for s in L.sections if s["start"] < len(L.bars)}
    sec_bounds = sorted((L.bars[s["start"]], L.bars[s["end"]] if s["end"] < len(L.bars) else L.nb)
                        for s in L.sections if s["start"] < len(L.bars))
    sec_end = np.array([next((e for a, e in reversed(sec_bounds) if a <= vb), L.nb) for vb in starts], float)
    is_sec = np.array([vb in lsecs for vb in starts])
    chorus = np.array([vb in lsecs and lsecs[vb].get("repeats", 1) > 1 for vb in starts])
    lname = [lsecs[vb].get("name") if vb in lsecs else None for vb in starts]
    NS = S + 1
    T = [None]
    for k in range(1, K):
        prev_bi, prev_n, _ = slots[k - 1]
        bi, _, sec = slots[k]
        adjacent = prev_bi + prev_n == bi
        i_sec_start = bi == sec["start"]
        inn = sec.get("name")
        M = np.zeros((NS, NS))
        ends = st + prev_n * spb
        pause = Lp[np.clip(np.round(ends).astype(int), 0, nL - 1)] < 0.5
        at_end = (np.abs(sec_end - ends) <= 1) | pause
        M[:S, NONE] = np.where(at_end, 0.1, -0.5 if stem == "vocals" else -0.2)
        enter = np.where(is_sec, 0.35 + (0.35 if i_sec_start else 0.0), -0.35)
        if i_sec_start:
            enter = enter + np.array([0.25 if is_sec[i] and ((lname[i] == "chorus" and inn in ("chorus", "drop"))
                                                              or lname[i] == inn) else 0.0 for i in range(S)])
        M[NONE, :S] = enter
        cont = (np.abs(ends[:, None] - st[None, :]) < 0.5) & adjacent
        back = (st[None, :] <= st[:, None]) & ~chorus[None, :]
        M[:S, :S] = np.where(cont, 0.9, enter[None, :] - 0.3 - 0.4 * back * (1.0 if stem == "vocals" else 0.3))
        T.append((M, cont))

    if stem == "vocals":
        reuse_pen = np.where(chorus, 0.45, 1.0)
    else:
        reuse_pen = np.full(S, 0.12 if stem == "drums" else 0.35)
    beam_w = 40
    beams = [(emit[0, s], [int(s)], np.zeros(S)) for s in np.where(emit[0] > -1e8)[0]]
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

    used = sum(slots[k][1] for k, s in enumerate(path) if s != NONE)
    own_bars = float(sum(iown[bi:bi + n].sum() for bi, n, _ in slots))
    out = dict(slots=slots, path=[None if s == NONE else starts[s] for s in path], score=total / K,
               bars=used, total_bars=sum(n for _, n, _ in slots), own_bars=own_bars, stem=stem)
    if pitched:
        hs = [harm_cell[k, s] for k, s in enumerate(path) if s != NONE]
        h = float(np.mean(hs)) if hs else 0.0
        out.update(harmony=h, harmony_pct=harmony_score(h, L, stem) if hs else 0.0, tonality=tonality(L, stem))
    return out


# ------------------------------------------------------------------ build project

def _layer_clips(L, plan, mL, bpb, t_of_bar, track_id, stem):
    """Merge consecutive continuing slots into clips; vocals also get pickups and tails."""
    spb = bpb / mL
    clips, cur = [], None
    for (bi, n, _), vb in zip(plan["slots"], plan["path"]):
        if vb is None:
            if cur:
                clips.append(cur)
                cur = None
            continue
        if cur and abs(cur["src"] + cur["len"] / mL - vb) < 0.5 and abs(cur["start"] + cur["len"] - t_of_bar(bi)) < 0.01:
            cur["len"] += n * bpb
            continue
        if cur:
            clips.append(cur)
        cur = P.clip(track_id, L.id, stem, t_of_bar(bi), vb, n * bpb, fade_out=0.05)
    if cur:
        clips.append(cur)
    if stem != "vocals":
        return clips
    if L.voicing() is not None:
        return _vocal_edges(L, clips, mL, spb)
    return _vocal_edges_coarse(L, clips, mL, bpb, spb)


def _vocal_edges_coarse(L, clips, mL, bpb, spb):
    """Pickups and tails from per-beat vocal presence (songs without melody features)."""
    von = L.on["vocals"]
    for i, c in enumerate(clips):
        prev_end = clips[i - 1]["start"] + clips[i - 1]["len"] if i else -1e9
        # pickup: vocal already sounding in the beats just before the phrase start
        s = int(round(c["src"]))
        k = 0
        while k < spb and s - k - 1 >= 0 and von[s - k - 1]:
            k += 1
        k_proj = min(k * mL, max(0.0, c["start"] - prev_end - 0.01), float(bpb), max(0.0, c["start"]))
        if k_proj >= 0.5:
            c["start"] -= k_proj
            c["src"] -= k_proj / mL
            c["len"] += k_proj
            c["fade_in"] = min(0.5, k_proj)
        # tail: let a held word ring past the phrase end if nothing follows right away
        nxt = clips[i + 1]["start"] if i + 1 < len(clips) else 1e9
        e = int(round(c["src"] + c["len"] / mL))
        t = 0
        while t < spb and e + t < len(von) and von[e + t]:
            t += 1
        t_proj = min((t + 1) * mL, 2.0, max(0.0, nxt - (c["start"] + c["len"]) - 0.01))
        if t_proj >= 0.5:
            c["len"] += t_proj
            c["fade_out"] = t_proj
    return clips


EDGE_OVERLAP = 0.5      # project beats two vocal clips may overlap (crossfaded) so neither edge chops a word
EDGE_PRE = 0.06         # s of breath kept before a line's first sound
EDGE_RELEASE = 0.2      # s a held word rings on after its voicing ends (fade starts as the voice stops)


def _vocal_edges(L, clips, mL, spb):
    """Pickups and tails from the vocal's own 10 ms voicing (L.voicing, as the cut costs see it).

    Pickup: a phrase start the voice is already running into is opened at the breath where that line begins
    (up to a bar back) - never part-way into the line before it, which a per-beat presence check can't tell
    apart. Tail: a word still sounding at the phrase end rings until it ends (up to 2 beats), and only then
    fades. Pickups go first (a clipped first word is the more audible fault); tails get the room left; where
    there isn't enough, the clips overlap by up to EDGE_OVERLAP beats and the earlier one fades out over it."""
    vm = L.voicing()
    fps = 100.0
    bix = np.arange(L.nb, dtype=float)
    sec = lambda b: float(np.interp(b, bix, L.beats))           # source beat -> seconds
    beat = lambda t: float(np.interp(t, L.beats, bix))          # seconds -> source beat
    fr = lambda t: int(np.clip(round(t * fps), 0, len(vm) - 1))
    planned_end = [c["start"] + c["len"] for c in clips]

    for i, c in enumerate(clips):
        room = min(max(0.0, c["start"] - (planned_end[i - 1] if i else -1e9) - 0.01), max(0.0, c["start"]))
        j, lim = fr(sec(c["src"])), fr(sec(c["src"] - spb))
        if not (j > 0 and vm[j - 1]):
            continue                                            # the phrase starts in a gap: nothing to pick up
        while j > lim and vm[j - 1]:
            j -= 1
        if j <= lim:                                            # mid-line entry (the planner paid for it): ease in
            c["fade_in"] = max(c.get("fade_in", 0.0), 0.5)
            continue
        g = max(lim, j - int(EDGE_PRE * fps))
        while g < j and vm[g]:
            g += 1
        want = (c["src"] - beat(g / fps)) * mL                  # project beats back to just before the line
        fade = max(0.05, (beat(j / fps) - beat(g / fps)) * mL)  # done as the voice begins
        k = min(want, room + (EDGE_OVERLAP if i else 0.0), max(0.0, c["start"]))
        if k < want - 0.01:                                     # no space for the whole line start: ease in, over
            fade = float(np.clip(want - k, 0.05, 0.5))          # about as much as is missing (a lost breath: barely)
        if k > 0.01:
            c["start"] -= k
            c["src"] -= k / mL
            c["len"] += k
        c["fade_in"] = max(c.get("fade_in", 0.0), fade)

    for i, c in enumerate(clips):
        end = c["start"] + c["len"]
        nxt = clips[i + 1]["start"] if i + 1 < len(clips) else 1e9
        room = nxt - end - 0.01
        if room < 0:                                            # the next line's pickup overlaps this clip: crossfade
            c["fade_out"] = max(c.get("fade_out", 0.0), -room)
            continue
        room += EDGE_OVERLAP if i + 1 < len(clips) else 0.0
        e_src = c["src"] + c["len"] / mL
        j, lim = fr(sec(e_src)), fr(sec(e_src + 2.0 / mL))
        if vm[j]:
            while j < lim and vm[j]:
                j += 1
            if j < lim:                                         # the word ends soon: let it, then fade the release
                q = j                                           # ... up to the next sound at most
                while q < min(len(vm), j + int(EDGE_RELEASE * fps)) and not vm[q]:
                    q += 1
                t = (beat(q / fps) - e_src) * mL
                fade = (beat(q / fps) - beat(j / fps - 0.05)) * mL
            else:                                               # the line runs on: fade it out over 2 beats
                t, fade = 2.0, 2.0
        else:                                                   # ends in a gap: the reverb rings out, faded -
            q, stop = j, fr(sec(e_src + min(1.0 * mL, 2.0) / mL))   # but not into the singer's next word
            while q < stop and not vm[q]:
                q += 1
            t = fade = max(0.0, (beat(max(j, q - 3) / fps) - e_src) * mL)
        t = min(t, room)
        if t > 0.01:
            c["len"] += t
            c["fade_out"] = min(t, max(fade, c.get("fade_out", 0.0)))
    return clips


def combo_bpm(I, layers):
    """Weighted geometric mean: the bed counts double."""
    logs, ws = [2 * math.log(I.bpm)], [2.0]
    for L in layers:
        m, _ = _best_mult(L.bpm, I.bpm)
        logs.append(math.log(L.bpm * m))
        ws.append(1.0)
    return round(math.exp(sum(logs) / sum(ws)) * 2) / 2


def bed_transpose(rel):
    """rel: [(relative shift, weight)] of pitched layers -> bed transpose minimising squared movement."""
    if not rel:
        return 0
    return min(range(-4, 5), key=lambda s: sum(w * (r + s) ** 2 for r, w in rel) + s * s)


def _replace_gain(I, L, stem, bed_section_energy=None, layer_section_energy=None):
    """Static replace-gain (unchanged default). When per-section energies are supplied, C.3 rides a per-section
    energy-envelope match on top so a layer's loudness tracks the bed's section dynamics (verse quieter,
    chorus louder) rather than one static gain. Omitting them reproduces the pre-change value exactly."""
    LI, LL = I.info["loudness"], L.info["loudness"]
    if stem == "vocals":
        # halfway between "as loud as the vocal it replaces" and "the vocal's own balance in its song"
        li_inst = LI.get("inst", LI["mix"])
        t_replace = LI["vocals"] if LI.get("vocals", -70) - LI["mix"] > -13 else li_inst - 4.0
        t_balance = li_inst + (LL["vocals"] - LL.get("inst", LL["mix"]))
        target = (t_replace + t_balance) / 2
    else:
        own = LI.get(stem, -70)
        target = own if own - LI["mix"] > -24 else LI["mix"] + (LL.get(stem, -30) - LL["mix"])
    g = float(np.clip(target - LL.get(stem, -30), -12, 12))
    if bed_section_energy is not None and layer_section_energy is not None:
        from . import blend
        return blend.section_gain(bed_section_energy, layer_section_energy, g)
    return g


def build_combo(I, layers, density=0.5, plans=None, bed_shift=None):
    """I: bed SongData. layers: {role: None (bed's own) | "none" | (SongData, relative shift)} for
    role in vocals/drums/bass. `plans` may hold precomputed plans keyed by role."""
    plans = dict(plans or {})
    bpb = I.bpb
    foreign = {r: v for r, v in layers.items() if isinstance(v, tuple)}
    bpm = combo_bpm(I, [L for L, _ in foreign.values()])
    mI, strI = _best_mult(I.bpm, bpm)
    rel = [(s, 1.2 if r == "vocals" else 1.0) for r, (L, s) in foreign.items() if r in PITCHED_ROLES]
    sI = bed_transpose(rel) if bed_shift is None else int(bed_shift)

    names = [L.title for L, _ in foreign.values()]
    proj = P.empty(" × ".join([I.title] + [n for n in dict.fromkeys(names) if n != I.title]), bpm, bpb)
    proj["songs"] = {I.id: dict(transpose=sI, mult=mI)}
    roles_desc = [dict(role="music", song=I.id, title=I.title, shift=sI, stretch=round(strI * 100, 1), mult=mI)]
    tr = {t["role"]: t for t in proj["tracks"]}
    tr["other"]["name"] = f"Other · {I.title[:28]}"

    # layer plans (need mult at the final tempo)
    for r, (L, s) in foreign.items():
        mL, strL = _best_mult(L.bpm, bpm)
        if r not in plans or plans[r].get("mult") != mL:
            plans[r] = plan_layer(L, I, r, s, mL, density)
            plans[r]["mult"] = mL
        shift = (s + sI) if r in PITCHED_ROLES else 0
        prev = proj["songs"].get(L.id)
        if prev and r in PITCHED_ROLES and prev.get("pitched_by") and prev["transpose"] != shift:
            raise ValueError("same song would need two different transposes")
        if not prev or r in PITCHED_ROLES:
            proj["songs"][L.id] = dict(transpose=shift if r in PITCHED_ROLES else (prev or {}).get("transpose", 0),
                                       mult=mL, pitched_by=r if r in PITCHED_ROLES else (prev or {}).get("pitched_by"))
        d = dict(role=r, song=L.id, title=L.title, shift=shift, stretch=round(strL * 100, 1), mult=mL,
                 bars=plans[r]["bars"])
        if r in PITCHED_ROLES:
            d.update(harmony_pct=int(round(plans[r]["harmony_pct"] * 100)), tonality=round(plans[r]["tonality"], 2))
        roles_desc.append(d)
    for r in ("vocals", "drums", "bass"):
        v = layers.get(r)
        if v is None and (r != "vocals" or I.on[r].mean() > 0.02):
            tr[r]["name"] = f"{r.capitalize()} · {I.title[:28]}"
            if r == "vocals":
                roles_desc.append(dict(role="vocals", song=I.id, title=I.title + " (own)", shift=sI, own=True))
        elif isinstance(v, tuple):
            tr[r]["name"] = f"{r.capitalize()} · {v[0].title[:28]}"
    vocal_on_top = isinstance(layers.get("vocals"), tuple) or layers.get("vocals") is None and I.has_real_vocals
    if vocal_on_top:
        tr["other"].update(pocket=2.5, duck=2.0)
        tr["bass"].update(duck=0.8)
        tr["vocals"].update(hpf=110.0)

    base_plan = next(iter(plans.values()), None)
    slots = base_plan["slots"] if base_plan else _slots(I, *I.info["content_bars"])
    if not slots:
        raise ValueError("bed song has no usable bars")
    # bed stems: one clip per stem per contiguous run of usable slots, packed back to back
    runs, cur = [], None
    for bi, n, _ in slots:
        if cur and cur[1] == bi:
            cur[1] = bi + n
        else:
            cur = [bi, bi + n]
            runs.append(cur)
    bed_stems = ["other"] + [r for r in ("drums", "bass", "vocals") if layers.get(r) is None]
    t_cursor, bar_time = 0.0, {}
    for a, b in runs:
        a_beat = I.bars[a]
        b_beat = I.bars[b] if b < len(I.bars) else I.nb
        for bi in range(a, b):
            bar_time[bi] = t_cursor + (I.bars[bi] - a_beat) * mI
        for r in bed_stems:
            if I.on[r].mean() > 0.02:
                proj["clips"].append(P.clip(tr[r]["id"], I.id, r, t_cursor, a_beat, (b_beat - a_beat) * mI,
                                            fade_out=0.05))
        t_cursor += (b_beat - a_beat) * mI
    for r, (L, s) in foreign.items():
        cl = _layer_clips(L, plans[r], plans[r]["mult"], bpb, lambda bi: bar_time[bi], tr[r]["id"], r)
        g = round(_replace_gain(I, L, r), 1)
        for c in cl:
            c["gain_db"] = g
        proj["clips"].extend(cl)
    for s in proj["songs"].values():
        s.pop("pitched_by", None)
    proj["master"]["gain_db"] = -1.5

    total_bars = sum(n for _, n, _ in slots)
    desc = dict(kind="combo", bed=I.id, bpm=bpm, roles=roles_desc, total_bars=total_bars,
                length_sec=round(t_cursor * 60 / bpm, 1))
    V = foreign.get("vocals")
    if V:   # fields the quick view shows
        pv = plans["vocals"]
        desc.update(vocal=V[0].id, inst=I.id, vocal_title=V[0].title, inst_title=I.title,
                    shift_vocal=V[1] + sI, shift_inst=sI, stretch_vocal=roles_desc_of(roles_desc, "vocals")["stretch"],
                    stretch_inst=round(strI * 100, 1), mult_vocal=pv["mult"], mult_inst=mI,
                    harmony_pct=int(round(pv["harmony_pct"] * 100)), tonality=round(pv["tonality"], 2),
                    vocal_bars=pv["bars"])
    _attach_transitions_and_blend(proj, desc, I, foreign, plans)
    desc["rating"] = _rating(desc, plans, I)
    proj["mashup"] = desc
    return proj, desc


def _bed_info(I):
    """The bed song's info dict in the shape transitions.build_plan / blend consume (bars/beats/bpb/
    beat_content/first_beat/last_beat/sections). Falls back gracefully when optional fields are absent."""
    info = dict(I.info)
    info.setdefault("bars", I.bars)
    info.setdefault("beats", list(I.beats))
    info.setdefault("bpb", I.bpb)
    info.setdefault("first_beat", int(I.info.get("first_beat", 0)))
    info.setdefault("last_beat", int(I.info.get("last_beat", I.nb - 1)))
    info.setdefault("sections", I.sections)
    return info


def _attach_transitions_and_blend(proj, desc, I, foreign, plans):
    """D.1 step 9: attach the TransitionPlan automation + blend summary + the `lyrics` desc line to a built
    combo. Purely additive — it writes to `proj["mashup"]`'s desc and leaves clips/tracks untouched, so a
    Phase-2-free project (no foreign stems, no lyrics) exports byte-identical. The TransitionPlan is a
    deterministic object derived from the bed's structure + the chosen vocal plan; the UI/renderer consume it
    as editable automation."""
    from . import transitions
    bed_info = _bed_info(I)

    # chosen path events: section boundaries from the bed's own structure + vocal entry/exit from the plan.
    path = []
    sections = bed_info.get("sections") or []
    bars = bed_info.get("bars") or []
    for k in range(len(sections) - 1):
        s, nxt = sections[k], sections[k + 1]
        try:
            boundary_bar = int(np.searchsorted(bars, nxt.get("start", 0)))
        except Exception:
            boundary_bar = 0
        path.append(dict(kind="section", beat=float(nxt.get("start", 0)),
                         energy=float(s.get("energy", 0.0)), next_energy=float(nxt.get("energy", 0.0)),
                         boundary_bar=boundary_bar))
    vp = plans.get("vocals")
    if vp and vp.get("slots") and vp.get("path"):
        prev = None
        for (bi, n, _), vb in zip(vp["slots"], vp["path"]):
            cur = vb is not None
            if cur != prev and prev is not None:
                path.append(dict(kind="vocal_entry" if cur else "vocal_exit",
                                 beat=float(I.bars[bi]) if bi < len(I.bars) else float(bi)))
            prev = cur

    try:
        plan = transitions.build_plan(path, bed_info, stem_plans=plans)
    except Exception:
        plan = {"version": transitions.RENDERER_VERSION, "strong_q": 0.0, "events": []}
    desc["transitions"] = plan
    desc["blend"] = {"congested_sections": [], "mode": "frequency-slot"}

    # the `lyrics` desc line (D.1 step 9): surfaced from the vocal song's lazy concept/lyrics, honestly
    # degrading to "not transcribed" when absent.
    V = foreign.get("vocals")
    desc["lyrics"] = _lyrics_desc_line(V[0] if V else None, I)


def _lyrics_desc_line(V, I):
    """Plain-language lyrics summary for the combo desc. V = vocal SongData (or None for a bed-own vocal)."""
    song = V if V is not None else I
    ly = getattr(song, "lyrics", None)
    co = getattr(song, "concept", None)
    if ly is None:
        return "Lyrics: not transcribed."
    q = (ly or {}).get("quality", {}) or {}
    conf = q.get("asr_conf", 0.0)
    mood = ((co or {}).get("tags") or {}).get("mood")
    if mood:
        return f"Lyrics: {mood} (ASR confidence {conf:.2f})."
    return f"Lyrics: transcribed (ASR confidence {conf:.2f})."


def roles_desc_of(rd, role):
    return next(d for d in rd if d["role"] == role)


def _rating(d, plans, I):
    """0-100 overall rating shown to the user."""
    harm, wsum = 0.0, 0.0
    for r, w in (("vocals", 0.7), ("bass", 0.3)):
        if r in plans and "harmony_pct" in plans[r]:
            harm += w * plans[r]["harmony_pct"]
            wsum += w
    h = harm / wsum if wsum else 0.85
    stretches = [abs(x["stretch"]) for x in d["roles"] if "stretch" in x]
    tempo = float(np.mean([max(0.0, 1 - s / 18) for s in stretches])) if stretches else 1.0
    shift = max(0.0, 1 - sum(abs(x.get("shift", 0)) for x in d["roles"] if not x.get("own")) * 0.06)
    covers = []
    if "vocals" in plans:
        covers.append(min(1.0, plans["vocals"]["bars"] / max(1, 0.45 * d["total_bars"])))
    for r in ("drums", "bass"):
        if r in plans and plans[r]["own_bars"] > 0:
            covers.append(min(1.0, plans[r]["bars"] / max(1.0, 0.8 * plans[r]["own_bars"])))
    cover = float(np.mean(covers)) if covers else 1.0
    return int(round(100 * (0.45 * h + 0.22 * tempo + 0.13 * shift + 0.2 * cover)))


def build_pair(vid, iid, shift_rel=None, density=0.5):
    """Project for a rated vocal-over-bed pairing (planned exactly as it was rated)."""
    from . import pairs
    V, I = get_song(vid), get_song(iid)
    res = pairs.evaluate(V, I, density, shifts=None if shift_rel is None else [int(shift_rel)], keep_plan=True)
    if not res.get("workable"):
        raise ValueError(res.get("reason", "not workable"))
    plan = dict(res["_plan"])
    plan.update(mult=res["mult_vocal"], harmony_pct=res["harmony"] / 100, tonality=V.stability)
    proj, desc = build_combo(I, {"vocals": (V, res["shift_rel"]), "drums": None, "bass": None}, density,
                             plans={"vocals": plan}, bed_shift=res["shift_bed"])
    comp = {k: res[k] for k in ("harmony", "tempo", "key", "structure", "groove", "acapella")}
    desc.update(rating=int(round(res["rating"])), components=comp, harmony_pct=res["harmony"],
                alternatives=res.get("alternatives", []), shift_rel=res["shift_rel"])
    for d in desc["roles"]:
        if d["role"] == "vocals" and not d.get("own"):
            d["harmony_pct"] = res["harmony"]
    proj["mashup"] = desc
    return proj, desc


# ------------------------------------------------------------------ search

OWN, NONE_ = "own", "none"


def generate_roles(roles, max_variants=500, density=0.5, progress=lambda f, m="": None, lyrics_mode="off"):
    """Role mode. roles = {"music": [song ids], "vocals": [ids | "own" | "none"], "drums": [ids | "own"],
    "bass": [ids | "own"]}. Every bed x candidate layer is planned once; EVERY combination is assembled and
    rated (best first). Vocals use the learned planner + rating (pairs.py); drums / bass layers are planned
    by energy / presence (bass also by chroma fit against the bed's music)."""
    from . import dupes, pairs
    music = [s for s in roles.get("music", []) if s not in (OWN, NONE_)]
    if not music:
        raise ValueError("pick at least one song for the music (bed)")
    cand = {r: list(dict.fromkeys(roles.get(r) or [OWN if r != "vocals" else NONE_])) for r in ("vocals", "drums", "bass")}
    ids = set(music) | {s for r in cand.values() for s in r if s not in (OWN, NONE_)}
    songs = {s: get_song(s) for s in ids}

    # plan every usable (bed, role, candidate)
    work = []
    for iid in music:
        I = songs[iid]
        for r, lst in cand.items():
            for lid in lst:
                if lid in (OWN, NONE_) or lid == iid or dupes.same_song(lid, iid):
                    continue
                L = songs[lid]
                if L.bpb != I.bpb or not L.has_stem(r):
                    continue
                if abs(_best_mult(L.bpm, I.bpm)[1]) > STRETCH_LIMIT[r]:
                    continue
                if r == "vocals":
                    work.append((I, r, L, None))
                else:
                    keys = [s for s, _ in ranked_shifts(L, I, r)[:2]] if r in PITCHED_ROLES else [0]
                    work += [(I, r, L, k) for k in keys]
    options = {}           # (bed id, role) -> [(factor 0..1, (L, shift), plan, extra)]
    for n, (I, r, L, k) in enumerate(work):
        progress(0.75 * n / max(1, len(work)), f"{r}: {L.title[:22]} on {I.title[:22]}")
        try:
            if r == "vocals":
                res = pairs.evaluate(L, I, density, keep_plan=True, lyrics_mode=lyrics_mode)
                if not res.get("workable"):
                    continue
                pl = dict(res["_plan"])
                pl.update(mult=res["mult_vocal"], harmony_pct=res["harmony"] / 100, tonality=L.stability)
                comp = {c: res[c] for c in ("harmony", "tempo", "key", "structure", "groove", "acapella")}
                options.setdefault((I.id, r), []).append((res["rating"] / 100, (L, res["shift_rel"]), pl, comp))
                continue
            mL, st = _best_mult(L.bpm, I.bpm)
            pl = plan_layer(L, I, r, k, mL, density)
            pl["mult"] = mL
        except ValueError:
            continue
        cover = min(1.0, pl["bars"] / max(1.0, 0.8 * pl["own_bars"])) if pl["own_bars"] > 0 else 0.5
        f = (0.55 + 0.45 * cover) / (1 + (abs(st) / (0.2 if r == "drums" else 0.15)) ** 4)   # drums stretch cleanly
        if r in PITCHED_ROLES:
            f *= (0.35 + 0.65 * pl["harmony_pct"]) * pairs.key_quality(k, 0)
        options.setdefault((I.id, r), []).append((f, (L, k), pl, None))

    combos = []
    for iid in music:
        I = songs[iid]
        per_role = {}
        for r, lst in cand.items():
            opts = sorted(options.get((iid, r), []), key=lambda o: -o[0])
            picked, seen = [], set()
            for o in opts:                        # best key per candidate song
                if o[1][0].id in seen:
                    continue
                seen.add(o[1][0].id)
                picked.append(o)
            choices = [(None, None, 1.0, None)] if OWN in lst or iid in lst else []
            if NONE_ in lst:
                choices.append(("none", None, 1.0, None))
            choices += [(o[1], o[2], o[0], o[3]) for o in picked]
            if not choices:
                choices = [(None, None, 1.0, None)] if r != "vocals" else [("none", None, 1.0, None)]
            per_role[r] = choices
        for combo in itertools.product(*(per_role[r] for r in ("vocals", "drums", "bass"))):
            layers, plans, f, comp = {}, {}, 1.0, None
            for r, (v, pl, fac, cp) in zip(("vocals", "drums", "bass"), combo):
                layers[r] = v
                if pl is not None:
                    plans[r] = pl
                    f *= fac
                if r == "vocals" and cp:
                    comp = cp
            if not plans:
                continue           # nothing replaced: that's just the original song
            combos.append((I, layers, plans, f, comp))
    if not combos:
        raise ValueError("no workable combination - check tempos (within ~20-25%, half/double time counts), "
                         "matching meters, and that vocal candidates actually have vocals (and aren't the same "
                         "song as the music)")
    combos.sort(key=lambda c: -c[3])
    combos = combos[:max_variants]
    variants = []
    for n, (I, layers, plans, f, comp) in enumerate(combos):
        progress(0.75 + 0.25 * n / len(combos), f"assembling combinations ({n + 1}/{len(combos)})")
        try:
            proj, desc = build_combo(I, layers, density, plans)
        except ValueError:
            continue
        desc["rating"] = int(round(100 * f))
        if comp:
            desc["components"] = comp
        proj["mashup"] = desc
        variants.append(dict(desc=desc, project=proj))
    progress(1.0, "done")
    variants.sort(key=lambda v: -v["desc"]["rating"])
    return variants


# ------------------------------------------------------------------ match finder

_song_cache = {}


def get_song(sid):
    """Cached SongData (reloaded when the song is re-analyzed)."""
    p = library.song_dir(sid) / "song.json"
    mt = p.stat().st_mtime if p.exists() else 0
    hit = _song_cache.get(sid)
    if hit and hit[0] == mt:
        return hit[1]
    sd = SongData(sid)
    _song_cache[sid] = (mt, sd)
    return sd


_NOTE = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]


def _key_relation(a, b):
    """Plain-words relationship between two detected keys."""
    ka, kb = a.info["key"], b.info["key"]
    ta = ka["tonic"] if ka["mode"] == "major" else (ka["tonic"] + 3) % 12     # compare via relative majors
    tb = kb["tonic"] if kb["mode"] == "major" else (kb["tonic"] + 3) % 12
    d = (tb - ta) % 12
    if d == 0:
        return "same key" if ka["mode"] == kb["mode"] else "relative major/minor (same notes)"
    if d in (5, 7):
        return "neighbouring key (one step on the circle of fifths)"
    return f"different key ({min(d, 12 - d)} semitone{'s' if min(d, 12 - d) > 1 else ''} apart)"


def _shift_words(s, whose="theirs"):
    if s == 0:
        return "fits as-is"
    return f"fits after moving {whose} {abs(s)} semitone{'s' if abs(s) > 1 else ''} {'up' if s > 0 else 'down'}"


_pair_cache = {}


def _pair_keys(L, I, stem):
    """key_scores with a lighter sample, cached per pair (invalidated when either song changes)."""
    k = (L.id, I.id, stem, _song_cache.get(L.id, (0,))[0], _song_cache.get(I.id, (0,))[0])
    if k not in _pair_cache:
        _pair_cache[k] = key_scores(L, I, stem, max_bars=24)
        if len(_pair_cache) > 20000:
            _pair_cache.pop(next(iter(_pair_cache)))
    return _pair_cache[k]


def match_list(ref_id, limit=400, density=0.5):
    """Rank every ready song against `ref_id`, with a plain-language breakdown per aspect.
    Uses the full mashup ratings (pairs.py) where they exist - both directions - and a quick estimate
    otherwise (flagged, so the UI can offer to rate those properly)."""
    from . import dupes, pairs
    R = get_song(ref_id)
    out = []
    ids = [s["id"] for s in library.list_songs() if s.get("status") == "ready"]
    rated = {k: v for k, v in pairs.load_all(ids, density).items() if ref_id in k}
    for s in library.list_songs():
        if s["id"] == ref_id or s.get("status") != "ready":
            continue
        try:
            C = get_song(s["id"])
        except Exception:
            continue
        if dupes.same_song(ref_id, C.id):
            out.append(dict(id=C.id, title=C.title, score=0, same_song=True, meter_ok=True, rated=True,
                            tempo=dict(score=0, their_bpm=round(C.bpm, 1), meet_bpm=0, stretch=0, mult=1,
                                       text="same song (duplicate / version) - not a mashup"),
                            key=dict(their=C.info["key"]["name"], camelot=C.info["key"]["camelot"], relation="same song"),
                            uses=[], has_vocals=C.has_real_vocals))
            continue
        # tempo: meet in the middle (ref counts double), half/double time allowed
        bpm = combo_bpm(R, [C])
        mC, strC = _best_mult(C.bpm, bpm)
        _, strR = _best_mult(R.bpm, bpm)
        worst = max(abs(strC), abs(strR))
        tempo_score = max(0.0, 1 - worst / 0.15)
        tempo_txt = ("same tempo" if worst < 0.005 else
                     f"{abs(strC) * 100:.1f}% {'faster' if strC > 0 else 'slower'} for them"
                     + (f", {abs(strR) * 100:.1f}% for you" if abs(strR) > 0.005 else ""))
        if mC != 1:
            tempo_txt += f" (their {C.bpm:.0f} BPM is {'double' if mC == 0.5 else 'half'} yours - the beats still line up)"
        # harmony: try every way the two could be combined, keep the best
        uses = []
        def add(kind, L, I, stem, weight=1.0):
            try:
                ks = _pair_keys(L, I, stem)
                i = int(np.argmax(ks - 0.004 * np.abs(SHIFTS)))
                sh, h = int(SHIFTS[i]), float(ks[i])
                if stem == "other":
                    # music on music: how far the best key stands out from a random key for this pair
                    score = float(np.clip((h - ks.mean()) / 0.1, 0, 1))
                else:
                    score = harmony_score(h, L, stem)
                uses.append(dict(kind=kind, shift=sh if L is C else -sh, harmony=round(score * 100),
                                 fit=weight * score - 0.05 * abs(sh),
                                 text=_shift_words(sh, "theirs" if L is C else "yours")))
            except Exception:
                pass
        if C.has_real_vocals:
            add("their vocals over your music", C, R, "vocals")
        if R.has_real_vocals:
            add("your vocals over their music", R, C, "vocals")
        add("layer the music together", C, R, "other", 0.85)
        if C.has_stem("bass"):
            add("their bass under your music", C, R, "bass")
        uses.sort(key=lambda u: -u["fit"])
        best = uses[0] if uses else None
        harm = best["fit"] if best else 0.0
        meter_ok = C.bpb == R.bpb
        score = (0.45 * harm + 0.4 * tempo_score + 0.15 * (1 - min(1, abs(best["shift"]) / 6) if best else 0))
        if not meter_ok:
            score *= 0.5
        score *= 0.8            # a quick estimate is optimistic next to a full rating
        # full ratings (vocal of one over the music of the other), when available
        full = []
        n_expected = int(C.has_real_vocals) + int(R.has_real_vocals)
        for (v, i), d in rated.items():
            if C.id not in (v, i):
                continue
            kind = "their vocals over your music" if v == C.id else "your vocals over their music"
            if d.get("workable"):
                full.append(dict(kind=kind, rating=d["rating"], harmony=d["harmony"], shift=d["shift_vocal"],
                                 text=f"rated {round(d['rating'])}/100 - tempo {d['tempo']}%, key {d['key']}%, "
                                      f"structure {d['structure']}%", fit=d["rating"] / 100))
            else:
                full.append(dict(kind=kind, rating=0, harmony=0, shift=0, text=d.get("reason", "not workable"), fit=0))
        is_rated = len(full) >= n_expected and n_expected > 0
        if full:
            full.sort(key=lambda u: -u["fit"])
            score = full[0]["fit"]
            uses = full + [u for u in uses if u["kind"] not in {f["kind"] for f in full}]
        out.append(dict(
            id=C.id, title=C.title, score=int(round(100 * score)), meter_ok=meter_ok, rated=is_rated,
            tempo=dict(score=int(round(100 * tempo_score)), their_bpm=round(C.bpm, 1), meet_bpm=bpm,
                       stretch=round(strC * 100, 1), mult=mC, text=tempo_txt),
            key=dict(their=C.info["key"]["name"], camelot=C.info["key"]["camelot"],
                     relation=_key_relation(R, C)),
            uses=[{k: v for k, v in u.items() if k != "fit"} for u in uses[:3]],
            has_vocals=C.has_real_vocals))
    out.sort(key=lambda m: (m.get("same_song", False), -m["score"]))
    unrated = sum(1 for m in out if not m.get("rated") and (m["has_vocals"] or R.has_real_vocals))
    return dict(ref=dict(id=R.id, title=R.title, bpm=round(R.bpm, 1), key=R.info["key"]["name"],
                         camelot=R.info["key"]["camelot"], has_vocals=R.has_real_vocals), matches=out[:limit],
                unrated=unrated)
