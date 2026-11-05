"""DJ brain: a language model directs a two-song mashup part by part, stem by stem; the engine plays it.

Each part of the script names where every stem comes from - drums, bass, instruments, a vocal and an optional
second vocal - each from a section of either song (or none), for a number of 2-bar measures, plus how the part
comes in and why. So a part can be A's drums + B's bass + A's instruments under B's vocal, with A's vocal
answering it.

The model can't hear harmony, so the engine works out what fits before it plans, in this mix's key: which vocal
sections sit well on which of the other song's instrument sections (the learned harmfit model, scored like the
pair ratings), which basslines work under the other song's chords, and which pairs of vocal sections take
turns instead of singing over each other. Those lists go into the brief as a menu; the arrangement - the arc,
the build, where the hooks land - is the model's job. The reply is schema-constrained, so it only ever names
sections that exist.

Then the engine plays the script and holds the musical rules: every stem starts on a measure, one tempo and one
key per song for the whole mix, vocals open in breaths (automash._vocal_edges), a bass that would clash with the
chords it's put under is swapped for the instruments' own, a vocal on chords it clashes with is left out, a second
vocal that would sing over the first is tucked under it or dropped, a stem with nothing in it is skipped - each
repair reported in the result, never silently trusted.
"""
import math

import numpy as np

from . import automash, harmfit, llm as llm_mod, pairs, planner
from . import project as P
from .automash import _best_mult, _replace_gain, _vocal_edges, compat

TRANSITIONS = ["cut", "crossfade", "filter_sweep", "echo_out", "riser", "drum_fill"]
MEASURE_CHOICES = [2, 4, 8, 12, 16]    # phrases: 1- and 6-measure parts made mixes choppy (user, Oct 17)
MAX_PARTS = 24
MIN_SECTION = 2            # measures: a shorter detector section is folded into the one before
MAX_SECTION = 8            # measures: a longer one is split into cue-able chunks
MAX_VOCAL_USES = 2         # times one vocal section may be used (the script is told; the engine enforces)
# what the engine lets through (harmony on the pair-rating scale: 0 = random song in a random key, 1 = as good
# as the song with itself)
VOCAL_MIN = 0.70           # a vocal on chords it fits worse than this is left out (was 0.30: "clashes", user Oct 17;
VOCAL_GOOD = VOCAL_MIN     # at 0.70 / 0.80 a pair still has 30-90 combinations to choose from) - and listed from here
BASS_MIN = 0.80            # a bass under the other song's chords below this is swapped for the chords' own bass
BASS_GOOD = BASS_MIN
DUET_TRADE = 0.40          # two vocals that sing at once more of the time than this trade lines instead
DUET_GOOD = DUET_TRADE       # ... at or under it they are listed as able to sing together

# script key -> (track role, stem)
ROLES = (("drums", "drums", "drums"), ("bass", "bass", "bass"), ("instruments", "other", "other"),
         ("vocal", "vocals", "vocals"), ("vocal2", "vocals2", "vocals"))

SYSTEM = """You are an experienced club DJ and mashup producer planning a mashup of two songs, A and B.
You write it as an ordered list of parts. For each part you pick where every stem comes from: DRUMS, BASS,
INSTRUMENTS (keys, guitars, synths), a VOCAL and an optional second vocal (VOCAL2) - each one a section id of
either song, or "none". Each stem plays from the start of its section for the part's number of measures
(1 measure = 2 bars); it may run on into the next section. All stems from the same section of the same song is
that song as it is. Mixing them is the mashup: e.g. A's drums + B's bass + A's instruments under B's vocal.

The engine has checked what fits in this mix's key and lists it under WHAT FITS TOGETHER. Vocals on the other
song's instruments, and a bass under the other song's instruments, should come from that list - other pairings
may clash and the engine will undo them. Drums from either song work under anything. VOCAL2 brings in the
other singer: the engine makes the two trade lines (call and response, switching every 1-2 measures at the
singers' breaths), so pick a VOCAL2 that also fits the part's instruments. Pairs listed under "Both vocals at
once" can even sing together.

How a good DJ blend is built - follow this closely, the listener is an audio engineer:
- Phrases, not fragments. Parts are 4, 8, 12 or 16 measures (8 to 32 bars); most are 8. A 2-measure part is only
  for a short build-up or break right before a big moment, at most once or twice in a mix. Use about the number of
  parts given under MIX - fewer, longer parts beat many short ones. A combination should play long enough to be
  heard as a groove before anything changes.
- Blend in layers. Between one part and the next, change one or two sources (e.g. bring B's drums in under A, then
  swap the bass, then hand over the vocal) - not all five at once, except for one deliberate drop or switch.
- Vocals on whole lines. The "sung lines start at" column says where each line begins inside its section. A vocal
  part should start its section where a line starts and last long enough for its lines to finish; a vocal part
  with no line starting in it is silence. Give most of the vocal time to the stronger direction under FIT, and use
  each vocal section at most twice.
- Only combinations the engine checked: vocals and basses on the other song's music from WHAT FITS TOGETHER
  (anything else clashes and is removed). Two vocals at once only where listed; otherwise VOCAL2 trades lines.
- An arc: open recognisably (one song, 8 measures), build, put the strongest HOOK combination at the peak about two
  thirds in, and end within 8 measures of the last vocal - no long instrumental tail.

Transitions say how a part comes in: cut (switch on the downbeat - best into a drop), crossfade (blend over a bar),
filter_sweep (the new music sweeps in through a filter), echo_out (the previous vocal echoes away), riser (a
build-up into a big part), drum_fill (a fill leads in). The first part's transition is ignored.

For each part write "why" first: at most 12 words on what it does for the mix."""


# ------------------------------------------------------------------ song sections

def _measures(S):
    """Bars that start S's measures inside its music (SongData.measure_starts)."""
    c0, c1 = S.info["content_bars"]
    ms = S.measure_starts()
    return [b for b in range(c0, min(c1, len(S.bars))) if ms[b]]


def _snap(M, bar):
    """Index of the measure start nearest `bar` (ties go later: a section the detector started one bar early)."""
    return int(np.argmin([abs(m - bar) - 1e-3 * (m >= bar) for m in M]))


def _level(x, cuts, words):
    return next((w for c, w in zip(cuts, words) if x < c), words[-1])


def sections(S, tag):
    """S's sections re-cut on its measures: [{id, name, m0, m1, energy, stems{...}, hook, lyric}]."""
    M = _measures(S)
    if not M:
        return [], M
    hooks = S.hook_starts()
    ly = S.lyrics
    q = (ly or {}).get("quality") or {}
    segs = (ly or {}).get("segments") or [] if q.get("usable") and q.get("asr_conf", 0) >= 0.3 else []
    raw = [s for s in S.sections if s.get("name") != "silence" and s["start"] < S.info["content_bars"][1]]
    # on measures, with the detector's runs of same-named sections merged (it cuts some songs into 2-bar
    # slivers), slivers folded into the part before, and long stretches split into chunks a DJ would cue
    cuts = []
    for s in raw:
        m0 = _snap(M, s["start"])
        if cuts and (cuts[-1][1] == s.get("name") or m0 - cuts[-1][0] < MIN_SECTION):
            continue
        if not cuts or m0 > cuts[-1][0]:
            cuts.append((m0, s.get("name", "part"), s))
    starts = []
    for k, (m0, _, _) in enumerate(cuts):
        m1 = cuts[k + 1][0] if k + 1 < len(cuts) else len(M)
        n = max(1, round((m1 - m0) / MAX_SECTION + 0.49))
        starts += [m0 + round(j * (m1 - m0) / n) for j in range(n)]
    out = []
    for k, m0 in enumerate(starts):
        m1 = starts[k + 1] if k + 1 < len(starts) else len(M)
        if m1 <= m0:
            continue
        b0, b1 = M[m0], (M[m1] if m1 < len(M) else S.info["content_bars"][1])
        src = max((c for c in cuts if c[0] <= m0), key=lambda c: c[0])[2]   # the section this chunk belongs to
        bars = range(b0, min(b1, len(S.bars)))
        stems = {st: float(np.mean([S.bar_presence(st)[b] for b in bars])) for st in ("drums", "bass", "other")}
        stems["vocals"] = float(np.mean([S.bar_presence("vocals")[b] for b in bars])) if S.has_real_vocals else 0.0
        t0, t1 = S.beats[S.bars[b0]], S.beats[min(S.bars[b1] if b1 < len(S.bars) else S.nb - 1, S.nb - 1)]
        words = [w["w"].strip() for sg in segs for w in sg.get("words", []) if t0 <= w["t"] < t1]
        out.append(dict(id=f"{tag}{len(out) + 1}", tag=tag, name=src.get("name", "part"), m0=m0, m1=m1,
                        energy=float(src.get("energy", 0.5)), stems=stems,
                        hook=any(S.bars[b] in hooks for b in bars), lyric=" ".join(words[:12])))
    return out, M


# ------------------------------------------------------------------ what fits together (engine-side)

class Fits:
    """Harmony of stems from one song against the other's, aligned measure on measure, in the mix's key."""

    def __init__(self, songs, Ms, tr, mult):
        self.songs, self.Ms, self.tr, self.mult = songs, Ms, tr, mult
        self.meta = harmfit.get().meta
        self.grid, self.sums = {}, {}
        for v, i in (("A", "B"), ("B", "A")):           # vocal of v on instruments of i
            V, I = songs[v], songs[i]
            g = None
            if V.has_real_vocals:
                try:
                    g = planner.Grid(V, I, mult[i] / mult[v])
                except ValueError:
                    g = None
            self.grid[v + i] = g
            if g is not None:
                num, den = g.info_sums(tr[v] - tr[i])
                self.sums[v + i] = (num, den, g.vocal,
                                    {bi: k for k, (bi, n, _) in enumerate(g.slots) if n == 2},
                                    {st: s for s, st in enumerate(g.starts)})
        self.vf = {t: self._voiced_per_beat(songs[t]) for t in "AB"}

    @staticmethod
    def _voiced_per_beat(S):
        vm = S.voicing()
        if vm is None:
            return S.on["vocals"].astype(float)
        fr = np.clip(np.round(np.asarray(S.beats) * 100).astype(int), 0, len(vm))
        cs = np.r_[0, np.cumsum(vm)]
        out = np.zeros(S.nb)
        for j in range(S.nb - 1):
            out[j] = (cs[fr[j + 1]] - cs[fr[j]]) / max(1, fr[j + 1] - fr[j])
        return out

    def vocal(self, vsec, isec, n):
        """Harmony (0..~1) of vocal section vsec on the other song's instruments isec over n measures, or None."""
        d = vsec["tag"] + isec["tag"]
        if d not in self.sums:
            return None
        num, den, voc, slot_of, start_of = self.sums[d]
        V, Mv, Mi = self.songs[vsec["tag"]], self.Ms[vsec["tag"]], self.Ms[isec["tag"]]
        tn = td = tv = 0.0
        for j in range(n):
            if isec["m0"] + j >= len(Mi) or vsec["m0"] + j >= len(Mv):
                break
            k, s = slot_of.get(Mi[isec["m0"] + j]), start_of.get(V.bars[Mv[vsec["m0"] + j]])
            if k is not None and s is not None:
                tn, td, tv = tn + num[k, s], td + den[k, s], tv + voc[k, s]
        if tv < 1.0:
            return None                                  # hardly any singing there to judge
        lo, hi = self.meta.get("plan_fit_random", self.meta["fit_random"]), self.meta.get("plan_fit_own", self.meta["fit_own"])
        h = float(np.clip((tn / td - lo) / (hi - lo), 0, 2.0)) if td > 1e-6 else pairs.H_NEUTRAL
        rel = float(np.clip(td / tv, 0, 1))
        return rel * h + (1 - rel) * pairs.H_NEUTRAL

    def bass(self, bsec, isec, n):
        """How well bass section bsec sits under the other song's instruments isec (1 = as in its own song)."""
        X, Y = self.songs[bsec["tag"]], self.songs[isec["tag"]]
        Mx, My = self.Ms[bsec["tag"]], self.Ms[isec["tag"]]
        ref, base = X.harmony_reference("bass")
        tgt, lay = Y.bar_chroma("other"), X.bar_chroma("bass")
        px, py = X.bar_presence("bass"), Y.bar_presence("other")
        h, w = 0.0, 0.0
        for j in range(2 * n):
            bx, by = Mx[bsec["m0"]] + j, My[isec["m0"]] + j
            if bx >= len(lay) or by >= len(tgt):
                break
            wt = px[bx] * py[by]
            h += wt * float(compat(tgt[by], lay[bx], self.tr[bsec["tag"]] - self.tr[isec["tag"]]))
            w += wt
        if w < 0.5:
            return None
        return float(np.clip((h / w - base) / max(0.03, ref - base), 0, 2.0))

    def overlap(self, a, b, n):
        """Share of the time two vocal sections (one per song) sing at once, aligned measure on measure."""
        A, B = self.songs[a["tag"]], self.songs[b["tag"]]
        ja, jb = A.bars[self.Ms[a["tag"]][a["m0"]]], B.bars[self.Ms[b["tag"]][b["m0"]]]
        L = 2 * A.bpb * n
        va, vb = self.vf[a["tag"]][ja:ja + L], self.vf[b["tag"]][jb:jb + L]
        k = min(len(va), len(vb))
        if k < 4:
            return None
        va, vb = va[:k], vb[:k]
        both, any_ = np.minimum(va, vb).sum(), np.maximum(va, vb).sum()
        return float(both / any_) if any_ > 1e-6 else None

    def menu(self, secs):
        """The WHAT FITS TOGETHER lists for the brief."""
        voc, bas, duet = [], [], []
        for v, i in (("A", "B"), ("B", "A")):
            for vs in secs[v]:
                for s in secs[i]:
                    n = min(vs["m1"] - vs["m0"], s["m1"] - s["m0"], 8)
                    if vs["stems"]["vocals"] >= 0.35 and s["stems"]["other"] >= 0.3:
                        h = self.vocal(vs, s, n)
                        if h is not None and h >= VOCAL_GOOD:
                            voc.append((h, vs["id"], s["id"]))
                    if vs["stems"]["bass"] >= 0.3 and s["stems"]["other"] >= 0.3:
                        h = self.bass(vs, s, n)
                        if h is not None and h >= BASS_GOOD:
                            bas.append((h, vs["id"], s["id"]))
        for a in secs["A"]:
            for b in secs["B"]:
                if a["stems"]["vocals"] >= 0.35 and b["stems"]["vocals"] >= 0.35:
                    o = self.overlap(a, b, min(a["m1"] - a["m0"], b["m1"] - b["m0"], 8))
                    if o is not None and o <= DUET_GOOD:
                        duet.append((o, a["id"], b["id"]))
        voc, bas, duet = sorted(voc, reverse=True)[:14], sorted(bas, reverse=True)[:10], sorted(duet)[:6]
        word = lambda h: "great" if h >= 1.0 else "good" if h >= 0.75 else "ok"
        lines = ["WHAT FITS TOGETHER (checked by the engine in this mix's key, best first):"]
        lines.append(" Vocal on the other song's instruments: " + (", ".join(
            f"{v} vocal on {i} ({word(h)})" for h, v, i in voc) or "nothing fits well - keep vocals on their own song"))
        lines.append(" Bass under the other song's instruments: " + (", ".join(
            f"{b} bass under {i} ({word(h)})" for h, b, i in bas) or "none - keep each song's bass with its own instruments"))
        lines.append(" Both vocals at once (they naturally take turns, so they can sing together): "
                     + (", ".join(f"{a} + {b}" for o, a, b in duet) or "none - VOCAL2 will trade lines with VOCAL"))
        lines.append(" Drums from either song work under anything (same tempo); swapping drums changes the groove.")
        return "\n".join(lines), dict(vocal=voc, bass=bas, duet=duet)


# ------------------------------------------------------------------ the brief and the reply format

def _heard(sid):
    try:
        from . import libmeta
        return libmeta.heard(sid)
    except Exception:                                   # a description is extra, never required
        return None


def _describe(S, tag, secs, M):
    k = S.info.get("key") or {}
    mood = ((S.concept or {}).get("tags") or {}).get("mood")
    voc = np.mean([s["stems"]["vocals"] for s in secs]) if secs else 0
    e = np.array([s["energy"] for s in secs]) if secs else np.zeros(1)
    lo, hi = np.quantile(e, 0.33), np.quantile(e, 0.8)
    amt = lambda x: _level(x, [0.1, 0.35, 0.65], ["-", "little", "some", "full"])
    lines = [f'SONG {tag} - "{S.title}": {S.bpm:.0f} BPM, {k.get("name", "unknown key")}, '
             f'{"vocal-heavy" if voc > 0.5 else "music-heavy" if voc < 0.25 else "vocals in parts"}'
             + (f", mood {mood}" if mood else "") + f", {len(M)} measures long."]
    h = _heard(S.id)
    if h:                       # an audio model's description (listen.py) - the blind test's winner over title guesses
        lines.append(f'  Sounds like (heard): {" / ".join(x for x in (h.get("genre"), h.get("subgenre")) if x)} - {h["sound"]}')
    lines.append(" id   section        measures   energy  drums  bass   instr  vocals  sung lines start at (measures in)  lyric")
    sung = song_lines(S) if S.has_real_vocals else None
    mb = 2 * S.bpb                                        # beats in a measure
    for s in secs:
        en = "peak" if s["energy"] >= hi and s["energy"] >= e.max() - 0.01 else _level(s["energy"], [lo, hi], ["low", "medium", "high"])
        st = s["stems"]
        b0 = float(S.bars[M[s["m0"]]])
        b1 = float(S.bars[M[s["m1"]]]) if s["m1"] < len(M) else float(S.info.get("last_beat", S.nb))
        starts = [(a - b0) / mb for a, b in (sung or []) if b0 - LINE_PICKUP <= a < b1]
        at = ("+" + " +".join(f"{max(0.0, x):g}" for x in [round(2 * v) / 2 for v in starts][:6])) if starts else "-"
        lines.append(f' {s["id"]:<4} {s["name"][:12]:<14} {s["m0"]:>3}-{s["m1"] - 1:<3} ({s["m1"] - s["m0"]:>2})  {en:<6}  '
                     f'{amt(st["drums"]):<6} {amt(st["bass"]):<6} {amt(st["other"]):<6} {amt(st["vocals"]):<6}  '
                     f'{at:<33} {"HOOK " if s["hook"] else ""}' + (f'"{s["lyric"]}"' if s["lyric"] else ""))
    return "\n".join(lines)


def brief(songs, secs, Ms, menu_text, bpm, target, prompt, fit=None):
    mbeats = 2 * songs["A"].bpb * 60.0 / bpm
    rate = lambda d: (fit or {}).get(d) if ((fit or {}).get(d) or {}).get("workable") else None
    ab, ba = rate("AB"), rate("BA")
    if ab or ba:                                          # which way round the vocals sit best, overall
        say = lambda f, v, i: f"{v}'s vocal over {i}'s music rates {round(f['rating'])}/100" if f else f"{v}'s vocal doesn't fit over {i}'s music"
        best = "B's vocal on A's music" if (ba and (not ab or ba["rating"] > ab["rating"])) else "A's vocal on B's music"
        fit_line = f"FIT: {say(ab, 'A', 'B')}; {say(ba, 'B', 'A')}. Give most of the vocal time to {best}."
    else:
        fit_line = "FIT: neither vocal fits over the other song's music - keep each vocal on its own song's instruments."
    n_parts = max(3, round(target / 8))
    return "\n".join([
        _describe(songs["A"], "A", secs["A"], Ms["A"]), "", _describe(songs["B"], "B", secs["B"], Ms["B"]), "",
        menu_text, "", fit_line, "",
        f"MIX: {bpm:g} BPM; one measure = 2 bars = {mbeats:.1f} s. LENGTH: the parts' measures must add up to "
        f"{target} (between {int(target * 0.9)} and {int(target * 1.1)}) - about {n_parts} parts.",
        "REQUEST: " + (f'"{prompt}"' if prompt else "none - use your judgement."),
    ])


def schema(ids):
    pick = {"type": "string", "enum": ids + ["none"]}
    part = {"type": "object", "properties": {
        "why": {"type": "string"},
        "drums": pick, "bass": pick, "instruments": pick, "vocal": pick, "vocal2": pick,
        "measures": {"type": "integer", "enum": MEASURE_CHOICES},
        "transition": {"type": "string", "enum": TRANSITIONS}},
        "required": ["why", "drums", "bass", "instruments", "vocal", "vocal2", "measures", "transition"]}
    return {"type": "object", "properties": {"title": {"type": "string"}, "parts": {"type": "array", "items": part}},
            "required": ["title", "parts"]}


# ------------------------------------------------------------------ the mix's tempo and key

def mix_setup(A, B):
    """(bpm, mult, transpose, fit): tempo halfway between the songs (geometric), the key relation from the
    better-rated vocal-over-instruments direction (its rating and harmony)."""
    mB0, _ = _best_mult(B.bpm, A.bpm)
    bpm = round(math.exp(0.5 * (math.log(A.bpm) + math.log(B.bpm * mB0))) * 2) / 2
    mult = {"A": _best_mult(A.bpm, bpm)[0], "B": _best_mult(B.bpm, bpm)[0]}
    fit = {}
    for d, (v, i) in (("AB", (A, B)), ("BA", (B, A))):
        try:
            fit[d] = pairs.evaluate(v, i, 0.5) if v.has_real_vocals else None
        except Exception as e:                          # one direction failing must not sink the mix
            fit[d] = dict(workable=False, reason=type(e).__name__)
    ok = [d for d in ("AB", "BA") if (fit[d] or {}).get("workable")]
    tr = {"A": 0, "B": 0}
    if ok:
        d = max(ok, key=lambda d: fit[d]["rating"])
        tr[d[1]] = int(fit[d]["shift_bed"])
        tr[d[0]] = int(fit[d]["shift_rel"] + fit[d]["shift_bed"])
    return bpm, mult, tr, fit


# ------------------------------------------------------------------ playing a script

def _clean_exit(S, src0, n_src, min_frac=0.5):
    """Source-beat length <= n_src ending where S's vocal can be cut cleanly (latest such beat after min_frac of
    it), else n_src (the clip edge then fades the line out)."""
    _, c_out = S.cut_cost()
    for e in range(int(math.floor(src0 + n_src)), int(math.ceil(src0 + min_frac * n_src)) - 1, -1):
        if 0 <= e < len(c_out) and c_out[e] < 0.3:
            return float(e - src0)
    return float(n_src)


LINE_BREATH = 0.30        # s of silence that ends a sung line
LINE_PICKUP = 1.5         # beats a line may start before its section (a pickup into it)
LINE_LONG = 4             # measures: a longer stretch without a breath is split where the voice dips
LINE_GAP = 0.25           # beats of air between one singer's line and the other's answer
LINE_JOIN = 2.0           # beats: lines this close in a row, where they fall naturally, play as one clip


def song_lines(S):
    """S's sung lines over the whole song, [(start beat, end beat)] - whole lines of lyric, so a vocal is never cut
    off or swapped mid-sentence. With a usable transcript a line is one of its lines (Whisper's segments), breaths
    inside it and all; the cut between two lines goes into the nearest real gap (a breath within 0.6 s, else the
    short silence between two words within 0.4 s - Whisper's times are only roughly right). Without one, the vocal
    is split at its breaths (>= LINE_BREATH s of silence). Either way a line running on past LINE_LONG measures is
    split at the gap nearest its middle. Cached per song; None without melody features (no 10 ms voicing)."""
    if "dj_lines" in S._bar:
        return S._bar["dj_lines"]
    vm = S.voicing()
    if vm is None:
        S._bar["dj_lines"] = None
        return None
    fps, n = 100, len(vm)
    raw = np.isfinite(S.mash["f0"])[:n]                   # the pitch track before gaps are closed: word gaps
    bix = np.arange(S.nb, dtype=float)
    beat = lambda fr: float(np.interp(fr / fps, S.beats, bix))
    long = LINE_LONG * 2 * S.bpb * 60.0 / S.bpm * fps
    out = []

    def add(a, b):
        idx = np.flatnonzero(vm[a:b])
        if not len(idx):
            return
        a, b = a + int(idx[0]), a + int(idx[-1]) + 1      # to the voice itself
        if b - a > long:
            gap = [i for i in range(a + int(long / 4), b - int(long / 4)) if not raw[i]]
            if gap:
                c = min(gap, key=lambda i: abs(i - (a + b) // 2))
                add(a, c)
                add(c + 1, b)
                return
        if b - a >= 0.25 * fps:                           # shorter: a breath or a click, not a line
            out.append((beat(a), beat(b)))

    def snap(t):
        """Frame of the cleanest cut near time t (s): in a breath if there is one close, else between words."""
        c = int(t * fps)
        for mask, w in ((vm, 0.6), (raw, 0.4)):
            gap = [i for i in range(max(0, int(c - w * fps)), min(n, int(c + w * fps))) if not mask[i]]
            if gap:
                return min(gap, key=lambda i: abs(i - c))
        return c

    ly = S.lyrics
    q = (ly or {}).get("quality") or {}
    segs = sorted(((ly or {}).get("segments") or []) if q.get("usable") and q.get("asr_conf", 0) >= 0.3 else [],
                  key=lambda g: g["t0"])
    on = vm.astype(np.int8)
    st = np.flatnonzero(np.diff(np.r_[0, on]) == 1)
    en = np.flatnonzero(np.diff(np.r_[on, 0]) == -1) + 1
    runs = []                                             # stretches of singing between breaths
    for a, b in zip(st, en):
        if runs and a - runs[-1][1] < LINE_BREATH * fps:
            runs[-1][1] = int(b)
        else:
            runs.append([int(a), int(b)])
    if segs:
        lo, hi = [snap(segs[0]["t0"] - 0.3)], []
        for g1, g2 in zip(segs, segs[1:]):
            if g2["t0"] - g1["t1"] > 1.0:                 # apart: each ends / starts at its own words
                hi.append(snap(g1["t1"] + 0.3))
                lo.append(snap(g2["t0"] - 0.3))
            else:                                         # back to back: one cut between them
                c = snap(0.5 * (g1["t1"] + g2["t0"]))
                hi.append(c)
                lo.append(c)
        hi.append(snap(segs[-1]["t1"] + 0.3))
        for a, b in zip(lo, hi):
            add(a, b)
        covered = np.zeros(n, bool)                       # singing the transcript missed: lines by breaths
        for a, b in zip(lo, hi):
            covered[a:b] = True
        for a, b in runs:
            if not covered[a:b].any():
                add(a, b)
        out.sort()
    else:
        for a, b in runs:
            add(a, b)
    S._bar["dj_lines"] = out
    return out


def _natural(lines_, v, m, t0, t1, meas):
    """The whole lines of vocal pick v that start in its section window, where they fall: [(project beat, a, b)].
    A line may start a little before the part (a pickup) and ring up to a measure into the next one."""
    out = []
    for a, b in lines_:
        if v["src0"] - LINE_PICKUP <= a < v["src0"] + v["n_src"]:
            p = t0 + (a - v["src0"]) * m
            if p + (b - a) * m <= t1 + meas:
                out.append((p, a, b))
    return out


def _call_response(L1, L2, v1, v2, mult, t0, t1, meas):
    """Two singers trading whole lines: each takes its next line in lyric order, moved by whole measures (so it
    keeps its place in the bar) to the first spot after the other singer's line has ended. A singer with nothing
    more to say lets the other carry on. -> [(0 | 1, project beat, a, b)]."""
    vs = (v1, v2)
    seq = [[(a, b) for a, b in L if v["src0"] - LINE_PICKUP <= a < v["src0"] + 2 * v["n_src"]]
           for L, v in ((L1, v1), (L2, v2))]
    idx, out = [0, 0], []
    first = lambda w: (seq[w][0][0] - vs[w]["src0"]) * mult[vs[w]["tag"]] if seq[w] else 1e9
    turn = 0 if first(0) <= first(1) else 1
    t = t0 - LINE_PICKUP
    while True:
        for w in (turn, 1 - turn):
            if idx[w] >= len(seq[w]):
                continue
            a, b = seq[w][idx[w]]
            m = mult[vs[w]["tag"]]
            nat = t0 + (a - vs[w]["src0"]) * m
            p = t + ((nat - t) % meas)
            if p + (b - a) * m > t1 + 0.5 * meas:
                idx[w] = len(seq[w])                      # its next line doesn't fit any more
                continue
            out.append((w, p, a, b))
            idx[w] += 1
            t = p + (b - a) * m + LINE_GAP
            turn = 1 - w
            break
        else:
            return out


def execute(script, songs, secs, Ms, fits, bpm, mult, tr, fit=None, prompt="", llm_name="", target=None):
    """script = {"title", "parts": [{why, drums, bass, instruments, vocal, vocal2, measures, transition}]}
    -> (project, desc)."""
    A, B = songs["A"], songs["B"]
    if A.bpb != B.bpb:
        raise ValueError("these two songs are in different meters")
    bpb = A.bpb
    by_id = {s["id"]: s for t in "AB" for s in secs[t]}
    parts = [p for p in (script.get("parts") or [])[:MAX_PARTS]
             if any(p.get(k) in by_id for k, _, _ in ROLES)]
    if not parts:
        raise ValueError("the plan had no usable parts")
    repairs = []
    total = sum(int(p["measures"]) for p in parts)
    if target and total < 0.8 * target:                   # small models undershoot: stretch, keeping proportions,
        f = target / total                                # but never past what a part's sections have left
        for p in parts:
            left = [len(Ms[by_id[p[k]]["tag"]]) - by_id[p[k]]["m0"] for k, _, _ in ROLES if p.get(k) in by_id]
            cap = max(int(p["measures"]), min(left)) if left else int(p["measures"])
            want = min(MEASURE_CHOICES, key=lambda c: abs(c - p["measures"] * f))
            p["measures"] = max([c for c in MEASURE_CHOICES if c <= min(want, cap)] or [int(p["measures"])])
        repairs.append(f"the plan was {total} measures for a {target}-measure mix - parts lengthened to "
                       f"{sum(p['measures'] for p in parts)}")

    proj = P.empty((script.get("title") or f"{A.title} × {B.title}")[:80], bpm, bpb)
    for t in "AB":
        proj["songs"][songs[t].id] = dict(transpose=tr[t], mult=mult[t])
    track = {t["role"]: t for t in proj["tracks"]}
    track["vocals2"] = None                               # added only if a second vocal is used
    inst = {t: songs[t].info["loudness"].get("inst", songs[t].info["loudness"]["mix"]) for t in "AB"}
    ref_lvl = float(np.mean(list(inst.values())))
    g_song = {t: float(np.clip(ref_lvl - inst[t], -6, 6)) for t in "AB"}   # the two songs at one level

    meas = 2 * bpb
    t_proj, done, used = 0.0, [], {}
    last = {}                                             # role -> (clip, song tag, src end) of the part before
    vox = []                                              # (role, tag, clip) for the breath pass
    lines_out = []                                        # vocal lines to place: {role, tag, p, a, b, g}
    for k, p in enumerate(parts):
        n = int(p["measures"])
        length = n * meas
        trans = p.get("transition", "cut") if k else "start"
        pick = {}
        for key, role, stem in ROLES:
            sid = p.get(key, "none")
            if sid not in by_id:
                continue
            sec = by_id[sid]
            tag, S, M = sec["tag"], songs[sec["tag"]], Ms[sec["tag"]]
            avail = len(M) - sec["m0"]
            src0 = float(S.bars[M[sec["m0"]]])
            n_src = min(n, avail) * meas / mult[tag]
            if n_src <= 0:
                continue
            if avail < n:
                repairs.append(f"part {k + 1}: {key} {sid} has only {avail} measure(s) left (asked {n})")
            on = S.on[stem][int(src0):int(src0 + n_src)]
            if stem == "vocals" and not S.has_real_vocals or not len(on) or on.mean() < 0.12:
                repairs.append(f"part {k + 1}: {sid} ({sec['name']}) has almost no {key.rstrip('2')} - left out")
                continue
            if stem == "vocals":
                used[sid] = used.get(sid, 0) + 1
                if used[sid] > MAX_VOCAL_USES:
                    repairs.append(f"part {k + 1}: vocal {sid} was already used {MAX_VOCAL_USES} times - left out")
                    continue
            pick[role] = dict(sid=sid, sec=sec, tag=tag, src0=src0, n_src=n_src, n=min(n, avail), stem=stem, note="")

        # ---- the engine's musical checks
        ins = pick.get("other")
        nn = lambda a, b: min(a["n"], b["n"], 8)
        if ins and pick.get("bass") and pick["bass"]["tag"] != ins["tag"]:
            h = fits.bass(pick["bass"]["sec"], ins["sec"], nn(pick["bass"], ins))
            if h is not None and h < BASS_MIN:
                old = pick.pop("bass")
                Si = songs[ins["tag"]]
                if Si.on["bass"][int(ins["src0"]):int(ins["src0"] + ins["n_src"])].mean() >= 0.12:
                    pick["bass"] = dict(ins, stem="bass", sid=ins["sid"], note="")
                    repairs.append(f"part {k + 1}: {old['sid']} bass clashes with {ins['sid']}'s chords "
                                   f"({round(100 * h)}%) - used {ins['sid']}'s own bass")
                else:
                    repairs.append(f"part {k + 1}: {old['sid']} bass clashes with {ins['sid']}'s chords - left out")
            elif h is not None:
                pick["bass"]["note"] = f"{min(100, round(100 * h))}%"
        for role in ("vocals", "vocals2"):
            v = pick.get(role)
            if v and ins and v["tag"] != ins["tag"]:
                h = fits.vocal(v["sec"], ins["sec"], nn(v, ins))
                if h is not None and h < VOCAL_MIN:
                    pick.pop(role)
                    repairs.append(f"part {k + 1}: {v['sid']} vocal clashes with {ins['sid']}'s chords "
                                   f"({round(100 * h)}%) - left out")
                elif h is not None:
                    v["note"] = f"{min(100, round(100 * h))}%"
        v1, v2 = pick.get("vocals"), pick.get("vocals2")
        if v2 and not v1:                                 # a lone second vocal is just the vocal
            pick["vocals"] = pick.pop("vocals2")
            v1, v2 = pick["vocals"], None
        if v1 and v2:
            if v1["sid"] == v2["sid"] or v1["tag"] == v2["tag"]:
                pick.pop("vocals2")
                repairs.append(f"part {k + 1}: a second vocal needs the other song - left out {v2['sid']}")
            else:
                o = fits.overlap(v1["sec"], v2["sec"], nn(v1, v2))
                L1, L2 = song_lines(songs[v1["tag"]]), song_lines(songs[v2["tag"]])
                if o is not None and o > DUET_TRADE:       # they'd sing over each other: trade whole lines
                    if L1 is None or L2 is None:
                        pick.pop("vocals2")
                        repairs.append(f"part {k + 1}: {v1['sid']} and {v2['sid']} would sing over each other "
                                       f"- left out {v2['sid']}")
                    else:
                        v1["trade"] = v2["trade"] = _call_response(L1, L2, v1, v2, mult, t_proj, t_proj + length, meas)
                        v2["note"] = (v2["note"] + " trading lines").strip()

        # ---- clips
        ref = (ins or pick.get("bass") or pick.get("drums") or v1 or {}).get("tag", "A")
        for role, v in pick.items():
            S, tag, stem = songs[v["tag"]], v["tag"], v["stem"]
            if stem == "vocals":
                g = (_replace_gain(songs[ref], S, "vocals") if tag != ref else 0.0) + g_song[ref]
                if role == "vocals2":
                    g -= 1.5 if v.get("trade") else 3.0
            else:
                g = (_replace_gain(songs[ref], S, stem) if tag != ref else 0.0) + g_song[ref]
            if role == "vocals2" and track["vocals2"] is None:
                track["vocals2"] = P.track("Vocals 2 · DJ mix", "vocals", hpf=110.0)
                proj["tracks"].append(track["vocals2"])
            L = song_lines(S) if stem == "vocals" else None
            if L is not None:                              # vocals: whole lines only
                if v.get("trade") is not None:
                    w = 0 if role == "vocals" else 1
                    got = [(p_, a, b) for who, p_, a, b in v["trade"] if who == w]
                else:
                    got = _natural(L, v, mult[tag], t_proj, t_proj + length, meas)
                lines_out += [dict(role=role, tag=tag, p=p_, a=a, b=b, g=round(g, 1)) for p_, a, b in got]
                v["note"] = (v["note"] + f" {len(got)} line{'s' if len(got) != 1 else ''}").strip()
                last.pop(role, None)
                continue
            nsrc = v["n_src"]
            if stem == "vocals" and v["src0"] + nsrc < S.info["last_beat"]:
                nsrc = _clean_exit(S, v["src0"], nsrc)
            prev = last.get(role)
            if prev and prev[1] == tag and abs(prev[2] - v["src0"]) < 0.5 and abs(prev[0]["start"] + prev[0]["len"] - t_proj) < 1e-6 \
                    and abs(prev[0]["gain_db"] - round(g, 1)) < 0.05:
                c = prev[0]                                # the same audio simply carries on: one clip
                c["len"] += nsrc * mult[tag]
            else:
                xf = bpb if (k and trans != "cut" and stem != "vocals" and prev
                             and v["src0"] - bpb / mult[tag] >= S.info["first_beat"]) else 0.0
                c = P.clip(track[role]["id"], S.id, stem, t_proj - xf, v["src0"] - xf / mult[tag],
                           nsrc * mult[tag] + xf, fade_in=xf, fade_out=0.05, gain_db=round(g, 1))
                proj["clips"].append(c)
                if xf and prev and abs(prev[0]["start"] + prev[0]["len"] - t_proj) < 1e-6:
                    prev[0]["fade_out"] = max(prev[0]["fade_out"], xf)
                if stem == "vocals":
                    vox.append((role, tag, c))
            last[role] = (c, tag, v["src0"] + nsrc)
        for role in [r for r in last if r not in pick]:   # a stem that stops here
            last.pop(role)
        show = lambda r: (f"{pick[r]['sid']}" + (f" ({pick[r]['note'].strip()})" if pick[r]["note"].strip() else "")) if r in pick else "-"
        done.append(dict(n=k + 1, start_measure=int(round(t_proj / meas)), measures=n, drums=show("drums"),
                         bass=show("bass"), instruments=show("other"), vocal=show("vocals"), vocal2=show("vocals2"),
                         transition=trans, why=p.get("why", "")[:140]))
        t_proj += length

    # vocal lines: per track in time order, a line that would start before the previous one has finished is
    # left out (a line is never cut); lines that simply follow each other where they fall play as one clip
    dropped = 0
    for role in ("vocals", "vocals2"):
        end, cur = -1e9, None
        for ln in sorted((x for x in lines_out if x["role"] == role), key=lambda x: x["p"]):
            m = mult[ln["tag"]]
            if ln["p"] < end - 0.1:
                dropped += 1
                continue
            if cur and cur["song"] == songs[ln["tag"]].id and cur["gain_db"] == ln["g"] \
                    and abs((ln["p"] - cur["start"]) - (ln["a"] - cur["src"]) * m) < 0.01 and ln["p"] - end <= LINE_JOIN:
                cur["len"] = (ln["b"] - cur["src"]) * m   # the same song carrying on, breath and all
            else:
                cur = P.clip(track[role]["id"], songs[ln["tag"]].id, "vocals", ln["p"], ln["a"], (ln["b"] - ln["a"]) * m,
                             fade_out=0.05, gain_db=ln["g"])
                proj["clips"].append(cur)
                vox.append((role, ln["tag"], cur))
            end = cur["start"] + cur["len"]
    if dropped:
        repairs.append(f"{dropped} vocal line(s) left out where they would have started before the line before ended")

    # breaths at every vocal edge, per song and track (clips were made back to back; edges reach into gaps)
    sung_end = {id(c): c["start"] + c["len"] for _, _, c in vox}      # where each clip's singing ends
    for role in ("vocals", "vocals2"):
        for t in "AB":
            cl = sorted([c for r, tg, c in vox if r == role and tg == t], key=lambda c: c["start"])
            if cl:
                proj["clips"] = [c for c in proj["clips"] if c not in cl]
                proj["clips"].extend(_vocal_edges(songs[t], cl, mult[t], bpb / mult[t]))
    # a line's ring-out stops where the other singer comes in (the line itself is never shortened)
    allv = sorted((c for _, _, c in vox), key=lambda c: c["start"])
    for c in allv:
        e0 = sung_end[id(c)]
        nxt = [d["start"] for d in allv if d is not c and (d["song"] != c["song"] or d["track"] != c["track"])
               and e0 - 0.05 <= d["start"] < c["start"] + c["len"]]
        if nxt:
            c["len"] = max(e0, min(nxt) + 0.1) - c["start"]
            c["fade_out"] = max(0.05, min(c["fade_out"], c["start"] + c["len"] - e0))
    if vox:
        track["other"].update(pocket=2.5, duck=2.0)
        track["bass"].update(duck=0.8)
        track["vocals"].update(hpf=110.0)
    for r_ in ("drums", "bass", "other", "vocals"):
        track[r_]["name"] = r_.capitalize() + " · DJ mix"
    proj["master"]["gain_db"] = -1.5
    desc = dict(kind="dj", title=proj["name"], songs={t: songs[t].id for t in "AB"},
                titles={t: songs[t].title for t in "AB"}, bpm=bpm, transpose=tr, prompt=prompt, llm=llm_name,
                parts=done, repairs=repairs, measures=int(round(t_proj / meas)), length_sec=round(t_proj * 60 / bpm, 1),
                fit={d: (None if not ((fit or {}).get(d) or {}).get("workable") else
                         dict(rating=fit[d]["rating"], harmony=fit[d]["harmony"])) for d in ("AB", "BA")})
    proj["mashup"] = desc
    return proj, desc


# ------------------------------------------------------------------ the whole thing

def run(song_ids, prompt="", minutes=3.0, seed=None, progress=lambda f, m="": None, model=None):
    """Two library songs + a request -> (project, desc). `model`: an llm client (default: llm.client())."""
    if len(song_ids) != 2:
        raise ValueError("a DJ mix takes exactly two songs")
    progress(0.02, "reading the songs")
    A, B = (automash.get_song(s) for s in song_ids)
    songs = {"A": A, "B": B}
    (secA, MA), (secB, MB) = sections(A, "A"), sections(B, "B")
    if not secA or not secB:
        raise ValueError("a song has no usable sections at its main tempo")
    secs, Ms = {"A": secA, "B": secB}, {"A": MA, "B": MB}
    progress(0.08, "finding the key and tempo")
    bpm, mult, tr, fit = mix_setup(A, B)
    progress(0.18, "checking what fits together")
    fits = Fits(songs, Ms, tr, mult)
    menu_text, menu = fits.menu(secs)
    target = max(8, int(round(minutes * bpm / (2 * A.bpb))))
    text = brief(songs, secs, Ms, menu_text, bpm, target, (prompt or "").strip()[:400], fit)
    ids = [s["id"] for s in secA + secB]
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}]
    model = model or llm_mod.client()
    progress(0.3, f"the DJ is planning ({model.name})")
    try:
        try:
            script = model.chat_json(messages, schema(ids), max_tokens=5000, temperature=0.45, seed=seed)
        except llm_mod.LlmError:
            script = model.chat_json(messages, schema(ids), max_tokens=8000, temperature=0.3, seed=seed)
    finally:
        model.unload()
    progress(0.85, "building the mix")
    proj, desc = execute(script, songs, secs, Ms, fits, bpm, mult, tr, fit, prompt, model.name, target)
    desc["brief"] = text
    progress(1.0, "done")
    return proj, desc
