"""Grid mashup: pick any stems from any songs; one song is the MASTER and everything else is fitted to it.

What the Auto-Mashup dialog's "Pick stems" tab builds. It writes an ordinary project - one lane per (song, stem),
clips the timeline can move, trim, mute and level like any others - so the arranged stems stay editable.

  master      plays as recorded, start to end, one clip per section. Its bar lines ARE the grid: every other
              song is placed against where the master's beats really fall, not against an ideal grid.
  drums /     "bar-locked": each master bar plays one of the other song's bars. Same meter: the bars simply
  bass /      run on. Other meter, two ways (picked by which needs less stretching, or by the caller):
  other         beats line up - beat for beat; each bar drops or repeats beats to fit (3 -> 4 plays 1 2 3 2)
                bars line up  - a whole bar over a whole bar (6/8 over 4/4 becomes a triplet feel)
              A layer that brings music (bass / other), or that is the backing because the master brings none,
              plays its bars in its own order, start to end. Drums under the master's music play where the
              master's own drums played (breakdowns stay breakdowns) and start each master section from one of
              their own sections of matching energy.
  staying     Two songs never keep the same time for long. follow = "stretch" (the default): a layer that slides
  in time     more than DRIFT from the master is sped up / slowed down a little, in as few long stretches as keep
              every bar line within DRIFT (clip warps: small changes are plain varispeed, bigger ones keep the
              pitch) - its audio is never cut. follow = "cut": nothing is stretched; the layer is re-anchored on
              a bar line instead (a small jump in its audio).
  vocals      whole lines (djremix.phrases: found from the vocal's level, never cut inside a word), each put with
              its bar line on a master bar line, pickup kept. Lines that follow each other keep their spacing
              where the meters agree; in another meter every line is re-anchored on the next free bar line. A
              line that would run past the end, or over the master's own singer, is left out - never cut short.
              A long line follows the master's tempo the same way a layer does. Several singers take turns by
              section: one sings the master's choruses (its hook), the others its verses.
  the grid    A song's own marks when it has them. A detector grid that switches between half / double time (or
              3:2) from section to section is re-counted on one pulse for the mix (beatfix) - never saved.

Everything is computed in source SECONDS and project beats. A song plays as recorded at one speed (see
project.song_tspec), so "source seconds per project beat" is one number per song, and a clip is fully described
by where it starts in the project, which moment of the song that is, and how long it runs.
"""
import math

import numpy as np

from . import project as P
from .config import STEMS

DRIFT = 0.015            # s a layer may slide from the master before it is made to follow (stretched, or re-anchored)
DRIFT_MAX = 0.05         # ... at most, when the grids themselves are rough (see Src.noise): marks placed 25 ms either
                         # side of the music can't be followed to 15 ms - that only makes the layer wobble
RATE_SPAN = (0.75, 1.34)  # speed changes a layer may be given to follow the master; past that it is cut instead
LURCH = 0.03             # a stretch of a bar or two that needs a speed this far from its neighbour's is a lurch ...
WIDE = 0.08              # ... spared if every bar line then stays within this share of a bar (see ease)
VOX_TOL = 0.03           # s a vocal line may slide before it is made to follow
VOX_SPAN = (0.95, 1.05)  # ... and the speed changes a voice may be given: a voice that lurches is worse than one
                         # that sits a little off, so past that the line gets one speed, or stays as recorded
VARI = dict(drums=0.03)  # a speed change up to this is plain varispeed (exact, nothing to render): drums 3 %,
VARI_PITCHED = 0.006     # anything with a pitch 0.6 % (10 cents); bigger changes keep the pitch (Rubber Band)
HOOK_SECTIONS = ("chorus", "drop", "hook")
LEAD = 0.012             # s a re-anchored clip starts before its bar line, so its 3 ms edge fade never eats a hit
MAX_STRETCH = 0.25       # a layer sped up / slowed down by more than this is built anyway, and reported
RATIOS = (1.0, 0.5, 2.0, 1 / 3, 3.0, 2 / 3, 1.5, 0.25, 4.0)
BEAT_M = (0.5, 1.0, 2.0)         # project beats one of the song's beats may span when the beats line up
BARS_PER = (0.5, 1.0, 2.0)       # the song's bars per master bar when the bars line up
BAR_BIAS = 0.03          # bars-line-up has to beat beats-line-up by this much (log speed) to be picked on its own
PRESENT = 0.4            # share of a bar a stem must sound in to count as playing there
MIN_PART_BARS = 2.0      # a joined stretch of singing may stop at one of its joins only after this much of it
BLOCK_GAP_BARS = 2.0     # lines further apart than this start a new block (placed as a group)
MAX_GAP_BARS = 8         # ... and at most this many bars of the original gap are kept between blocks
SKIP_SECTIONS = ("intro", "outro", "silence")
GLUE = 0.25              # s: lines this close are one stretch of singing - they stay one clip, never pulled apart
HIT_REACH = 0.06         # s either side of a beat mark a drum hit is looked for
HIT_MAX = 0.03           # ... and the most a song is nudged so its hits (not its marks) meet the master's


# ------------------------------------------------------------------ a song, as plain data

class Src:
    """One song as the planner sees it. Built from the library by from_song(); tests build it by hand.

    marks_t / marks_d  the song's grid: beat times (s) and bar-line flags (the user's marks, else the detector's)
    noise              how rough that grid is (s): what a layer following it, or it following a master, must allow for
    info_beats         the song's stored beat list - what clip.src counts in
    info_bpm           its stored tempo - what a mult is measured against
    sections           [{t0, t1, name, energy, stems: {stem: 0..1}}]
    on                 {stem: bool per stored beat}; energy: 0..1 per stored beat
    phrases            [(t0 s, t1 s, words[, how many lines share these words])] whole vocal lines, or None
    hit                s its drum hits fall after its beat marks (hits_vs_marks), or None when its drums don't say
    note               something the user should know about this song's grid (e.g. it was re-counted), or None
    """

    def __init__(self, sid, title, marks_t, marks_d, info_beats, info_bpm, bpb=4, content=None, sections=None,
                 on=None, energy=None, phrases=None, has=None, loudness=None, hit=None, note=None):
        self.id, self.title = sid, title
        self.hit, self.note = hit, note
        self.bt = np.asarray(marks_t, float)
        d = np.asarray(marks_d, int)
        self.ib = np.asarray(info_beats, float)
        self.info_bpm = float(info_bpm)
        down = np.flatnonzero(d)
        if len(self.bt) < 4 or len(down) < 2:
            raise ValueError(f"{title}: no usable beat grid")
        self.down = down
        counts = np.diff(down)
        self.bpb = int(bpb or 0) or int(np.bincount(counts).argmax())
        per = np.diff(self.bt)
        self.period = float(np.median(per))
        self.marks_bpm = 60.0 / self.period
        r = self.marks_bpm / self.info_bpm
        near = min(RATIOS, key=lambda x: abs(math.log(r / x)))
        self.ratio = near if abs(math.log(r / near)) < 0.05 else 1.0     # marked beats per stored-tempo beat
        self.beat_bpm = self.info_bpm * self.ratio                        # the tempo of the MARKED beat
        lo, hi = content if content else (float(self.bt[down[0]]), float(self.bt[-1]) + self.period)
        self.t0, self.t1 = float(lo), float(hi)
        self.sections = list(sections or [dict(t0=self.t0, t1=self.t1, name="", energy=0.5, stems={})])
        self.on = on or {}
        self.energy = None if energy is None else np.asarray(energy, float)
        self.phrases = phrases
        self.has = dict(has or {})
        self.loudness = loudness or {}
        # bars: (first mark index, mark count) for every bar line; the last runs to the end of the marks
        ends = list(down[1:]) + [len(self.bt)]
        self.bars = [(int(a), int(b - a)) for a, b in zip(down, ends) if b > a]
        self.bar_t = np.array([self.mark_time(a) for a, _ in self.bars] + [self.mark_time(self.bars[-1][0] + self.bars[-1][1])])
        # how rough the grid is: how far a bar line typically sits from a straight line through its 8 neighbours (s).
        # 0 for a song on a click; a few ms for careful marks; 20 ms and more when the detector was guessing
        bl, res = self.bar_t, []
        for i in range(4, len(bl) - 4):
            res.append(abs(bl[i] - (bl[i - 4:i].sum() + bl[i + 1:i + 5].sum()) / 8.0))
        self.noise = float(np.median(res)) if res else 0.0

    # -- time <-> the grid
    def mark_time(self, i):
        """Seconds of (fractional) mark index i, carrying on past either end at the edge tempo."""
        return float(P._beat_to_time(i, self.bt))

    def src_index(self, t):
        """clip.src for the moment t (s): its place in the stored beat list."""
        return float(P._time_to_beat(t, self.ib))

    def bar_time(self, x):
        """Seconds of bar position x (bar index + fraction of that bar, by its own beats); runs on past the end."""
        k = int(np.clip(math.floor(x), 0, len(self.bars) - 1))
        a, n = self.bars[k]
        return self.mark_time(a + (x - k) * n)

    def content_bars(self):
        """Indices of the bars inside the song's content (not the silence the grid was extended over)."""
        return [k for k in range(len(self.bars)) if self.t0 - 1e-3 <= self.bar_t[k] < self.t1 - 1e-3]

    def section_at(self, t):
        for s in self.sections:
            if s["t0"] - 1e-3 <= t < s["t1"] - 1e-3:
                return s
        return min(self.sections, key=lambda s: min(abs(t - s["t0"]), abs(t - s["t1"])))

    def presence(self, stem, t0, t1):
        """0..1 share of [t0, t1) the stem sounds in (1 when nothing is known about it)."""
        on = self.on.get(stem)
        if on is None or not len(on):
            return 1.0
        a = int(np.clip(math.floor(self.src_index(t0) + 1e-6), 0, len(on) - 1))
        b = int(np.clip(math.ceil(self.src_index(t1) - 1e-6), a + 1, len(on)))
        return float(np.mean(on[a:b]))

    def level(self, t0, t1):
        if self.energy is None or not len(self.energy):
            return 0.5
        a = int(np.clip(math.floor(self.src_index(t0) + 1e-6), 0, len(self.energy) - 1))
        b = int(np.clip(math.ceil(self.src_index(t1) - 1e-6), a + 1, len(self.energy)))
        return float(np.mean(self.energy[a:b]))

    def plays(self, stem):
        """Does the song have enough of this stem to use it (or to be replaced in it)?"""
        return bool(self.has.get(stem, True))


def from_song(S, fix=True):
    """Src for a library song (automash.SongData). fix: put a detector grid that changes tempo right for the mix
    where the audio says how (_fixed_grid); the user's own marks are never touched."""
    from . import timing
    info = S.info
    drums = _drum_reader(S)
    marks, note = info.get("user_marks"), None
    if not marks:
        marks = timing.detected_marks(info)
        fixed = _fixed_grid(S, info, drums) if fix else None
        if fixed:
            marks, note = fixed
    ib = np.asarray(info["beats"], float)
    bars = list(info["bars"])
    per = float(np.median(np.diff(ib))) if len(ib) > 1 else 0.5
    end = float(info.get("duration") or ib[-1] + per)
    bar_s = lambda b: float(ib[bars[b]]) if 0 <= b < len(bars) else end
    c0, c1 = info.get("content_bars") or (0, len(bars))
    secs = []
    for s in info.get("sections") or []:
        if s.get("name") == "silence":
            continue
        secs.append(dict(t0=bar_s(s["start"]), t1=bar_s(s["end"]), name=s.get("name") or s.get("label") or "",
                         energy=float(s.get("energy", 0.5)), stems=dict(s.get("stems") or {})))
    phr = None
    if S.has_real_vocals:
        try:
            from . import djremix
            got = djremix.phrases(S)
            if got:
                phr = [(p["f0"] / 100.0, p["f1"] / 100.0, p.get("words", ""), int(p.get("repeats") or 1)) for p in got]
        except Exception:
            phr = None
        if phr is None:
            phr = _coarse_phrases(S.on["vocals"], ib)
    return Src(S.id, S.title, marks["t"], marks["d"], ib, float(info["bpm"]), bpb=marks.get("bpb") or info.get("bpb"),
               content=(bar_s(c0), bar_s(c1)), sections=secs or None, on={s: np.asarray(S.on[s], bool) for s in STEMS},
               energy=S.beat_energy, phrases=phr, has={s: bool(S.has_stem(s)) for s in STEMS},
               loudness=dict(info.get("loudness") or {}), hit=_song_hit(S, marks["t"], bar_s(c0), bar_s(c1), drums),
               note=note)


def _drum_reader(S):
    """() -> (mono drum stem, sr) or (None, None); the file is read at most once, and only if somebody asks."""
    got = []

    def read():
        if not got:
            y = sr = None
            try:
                if S.has_stem("drums"):
                    import soundfile as sf
                    from . import library
                    files = sorted((library.song_dir(S.id) / "stems").glob("drums.*"))
                    if files:
                        y, sr = sf.read(str(files[0]), dtype="float32", always_2d=True)
                        y = y.mean(axis=1)
            except Exception:
                y = sr = None
            got.append((y, sr))
        return got[0]
    return read


# ------------------------------------------------------------------ a grid the mix can trust
#
# A song's own marks (the user's) are used as they are. A detector grid is used as it is too, unless it changes
# tempo along the way - then one of these is usually true, and the first two are checked against the audio:
#   1. the music keeps ONE tempo and the detector lost it for a while (a quiet intro, a breakdown, a change of feel):
#      steady_fit() finds that out from the music's onsets, and the mix gets a steady grid (steady_marks);
#   2. the detector counted a stretch at another level than the stretch next to it. tidy() walks outward from the
#      song's longest stretch (taken to be counted right) and judges every stretch against the one before it:
#        half / double time, 3 against 2 ...  -> re-counted on the neighbour's pulse (its own beats are kept)
#        within KEEP_SPAN of the neighbour     -> a real change of tempo: kept, and followed
#        anything else (4 against 3, no beat)  -> the detector lost it: even beats at the neighbour's tempo, so the
#                                                 song simply plays on through it instead of being cut bar by bar
# Nothing is saved anywhere: the song's grid stays as it is.

STEADY_SHARE = 0.9       # share of the music's pulse that sits on one unbroken tempo for the song to count as steady
STEADY_COVER = 0.6       # ... the share of the music OUTSIDE the trusted stretch that has that pulse too
STEADY_SNR = 5.0         # ... and how far that tempo stands out from the tempos around it
STEADY_AGREE = 0.02      # s the steady grid may sit from the detector's marks in its longest stretch (median)
KEEP_SPAN = (0.8, 1.25)  # a stretch this close to its neighbour's tempo is a tempo change, not a miscount


def stretches(beats):
    """The grid's stretches of steady tempo and what to make of each: ([(a, b, bpm, what, tempo it really runs at)],
    index of the longest). what = ("count", k): counted at k x its neighbour's pulse (1.0 = the same pulse);
    ("keep", 1.0): a tempo of its own; ("even", None): no usable count. Judged outward from the longest stretch."""
    from . import beatfix
    b = np.asarray(beats, float)
    if len(b) < 2 * beatfix.MIN_SEC + 2:
        return [], None
    _, segs = beatfix.plan(b)
    main = max(range(len(segs)), key=lambda i: b[segs[i][1]] - b[segs[i][0]])
    what, real = {main: ("count", 1.0)}, {main: float(segs[main][2])}
    for i in list(range(main - 1, -1, -1)) + list(range(main + 1, len(segs))):
        ref = real[i + 1] if i < main else real[i - 1]
        r = segs[i][2] / ref
        k = min(beatfix.RATIOS, key=lambda q: abs(math.log(r / q)))
        if abs(math.log(r / k)) < beatfix.TOL:
            what[i], real[i] = ("count", k), float(segs[i][2] / k)
        elif KEEP_SPAN[0] <= r <= KEEP_SPAN[1]:
            what[i], real[i] = ("keep", 1.0), float(segs[i][2])
        else:
            what[i], real[i] = ("even", None), float(ref)
    return [(a, c, float(t), what[i], real[i]) for i, (a, c, t, _) in enumerate(segs)], main


def needs_tidy(beats):
    """(stretches counted at another level than their neighbour, stretches with no usable count)."""
    segs = stretches(beats)[0]
    return (sum(1 for x in segs if x[3][0] == "count" and x[3][1] != 1.0), sum(1 for x in segs if x[3][0] == "even"))


def tidy(beats):
    """The beats with every stretch put on its neighbour's pulse where it was miscounted (see above). -> new beats"""
    b = np.asarray(beats, float)
    segs, _ = stretches(b)
    if not segs:
        return b
    pos = np.zeros(len(b))
    for a, c, t, (kind, k), real in segs:
        pos[a] = round(pos[a])                               # every stretch starts on a whole beat
        if kind == "even":
            n = max(1, int(round((b[c] - b[a]) * real / 60.0)))
            pos[a + 1:c + 1] = pos[a] + n * (b[a + 1:c + 1] - b[a]) / (b[c] - b[a])
        else:
            pos[a + 1:c + 1] = pos[a] + np.arange(1, c - a + 1) / (k or 1.0)
    return np.interp(np.arange(0, int(math.floor(pos[-1] + 1e-9)) + 1, dtype=float), pos, b)


def onset_train(y, sr, step=0.004):
    """(rise of the level in steps of about `step` s, the exact step in s)."""
    d = max(1, int(round(sr * step)))
    y = np.asarray(y, np.float32)
    e = np.abs(y[:len(y) // d * d]).reshape(-1, d).mean(axis=1)
    return np.maximum(0, np.diff(e, prepend=e[:1])), d / float(sr)


def steady_fit(e, step, bpm0, span=0.012, harm=(1, 2, 4), win=16.0, main=None):
    """Does the music keep one tempo from start to end? e: onset train (onset_train) of the music; main: (from s,
    to s) of the stretch the grid is trusted in (default: all of it).

    The onset train is summed at a beat rate and its double / quadruple (the size of the Fourier component: big when
    onsets keep falling at the same place in the beat). Scanned finely around bpm0 for the tempo where the WHOLE
    song adds up best; then the same sum is taken window by window.
    -> (tempo BPM,
        share 0..1 of the windows' pulse that adds up over the whole song - 1 when every window that has the pulse
          has it at the same place in the beat, start to end (nothing slipped, sped up or slowed down),
        cover 0..1 of the music (by onsets) outside the trusted stretch that has this pulse too, as strongly as
          the trusted stretch does - low when the rest of the song runs at another tempo,
        how many times the peak stands above the scan's median)."""
    e = np.asarray(e, float)
    blk = max(1, int(round(0.25 / step)))
    nb = len(e) // blk
    nw = max(1, int(round(win / 0.25)))
    if nb < 4 * nw or e.sum() <= 0:
        return float(bpm0), 0.0, 0.0, 0.0
    t = np.arange(nb * blk) * step
    tb = (np.arange(nb) + 0.5) * blk * step
    z = [(e[:nb * blk] * np.exp(-2j * np.pi * bpm0 * h / 60.0 * t)).reshape(nb, blk).sum(axis=1) for h in harm]

    def size(bpms):
        m = np.zeros(len(bpms))
        for h, zh in zip(harm, z):
            df = (np.asarray(bpms) - bpm0) * h / 60.0
            for a in range(0, len(df), 64):
                m[a:a + 64] += np.abs(np.exp(-2j * np.pi * np.outer(df[a:a + 64], tb)) @ zh)
        return m

    coarse = np.arange(bpm0 * (1 - span), bpm0 * (1 + span), 0.01)
    mc = size(coarse)
    i = int(np.argmax(mc))
    fine = coarse[i] + np.arange(-0.02, 0.02001, 0.0005)
    bpm = float(fine[int(np.argmax(size(fine)))])
    snr = float(mc[i] / (np.median(mc) + 1e-12))
    starts = range(0, nb - nw + 1, nw)
    tot = np.array([e[a * blk:(a + nw) * blk].sum() for a in starts])
    whole, parts = 0.0, np.zeros(len(tot))
    for h, zh in zip(harm, z):
        c = zh * np.exp(-2j * np.pi * (bpm - bpm0) * h / 60.0 * tb)
        cw = np.array([c[a:a + nw].sum() for a in starts])
        whole += abs(cw.sum())
        parts += np.abs(cw)
    share = float(whole / (parts.sum() + 1e-12))
    mid = (np.array(list(starts)) + nw / 2.0) * 0.25
    sounding = tot > 0.05 * np.median(tot[tot > 0])
    trusted = sounding if main is None else sounding & (mid >= main[0]) & (mid <= main[1])
    if not trusted.any():
        trusted = sounding
    per = parts / (tot + 1e-12)                               # pulse per onset, window by window
    has = sounding & (per >= 0.4 * np.median(per[trusted]))
    rest = sounding & ~trusted
    if tot[rest].sum() < 0.05 * tot[sounding].sum():          # (next to nothing outside it: judge all of it)
        rest = sounding
    cover = float(tot[has & rest].sum() / (tot[rest].sum() + 1e-12))
    return bpm, share, cover, snr


def bar_flags(t, downs, bpb, main, trusted=()):
    """Bar-line flags for new beats t from the old grid's bar lines `downs` (s): the count most of the old bar lines
    fall on inside `main` (from s, to s) - and inside every other `trusted` stretch (counted on the same pulse) the
    count most of ITS bar lines fall on, when they clearly agree: a song that drops or adds a beat somewhere has
    its bar lines right on both sides of it."""
    t = np.asarray(t, float)
    downs = np.asarray(downs, float)

    def count(lo, hi):
        sel = downs[(downs >= lo - 1e-6) & (downs <= hi + 1e-6)]
        if not len(sel) or not len(t):
            return 0, 0, 0
        idx = np.array([int(np.argmin(np.abs(t - x))) for x in sel])
        c = np.bincount(idx % bpb, minlength=bpb)
        return int(c.argmax()), int(c.max()), int(len(sel))
    phase = np.full(len(t), count(*main)[0] if main else count(-1e9, 1e9)[0], int)
    for lo, hi in trusted:
        ph, top, n = count(lo, hi)
        if n >= 4 and top >= 0.7 * n:
            phase[(t >= lo - 1e-6) & (t < hi - 1e-6)] = ph
    return [1 if (i - int(phase[i])) % bpb == 0 else 0 for i in range(len(t))]


def steady_marks(beats, downs, bpb, a, c, bpm, t0, t1, trusted=()):
    """A steady grid at `bpm` through beats[a..c] (the stretch that is counted right), over [t0, t1] s.
    -> ({t, d, s, bpb}, median s the new marks sit from the old ones inside that stretch)."""
    b = np.asarray(beats, float)
    p = 60.0 / float(bpm)
    ph = np.exp(2j * np.pi * b[a:c + 1] / p)                 # where in the beat each trusted mark sits; the grid goes
    off = float(np.angle(ph.mean()) / (2 * np.pi) * p)        # through where most of them do (a dropped or doubled
    k = np.arange(math.ceil((t0 - off) / p - 1e-9), math.floor((t1 - off) / p + 1e-9) + 1)      # beat changes nothing)
    t = off + k * p
    agree = float(np.median(np.abs((b[a:c + 1] - off + p / 2) % p - p / 2)))
    d = bar_flags(t, downs, bpb, (float(b[a]), float(b[c])), trusted)
    return dict(t=[round(float(x), 5) for x in t], d=d, s=[1] * len(t), bpb=int(bpb)), agree


def _music_train(S):
    """Onset train of the song's music (drums + bass + other stems), or (None, None)."""
    import soundfile as sf
    from . import library
    e = step = None
    for stem in ("drums", "bass", "other"):
        if not S.has_stem(stem):
            continue
        files = sorted((library.song_dir(S.id) / "stems").glob(stem + ".*"))
        if not files:
            continue
        y, sr = sf.read(str(files[0]), dtype="float32", always_2d=True)
        x, step = onset_train(y.mean(axis=1), sr)
        if e is None:
            e = x
        else:
            n = min(len(e), len(x))
            e = e[:n] + x[:n]
    return e, step


def _fixed_grid(S, info, drums):
    """(marks, note) when the song's detector grid changes tempo along the way and there is something to put right
    (see above), else None. For this mix only. Kept on S (the marks, not the audio)."""
    ib = info["beats"]
    key = ("grid_fix", len(ib), round(float(ib[0]), 3) if len(ib) else 0, round(float(ib[-1]), 3) if len(ib) else 0)
    if key in S._bar:
        return S._bar[key]
    out = None
    try:
        from . import beatfix
        dur = float(info.get("duration") or ib[-1])
        det = np.array([x for x in ib if -1e-6 <= x <= dur + 1e-6])
        bpb = int(info.get("bpb") or 4)
        downs = [ib[j] for j in info.get("bars") or [] if 0 <= j < len(ib) and -1e-6 <= ib[j] <= dur + 1e-6]
        segs, main = stretches(det)
        if len(segs) > 1:
            a, c, bpm0 = segs[main][:3]
            span = (float(det[a]), float(det[c]))
            same = [(float(det[x[0]]), float(det[x[1]])) for i, x in enumerate(segs) if x[3] == ("count", 1.0) and i != main]
            own = " (the song's own grid is unchanged - the Beat grid window is where to fix it for good)"
            e, step = _music_train(S)
            if e is not None:
                bpm, share, cover, snr = steady_fit(e, step, bpm0, main=span)
                if share >= STEADY_SHARE and cover >= STEADY_COVER and snr >= STEADY_SNR:
                    marks, agree = steady_marks(det, downs, bpb, a, c, bpm, 0.0, dur, same)
                    if agree <= STEADY_AGREE and len(marks["t"]) >= 8:
                        out = (marks, f"its beat grid changes tempo {len(segs) - 1} time{'s' if len(segs) != 2 else ''}, but "
                                      f"the music keeps one tempo ({bpm:.2f} BPM) from start to end: a steady grid is used "
                                      "for this mix" + own)
            n, lost = needs_tidy(det)
            if out is None and (n or lost):
                new = tidy(det)
                y, sr = drums()
                if y is not None and len(new) > 20:
                    new = beatfix._on_hits(new, y, sr)
                if len(new) >= 8:
                    marks = dict(t=[round(float(x), 5) for x in new], d=bar_flags(new, downs, bpb, span, same), s=[1] * len(new), bpb=bpb)
                    if sum(marks["d"]) >= 2:
                        bits = ([f"counted {n} stretch{'es' if n != 1 else ''} in half / double time or 3 against 2"] if n else []) + \
                               ([f"lost the beat in {lost} stretch{'es' if lost != 1 else ''}"] if lost else [])
                        out = (marks, "its beat grid " + " and ".join(bits) + ": put back on the pulse of the stretch next "
                                      "to it for this mix" + own)
    except Exception:
        out = None
    S._bar[key] = out
    return out


def hits_vs_marks(y, sr, marks_t, t0=0.0, t1=1e9):
    """Seconds a song's drum hits fall after its beat marks: for every mark in [t0, t1) the strongest onset within
    HIT_REACH, the median over the marks that have a real hit there. None when too few do, or they scatter.

    Marks are placed by ear or by the detector, a few ms either side of the hit and by a different amount in every
    song (measured on four library songs: -13 .. +13 ms, each song steady within a few ms). Lining two songs up
    mark to mark leaves that difference as a flam; lining up their hits doesn't."""
    y = np.asarray(y, np.float32)
    step = max(1, int(sr // 2000))                           # 0.5 ms
    n = len(y) // step
    if n < 2000:
        return None
    e = np.abs(y[:n * step]).reshape(n, step).mean(axis=1)
    k = np.hanning(7)
    e = np.convolve(e, k / k.sum(), "same")
    on = np.maximum(0, np.diff(e, prepend=e[0]))
    top = float(on.max())
    if top <= 0:
        return None
    thr = max(float(np.percentile(on, 99.0)), 0.05 * top)     # a real hit: among the loudest 1 % of moments
    w = int(HIT_REACH * sr / step)
    got = []
    for t in marks_t:
        c = int(round(t * sr / step))
        if not (t0 <= t < t1) or c - w < 0 or c + w >= n:
            continue
        seg = on[c - w:c + w + 1]
        j = int(np.argmax(seg))
        if seg[j] >= thr:
            got.append((j - w) * step / sr)
    if len(got) < 24:
        return None
    q1, med, q3 = np.percentile(got, [25, 50, 75])
    if q3 - q1 > 0.015:
        return None
    return float(np.clip(med, -HIT_MAX, HIT_MAX))


def _song_hit(S, marks_t, t0, t1, drums=None):
    """hits_vs_marks for a library song's drum stem (worked out once per song and grid, kept on S)."""
    key = ("grid_hit", len(marks_t), round(float(marks_t[0]), 3) if len(marks_t) else 0,
           round(float(marks_t[-1]), 3) if len(marks_t) else 0)
    if key not in S._bar:
        hit = None
        try:
            y, sr = (drums or _drum_reader(S))()
            if y is not None:
                hit = hits_vs_marks(y, sr, marks_t, t0, t1)
        except Exception:
            hit = None
        S._bar[key] = hit
    return S._bar[key]


def _coarse_phrases(on, beats):
    """Lines from per-beat vocal presence, for songs without melody features: each run of sounding beats."""
    on = np.asarray(on, bool)
    out, a = [], None
    for i, v in enumerate(np.r_[on, False]):
        if v and a is None:
            a = i
        elif not v and a is not None:
            if i - a >= 2:
                out.append((float(beats[a]), float(beats[min(i, len(beats) - 1)]), ""))
            a = None
    return out


# ------------------------------------------------------------------ tempo: how a layer's beats meet the master's

def tempo_fit(L, bpm, master_bpb, fit="auto"):
    """How song L runs against a master at `bpm` with `master_bpb` beats a bar.

    -> dict(M, fit, stretch, mult, per_bar, unit, slots): M = project beats one of L's (marked) beats spans; fit
    "beat" (beats line up) or "bar" (bars line up, beats in a cross-rhythm); stretch = speed change (+ = faster);
    mult = what the project stores for the song; per_bar = L's bars per master bar (0.5 / 1 / 2 when its bars
    simply run on, else None: every bar is re-metered beat by beat, `unit` of L's bars at a time, which hold
    `slots` project beats)."""
    cands = [(m, "beat") for m in BEAT_M]
    if L.bpb != master_bpb:
        for rho in BARS_PER:
            m = master_bpb / (L.bpb * rho)
            if all(abs(m - x) > 1e-6 for x, _ in cands) and 0.2 <= m <= 4.0:
                cands.append((m, "bar"))
    if fit in ("beat", "bar") and any(c[1] == fit for c in cands):
        cands = [c for c in cands if c[1] == fit]

    def cost(c):
        m, kind = c
        return abs(math.log(bpm / (L.beat_bpm * m))) + (BAR_BIAS if kind == "bar" else 0.0)
    M, kind = min(cands, key=cost)
    rho = master_bpb / (L.bpb * M)
    per_bar = next((x for x in BARS_PER if abs(rho - x) < 0.02), None)
    q = L.bpb * M                                            # project-beat slots in one of L's bars
    unit = 1 if abs(q - round(q)) < 1e-6 else 2             # (an odd bar at double speed: two bars make a unit)
    return dict(M=M, fit=kind, stretch=bpm / (L.beat_bpm * M) - 1.0, mult=M * L.ratio, per_bar=per_bar,
                unit=unit, slots=int(round(q * unit)))


def beat_map(q, n):
    """Which of a bar's q beat slots plays on each of n master beats: the tail is dropped, or inner beats are
    repeated (3 -> 4 gives 0 1 2 1, so the bar still starts on its own 1 and never repeats it)."""
    q, n = max(1, int(q)), max(1, int(n))
    if q >= n:
        return list(range(n))
    return list(range(q)) + [1 + (i % max(q - 1, 1)) if q > 1 else 0 for i in range(n - q)]


# ------------------------------------------------------------------ the master grid

class Grid:
    """The master song laid on the project: bar k's line at G[k] project beats, its beats at beats[k]."""

    def __init__(self, Mst, mult, lead=0.0):
        self.M = Mst
        self.spb = (60.0 / Mst.info_bpm) / mult            # source seconds per project beat (= real seconds)
        self.ks = Mst.content_bars()
        if len(self.ks) < 2:
            raise ValueError(f"{Mst.title}: too few bars")
        self.t_start = float(Mst.bar_t[self.ks[0]])
        self.lead = float(lead)
        self.G = np.array([self.pos(Mst.bar_t[k]) for k in self.ks] + [self.pos(Mst.bar_t[self.ks[-1] + 1])])
        self.beats = []
        for k in self.ks:
            a, n = Mst.bars[k]
            self.beats.append(np.array([self.pos(Mst.mark_time(a + i)) for i in range(n + 1)]))
        self.total = float(self.G[-1])
        self.sec = [Mst.section_at(Mst.bar_t[k] + 1e-3) for k in self.ks]
        self.sec_start = [i == 0 or self.sec[i] is not self.sec[i - 1] for i in range(len(self.ks))]

    def pos(self, t):
        return (float(t) - self.t_start) / self.spb + self.lead

    def time(self, u):
        return (float(u) - self.lead) * self.spb + self.t_start

    def n(self):
        return len(self.ks)

    def all_beats(self):
        return np.unique(np.concatenate([b[:-1] for b in self.beats] + [self.G[-1:]]))


# ------------------------------------------------------------------ bar-locked layers (drums, bass, instruments)

def pair_bars(grid, L, stems, want, step=1.0, in_order=False):
    """Which of L's bars each master bar plays: [bar position or None], one per master bar.

    in_order (a layer that brings music - bass / other - or that IS the backing because the master brings none):
    L's bars simply run on from its first sounding bar, round again when they run out; nothing is skipped, so the
    song keeps its own shape. Otherwise (drums under the master's music): bars none of `stems` sounds in are
    skipped, and at a master section change the layer jumps to one of its own sections of matching energy unless
    the bars it would play next already match."""
    cb = L.content_bars()
    if not cb:
        return [None] * grid.n()
    lo, hi = cb[0], cb[-1] + 1

    def usable(x):
        if not (lo <= x and x + step <= hi + 1e-6):
            return False
        t0, t1 = L.bar_time(x), L.bar_time(x + step)
        return max(L.presence(s, t0, t1) for s in stems) >= PRESENT

    if in_order:
        x = next((float(k) for k in range(lo, hi) if usable(float(k))), float(lo))
        out = []
        for i in range(grid.n()):
            if x + step > hi + 1e-6:
                x = float(lo)
            out.append(x if want[i] else None)
            x += step
        return out

    secs = [s for s in L.sections if s.get("name") not in ("silence",)]
    L_e = np.array([s["energy"] for s in secs] or [0.5])
    M_e = np.array([s["energy"] for s in grid.M.sections] or [0.5])
    rank = lambda e, arr: float(np.mean(arr <= e + 1e-9))

    def first_usable(s):
        k = int(np.searchsorted(L.bar_t, s["t0"] - 1e-3))
        x = float(max(lo, k))
        while x + step <= hi + 1e-6 and L.bar_t[int(x)] < s["t1"] - 1e-3:
            if usable(x):
                return x
            x += max(step, 1.0)
        return None

    starts = [(rank(s["energy"], L_e), first_usable(s)) for s in secs]
    starts = [(q, x) for q, x in starts if x is not None]

    def advance(x):
        y = x + step
        for _ in range(4 * (hi - lo) + 8):
            if y + step > hi + 1e-6:
                y = float(lo)                                   # ran out: round again from the top
            if usable(y):
                return y
            y += max(step, 1.0)
        return None

    out, cur = [], None
    for i in range(grid.n()):
        if not want[i]:
            out.append(None)
            continue
        q_m = rank(grid.sec[i]["energy"], M_e)
        nxt = advance(cur) if cur is not None else None
        fresh = cur is None or grid.sec_start[i] or (i and not want[i - 1])
        if nxt is not None and fresh and starts:
            q_n = rank(L.section_at(L.bar_time(nxt) + 1e-3)["energy"], L_e)
            if abs(q_n - q_m) > 0.34:
                nxt = None                                      # what comes next is the wrong size: jump
        if nxt is None:
            nxt = min(starts, key=lambda s: (abs(s[0] - q_m), s[1]))[1] if starts else None
        out.append(nxt)
        cur = nxt if nxt is not None else cur
    return out


def layer_pieces(grid, L, tf, pairs):
    """Pieces [(U0, U1, source time at U0)] for a bar-locked layer: per master bar where its bars run on, per
    master beat where every bar is re-metered."""
    M, per_bar = tf["M"], tf["per_bar"]
    out = []
    if per_bar:
        for i, x in enumerate(pairs):
            if x is not None:
                out.append((float(grid.G[i]), float(grid.G[i + 1]), L.bar_time(x)))
        return out
    for i, x in enumerate(pairs):
        if x is None:
            continue
        g = grid.beats[i]
        bm = beat_map(tf["slots"], len(g) - 1)
        a, _ = L.bars[int(x)]
        for j, s in enumerate(bm):
            out.append((float(g[j]), float(g[j + 1]), L.mark_time(a + s / M)))
    return out


def merge_pieces(pieces, spb, tol=DRIFT):
    """follow = "cut": join pieces that simply carry on -> [(U0, U1, t0, None)], the audio running unbroken from t0.
    A piece is joined while the moment it asks for is within `tol` s of where the running clip already is - so a
    steady layer is one long clip, and a drifting one is re-anchored (a small jump in its audio) when the slide
    would be heard. Nothing is ever stretched."""
    out = []
    for u0, u1, t in pieces:
        if out:
            a0, a1, at, _ = out[-1]
            if abs(u0 - a1) < 1e-6 and abs(t - (at + (a1 - a0) * spb)) <= tol:
                out[-1] = (a0, u1, at, None)
                continue
        out.append((u0, u1, t, None))
    return out


def runs_of(pieces, spb):
    """Pieces grouped into runs whose audio carries on: the next piece starts about where this one ends (at any
    speed within RATE_SPAN of the song's own)."""
    runs = []
    for u0, u1, t in pieces:
        if runs:
            a0, a1, at = runs[-1][-1]
            rate = (t - at) / max((a1 - a0) * spb, 1e-9)
            if abs(u0 - a1) < 1e-6 and RATE_SPAN[0] <= rate <= RATE_SPAN[1]:
                runs[-1].append((u0, u1, t))
                continue
        runs.append([(u0, u1, t)])
    return runs


def fit_points(U, T, tol):
    """The fewest straight stretches through the points (project beat U[i] <-> source second T[i]) that pass every
    point within `tol` s: indices of the points kept as corners (greedy, always the first and the last)."""
    U, T = np.asarray(U, float), np.asarray(T, float)
    keep, i, n = [0], 0, len(U)
    while i < n - 1:
        j = i + 1
        while j + 1 < n:
            k = np.arange(i + 1, j + 1)
            line = T[i] + (T[j + 1] - T[i]) * (U[k] - U[i]) / (U[j + 1] - U[i])
            if np.max(np.abs(T[k] - line)) > tol:
                break
            j += 1
        keep.append(j)
        i = j
    return keep


def calm(x, width=7):
    """A running median: marks are placed a few ms either side of where the music is, each one differently, and a
    layer that followed every one of them would wobble. The slow slide between two songs passes through unchanged,
    a real jump stays a jump; one stray mark is ignored."""
    x = np.asarray(x, float)
    if len(x) < width:
        return x
    n, h = len(x), width // 2
    out = []
    for i in range(n):
        k = min(h, i, n - 1 - i)                             # the same reach both ways, so a steady slide is untouched
        out.append(np.median(x[i - k:i + k + 1]))
    return np.array(out)


def ease(keep, U, T, spb, wide, lurch=None, short=2):
    """Corners (fit_points) with the lurches taken out: where a stretch of one or two bars would need a speed more
    than LURCH away from the stretch beside it, the two become one - as long as every bar line in it stays within
    `wide` s. A layer following a song whose bars alternate long and short (an odd meter the grid counts in 4), or
    one mark that sits wrong, then keeps an even speed and meets the master every few bars, instead of speeding
    up and slowing down bar by bar. A change of tempo that lasts is still followed."""
    lurch = LURCH if lurch is None else lurch
    keep = list(keep)
    rate = lambda i, j: (T[j] - T[i]) / ((U[j] - U[i]) * spb)
    again = True
    while again and len(keep) > 2:
        again = False
        for x in range(1, len(keep) - 1):
            i, k, j = keep[x - 1], keep[x], keep[x + 1]
            if min(k - i, j - k) > short or abs(rate(i, k) - rate(k, j)) <= lurch:
                continue
            idx = np.arange(i + 1, j)
            line = T[i] + (T[j] - T[i]) * (U[idx] - U[i]) / (U[j] - U[i])
            if np.max(np.abs(T[idx] - line)) <= wide:
                del keep[x]
                again = True
                break
    return keep


def follow_runs(pieces, spb, tol=DRIFT, wide=0.0):
    """follow = "stretch": -> [(U0, U1, t0, stretches or None)], one item per run of audio that carries on.

    A run that stays within `tol` s of the master on its own is left exactly as recorded (stretches None). One
    that slides is NOT cut: it gets as few constant-speed stretches [(source s from, to, speed)] as keep every
    bar line within `tol` - the song follows the master's tempo the way a warp would, with no jump in its audio.
    wide: how far (s) a bar line may sit off to spare the layer a lurch (see ease). Only where a stretch would have
    to leave RATE_SPAN are those bars cut instead (re-anchored bar by bar); the bars either side still follow."""
    out = []
    for run in runs_of(pieces, spb):
        U = np.array([p[0] for p in run] + [run[-1][1]])
        T = [p[2] for p in run]
        last = 1.0
        if len(run) > 1:
            last = float(np.clip((T[-1] - T[-2]) / ((U[-2] - U[-3]) * spb), *RATE_SPAN))
        T = np.array(T + [T[-1] + (U[-1] - U[-2]) * spb * last])
        T = (T[0] + (U - U[0]) * spb) + calm(T - (T[0] + (U - U[0]) * spb))
        if np.max(np.abs(T - (T[0] + (U - U[0]) * spb))) <= tol:
            out.append((float(U[0]), float(U[-1]), float(T[0]), None))
            continue
        keep = fit_points(U, T, tol)
        if wide > tol:
            keep = ease(keep, U, T, spb, wide)
        rates = [(T[j] - T[i]) / ((U[j] - U[i]) * spb) for i, j in zip(keep[:-1], keep[1:])]
        ok = [RATE_SPAN[0] <= r <= RATE_SPAN[1] for r in rates]
        k = 0
        while k < len(rates):
            m = k
            while m < len(rates) and ok[m] == ok[k]:
                m += 1
            i, j = keep[k], keep[m]
            if ok[k]:
                st = [(float(T[keep[x]]), float(T[keep[x + 1]]), float(rates[x])) for x in range(k, m) if abs(rates[x] - 1.0) > 1e-4]
                out.append((float(U[i]), float(U[j]), float(T[i]), st or None))
            else:
                out += merge_pieces(run[i:j], spb, tol)          # too far to stretch: these bars are cut instead
            k = m
    return out


def lead_in(items, spb, lead_s=LEAD):
    """Start every anchored clip a few ms before its line (and end the one before it there), so the edge fade
    falls just ahead of the hit instead of on it."""
    lb = lead_s / spb
    out = []
    for u0, u1, t, st in items:
        room = min(lb, max(0.0, u0))
        if out and out[-1][1] > u0 - room - 1e-9:
            if out[-1][1] - room - out[-1][0] > 4 * lb:
                out[-1] = (out[-1][0], u0 - room) + out[-1][2:]
            else:
                room = max(0.0, u0 - out[-1][1])
        out.append((u0 - room, u1, t - room * spb, st))
    return out


def split_at(u0, u1, t, cuts):
    """One as-recorded clip cut at the master's section lines (each part carries on exactly where the last stopped).
    -> [(a, b, t, beats from the clip's start, is the first part, is the last part)]"""
    edges = [u0] + [c for c in cuts if u0 + 0.5 < c < u1 - 0.5] + [u1]
    return [(a, b, t, a - u0, a == u0, b == u1) for a, b in zip(edges[:-1], edges[1:])]


def src_at(u0, t0, stretches, u, spb):
    """The source second an item that starts at project beat u0 / source second t0 has reached at project beat u:
    as recorded (spb source seconds a beat), except inside its stretches [(source s from, to, speed)]."""
    cu, ct = float(u0), float(t0)
    for ta, tb, r in stretches or ():
        if tb <= ct:
            continue
        if ta > ct:
            du = (ta - ct) / spb
            if u <= cu + du:
                return ct + (u - cu) * spb
            cu, ct = cu + du, ta
        du = (tb - ct) / (spb * r)
        if u <= cu + du:
            return ct + (u - cu) * spb * r
        cu, ct = cu + du, tb
    return ct + (u - cu) * spb


def warp_list(L, stretches, stem):
    """Stretches [(source s from, to, speed)] as clip["warps"] (warps.py). A small speed change is plain varispeed
    (the pitch moves with it by a few cents; exact, nothing to render), a bigger one keeps the pitch."""
    vari = VARI.get(stem, VARI_PITCHED)
    return [dict(a=round(L.src_index(ta), 6), b=round(L.src_index(tb), 6), r=round(r, 6), kp=bool(abs(r - 1.0) > vari))
            for ta, tb, r in stretches]


# ------------------------------------------------------------------ vocals: whole lines on bar lines

def line_anchor(L, t):
    """The bar line (s) a line starting at t belongs to: the nearest one - a pickup stays ahead of its bar, a line
    starting on beat 2 stays there."""
    return float(L.bar_t[int(np.argmin(np.abs(L.bar_t - t)))])


def lines_of(L):
    """L's lines [(t0, t1, words, parts)] with the ones that run straight on joined into one: a long stretch of
    singing that was split for the DJ planner (djremix.PH_MAX_BARS), or two lines with no real breath between them.
    Kept apart they could land on different bars, which cuts the singer off mid-phrase. parts: the pieces it was
    joined from [(t0, t1)] - the only places inside it where the singer eases off at all."""
    out = []
    for ph in sorted(L.phrases or []):
        t0, t1, words = ph[0], ph[1], ph[2]
        if t1 - t0 < 0.05:
            continue
        if out and t0 - out[-1][1] <= GLUE:
            out[-1] = (out[-1][0], max(out[-1][1], t1), (out[-1][2] + " " + words).strip(), out[-1][3] + [(t0, t1)])
        else:
            out.append((t0, t1, words, [(t0, t1)]))
    return out


def vocal_blocks(L):
    """L's lines grouped into blocks [[(t0, t1, words, anchor s, parts)]]: lines less than BLOCK_GAP_BARS apart."""
    bar = L.period * L.bpb
    blocks = []
    for t0, t1, words, parts in lines_of(L):
        item = (t0, t1, words, line_anchor(L, t0), parts)
        if blocks and t0 - blocks[-1][-1][1] <= BLOCK_GAP_BARS * bar:
            blocks[-1].append(item)
        else:
            blocks.append([item])
    return blocks


def block_kind(L, block):
    """"hook" or "verse": a hook is mostly lines the singer repeats (djremix.phrases counts how many lines share a
    line's words), or sits in one of the song's own chorus / drop sections."""
    a, b = block[0][0], block[-1][1]
    rep = tot = 0.0
    for ph in L.phrases or []:
        o = min(b, ph[1]) - max(a, ph[0])
        if o > 0:
            tot += o
            if len(ph) > 3 and (ph[3] or 1) >= 2:
                rep += o
    if tot > 0 and rep / tot >= 0.5:
        return "hook"
    return "hook" if L.section_at(0.5 * (a + b)).get("name") in HOOK_SECTIONS else "verse"


def vocal_zone(grid, own_vocal):
    """(bars a block may start in, bars a line may sound in). Where the master's own singer was (when that vocal
    isn't in the mix), else every section that isn't an intro / outro; under the master's own vocal: its gaps."""
    Mst, n = grid.M, grid.n()
    bar = lambda i: (Mst.bar_t[grid.ks[i]], Mst.bar_t[grid.ks[i] + 1])
    busy = np.array([bool(own_vocal and Mst.presence("vocals", *bar(i)) >= PRESENT) for i in range(n)])
    body = np.array([grid.sec[i].get("name") not in SKIP_SECTIONS for i in range(n)])
    if not body.any():
        body[:] = True
    start = body & ~busy
    if not own_vocal and Mst.plays("vocals"):
        sung = np.array([float(grid.sec[i].get("stems", {}).get("vocals", 0.0)) >= 0.25 for i in range(n)])
        if sung.mean() >= 0.2:
            start = sung
    return start, ~busy


def _put_block(grid, L, tf, block, k, beats, sound_ok, limit=None):
    """One block with its first line's bar line on master bar k. Every line sits on the master beat (same meter)
    or bar line (another meter) nearest where it would fall, and moves on a bar at a time until it's clear of the
    line before it. A line that would pass the end, or sound over the master's own singer, is left out whole. The
    block stops at the first line that would pass `limit`: that line and the ones after it come back as `rest`.
    A long stretch of singing that was joined from several pieces (lines_of) may stop at the last of its own joins
    that still fits, once at least MIN_PART_BARS of it have been sung - the rest of it heads `rest`. Never anywhere
    else inside a line.
    -> ([(U0, U1, t0, words, anchor U, anchor s)], lines left out, where the block ends, rest)."""
    G, n = grid.G, grid.n()
    spb = (60.0 / L.info_bpm) / tf["mult"]
    out, lost, rest = [], 0, []
    prev_u = prev_anchor = prev_end = None
    for j_line, (t0, t1, words, anc, parts) in enumerate(block):
        want = G[k] if prev_u is None else prev_u + (anc - prev_anchor) / spb
        if tf["per_bar"]:
            u = float(beats[int(np.argmin(np.abs(beats - want)))])
            if abs(u - want) > 0.3:
                u = float(want)
        else:
            u = float(G[int(np.argmin(np.abs(G - want)))])
        a, b = u - (anc - t0) / spb, u + (t1 - anc) / spb
        while prev_end is not None and a < prev_end + 0.05:       # clear of the line before: a bar on
            j = max(1, int(np.searchsorted(G, u + 1e-6, side="right")))
            if j >= len(G):
                break
            step = float(G[j] - G[j - 1])
            u, a, b = u + step, a + step, b + step
        if limit is not None and b > limit + 1e-6 and b <= grid.total + 1e-6:
            rest = list(block[j_line:])
            stop = None
            for (p0, p1), (q0, q1) in zip(parts, parts[1:]):
                if u + (p1 - anc) / spb <= limit + 1e-6 and p1 - t0 >= MIN_PART_BARS * L.period * L.bpb:
                    stop = (p1, q0)
            if stop:
                b = u + (stop[0] - anc) / spb
                i0 = int(np.clip(np.searchsorted(G, a + 0.25, side="right") - 1, 0, n - 1))
                i1 = int(np.clip(np.searchsorted(G, b - 0.25, side="left"), i0 + 1, n))
                if sound_ok[i0:i1].all():
                    out.append((a, b, t0, words, u, anc))
                    prev_end = b
                    later = (stop[1], t1, words, line_anchor(L, stop[1]), [x for x in parts if x[0] >= stop[1] - 1e-6])
                    rest = [later] + list(block[j_line + 1:])
            break
        if b > grid.total + 1e-6 or (a < -1e-6 and prev_u is not None):
            lost += 1
            continue
        i0 = int(np.clip(np.searchsorted(G, a + 0.25, side="right") - 1, 0, n - 1))
        i1 = int(np.clip(np.searchsorted(G, b - 0.25, side="left"), i0 + 1, n))
        if not sound_ok[i0:i1].all():
            lost += 1
            continue
        out.append((a, b, t0, words, u, anc))
        prev_u, prev_anchor, prev_end = u, anc, b
    return out, lost, prev_end, rest


def place_vocals(grid, layers, own_vocal=False, turns=True):
    """layers: [(L, tempo fit)] -> ({song id: [(U0, U1, t0, words, anchor U, anchor s)]}, {song id: lines left
    out}, {song id: "chorus" | "verses"} when the singers take turns).

    One singer: its blocks in order, each starting on a bar two bars apart from its section's start, the breaks
    between blocks kept (up to MAX_GAP_BARS).
    Several singers, turns on: the master's chorus / drop sections get the hook of ONE of them - the singer with
    the most repeated lines - every time (a hook may repeat), and its other sections get the other singers'
    verses, a new singer each time, each singer's verses in their own order (when they run out, the hook's singer
    has verses too). Sections of one kind that follow each other count as one stretch. A block stays inside its
    stretch (it may run one bar over): a verse that doesn't fit carries on, from the line it stopped at, the next
    time it is that singer's turn - no line is ever shortened.
    Several singers when that can't be done (no chorus in the master, nobody has a hook), or turns off: blocks
    are dealt out in turn, back to back."""
    start_ok, sound_ok = vocal_zone(grid, own_vocal)
    n, G = grid.n(), grid.G
    sec_first = {}
    for i in range(n):
        sec_first.setdefault(id(grid.sec[i]), i)
    allowed = [i for i in range(n) if start_ok[i] and (i - sec_first[id(grid.sec[i])]) % 2 == 0]
    heads = [i for i in allowed if grid.sec_start[i]] or allowed
    beats = grid.all_beats()
    queues = [(L, tf, vocal_blocks(L)) for L, tf in layers]
    fresh = lambda: ({L.id: [] for L, _, _ in queues}, {L.id: 0 for L, _, _ in queues})
    placed, dropped = fresh()
    parts = {}
    pre = lambda L, tf, block: max(0.0, (block[0][3] - block[0][0]) / ((60.0 / L.info_bpm) / tf["mult"]))
    bar_len = (grid.total - grid.lead) / max(1, n)

    # ---- several singers taking turns by section
    secs, i = [], 0
    while i < n:
        j = i + 1
        while j < n and not grid.sec_start[j]:
            j += 1
        if start_ok[i:j].any():
            chorus = grid.sec[i].get("name") in HOOK_SECTIONS
            if secs and secs[-1][1] == i and secs[-1][2] == chorus:
                secs[-1] = (secs[-1][0], j, chorus)              # the same kind again: one stretch
            else:
                secs.append((i, j, chorus))
        i = j
    kinds = {L.id: [block_kind(L, b) for b in blocks] for L, _, blocks in queues}
    hook_s = {L.id: sum(b[-1][1] - b[0][0] for b, k in zip(blocks, kinds[L.id]) if k == "hook") for L, _, blocks in queues}
    if turns and len(queues) > 1 and any(c for _, _, c in secs) and not all(c for _, _, c in secs) and max(hook_s.values()) > 0:
        ci = max(range(len(queues)), key=lambda q: hook_s[queues[q][0].id])
        C, Ctf, Cblocks = queues[ci]
        hooks = [b for b, k in zip(Cblocks, kinds[C.id]) if k == "hook"]
        order = [q for q in range(len(queues)) if q != ci]               # the verses go round the other singers
        verses = {q: [list(b) for b, k in zip(queues[q][2], kinds[queues[q][0].id]) if k == "verse" or q != ci]
                  for q in range(len(queues))}
        cursor, first, hi, turn = 0.0, True, 0, 0
        for i0, i1, chorus in secs:
            limit = float(G[i1]) + bar_len
            who = None
            for _ in range(64):
                if chorus:
                    L, tf, block = C, Ctf, hooks[hi % len(hooks)]
                else:
                    if who is not None and not verses[who]:
                        break                                    # this singer is done: nobody else starts mid-stretch
                    if who is None:
                        who = next((order[(turn + s) % len(order)] for s in range(len(order))
                                    if verses[order[(turn + s) % len(order)]]), None)
                        if who is None and verses[ci]:
                            who = ci                             # they have run out: the hook's singer has verses too
                    if who is None:
                        break
                    L, tf, block = queues[who][0], queues[who][1], verses[who][0]
                room = 0.0 if first else pre(L, tf, block)
                k = next((x for x in allowed if i0 <= x < i1 and G[x] - room >= cursor - 1e-6), None)
                if k is None:
                    break
                got, lost, end, rest = _put_block(grid, L, tf, block, k, beats, sound_ok, limit)
                if not got:
                    break
                first = False
                placed[L.id] += got
                dropped[L.id] += lost
                cursor = end
                if chorus:
                    hi += 1
                elif rest:
                    verses[who][0] = rest                        # the verse carries on here at this singer's next turn
                else:
                    verses[who].pop(0)
                if rest or float(G[i1]) - end < 4 * bar_len:     # room for another block only if a good part is left
                    break
            if not chorus and who in order:
                turn = (order.index(who) + 1) % len(order)
        parts = {queues[q][0].id: ("chorus" if q == ci else "verses") for q in range(len(queues))}
        # taking turns only stands if it works out: every singer gets to sing, and not much less of them is heard
        # than when their lines are simply dealt out (long lines don't fit short sections)
        turn_plan = (placed, dropped, parts)
        placed, dropped = fresh()
        parts = {}
        queues = [(L, tf, vocal_blocks(L)) for L, tf in layers]
    else:
        turn_plan = None

    # ---- one singer, or blocks dealt out in turn
    cursor, first = 0.0, True                                   # project beat the next block may start at
    while any(q[2] for q in queues):
        for L, tf, blocks in queues:
            if not blocks:
                continue
            block = blocks.pop(0)
            pool = heads if first else allowed
            room = 0.0 if first else pre(L, tf, block)           # the very first pickup may open the mix early
            k = next((i for i in pool if G[i] - room >= cursor - 1e-6), None)
            if k is None:
                k = next((i for i in allowed if G[i] - room >= cursor - 1e-6), None)
            if k is None:
                dropped[L.id] += len(block)
                continue
            first = False
            got, lost, end, _ = _put_block(grid, L, tf, block, k, beats, sound_ok)
            placed[L.id] += got
            dropped[L.id] += lost
            if end is not None:
                gap = 0.0                                        # one singer: the breaks between blocks are kept
                if blocks and len(queues) == 1:                  # (several take turns back to back)
                    gap = min(MAX_GAP_BARS, max(0.0, (blocks[0][0][0] - block[-1][1]) / (L.period * L.bpb)))
                cursor = end + gap * bar_len
    if turn_plan:
        sung = lambda pl: sum(b - a for v in pl.values() for a, b, *_ in v)
        if all(turn_plan[0][L.id] for L, _, _ in queues) and sung(turn_plan[0]) >= 0.35 * sung(placed):
            return turn_plan
        parts = {L.id: "dealt" for L, _, _ in queues}            # (told apart from "turns were never on the cards")
    return placed, dropped, parts


def follow_line(grid, L, tf, c, tol=VOX_TOL):
    """A placed vocal clip made to follow the master's tempo. Its anchor (c["_anc"]: project beat, source second)
    stays where it is; every bar line of the singer's own song after and before it is tied to the master beat it
    belongs on, and the clip is given the fewest speed changes that keep those within `tol` s. A line the master
    doesn't pull away from is left exactly as recorded."""
    u_a, t_a = c["_anc"]
    spb = (60.0 / L.info_bpm) / tf["mult"]
    B = grid.all_beats()
    i_m = float(P._time_to_beat(u_a, B))
    i_s = float(P._time_to_beat(t_a, L.bt))
    t0 = float(P._beat_to_time(c["src"], L.ib))
    t1 = t0 + c["len"] * spb
    n0 = math.floor((float(P._time_to_beat(t0, L.bt)) - i_s) / L.bpb)
    n1 = math.ceil((float(P._time_to_beat(t1, L.bt)) - i_s) / L.bpb)
    ns = np.arange(n0, max(n1, n0 + 1) + 1) * L.bpb
    T = np.array([L.mark_time(i_s + x) for x in ns])
    U = np.array([float(P._beat_to_time(i_m + x * tf["M"], B)) for x in ns])
    T = t_a + (U - u_a) * spb + calm(T - (t_a + (U - u_a) * spb))
    if np.max(np.abs((U - u_a) * spb - (T - t_a))) <= tol:
        return 0
    keep = ease(fit_points(U, T, tol), U, T, spb, WIDE * L.period * L.bpb)
    Tk, Uk = T[keep], U[keep]
    rates = np.diff(Tk) / (np.diff(Uk) * spb)
    if rates.min() < VOX_SPAN[0] or rates.max() > VOX_SPAN[1]:
        # the grids disagree too much somewhere in the line to follow bar by bar: one speed for the whole line (the
        # anchor stays put) - if that is a gentle one and leaves the line closer than it was
        du, dt_ = U - u_a, T - t_a
        if float(np.sum(du * du)) <= 0:
            return 0
        r = float(np.sum(dt_ * du) / (spb * np.sum(du * du)))
        was, now = np.max(np.abs(du * spb - dt_)), np.max(np.abs(du * spb * r - dt_))
        if not (VOX_SPAN[0] <= r <= VOX_SPAN[1]) or now >= 0.7 * was:
            return 0                                            # leave it as recorded
        Tk, Uk, rates = np.array([t_a, t_a + r * spb]), np.array([u_a, u_a + 1.0]), np.array([r])
    def at(t):                                                  # project beat of source second t (runs on past the ends)
        if t <= Tk[0]:
            return float(Uk[0] + (t - Tk[0]) / (spb * rates[0]))
        if t >= Tk[-1]:
            return float(Uk[-1] + (t - Tk[-1]) / (spb * rates[-1]))
        return float(np.interp(t, Tk, Uk))
    edges = [-1e9] + list(Tk[1:-1]) + [1e9]
    st = [(max(t0, a), min(t1, b), float(r)) for a, b, r in zip(edges[:-1], edges[1:], rates)
          if abs(r - 1.0) > 1e-4 and min(t1, b) - max(t0, a) > 1e-3]
    if not st:
        return 0
    c["start"] = round(max(0.0, at(t0)), 6)
    c["len"] = round(max(0.05, at(t1) - at(t0)), 6)
    c["warps"] = warp_list(L, st, "vocals")
    return len(st)


# ------------------------------------------------------------------ the project

def _short(title):
    return title.split(" - ")[0][:24]


def pick_master(picks, songs):
    """The song that brings chords and bass (they carry the bar structure), else the one bringing the most."""
    def score(p):
        st = set(p["stems"])
        return (len(st & {"other", "bass"}), "other" in st, len(st), -picks.index(p))
    return max(picks, key=score)["song"]


def build_project(songs, picks, master=None, fits=None, shifts=None, gains=None, name=None, opts=None, edges=None):
    """songs {id: Src}; picks [{song, stems}] in the order given; fits {id: "auto" | "beat" | "bar"};
    shifts {id: semitones}; gains {(id, stem): dB}; opts {follow: "stretch" (default) | "cut", turns: bool};
    edges(project): trims the guest vocal clips' edges before they are made to follow the master.
    -> (project, desc)."""
    picks = [dict(song=p["song"], stems=[s for s in STEMS if s in set(p.get("stems") or [])]) for p in picks]
    picks = [p for p in picks if p["stems"] and p["song"] in songs]
    if not picks:
        raise ValueError("pick at least one stem")
    master = master if master in {p["song"] for p in picks} else pick_master(picks, songs)
    Mst = songs[master]
    fits, shifts, gains, opts = fits or {}, shifts or {}, gains or {}, opts or {}
    stretchy = opts.get("follow", "stretch") != "cut"
    m_mult = Mst.ratio
    bpm = round(Mst.info_bpm * m_mult, 4)
    bpb = Mst.bpb
    own = set(next(p["stems"] for p in picks if p["song"] == master))
    led = bool(own & {"other", "bass"})                      # the master brings the music; else a layer is the backing

    tfs = {p["song"]: tempo_fit(songs[p["song"]], bpm, bpb, fits.get(p["song"], "auto")) for p in picks if p["song"] != master}
    # a first line's pickup may start before bar 1: the mix then opens a whole bar (or two) earlier
    lead = 0.0
    voc = [(songs[p["song"]], tfs[p["song"]]) for p in picks if p["song"] != master and "vocals" in p["stems"]
           and songs[p["song"]].phrases]
    grid = Grid(Mst, m_mult)
    placed, dropped, parts = place_vocals(grid, voc, "vocals" in own, opts.get("turns", True)) if voc else ({}, {}, {})
    low = min([x[0] for v in placed.values() for x in v] + [0.0])
    if low < -1e-6:
        bar = float(grid.G[1] - grid.G[0])
        lead = math.ceil(-low / bar) * bar
        grid = Grid(Mst, m_mult, lead)
        placed = {sid: [(a + lead, b + lead, t, w, u + lead, anc) for a, b, t, w, u, anc in v] for sid, v in placed.items()}

    titles = [songs[p["song"]].title for p in picks]
    proj = P.empty(name or " × ".join(dict.fromkeys([Mst.title] + titles)), bpm, bpb)
    proj["songs"][master] = dict(transpose=int(shifts.get(master, 0)), mult=m_mult)
    base = {t["role"]: t for t in proj["tracks"]}
    taken = set()
    cuts = sorted({float(grid.G[i]) for i in range(grid.n()) if grid.sec_start[i]} | {grid.total})

    def lane(sid, stem):
        nm = f"{stem.capitalize()} · {_short(songs[sid].title)}"
        if stem in base and stem not in taken:
            taken.add(stem)
            base[stem]["name"] = nm
            tk = base[stem]
        else:
            tk = P.track(nm, stem)
            proj["tracks"].append(tk)
        tk["gain_db"] = round(float(gains.get((sid, stem), 0.0)), 1)
        return tk

    def add(tk, sid, stem, u0, u1, t, fade_in=0.0, fade_out=0.0, **kw):
        if u1 - u0 < 1e-3:
            return None
        e0, e1 = round(u0, 6), round(u1, 6)                  # the same edge for the clip ending and the one starting
        kw = {k: v for k, v in kw.items() if v}
        c = P.clip(tk["id"], sid, stem, e0, round(songs[sid].src_index(t), 6), round(e1 - e0, 9),
                   fade_in=fade_in, fade_out=fade_out, **kw)
        proj["clips"].append(c)
        return c

    # ---- the master, as recorded: one clip per section, each stem only where it plays
    layers, notes = [], []
    spb_m = grid.spb
    for stem in [s for s in STEMS if s in own]:
        tk = lane(master, stem)
        for i in range(grid.n()):
            if not grid.sec_start[i]:
                continue
            j = next((x for x in range(i + 1, grid.n()) if grid.sec_start[x]), grid.n())
            sec = grid.sec[i]
            if float(sec.get("stems", {}).get(stem, 1.0)) <= 0.05:
                continue
            u0, u1 = float(grid.G[i]), float(grid.G[j])
            add(tk, master, stem, u0, u1, grid.time(u0), sec=sec.get("name"))
    layers.append(dict(song=master, title=Mst.title, role="master", stems=sorted(own, key=STEMS.index), mult=m_mult,
                       stretch=0.0, shift=int(shifts.get(master, 0)), bpb=Mst.bpb))
    if Mst.note:
        notes.append(f"{Mst.title}: {Mst.note}")
    band = [p for p in picks if p["song"] != master and set(p["stems"]) & {"other", "bass"}]
    if own == {"vocals"} and len(band) == 1:
        notes.append(f"{Mst.title} is the master but brings only its vocal, so {songs[band[0]['song']].title}'s whole band is "
                     f"being fitted to a voice. With {songs[band[0]['song']].title} as the master the band plays as recorded "
                     "and the vocal lines are placed on its bars instead")

    # ---- everything else
    vox = []                                                 # (clip, song, tempo fit, reach): guest lines, to follow the master
    for p in picks:
        sid = p["song"]
        if sid == master:
            continue
        L, tf = songs[sid], tfs[sid]
        spb = (60.0 / L.info_bpm) / tf["mult"]
        # hits on hits, not marks on marks: how much later this song's drums fall after its marks than the master's do
        dt = 0.0 if L.hit is None or Mst.hit is None else float(np.clip(L.hit - Mst.hit * spb / spb_m, -HIT_MAX, HIT_MAX))
        proj["songs"][sid] = dict(transpose=int(shifts.get(sid, 0)), mult=round(tf["mult"], 6))
        reach = float(np.clip(2.0 * math.hypot(L.noise, Mst.noise), DRIFT, DRIFT_MAX))      # how closely to follow
        d = dict(song=sid, title=L.title, role="layer", stems=list(p["stems"]), mult=round(tf["mult"], 6), fit=tf["fit"],
                 stretch=round(tf["stretch"] * 100, 1), shift=int(shifts.get(sid, 0)), bpb=L.bpb, clips=0,
                 nudge_ms=round(-dt * 1000.0, 1), follows=0, cuts=0, moves=0, reach_ms=round(reach * 1000.0, 1))
        if L.note:
            notes.append(f"{L.title}: {L.note}")
        locked = [s for s in p["stems"] if s != "vocals"]
        if locked:
            n = grid.n()
            in_order = (not led) or bool(set(locked) & {"other", "bass"})
            want_any = np.zeros(n, bool)
            wants = {}
            for stem in locked:
                bar = lambda i: (Mst.bar_t[grid.ks[i]], Mst.bar_t[grid.ks[i] + 1])
                w = np.ones(n, bool)
                if led and Mst.plays(stem) and stem not in own:      # replacing the master's own: only where that played
                    w = np.array([Mst.presence(stem, *bar(i)) >= PRESENT for i in range(n)])
                    if w.mean() < 0.15:
                        w[:] = True
                wants[stem] = w
                want_any |= w
            step = tf["per_bar"] or float(tf["unit"])
            pairs = pair_bars(grid, L, locked, want_any, step, in_order)
            if not tf["per_bar"]:
                d["beats"] = [b + 1 for b in beat_map(tf["slots"], bpb)]
            # ONE time map for the song: every stem it brings is cut out of the same run of audio, stretched the same
            # way at the same moments - its drums, bass and instruments stay exactly as tight as they were recorded
            pieces = [(u0, u1, t + dt) for u0, u1, t in layer_pieces(grid, L, tf, pairs)]
            items = lead_in(follow_runs(pieces, spb, reach, WIDE * L.period * L.bpb) if stretchy else
                            merge_pieces(pieces, spb, reach), spb)
            for k, (u0, u1, t, st) in enumerate(items):
                d["follows"] += len(st or ())
                if k and abs(u0 - items[k - 1][1]) < 1e-6:
                    p0, p1, pt, pst = items[k - 1]               # the audio jumps with no gap before it: by how much?
                    far = abs(t - src_at(p0, pt, pst, p1, spb)) > 0.5 * L.period * L.bpb
                    d["moves" if far else "cuts"] += 1           # to another part of the song / back into time
            lb = LEAD / spb
            for stem in locked:
                on = []                                          # [from, to) project beats this stem plays
                for i, x in enumerate(pairs):
                    if x is not None and wants[stem][i] and L.presence(stem, L.bar_time(x), L.bar_time(x + step)) >= 0.15:
                        if on and abs(on[-1][1] - float(grid.G[i])) < 1e-6:
                            on[-1][1] = float(grid.G[i + 1])
                        else:
                            on.append([float(grid.G[i]), float(grid.G[i + 1])])
                tk = lane(sid, stem)
                for u0, u1, t, st in items:
                    for oa, ob in on:
                        a_, b_ = (u0 if oa - u0 <= lb + 1e-6 else oa), min(ob, u1)
                        if b_ - a_ < 1e-3 or oa >= u1 or ob <= u0:
                            continue
                        ta = src_at(u0, t, st, a_, spb)
                        if st:
                            tb = src_at(u0, t, st, b_, spb)
                            c = add(tk, sid, stem, a_, b_, ta, fade_out=0.05)
                            if c:
                                w = warp_list(L, [x for x in st if x[1] > ta + 1e-6 and x[0] < tb - 1e-6], stem)
                                if w:
                                    c["warps"] = w
                                d["clips"] += 1
                            continue
                        for a, b, t_, off, head, tail in split_at(a_, b_, ta, cuts):
                            if add(tk, sid, stem, a, b, t_ + off * spb, fade_out=0.05 if tail else 0.0):
                                d["clips"] += 1
            d["bars"] = int(sum(x is not None for x in pairs))
        if "vocals" in p["stems"]:
            tk = lane(sid, "vocals")
            n_lines = 0
            for a, b, t, w, u, anc in placed.get(sid, []):
                c = add(tk, sid, "vocals", max(0.0, a - dt / spb), b - dt / spb, t, fade_out=0.05, line=w[:80])
                if c:
                    c["_anc"] = (u - dt / spb, anc)
                    vox.append((c, L, tf, max(VOX_TOL, reach)))
                    n_lines += 1
            d["lines"] = n_lines
            d["left_out"] = int(dropped.get(sid, 0))
            d["clips"] += n_lines
            if parts.get(sid) in ("chorus", "verses"):
                d["part"] = parts[sid]
            if not L.phrases:
                notes.append(f"{L.title}: no vocal lines found")
            elif d["left_out"]:
                notes.append(f"{L.title}: {d['left_out']} line(s) not used (no room where they belong, or over the master's singer)")
        if abs(tf["stretch"]) > MAX_STRETCH:
            notes.append(f"{L.title}: plays {abs(d['stretch']):.0f}% {'faster' if tf['stretch'] > 0 else 'slower'} than "
                         "recorded to meet the master's tempo - expect to hear it")
        layers.append(d)

    if any(v == "dealt" for v in parts.values()):
        notes.append("Taking turns by section didn't work out with these singers (a singer would have been left out, or most "
                     "of the lines had no section long enough): their lines are dealt out one block after another instead")
    # ---- guest lines: edges first (the app's own breath / tail rules), then follow the master, then tidy up
    if edges:
        edges(proj)
    by_song = {x["song"]: x for x in layers}
    for c, L, tf, tol in vox:
        if stretchy and "_anc" in c:
            by_song[L.id]["follows"] += follow_line(grid, L, tf, c, tol)
    for c in proj["clips"]:
        c.pop("_anc", None)
    for tk in proj["tracks"]:
        if tk["role"] != "vocals":
            continue
        cl = sorted((c for c in proj["clips"] if c["track"] == tk["id"]), key=lambda c: c["start"])
        for a, b in zip(cl, cl[1:]):
            if a["start"] + a["len"] > b["start"] + 1e-6:        # a tail must never run under the next line
                a["len"] = round(max(0.05, b["start"] - a["start"]), 6)
                a["fade_out"] = min(max(a.get("fade_out", 0.05), 0.1), a["len"])

    if any(c["stem"] == "vocals" for c in proj["clips"]):     # room for a vocal on top (as automash.build_combo)
        for t in proj["tracks"]:
            if t["role"] == "other":
                t.update(pocket=2.5, duck=2.0)
            elif t["role"] == "bass":
                t.update(duck=0.8)
            elif t["role"] == "vocals":
                t.update(hpf=110.0)
    proj["master"]["gain_db"] = -1.5
    desc = dict(kind="grid", master=master, bpm=bpm, bpb=bpb, bars=grid.n(), length_sec=round(grid.total * 60.0 / bpm, 1),
                follow="stretch" if stretchy else "cut", layers=layers, notes=notes)
    proj["mashup"] = desc
    return proj, desc


# ------------------------------------------------------------------ from the library

def _key_shifts(picks, master, sd):
    """Semitones for every song bringing pitched stems, against the song whose music carries the mix: the best
    bar-level fit (automash.ranked_shifts); an unpitched vocal (rap, screams) keeps its key."""
    from . import automash
    pitched = lambda p: [s for s in ("vocals", "other", "bass") if s in p["stems"]]
    order = sorted(picks, key=lambda p: (p["song"] != master, "other" not in p["stems"], "bass" not in p["stems"]))
    ref = next((p["song"] for p in order if set(p["stems"]) & {"other", "bass"}), None) or \
        next((p["song"] for p in order if pitched(p)), None)
    out = {p["song"]: 0 for p in picks}
    if ref is None:
        return out
    for p in picks:
        st = pitched(p)
        if p["song"] == ref or not st:
            continue
        stem = st[0]
        try:
            if stem == "vocals" and automash.tonality(sd[p["song"]], "vocals") < 0.25:
                continue
            out[p["song"]] = int(automash.ranked_shifts(sd[p["song"]], sd[ref], stem)[0][0])
        except Exception:
            out[p["song"]] = 0
    return out


def _gains(picks, master, sd):
    from . import automash
    out = {}
    for p in picks:
        if p["song"] == master:
            continue
        for stem in p["stems"]:
            try:
                out[(p["song"], stem)] = automash._replace_gain(sd[master], sd[p["song"]], stem)
            except Exception:
                out[(p["song"], stem)] = 0.0
    return out


def build(picks, master=None, fits=None, name=None, opts=None):
    """picks [{song: id, stems: [...]}] (library songs) -> (project, desc). See the module text.
    opts {follow: "stretch" | "cut", turns: bool, grids: "fix" | "asis"}."""
    from . import automash
    picks = [dict(song=p["song"], stems=[s for s in STEMS if s in set(p.get("stems") or [])]) for p in picks or []]
    picks = [p for p in picks if p["stems"]]
    if not picks:
        raise ValueError("pick at least one stem")
    if len({p["song"] for p in picks}) != len(picks):
        raise ValueError("a song is listed twice")
    sd = {p["song"]: automash.get_song(p["song"]) for p in picks}
    songs = {sid: from_song(S, (opts or {}).get("grids", "fix") != "asis") for sid, S in sd.items()}
    master = master if master in songs else pick_master(picks, songs)
    return build_project(songs, picks, master, fits or {}, _key_shifts(picks, master, sd), _gains(picks, master, sd), name,
                         opts, edges=lambda proj: _edges(proj, master, sd, songs))


def _edges(proj, master, sd, songs):
    """Guest lines get the same edges as everywhere else in the app (automash._vocal_edges): opened at the breath
    before the first word, a held last word left to ring - within the room the lane has, never past the mix's end,
    and never into the singer's NEXT line (which may now sit somewhere else: its first word would poke through)."""
    from . import automash
    total = max([c["start"] + c["len"] for c in proj["clips"] if c["song"] == master] or [0.0])
    for sid, S in sd.items():
        if sid == master:
            continue
        cl = sorted((c for c in proj["clips"] if c["song"] == sid and c["stem"] == "vocals"), key=lambda c: c["start"])
        if not cl:
            continue
        L = songs[sid]
        m = float(proj["songs"][sid]["mult"])
        spb = (60.0 / L.info_bpm) / m
        when = lambda c: float(P._beat_to_time(c["src"], L.ib))
        pieces = sorted(x for ln in lines_of(L) for x in ln[3])     # every piece of singing, joined or not
        own = {}                                               # clip -> (where its singing ends, where the next begins)
        for c in cl:
            end = when(c) + c["len"] * spb                     # (as placed: a whole line, or up to one of its joins)
            k = min(range(len(pieces)), key=lambda i: abs(pieces[i][1] - end)) if pieces else None
            if k is not None and abs(pieces[k][1] - end) < 0.05:
                own[id(c)] = (pieces[k][1], pieces[k + 1][0] if k + 1 < len(pieces) else None)
        try:
            if S.voicing() is not None:
                automash._vocal_edges(S, cl, m, proj["bpb"] / m)
        except Exception:
            pass
        for c in cl:
            end, nxt = own.get(id(c), (None, None))
            if nxt is not None:
                stop = max(end, nxt - 0.03)                    # ring on up to just before the next line's first sound
                if when(c) + c["len"] * spb > stop + 1e-6:
                    c["len"] = max(0.05, (stop - when(c)) / spb)
                    c["fade_out"] = float(np.clip((stop - end) / spb, 0.05, max(0.05, c.get("fade_out", 0.05))))
            if total and c["start"] + c["len"] > total:
                c["len"] = max(0.05, total - c["start"])
                c["fade_out"] = min(max(c.get("fade_out", 0.05), 1.0), c["len"])
