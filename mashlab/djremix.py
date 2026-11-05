"""DJ remix: one song is the BED, played as recorded - its drums, bass and instruments together, its own vocal in
place where the plan switches it on - and the other song, the GUEST, goes on top line by line: verses where the
bed's vocal is out, short hooks looped over it, stutters into a section, a one-bar break before the drop.
Optionally a third song's drums carry the beat, with the bed's own drums under them.

Modelled on a reference mix by the author (an electronic track as the bed, a rap on top, a third song's
drums), which he held up as what a DJ mix should be (Oct 17). djbrain's part-by-part stem swap lost to it: one
song's bass under the other's instruments exposed separation artifacts, and its vocals stopped wherever a part
ended, mid-word. Here the bed's stems stay together and every vocal is a whole phrase.

The language model plans what it can judge from the brief: the bed's path (which sections, how long, where the
drop is, where the bed's vocal is on) and which lines go where - by their words, length and repeats. The engine
picks the bed by harmonic fit before planning, finds each phrase and its clean cuts, keeps every line's pickup and
place in the bar, checks guest lines against the chords they land on, and reports each repair.
"""
import math
import re

import numpy as np

from . import automash, djbrain, library, llm as llm_mod
from . import project as P
from .automash import _best_mult, _vocal_edges

PART_BARS = list(range(2, 50, 2))   # even bar counts, so a part can end right where a section does
DRUMS = ["bed", "donor", "both", "none"]
MAX_PARTS = 16
MAX_LINES = 48
MAX_BAR = 31               # a line's bar inside its part
STUTTER = [0, 2, 4, 6]
EVERY = [1, 2, 4, 8]
SLICE = 0.5                # project beats per stutter slice (an eighth note, as in the reference mix)
VOCAL_FLOOR = 0.5          # a guest line on chords it fits worse than this (pair-rating scale) is left out
PH_MIN = 2.0               # source beats: a voiced bit shorter than this (ad-lib, breath, bleed) is not a phrase
PH_MIN_WORDS = 1.0         # ... unless the transcript has words in it
PH_MAX_BARS = 16           # a longer phrase is split at the quietest point near one of its bar lines (only then:
                           # a cut inside a sung line is heard as one, however quiet the spot - user, Oct 17)
GAP_DB = 24.0              # the voice counts as stopped this far under its loud level ...
GAP_MIN = 0.15             # ... for at least this long (s)
TRIM_DB = 30.0             # a phrase starts / ends where its vocal comes within this of the phrase's level
EDGE_REACH = 30            # frames (10 ms) a phrase may reach past its voicing for an unvoiced attack or ending
PICKUP = 1.0               # source beats a bed line may start before its part and still belong to it

# The reference mix, as relations that carry over to other songs: the bed's tracks at the user's settings, the
# guest's vocal 2 dB under the bed vocal's own level (the bed vocal itself 5.7 dB down, under it), and the drum
# donor 5 dB under the bed's own drums (which, layered under the donor, sit at -9.5).
TEMPLATE = {"drums": dict(gain_db=0.8), "bass": dict(gain_db=-4.3, lpf=8000.0), "other": dict(gain_db=-1.4, hpf=90.0),
            "vocals": dict(gain_db=-5.7, duck=2.0), "guest": dict(lpf=8000.0), "guest2": dict(gain_db=-3.3)}
GUEST_VS_BED_VOCAL = -2.0
DONOR_VS_BED_DRUMS = -5.0
UNDER_DB = -9.5

SYSTEM = """You are a club DJ and remix producer. You make a remix: one song is the BED - its music, played as
recorded - and the other song, the GUEST, sings or raps on top of it, line by line. The listener is an audio
engineer who made a remix this way and loves it; build yours the same way.

THE BED (parts, in order)
- The bed's drums, bass and instruments always play together, as recorded. Most of the bed is one or two long
  stretches of the song as it goes; a part changes what's on (vocal, drums, music) - a short part (2-4 bars) is for
  an opening or a breakdown.
- A part plays from the start of its section and runs on through the song (past the section's end), for any even
  number of bars. section "continue" carries on from exactly where the part before ended - use it for every part
  that isn't a jump. Naming a section jumps there; jump rarely - once or twice in a mix, e.g. to the bed's big
  chorus at the drop.
- bed_vocal switches the bed's own vocal on, in place, for the part. Switch it OFF where the guest has a verse, so
  the guest is heard over the bed's music alone; keep it on where the bed's hook carries, with short guest lines
  over it.
- drop=true on the part where the mix peaks: in the bar before it the music and drums stop (a one-bar break - only
  vocals), then everything hits at once.
- music=false: drums (and vocals) only - for a 2-4 bar opening or a breakdown.
- drums: "bed" = the bed's own drums; "none"; if a DRUMS song is given, "donor" = its drums instead and "both" = its
  drums with the bed's quietly under them.

THE LINES (vocal lines placed on top)
- Each line is a whole sung or rapped phrase; place it by part (1 = first) and bar (0 = the part's first bar). The
  engine keeps the line's pickup and its place in the bar, and never cuts a word.
- Guest verses: several guest lines in a row, each at the bar where the one before ends (its bar + its length,
  rounded up), in parts where bed_vocal is off.
- Hooks: a short line (1-2 bars, one the song repeats) looped as a motif - repeat 2-4 times, every 2 or 4 bars -
  over the bed's vocal or leading into the drop.
- stutter (2-6 slices of the line's first word: "st-st-stutter") leads a line in at a section change or right
  before the drop. Use it 2-4 times in a mix, not on every line.
- The guest's strongest line (its chorus / best-known line) goes at the peak, after the drop.
- Never put a guest line over a bed section listed in its "clashes on" column.
- The bed's own lines can be placed too, out of place - e.g. its hook teased over the opening drums.
- Most bars should have a vocal (the bed's in place, or a placed line); never more than two at once.

AN ARC: open recognisably (e.g. 2 bars of drums with a hook on top, then the bed from its start), the guest comes in
within 8-16 bars, build, a drop with the guest's strongest line about two thirds in, end on a strong line over the
bed's last big stretch - no long tail.

THE LISTENER'S OWN REMIX, for reference (an electronic track as the bed, a rap on top, 77 bars):
 bars 1-2   drums only, the bed's hook line teased on top
 bars 3-36  the bed from its start, one 34-bar stretch. Its vocal on for bars 3-12, off 13-20 under a guest verse,
            on again 21-29 with the guest rapping over it, off 30-36 under the next guest verse; a short guest phrase
            looped every 4 bars (bars 5 and 9); a stutter leading into the guest's next verse (bar 28).
 bar 36     one-bar break: only the guest's vocal.
 bars 37-77 the drop: the bed's chorus section, one 40-bar stretch, its vocal back on; the guest phrase looped again
            (bars 37, 41); the guest's chorus lines from bar 45, led in by a stutter; the guest's last chorus to the
            end over the bed's final stretch.

For each part write "why" first: at most 12 words on what it does for the mix."""


# ------------------------------------------------------------------ phrases (whole lines of a vocal)

def _venv(S):
    """10 ms RMS (dB) of S's vocal stem, cached on S; None without one."""
    if "rx_env" not in S._bar:
        import soundfile as sf
        files = sorted((library.song_dir(S.id) / "stems").glob("vocals.*"))
        env = None
        if files:
            x, sr = sf.read(str(files[0]), dtype="float32", always_2d=True)
            x = x.mean(1)
            hop = sr // 100
            n = len(x) // hop
            env = 20 * np.log10(np.sqrt(np.mean(x[:n * hop].reshape(n, hop) ** 2, 1)) + 1e-6)
        S._bar["rx_env"] = env
    return S._bar["rx_env"]


def words_of(sid):
    """The song's words [(t s, word, line no.)] - your typed lyrics (lined up) first, else Whisper's when usable."""
    from . import lyrics as lyr, userlyrics
    segs, src = userlyrics.segments(sid)
    if src == "whisper" and not ((lyr.load(sid) or {}).get("quality") or {}).get("usable"):
        return []
    out = [(float(w["t"]), w["w"].strip(), k) for k, g in enumerate(segs) for w in g.get("words", [])
           if str(w.get("w", "")).strip()]
    return sorted(out)


_SELF = {"i", "i'm", "i'll", "i've", "i'd"}


def _strength(w1, w2):
    """How much the gap between two words reads as a line break: 3 = a new transcript line, 2 = a sentence ends,
    1 = a clause (comma) or a capitalised line start, 0 = mid-sentence."""
    (_, a, la), (_, b, lb) = w1, w2
    s = 2 if a[-1:] in ".?!" else 1 if (a[-1:] in ",;:" or (b[:1].isupper() and b.lower().strip(",.?!") not in _SELF)) else 0
    return max(s, 3 if la != lb else 0)


def split_phrases(vm, env, bar_frames, words, beat_s, max_bars=PH_MAX_BARS, bpb=4, fps=100):
    """Frames [(f0, f1)] of whole phrases (see phrases()). vm = voicing mask, env = vocal level (dB), both at fps;
    bar_frames = the song's bar lines (frames); words = [(t s, word, line no.)], may be empty; beat_s = s per beat."""
    n = min(len(vm), len(env))
    vm, env = np.asarray(vm[:n], bool), np.asarray(env[:n], float)
    sm = np.convolve(env, np.ones(3) / 3, "same")                    # 30 ms

    def quiet(f0, f1):
        f0, f1 = max(0, int(f0)), min(n, int(f1))
        return f0 + int(np.argmin(sm[f0:f1])) if f1 - f0 > 1 else int(np.clip(f0, 0, n - 1))

    # stretches of singing: the vocal's LEVEL within GAP_DB of its loud level (pitch tracking alone drops out on
    # held, breathy or effected notes, and a cut there is mid-word), gaps shorter than GAP_MIN closed
    ref = float(np.percentile(sm[vm] if vm.any() else sm, 95))
    act = sm > ref - GAP_DB
    if act.any():
        from scipy.ndimage import binary_closing
        act = binary_closing(np.r_[False, act, False], np.ones(int(GAP_MIN * fps), bool))[1:-1]
    on = act.astype(np.int8)
    runs = [[int(a), int(b)] for a, b in zip(np.flatnonzero(np.diff(np.r_[0, on]) == 1),
                                             np.flatnonzero(np.diff(np.r_[on, 0]) == -1) + 1)]
    short = 2 * beat_s * fps
    merged = []                                       # a fragment under 2 beats joins its neighbour across a short gap
    for r in runs:
        if merged and r[0] - merged[-1][1] < 0.4 * fps and (r[1] - r[0] < short or merged[-1][1] - merged[-1][0] < short):
            merged[-1][1] = r[1]
        else:
            merged.append(r)
    wf = [(int(round(t * fps)), w, k) for t, w, k in words]
    long_ = max_bars * bpb * beat_s * fps
    bf = np.asarray(bar_frames, float)

    def cands(a, b):
        """Possible cuts inside [a, b): (frame, strength) - just before a word that may start a line (the quietest
        30 ms between it and the word before), and, weakest, just before each bar line (where a pickup starts) for
        stretches the transcript has no words in."""
        ws = [x for x in wf if a - 15 <= x[0] < b]
        out = []
        for w1, w2 in zip(ws, ws[1:]):
            if w2[0] - w1[0] >= 8:
                out.append((quiet(max(w1[0] + 8, w2[0] - 35), w2[0] + 3), _strength(w1, w2)))
        for f in bf[(bf > a) & (bf < b)]:
            out.append((quiet(f - 0.75 * beat_s * fps, f + 0.25 * beat_s * fps), -1))
        return out

    def split(a, b, cs):
        if b - a <= long_:
            return [(a, b)]
        inner = [x for x in cs if a + (b - a) / 5 < x[0] < b - (b - a) / 5] or [x for x in cs if a + 50 < x[0] < b - 50]
        if not inner:
            return [(a, b)]
        bar = lambda c: round(float(np.min(np.abs(bf - c))) / (beat_s * fps)) if len(bf) else 0
        # the deepest dip first (in 3 dB steps: the nearest thing to a breath), then where the words say a line
        # ends, then nearest a bar line, then nearest the middle
        c = min(inner, key=lambda x: (round(sm[x[0]] / 3), -x[1], bar(x[0]), abs(x[0] - (a + b) / 2)))[0]
        return split(a, c, cs) + split(c, b, cs)

    pieces = []                                       # (start, end, start is a breath edge, end is a breath edge)
    for a, b in merged:
        cut = split(a, b, cands(a, b))
        for j, (a2, b2) in enumerate(cut):
            pieces.append((a2, b2, j == 0, j == len(cut) - 1))
    out = []
    for i, (a2, b2, open_a, open_b) in enumerate(pieces):
        seg = sm[a2:b2]
        if len(seg) < 5:
            continue
        thr = np.percentile(seg, 90) - TRIM_DB
        if open_a:                                    # the voicing starts late on an unvoiced attack ("s", "p", "t"):
            lim = max(0, a2 - EDGE_REACH, pieces[i - 1][1] if i else 0)   # reach back to where the sound starts
            while a2 > lim and sm[a2 - 1] > thr:
                a2 -= 1
        if open_b:                                    # ... and on to where a final consonant ends
            lim = min(n, b2 + EDGE_REACH, pieces[i + 1][0] if i + 1 < len(pieces) else n)
            while b2 < lim and sm[b2] > thr:
                b2 += 1
        loud = np.flatnonzero(sm[a2:b2] > thr)
        if len(loud):
            out.append((a2 + int(loud[0]), a2 + int(loud[-1]) + 1))
    return out


def phrases(S):
    """S's vocal as whole phrases [{a, b, words, repeats, f0, f1}] (a, b in source beats, f0/f1 at 100 fps).

    Lines are what the singer sings between breaths (S.voicing); a line running past PH_MAX_BARS bars - most
    sung vocals barely breathe - is split where its words say a line ends: a new transcript line first, then a
    sentence end, then a comma or a capitalised line start, preferring the one nearest a bar line, never right at
    the edges (so no 1-word scraps). The cut goes in the quietest 30 ms before that word. Without words a long
    line is split just before a bar line, where a pickup starts. Each phrase is trimmed to its voice; repeats = how
    many phrases share most of its words (a hook). Cached per song (and per version of its words); None without
    melody features or a vocal stem."""
    from . import userlyrics
    key = ("rx_phrases", userlyrics.mtime(S.id))
    if key in S._bar:
        return S._bar[key]
    vm, env = S.voicing(), _venv(S)
    if vm is None or env is None or not S.has_real_vocals:
        S._bar[key] = None
        return None
    words = words_of(S.id)
    bar_frames = [float(S.beats[b]) * 100 for b in S.bars if b < S.nb]
    beat = lambda fr: float(np.interp(fr / 100.0, S.beats, np.arange(S.nb, dtype=float)))
    out = []
    cut = split_phrases(vm, env, bar_frames, words, 60.0 / S.bpm, bpb=S.bpb)
    for j, (f0, f1) in enumerate(cut):                 # each word to one phrase: the one it starts in (Whisper's
        nxt = cut[j + 1][0] if j + 1 < len(cut) else 1e12   # times run a little early: 0.2 s of slack)
        ws = [w for t, w, _ in words if f0 / 100 - 0.2 <= t < min(f1 / 100, nxt / 100 - 0.2)]
        a, b = beat(f0), beat(f1)
        if b - a >= (PH_MIN_WORDS if ws else PH_MIN):
            out.append(dict(a=a, b=b, words=" ".join(ws), f0=f0, f1=f1))
    toks = [set(re.sub(r"[^a-z' ]", " ", p["words"].lower()).split()) for p in out]
    for i, p in enumerate(out):
        t = toks[i]
        p["repeats"] = 1 if len(t) < 2 else sum(1 for u in toks if len(u) >= 2 and len(t & u) / len(t | u) >= 0.6)
    S._bar[key] = out
    return out


def _onset(S, p):
    """Source beat where phrase p's first sound starts (for stutter slices: they must start on the attack)."""
    env = _venv(S)
    sm = np.convolve(env, np.ones(3) / 3, "same")
    f0 = int(p["f0"])
    lvl = np.percentile(sm[f0:min(len(sm), f0 + 100)], 90) if f0 < len(sm) else 0.0
    w = np.flatnonzero(sm[max(0, f0 - 8):f0 + 30] >= lvl - 18)
    f = max(0, f0 - 8) + (int(w[0]) if len(w) else 8) - 1
    return float(np.interp(f / 100.0, S.beats, np.arange(S.nb, dtype=float)))


def _anchor(S, a):
    """The bar line (source beat) a phrase starting at beat a belongs to: the nearest one, so a pickup keeps its
    place before the bar and a line starting on beat 2 keeps that."""
    bars = np.asarray([b for b in S.bars if b < S.nb] or [0], float)
    return float(bars[int(np.argmin(np.abs(bars - a)))])


# ------------------------------------------------------------------ the brief

def _sec_of(secs, M, S, a, mult):
    """(section, project bars from its start to source beat a) for the section that holds beat a, or (None, 0)."""
    for s in secs:
        b0 = S.bars[M[s["m0"]]]
        b1 = S.bars[M[s["m1"]]] if s["m1"] < len(M) else S.info.get("last_beat", S.nb) + 1
        if b0 - PICKUP <= a < b1:
            return s, max(0.0, (a - b0) * mult / S.bpb)
    return None, 0.0


def _bars_of(s, M, S, mult):
    b0 = S.bars[M[s["m0"]]]
    b1 = S.bars[M[s["m1"]]] if s["m1"] < len(M) else S.info.get("last_beat", S.nb) + 1
    return (b1 - b0) * mult / S.bpb


def clash_table(fits, secs, guest, bed):
    """{guest section id: [bed section ids its vocal clashes on]} (engine-checked, in the mix's key)."""
    out = {}
    for g in secs[guest]:
        if g["stems"]["vocals"] < 0.2:
            continue
        bad = []
        for s in secs[bed]:
            h = fits.vocal(g, s, min(g["m1"] - g["m0"], s["m1"] - s["m0"], 8))
            if h is not None and h < VOCAL_FLOOR:
                bad.append(s["id"])
        out[g["id"]] = bad
    return out


def line_ids(songs, secs, Ms, phr, mult, tags):
    """{line id: (tag, phrase, section, bars into the section)} - ids like A7 / B12, in song order."""
    out = {}
    for t in tags:
        for k, p in enumerate(phr.get(t) or []):
            sec, off = _sec_of(secs[t], Ms[t], songs[t], p["a"], mult[t])
            out[f"{t}{k + 1}"] = (t, p, sec, off)
    return out


def brief(songs, secs, Ms, phr, lines, roles, clashes, bpm, mult, target, prompt):
    bed, guest, donor = roles["bed"], roles["guest"], roles.get("donor")
    Sb = songs[bed]
    e = np.array([s["energy"] for s in secs[bed]] or [0.5])
    lo, hi = np.quantile(e, 0.33), np.quantile(e, 0.8)
    amt = lambda x: djbrain._level(x, [0.1, 0.35, 0.65], ["-", "little", "some", "full"])

    def head(t, what):
        S = songs[t]
        k = S.info.get("key") or {}
        out = [f'{what}: SONG {t} - "{S.title}": {S.bpm:.0f} BPM (plays at {bpm:g}), {k.get("name", "unknown key")}.']
        h = djbrain._heard(S.id)
        if h:
            out.append(f'  Sounds like (heard): {" / ".join(x for x in (h.get("genre"), h.get("subgenre")) if x)} - {h["sound"]}')
        return out

    def line_rows(t, with_clash):
        rows = []
        for lid, (tg, p, sec, off) in lines.items():
            if tg != t:
                continue
            ln = (p["b"] - p["a"]) * mult[t] / Sb.bpb
            where = f'{sec["id"]} +{off:.0f}' if sec else "-"
            rep = f'x{p["repeats"]}' if p["repeats"] > 1 else ""
            cl = ""
            if with_clash and sec and clashes.get(sec["id"]):
                cl = "clashes on " + ",".join(clashes[sec["id"]])
            rows.append(f' {lid:<5} {where:<9} {ln:4.1f}   {rep:<4} {cl:<22} '
                        + (f'"{p["words"][:90]}"' if p["words"] else "(no words heard)"))
        return rows

    L = head(bed, "BED")
    L.append(" section   name           bars  energy  drums  bass   instr  vocals")
    for s in secs[bed]:
        en = "peak" if s["energy"] >= hi and s["energy"] >= e.max() - 0.01 else djbrain._level(s["energy"], [lo, hi], ["low", "medium", "high"])
        st = s["stems"]
        L.append(f' {s["id"]:<9} {s["name"][:12]:<14} {_bars_of(s, Ms[bed], Sb, mult[bed]):4.0f}  {en:<6}  '
                 f'{amt(st["drums"]):<6} {amt(st["bass"]):<6} {amt(st["other"]):<6} {amt(st["vocals"]):<6}'
                 + ("  HOOK" if s["hook"] else ""))
    L.append(" The bed's own lines (they play in place where bed_vocal is on):")
    L.append(" line  section+bar bars repeats                  words")
    L += line_rows(bed, False) or [" (no vocal)"]
    L.append("")
    L += head(guest, "GUEST")
    L.append(" line  section+bar bars repeats clashes on          words")
    L += line_rows(guest, True) or [" (no vocal)"]
    if donor:
        L += ["", f'DRUMS: SONG {donor} - "{songs[donor].title}": its drums can replace the bed\'s ("donor") or '
                  f'layer over them ("both").']
    n_parts = max(3, round(target / 11))
    L += ["", f"MIX: {bpm:g} BPM, one bar = {60.0 / bpm * Sb.bpb:.2f} s. LENGTH: the parts' bars must add up "
              f"to {target} (between {int(target * 0.9)} and {int(target * 1.1)}) - about {n_parts}-{n_parts + 2} parts.",
          "REQUEST: " + (f'"{prompt}"' if prompt else "none - use your judgement.")]
    return "\n".join(L)


def schema(sec_ids, ids, has_donor):
    part = {"type": "object", "properties": {
        "why": {"type": "string"},
        "section": {"type": "string", "enum": sec_ids + ["continue"]},
        "bars": {"type": "integer", "enum": PART_BARS},
        "music": {"type": "boolean"},
        "drums": {"type": "string", "enum": DRUMS if has_donor else ["bed", "none"]},
        "bed_vocal": {"type": "boolean"},
        "drop": {"type": "boolean"}},
        "required": ["why", "section", "bars", "music", "drums", "bed_vocal", "drop"]}
    line = {"type": "object", "properties": {
        "line": {"type": "string", "enum": ids},
        "part": {"type": "integer", "enum": list(range(1, MAX_PARTS + 1))},
        "bar": {"type": "integer", "enum": list(range(0, MAX_BAR + 1))},
        "repeat": {"type": "integer", "enum": [1, 2, 3, 4]},
        "every": {"type": "integer", "enum": EVERY},
        "stutter": {"type": "integer", "enum": STUTTER}},
        "required": ["line", "part", "bar", "repeat", "every", "stutter"]}
    return {"type": "object", "properties": {
        "title": {"type": "string"},
        "parts": {"type": "array", "items": part},
        "lines": {"type": "array", "items": line}},
        "required": ["title", "parts", "lines"]}


# ------------------------------------------------------------------ setting up

def roles_and_keys(songs, fit, tr, donor=None):
    """Bed and guest from the fit: the bed is the music the other's vocal sits on best (the first song when neither
    fits). The bed keeps its key; only the guest's vocal moves."""
    rate = lambda d: fit[d]["rating"] if (fit.get(d) or {}).get("workable") else -1
    if rate("BA") > rate("AB"):
        bed, guest, d = "A", "B", "BA"
    elif rate("AB") >= 0:
        bed, guest, d = "B", "A", "AB"
    else:
        bed, guest, d = "A", "B", None
    if d and abs(fit[d]["shift_rel"]) > 4:              # a big move: split between them, as the fit chose
        t = {bed: int(fit[d]["shift_bed"]), guest: int(fit[d]["shift_rel"] + fit[d]["shift_bed"])}
    else:
        t = {bed: 0, guest: int(fit[d]["shift_rel"]) if d else int(tr[guest] - tr[bed])}
    if donor:
        t[donor] = 0
    roles = dict(bed=bed, guest=guest)
    if donor:
        roles["donor"] = donor
    return roles, t


def _match(bed, other, stem, offset):
    """Clip gain (dB) putting `other`'s stem `offset` dB from the bed's own (the bed's vocal: or its music - 4 dB)."""
    LB, LO = bed.info["loudness"], other.info["loudness"]
    ref = LB.get(stem, -70)
    if stem == "vocals" and ref - LB["mix"] <= -13:
        ref = LB.get("inst", LB["mix"]) - 4.0
    return float(np.clip(ref + offset - LO.get(stem, -30), -12, 12))


# ------------------------------------------------------------------ playing a plan

class _Lane:
    """One track's clips; a clip continuing the one before (same audio, back to back) extends it."""

    def __init__(self, proj, track):
        self.proj, self.track, self.clips = proj, track, []

    def add(self, sid, stem, t0, t1, s0, m, gain=0.0, merge=True, **kw):
        if t1 - t0 < 1e-3:
            return None
        c = self.clips[-1] if self.clips else None
        if (merge and c and c["song"] == sid and abs(c["start"] + c["len"] - t0) < 1e-4
                and abs(c["src"] + c["len"] / m - s0) < 1e-3 and abs(c["gain_db"] - round(gain, 1)) < 0.05):
            c["len"] = t1 - c["start"]
            return c
        c = P.clip(self.track["id"], sid, stem, t0, s0, t1 - t0, gain_db=round(gain, 1), fade_out=0.05, **kw)
        self.clips.append(c)
        self.proj["clips"].append(c)
        return c

    def free(self, t0, t1):
        return all(c["start"] + c["len"] <= t0 + 0.05 or c["start"] >= t1 - 0.05 for c in self.clips)


def _donor_runs(D, Md, secs_d, parts, plan, mult_d, bpb):
    """Source start beat of the donor's drums for each part that uses them: continuing from the part before where it
    can, else the donor section whose energy matches the part's (its strongest after a drop)."""
    cand = [s for s in secs_d if s["stems"]["drums"] >= 0.6] or secs_d
    if not cand:
        return {}
    es = np.array([s["energy"] for s in cand])
    end = float(D.info.get("last_beat", D.nb))
    out, pos = {}, None
    for k, (p, pl) in enumerate(zip(parts, plan)):
        if p["drums"] not in ("donor", "both"):
            pos = None
            continue
        need = pl["len"] / mult_d
        if pos is not None and not p["drop"] and pos + need <= end:
            out[k], pos = pos, pos + need
            continue
        if p["drop"]:
            pick = max(cand, key=lambda s: (s["energy"], -s["m0"]))
        else:
            q = pl["energy_q"]
            pick = min(cand, key=lambda s: (abs(np.mean(es <= s["energy"]) - q), s["m0"]))
        s0 = float(D.bars[Md[pick["m0"]]])
        if s0 + need > end:
            s0 = max(float(D.info.get("first_beat", 0)), end - need)
        out[k], pos = s0, s0 + need
    return out


def execute(script, songs, secs, Ms, phr, fits, bpm, mult, tr, roles, clashes, prompt="", llm_name="", target=None):
    """script = {title, parts: [{why, section, bars, music, drums, bed_vocal, drop}],
    lines: [{line, part, bar, repeat, every, stutter}]} -> (project, desc)."""
    bed, guest, donor = roles["bed"], roles["guest"], roles.get("donor")
    Sb, Sg = songs[bed], songs[guest]
    if Sb.bpb != Sg.bpb:
        raise ValueError("these two songs are in different meters")
    bpb, mb = Sb.bpb, mult[bed]
    Mb = Ms[bed]
    by_id = {s["id"]: s for s in secs[bed]}
    parts = [dict(p) for p in (script.get("parts") or [])[:MAX_PARTS] if p.get("section") in by_id or p.get("section") == "continue"]
    if parts and parts[0]["section"] == "continue":
        parts[0]["section"] = secs[bed][0]["id"]
    if not parts:
        raise ValueError("the plan had no usable parts")
    repairs = []
    if donor is None:
        for k, p in enumerate(parts):
            if p.get("drums") in ("donor", "both"):
                p["drums"] = "bed"
    e = np.array([s["energy"] for s in secs[bed]])

    # ---- the bed's path: where each part plays from (continuing where the last one ended, when it names that spot)
    end_src = float(Sb.bars[Sb.info["content_bars"][1]] if Sb.info["content_bars"][1] < len(Sb.bars)
                    else Sb.info.get("last_beat", Sb.nb - 1) + 1)
    def sec_at(s):
        """The bed section playing at source beat s."""
        return max((x for x in secs[bed] if Sb.bars[Mb[x["m0"]]] <= s + 1e-6), key=lambda x: x["m0"], default=secs[bed][0])

    plan, t, prev_end = [], 0.0, None
    for k, p in enumerate(parts):
        if p["section"] == "continue" and prev_end is not None:
            s0 = prev_end
            sec = sec_at(s0)
        else:
            sec = by_id[p["section"]]
            s0 = float(Sb.bars[Mb[sec["m0"]]])
            if prev_end is not None and abs(prev_end - s0) <= 2 * bpb / mb + 1e-6:
                s0 = prev_end                              # a section starting within 2 bars of where the last part
                sec = sec_at(s0)                           # ended: it carries on rather than skip or repeat a bar
        L = int(p["bars"]) * bpb
        if s0 + L / mb > end_src + 1e-6:
            left = int((end_src - s0) * mb // bpb)
            if left < 1:
                repairs.append(f"part {k + 1}: {sec['id']} has no bars left in the song - left out")
                continue
            repairs.append(f"part {k + 1}: {sec['id']} has only {left} bars left in the song (asked {p['bars']})")
            L = left * bpb
        cont = prev_end is not None and abs(prev_end - s0) < 1e-6
        plan.append(dict(part=p, sec=sec, t0=t, t1=t + L, len=L, s0=s0, s1=s0 + L / mb, cont=cont,
                         energy_q=float(np.mean(e <= sec["energy"]))))
        t += L
        prev_end = s0 + L / mb
    parts = [pl["part"] for pl in plan]
    total = t
    if target and abs(total / bpb - target) > 0.2 * target:
        repairs.append(f"the plan is {round(total / bpb)} bars (asked {target})")

    # a line placed at the very start whose pickup begins before bar 1: the mix opens on that pickup alone (up to 2
    # bars of it), so the line keeps its first word and the music still lands on the bar
    ids = line_ids(songs, secs, Ms, phr, mult, [bed, guest])
    earliest = 0.0
    for ln in (script.get("lines") or [])[:MAX_LINES]:
        k = int(ln.get("part") or 1) - 1
        if ln.get("line") in ids and 0 <= k < len(plan):
            tg, ph, _, _ = ids[ln["line"]]
            earliest = min(earliest, plan[k]["t0"] + int(ln.get("bar") or 0) * bpb
                           + (ph["a"] - _anchor(songs[tg], ph["a"])) * mult[tg])
    lead = min(2 * bpb, math.ceil(-earliest / bpb) * bpb) if earliest < 0 else 0.0
    if lead:
        for pl in plan:
            pl["t0"] += lead
            pl["t1"] += lead
        total += lead

    proj = P.empty((script.get("title") or f"{Sb.title} × {Sg.title}")[:80], bpm, bpb)
    for tg in [x for x in (bed, guest, donor) if x]:
        proj["songs"][songs[tg].id] = dict(transpose=tr[tg], mult=mult[tg])
    trk = {t_["role"]: t_ for t_ in proj["tracks"]}
    for role, kw in TEMPLATE.items():
        if role in trk:
            trk[role].update(kw)
    short = lambda S: S.title.split(" - ")[0][:24]
    for role in ("drums", "bass", "other", "vocals"):
        trk[role]["name"] = f"{role.capitalize()} · {short(Sb)}"
    lane = {r: _Lane(proj, trk[r]) for r in ("drums", "bass", "other", "vocals")}

    def new_lane(name, role, **kw):
        tk = P.track(name, role, **kw)
        proj["tracks"].append(tk)
        return _Lane(proj, tk)

    if donor:
        lane["donor"] = new_lane(f"Drums · {short(songs[donor])}", "drums")
    g_donor = _match(Sb, songs[donor], "drums", DONOR_VS_BED_DRUMS) if donor else 0.0
    d_start = _donor_runs(songs[donor], Ms[donor], secs[donor], parts, plan, mult[donor], bpb) if donor else {}

    # ---- bed music, drums, and the bed's vocal in place
    bed_ph = phr.get(bed) or []
    taken = set()
    bed_vox = []                                           # (index, t0) of bed phrases played in place
    for k, (p, pl) in enumerate(zip(parts, plan)):
        brk = bpb if (k + 1 < len(parts) and parts[k + 1]["drop"] and pl["len"] > bpb) else 0.0
        t0, t1, s0 = pl["t0"], pl["t1"] - brk, pl["s0"]
        if p["music"]:
            for r in ("bass", "other"):
                lane[r].add(Sb.id, r, t0, t1, s0, mb)
        if p["drums"] in ("bed", "both"):
            lane["drums"].add(Sb.id, "drums", t0, t1, s0, mb, gain=UNDER_DB if p["drums"] == "both" else 0.0)
        if p["drums"] in ("donor", "both") and k in d_start:
            lane["donor"].add(songs[donor].id, "drums", t0, t1, d_start[k], mult[donor], gain=g_donor)
        if p["bed_vocal"]:
            nxt_cont = k + 1 < len(plan) and plan[k + 1]["cont"]
            limit = plan[k + 1]["t1"] if nxt_cont else pl["t1"] + 0.5 * bpb
            for i, ph in enumerate(bed_ph):
                if i in taken or not (s0 - PICKUP <= ph["a"] < pl["s1"]):
                    continue
                a_t, b_t = t0 + (ph["a"] - s0) * mb, t0 + (ph["b"] - s0) * mb
                if b_t > min(limit, total) + 1e-6 or a_t < -1e-6:
                    continue                               # would be cut off where the bed jumps or the mix ends
                taken.add(i)
                bed_vox.append((i, a_t, k))
    # one clip per run of bed lines that follow each other in place, in the same or the next part (what's between
    # them is the song as recorded) - never across a part with the bed's vocal off
    run_end, run_k = None, None
    for i, a_t, k in sorted(bed_vox, key=lambda x: x[1]):
        ph = bed_ph[i]
        b_t = a_t + (ph["b"] - ph["a"]) * mb
        c = lane["vocals"].clips[-1] if lane["vocals"].clips else None
        if c and run_end is not None and a_t - run_end <= 2 * bpb and k - run_k <= 1 and \
                abs((a_t - c["start"]) - (ph["a"] - c["src"]) * mb) < 0.01:
            c["len"] = b_t - c["start"]
        else:
            lane["vocals"].add(Sb.id, "vocals", a_t, b_t, ph["a"], mb, merge=False)
        run_end, run_k = b_t, k

    # ---- placed lines
    starts = [pl["t0"] for pl in plan]
    g_guest = _match(Sb, Sg, "vocals", GUEST_VS_BED_VOCAL)
    lane["guest"] = new_lane(f"Vocals · {short(Sg)}", "vocals", **TEMPLATE["guest"])
    lane["guest2"] = new_lane(f"Vocals 2 · {short(Sg)}", "vocals", **TEMPLATE["guest2"])
    slices, lead = [], {}
    script_lines = (script.get("lines") or [])[:MAX_LINES]
    placed = [None] * len(script_lines)

    def first(j):                                          # the drop's own lines get their place first: a line
        ln = script_lines[j]                               # before it running long mustn't push them out
        k = int(ln.get("part") or 1) - 1
        return (0 if 0 <= k < len(parts) and parts[k]["drop"] and int(ln.get("bar") or 0) < 4 else 1, j)
    for j in sorted(range(len(script_lines)), key=first):
        ln = script_lines[j]
        lid = ln.get("line")
        rec = dict(line=lid, part=ln.get("part"), bar=ln.get("bar"), repeat=int(ln.get("repeat") or 1),
                   every=int(ln.get("every") or 4), stutter=int(ln.get("stutter") or 0), placed=0, note="")
        placed[j] = rec
        if lid not in ids:
            rec["note"] = "no such line"
            continue
        k = int(ln.get("part") or 1) - 1
        if not 0 <= k < len(plan):
            rec["note"] = f"part {k + 1} isn't in the mix"
            continue
        tg, ph, sec, _ = ids[lid]
        S, m = songs[tg], mult[tg]
        rec["words"] = ph["words"]
        anc = _anchor(S, ph["a"])
        notes = []
        first = starts[k] + int(ln.get("bar") or 0) * bpb + (ph["a"] - anc) * m
        shift = math.ceil(-first / bpb) * bpb if first < 0 else 0.0
        if shift:                                          # its pickup would start before the mix: all a bar later
            notes.append(f"moved {round(shift / bpb)} bar(s) later - its pickup starts before the mix")
        for r in range(max(1, min(4, rec["repeat"]))):
            bar_t = starts[k] + (int(ln.get("bar") or 0) + r * rec["every"]) * bpb + shift
            p0 = bar_t + (ph["a"] - anc) * m
            L = (ph["b"] - ph["a"]) * m
            if p0 + L > total + 0.5 * bpb:
                notes.append("runs past the end of the mix")
                continue
            at = max(j for j, s in enumerate(starts) if s <= max(0.0, bar_t))
            here = sec_at(plan[at]["s0"] + (p0 - plan[at]["t0"]) / mb)   # the bed section it lands on
            if tg == bed and plan[at]["part"]["bed_vocal"] and \
                    any(abs((p0 - c["start"]) - (ph["a"] - c["src"]) * m) < 0.05 and c["start"] - 0.05 <= p0 < c["start"] + c["len"]
                        for c in lane["vocals"].clips):
                notes.append("already sung there in place")
                continue
            if tg == guest and here["id"] in clashes.get(sec["id"] if sec else "", []):
                notes.append(f"clashes with {here['id']}'s chords")   # the same section-level check the brief listed
                continue
            ns = rec["stutter"] if r == 0 else 0
            on_src = _onset(S, ph) if ns else ph["a"]
            on_t = p0 + (on_src - ph["a"]) * m
            g0 = on_t - ns * SLICE
            order = (["vocals"] if tg == bed else []) + ["guest", "guest2"]
            ln_ = next((lane[x] for x in order if lane[x].free(g0, p0 + L)), None)
            if ln_ is None:
                notes.append("two lines already sing there")
                continue
            gain = g_guest if tg == guest else (TEMPLATE["vocals"]["gain_db"] if ln_ is not lane["vocals"] else 0.0)
            for j in range(ns):
                c = ln_.add(S.id, "vocals", g0 + j * SLICE, g0 + (j + 1) * SLICE, on_src, m, gain=gain, merge=False)
                c["fade_out"] = 0.03
                slices.append(c)
            c = ln_.add(S.id, "vocals", p0, p0 + L, ph["a"], m, gain=gain, merge=False)
            c["line"] = lid
            if ns:
                lead[id(c)] = on_t                         # its pickup mustn't reach back over the slices
            rec["placed"] += 1
        if notes:
            rec["note"] = "; ".join(sorted(set(notes)))
            repairs.append(f"line {lid} (part {k + 1}, bar {rec['bar']}): " + rec["note"]
                           + (f" - {rec['placed']} of {rec['repeat']} placed" if rec["placed"] else " - left out"))

    # ---- vocal edges: pickups from the breath before a line, held words let ring (never the stutter slices)
    sl = {id(c) for c in slices}
    for key in ("vocals", "guest", "guest2"):
        for tg in (bed, guest):
            S = songs[tg]
            cl = sorted((c for c in lane[key].clips if c["song"] == S.id and id(c) not in sl), key=lambda c: c["start"])
            if cl:
                _vocal_edges(S, cl, mult[tg], bpb / mult[tg])
    for key in ("vocals", "guest", "guest2"):
        for c in lane[key].clips:
            st = lead.get(id(c))
            if st is not None and c["start"] < st:         # a stuttered line starts on its attack, after the slices
                cut = st - c["start"]
                c["start"], c["src"], c["len"] = st, c["src"] + cut / mult[bed if c["song"] == Sb.id else guest], c["len"] - cut
                c["fade_in"] = 0.0
            if c["start"] + c["len"] > total:              # nothing rings past the end of the mix
                c["len"] = max(0.05, total - c["start"])
                c["fade_out"] = min(max(c.get("fade_out", 0.05), 1.0), c["len"])
    base = {id(lane[r].track) for r in ("drums", "bass", "other", "vocals")}
    used = {c["track"] for c in proj["clips"]}
    proj["tracks"] = [tk for tk in proj["tracks"] if id(tk) in base or tk["id"] in used]

    meas = 2 * bpb
    done = [dict(n=k + 1, bar=int(round(pl["t0"] / bpb)) + 1, bars=int(round(pl["len"] / bpb)), section=pl["sec"]["id"],
                 name=pl["sec"]["name"], music=bool(p["music"]), drums=p["drums"], bed_vocal=bool(p["bed_vocal"]),
                 drop=bool(p["drop"]), carries_on=pl["cont"], why=str(p.get("why", ""))[:140])
            for k, (p, pl) in enumerate(zip(parts, plan))]
    titles = {tg: songs[tg].title for tg in (bed, guest, donor) if tg}
    desc = dict(kind="dj-remix", title=proj["name"], songs={tg: songs[tg].id for tg in titles}, titles=titles,
                roles=roles, bpm=bpm, transpose=tr, prompt=prompt, llm=llm_name, parts=done, lines=placed,
                repairs=repairs, bars=int(round(total / bpb)), measures=int(round(total / meas)),
                length_sec=round(total * 60 / bpm, 1))
    proj["mashup"] = desc
    return proj, desc


# ------------------------------------------------------------------ the whole thing

def setup(song_ids, drums=None):
    """Songs, sections, phrases, tempo, keys, roles and the fit checks for a remix - everything but the plan."""
    if len(song_ids) != 2:
        raise ValueError("a DJ mix takes exactly two songs")
    songs = {"A": automash.get_song(song_ids[0]), "B": automash.get_song(song_ids[1])}
    if drums:
        songs["C"] = automash.get_song(drums)
    secs, Ms = {}, {}
    for t, S in songs.items():
        secs[t], Ms[t] = djbrain.sections(S, t)
    if not secs["A"] or not secs["B"]:
        raise ValueError("a song has no usable sections at its main tempo")
    bpm, mult, tr, fit = djbrain.mix_setup(songs["A"], songs["B"])
    roles, tr = roles_and_keys(songs, fit, tr, "C" if drums else None)
    if drums:
        mult["C"] = _best_mult(songs["C"].bpm, bpm)[0]
    if not songs[roles["guest"]].has_real_vocals:        # an instrumental can only be the bed
        roles["bed"], roles["guest"] = roles["guest"], roles["bed"]
        shift = tr[roles["guest"]] - tr[roles["bed"]]
        tr[roles["bed"]], tr[roles["guest"]] = 0, shift
    fits = djbrain.Fits(songs, Ms, tr, mult)
    phr = {t: phrases(songs[t]) for t in (roles["bed"], roles["guest"])}
    clashes = clash_table(fits, secs, roles["guest"], roles["bed"])
    return dict(songs=songs, secs=secs, Ms=Ms, bpm=bpm, mult=mult, tr=tr, fit=fit, roles=roles, fits=fits, phr=phr,
                clashes=clashes)


def transcribe_missing(song_ids, progress=lambda f, m="": None):
    """Whisper the vocals of songs that have no words yet (once per song: lyrics.json, as the Lyrics button does),
    so the planner can pick lines by what they say. A song whose transcription fails is planned without words."""
    from . import lyrics, userlyrics
    for k, sid in enumerate(song_ids):
        S = automash.get_song(sid)
        if not S.has_real_vocals or userlyrics.segments(sid)[1] is not None:
            continue
        progress(k / max(1, len(song_ids)), f"hearing the words of {S.title[:40]}")
        try:
            lyrics.ensure(sid)
        except Exception:
            continue
        S.__dict__.pop("_lyrics", None)                # SongData caches "no transcript" otherwise


def run(song_ids, prompt="", minutes=3.0, seed=None, progress=lambda f, m="": None, model=None, drums=None):
    """Two library songs (+ an optional third for drums) + a request -> (project, desc)."""
    progress(0.01, "reading the songs")
    transcribe_missing(song_ids, lambda f, m="": progress(0.01 + 0.15 * f, m))
    progress(0.16, "reading the songs")
    x = setup(song_ids, drums)
    songs, secs, Ms, roles, mult = x["songs"], x["secs"], x["Ms"], x["roles"], x["mult"]
    if not x["phr"].get(roles["guest"]):
        raise ValueError(f'{songs[roles["guest"]].title} has no vocal lines to put on top')
    bpm = x["bpm"]
    target = max(16, int(round(minutes * bpm / songs["A"].bpb / 4)) * 4)
    progress(0.2, "finding the lines")
    lines = line_ids(songs, secs, Ms, x["phr"], mult, [roles["bed"], roles["guest"]])
    text = brief(songs, secs, Ms, x["phr"], lines, roles, x["clashes"], bpm, mult, target, (prompt or "").strip()[:400])
    sch = schema([s["id"] for s in secs[roles["bed"]]], list(lines), bool(drums))
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}]
    model = model or llm_mod.client()
    progress(0.3, f"the DJ is planning ({model.name})")
    try:
        try:
            script = model.chat_json(messages, sch, max_tokens=6000, temperature=0.45, seed=seed)
        except llm_mod.LlmError:
            script = model.chat_json(messages, sch, max_tokens=9000, temperature=0.3, seed=seed)
    finally:
        model.unload()
    progress(0.85, "building the mix")
    proj, desc = execute(script, songs, secs, Ms, x["phr"], x["fits"], bpm, mult, x["tr"], roles, x["clashes"],
                         prompt, model.name, target)
    desc["fit"] = {d: (None if not ((x["fit"] or {}).get(d) or {}).get("workable") else
                       dict(rating=x["fit"][d]["rating"], harmony=x["fit"][d]["harmony"])) for d in ("AB", "BA")}
    desc["brief"] = text
    desc["script"] = script
    progress(1.0, "done")
    return proj, desc
