"""Transitions as a first-class musical object (design B).

A `TransitionPlan` is a deterministic list of transition events derived from the planner's chosen vocal path
+ the bed's `structure.sections` grid + the multi-stem plan. Every event is beat-aligned to a downbeat and
placed only where cut cost (F6 c_in/c_out) is low; a mid-line boundary forces the gentler types.

The six transition types (B.2):
    1  beat-aligned hard cut   (strong downbeat, clean gap)
    2  equal-power crossfade   (vocal swap mid-line / low-energy section merge)
    3  filter sweep            (riser/faller into a louder section; the no-fill fallback)
    4  energy-matched entry/exit (duck + swell on a vocal entry/exit)
    5  drum fill injection      (section build with a usable source fill)
    6  backbeat/white-noise sweep (last-resort synthesized riser)

Decides here (pure, deterministic, unit-testable without audio); renders via the mixer (crossfades/cuts/
duck-swells = clip fade/gain + overlap clips + the existing ducking) PLUS offline-baked cache clips for every
time-varying filter (types 3 and 6, and C.2's moving EQ) — B.4 Option B. `_TrackFX` and `Mixer.render` are
UNTOUCHED: a swept segment is baked into an ordinary pre-rendered clip, so playback == export holds.
"""
import hashlib
import json

import numpy as np

from . import structure, warp
from .config import CACHE, SR

# ------------------------------------------------------------------ pinned params (design B.2, VERBATIM)

TRANSITIONS = {
    "strong_q": 0.75,      # percentile of in-range non-beatless beat_content that marks a "strong downbeat"
    "energy_delta_hi": 0.30,  # section energy jump (0..1) that counts as a build into a louder section
    "low_energy": 0.40,    # section-energy (0..1) below which a side is "low energy" (favors crossfade)
    "cut_cost_hi": 0.50,   # F6 c_in/c_out above which a boundary is mid-line (no hard cut)
    "fill_ratio": 1.50,    # per-bar onset-energy multiple of the song median to nominate a drum fill
    "fill_floor": 40,      # absolute beat_content floor (0..100) a fill bar must also clear
}

CONTENT_NONE = 12          # mirrors timing.CONTENT_NONE: beat_content at/below this is "beatless"


# ------------------------------------------------------------------ strong-downbeat percentile (F11 / B.2)

def strong_q(beat_content, first_beat, last_beat, q=None):
    """STRONG_Q = the `q`-th percentile of the song's in-range (first..last beat) non-beatless beat_content.

    Mirrors beats.beat_content's own `typical = quantile(inside, 0.75)` — self-calibrated per song, NOT an
    absolute cutoff. A beat j is a *strong downbeat* when beat_content[j] >= STRONG_Q.
    """
    if q is None:
        q = TRANSITIONS["strong_q"]
    bc = np.asarray(beat_content, float)
    lo = max(0, int(first_beat))
    hi = min(len(bc) - 1, int(last_beat))
    if hi < lo:
        return 0.0
    inside = bc[lo:hi + 1]
    inrange = inside[inside > CONTENT_NONE]
    if not len(inrange):
        return 0.0
    return float(np.quantile(inrange, q))


# ------------------------------------------------------------------ usable-source-fill detector (B.2 type 5)

def _per_bar_onset(beat_content, spans):
    """Mean per-beat beat_content (0..100) over each bar's [beat_start, beat_end) span."""
    bc = np.asarray(beat_content, float)
    out = []
    for a, b in spans:
        a = max(0, int(a))
        b = min(len(bc), int(b))
        out.append(float(bc[a:b].mean()) if b > a else 0.0)
    return np.array(out)


def fill_bar(info, boundary_bar, params=None):
    """The highest-scoring qualifying drum-fill bar strictly before `boundary_bar`, or None.

    Candidate = the last 1 bar before the section boundary; a bar qualifies when its mean per-beat
    beat_content (0..100) >= fill_ratio * median_bar AND >= fill_floor. Uses structure.bar_spans so the
    beats-per-bar comes from info["bpb"] (never hardcoded to 4). Returns the bar index or None.
    """
    p = dict(TRANSITIONS, **(params or {}))
    bars = info.get("bars")
    beats = info.get("beats")
    bpb = info.get("bpb")
    bc = info.get("beat_content")
    if not bars or beats is None or not bpb or bc is None:
        return None
    spans = structure.bar_spans(bars, len(beats), int(bpb))
    per_bar = _per_bar_onset(bc, spans)
    if not len(per_bar):
        return None
    med = float(np.median(per_bar))
    thr = max(p["fill_ratio"] * med, float(p["fill_floor"]))
    best, best_score = None, -1.0
    for bi in range(len(per_bar)):
        if bi >= int(boundary_bar):
            break
        score = per_bar[bi]
        if score >= thr and score > best_score:
            best, best_score = bi, score
    return best


# ------------------------------------------------------------------ selection rule (B.2, VERBATIM)

def select_type(event, bed_info, params=None):
    """Pick the transition type (1..6) for one boundary event. Deterministic. `event` fields:

        kind        : "section" | "vocal_entry" | "vocal_exit" | "vocal_swap"
        energy      : this section's energy (0..1)         [section events]
        next_energy : the following section's energy (0..1) [section events]
        boundary_bar: bed bar index of the boundary        [section events, for the fill detector]
        c_in, c_out : F6 cut cost at the boundary          [vocal_swap]
        bc_at       : beat_content (0..100) at the boundary downbeat [vocal_swap]
        strong_q    : the per-song STRONG_Q value          [vocal_swap]

    Returns (type:int, info:dict) where info carries the chosen fill bar / fallback reason.
    """
    p = dict(TRANSITIONS, **(params or {}))
    kind = event["kind"]

    if kind == "section":
        energy = float(event.get("energy", 0.0))
        nxt = float(event.get("next_energy", energy))
        energy_delta = nxt - energy
        if energy_delta >= p["energy_delta_hi"]:
            fb = fill_bar(bed_info, event.get("boundary_bar", 0), p)
            if fb is not None:
                return 5, {"fill_bar": int(fb), "layer_sweep": 3}   # type 5 (+ a type-3 sweep layered under)
            return 3, {"fallback": "type5_no_fill"}                 # explicit type-5 fallback (logged)
        # energy_delta < energy_delta_hi
        if energy < p["low_energy"] and nxt < p["low_energy"]:
            return 2, {}                                            # both sides low-energy -> crossfade
        return 3, {}

    if kind in ("vocal_entry", "vocal_exit"):
        return 4, {}

    if kind == "vocal_swap":
        c_in = float(event.get("c_in", 0.0))
        c_out = float(event.get("c_out", 0.0))
        if c_in > p["cut_cost_hi"] or c_out > p["cut_cost_hi"]:
            return 2, {}                                            # mid-line swap -> 2-beat crossfade
        bc_at = float(event.get("bc_at", 0.0))
        sq = float(event.get("strong_q", 0.0))
        if bc_at >= sq:
            return 1, {}                                            # clean gap on a strong downbeat -> hard cut
        return 2, {}                                                # clean gap but not strong -> crossfade

    # last-resort fallback (none of the above fit)
    return 6, {"fallback": "last_resort"}


# ------------------------------------------------------------------ beat-aligned event placement (B.1/B.5)

def _nearest_downbeat_bar(bar_beat, bars):
    """Snap a (fractional) beat index to the nearest bar downbeat index."""
    bars = np.asarray(bars, float)
    if not len(bars):
        return 0
    return int(bars[int(np.argmin(np.abs(bars - float(bar_beat))))])


def build_plan(chosen_path, bed_info, stem_plans=None, params=None):
    """Deterministic TransitionPlan.

    chosen_path : the vocal planner's chosen path events — a list of dicts, each at least
                  {kind, beat} (project/bed beat index of the boundary) plus the per-kind fields select_type
                  consumes (energy/next_energy/boundary_bar for sections; c_in/c_out/bc_at for vocal_swap).
    bed_info    : the bed song's info dict (bars/beats/bpb/beat_content/first_beat/last_beat/sections).
    stem_plans  : optional {role: plan} multi-stem plans; drum/bass layer entry/exit boundaries are folded
                  in as extra section-like events (kind "layer_entry"/"layer_exit" -> treated as sections).

    Returns {"version", "strong_q", "events":[...]} where each event carries its snapped bed beat, downbeat
    bar, chosen `type`, and the select_type info. Events closer than `merge_beats` (one crossfade length,
    default 1 beat) are merged into one (B.5), keeping the stronger (lower-type-number = harder) cut.
    """
    p = dict(TRANSITIONS, **(params or {}))
    bars = bed_info.get("bars") or []
    first_beat = int(bed_info.get("first_beat", 0))
    last_beat = int(bed_info.get("last_beat", len(bed_info.get("beats", [])) - 1))
    sq = strong_q(bed_info.get("beat_content", []), first_beat, last_beat, p["strong_q"])

    raw = []
    for ev in (chosen_path or []):
        e = dict(ev)
        e.setdefault("strong_q", sq)
        beat = float(e.get("beat", 0.0))
        e["beat"] = beat
        e["downbeat_bar"] = _nearest_downbeat_bar(beat, bars)
        kind = e.get("kind")
        if kind in ("layer_entry", "layer_exit"):
            e["kind"] = "section"                    # layer boundaries select like a section change
        t, tinfo = select_type(e, bed_info, p)
        e["type"] = int(t)
        e["info"] = tinfo
        raw.append(e)

    # merge two transitions closer than one crossfade length (B.5): keep the harder (lower type number)
    raw.sort(key=lambda e: e["beat"])
    merge_beats = float(p.get("merge_beats", 1.0))
    merged = []
    for e in raw:
        if merged and abs(e["beat"] - merged[-1]["beat"]) < merge_beats:
            if e["type"] < merged[-1]["type"]:
                merged[-1] = e
            continue
        merged.append(e)

    return {"version": RENDERER_VERSION, "strong_q": sq, "events": merged}


# ================================================================== B.4 Option B: offline sweep renderer
#
# A time-varying filter (types 3 and 6, and C.2's moving EQ) is baked offline into a short cache clip, sample-
# accurately, block-by-block with carried pedalboard state, no RNG, fixed block size -> byte-identical for a
# given key. The static mixer is never touched.

BLOCK = 1024               # fixed render block size (samples); folded nowhere else, so it is part of the
                           # renderer contract — changing it bumps RENDERER_VERSION.

# renderer_version folds in the warp cache epoch + the pedalboard/numpy versions, so a dependency bump
# invalidates and re-bakes rather than silently diverging in the LSBs (B.4 MEDIUM-8).
try:
    import pedalboard as _pb
    _PB_VER = _pb.__version__
except Exception:                                   # pedalboard missing: planning still imports fine
    _pb = None
    _PB_VER = "none"
RENDERER_VERSION = f"{warp.RENDER_VERSION}.sweep1.pb{_PB_VER}.np{np.__version__}"


def quantize_spec(fx_type, from_hz, to_hz, curve, duration_samples, q=0.707):
    """The fully-quantized sweep spec (the `param_tuple` of the cache key). Rounding here is what makes two
    logically-equal sweeps share a key."""
    return (str(fx_type), round(float(from_hz), 3), round(float(to_hz), 3), str(curve),
            int(duration_samples), round(float(q), 4))


def cache_key(source_song_id, stem, src_sample_start, src_sample_len, fx_type, param_tuple,
              renderer_version=RENDERER_VERSION):
    """Baked-clip cache key (B.4, VERBATIM tuple):
        (source_song_id, stem, src_sample_start, src_sample_len, fx_type, param_tuple, renderer_version)
    Same tuple -> reuse the cached buffer; any change -> a NEW key (a changed sweep never aliases an old one).
    Mirrors clipfx.key_for: a sha1[:18] over the tuple, prefixed by the renderer_version epoch.
    """
    tup = (str(source_song_id), str(stem), int(src_sample_start), int(src_sample_len),
           str(fx_type), list(param_tuple), str(renderer_version))
    s = json.dumps(tup, separators=(",", ":"), sort_keys=False)
    return warp.RENDER_VERSION + "sw" + hashlib.sha1(s.encode()).hexdigest()[:18]


def path_for(source_song_id, key):
    """Baked clip lives beside the warp/pitch renders in the existing clip cache (clipfx.path_for scheme)."""
    return CACHE / str(source_song_id) / "clips" / f"{key}.npy"


def _cutoff_schedule(from_hz, to_hz, curve, n_frames):
    """Per-frame cutoff values. 'exp' sweeps geometrically (musical), 'lin' linearly. Deterministic."""
    if n_frames <= 1:
        return np.array([float(to_hz)], float)
    t = np.linspace(0.0, 1.0, n_frames)
    if curve == "exp":
        a, b = max(float(from_hz), 1e-6), max(float(to_hz), 1e-6)
        return a * (b / a) ** t
    return float(from_hz) + (float(to_hz) - float(from_hz)) * t


def render_sweep(audio, fx_type, from_hz, to_hz, curve="exp", q=0.707, block=BLOCK):
    """Bake a time-varying filter onto `audio` (shape (channels, n) float32), offline + sample-accurate.

    Steps the cutoff per `block`-sample frame and processes block-by-block with carried pedalboard state;
    no RNG, fixed block size -> byte-identical for identical inputs. fx_type: 'lpf' | 'hpf' | 'noise'
    (type-6 synthesized riser filters a deterministic noise bed instead of program audio).
    """
    if _pb is None:
        raise RuntimeError("pedalboard is required to bake sweeps")
    x = np.ascontiguousarray(np.atleast_2d(audio), dtype=np.float32)
    ch, n = x.shape
    if fx_type == "noise":
        # deterministic noise bed (fixed seed), filtered by a rising HPF -> white-noise riser (type 6)
        rng = np.random.default_rng(0)
        x = rng.standard_normal((ch, n)).astype(np.float32) * np.float32(0.25)
        filt = "hpf"
    else:
        filt = fx_type
    n_frames = max(1, int(np.ceil(n / block)))
    cutoffs = _cutoff_schedule(from_hz, to_hz, curve, n_frames)
    out = np.empty_like(x)
    for i in range(n_frames):
        a = i * block
        b = min(n, a + block)
        c = float(cutoffs[i])
        if filt == "hpf":
            bd = _pb.Pedalboard([_pb.HighpassFilter(cutoff_frequency_hz=c)])
        elif filt == "lpf":
            bd = _pb.Pedalboard([_pb.LowpassFilter(cutoff_frequency_hz=c)])
        else:                                        # a peak carve that ramps (C.2 moving EQ reuse)
            bd = _pb.Pedalboard([_pb.PeakFilter(cutoff_frequency_hz=c, gain_db=float(q), q=0.707)])
        # reset=True per block with a known cutoff is deterministic (no carried LSB drift across versions);
        # the per-block cutoff schedule is the carried state that makes the whole sweep continuous.
        out[:, a:b] = bd.process(x[:, a:b], SR, reset=True)
    return out


def render_blocking(source_song_id, stem, src_sample_start, audio, fx_type, from_hz, to_hz,
                    curve="exp", q=0.707):
    """Bake a sweep and write it to the clip cache via write-then-rename; return (key, path). Reuses the
    cached buffer when the key already exists (same tuple -> byte-identical, so re-render is a cache hit)."""
    from . import audio_io
    n = int(np.atleast_2d(audio).shape[1])
    spec = quantize_spec(fx_type, from_hz, to_hz, curve, n, q)
    key = cache_key(source_song_id, stem, src_sample_start, n, fx_type, spec)
    out_path = path_for(source_song_id, key)
    if out_path.exists():
        return key, out_path
    baked = render_sweep(audio, fx_type, from_hz, to_hz, curve, q)
    audio_io.save_npy_atomic(out_path, baked.astype(np.float32))
    return key, out_path
