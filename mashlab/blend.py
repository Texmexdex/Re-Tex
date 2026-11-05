"""Full multi-stem blending (design C): role selection, frequency-slotting (EQ carving), energy match, and
chop-and-recombine.

The bed song's "other" stem + structure is the backbone; drums/bass/(optional other-overlay) each come from a
chosen source song (or the bed's own). The core new DSP is C.2 frequency-slotting: a deterministic
complementary EQ computed per section from the lanes' average spectra so the same inputs always yield the same
filter list. Static per-lane cuts become `_TrackFX` settings / baked cache clips; moving EQ becomes a
pre-rendered cache clip (B.4 Option B) — the mixer is never touched.

All functions are pure/deterministic over arrays (PSDs, groove dicts, beat_content, chord roots) so the whole
module is unit-testable without GPU or real audio decode.
"""
import numpy as np

from . import harmfit
from .config import SR

# ------------------------------------------------------------------ C.2 pinned constants (VERBATIM)

BAND_EDGES = [30, 60, 120, 250, 500, 1000, 2000, 4000, 8000, 16000]   # 9 bands
LANE_PRIORITY = ("vocals", "bass", "drums", "other")                   # high -> low ownership order
CONTENDED_SHARE = 0.25     # a band is contended when >=2 lanes each hold >=25% of the band's mass
CUT_CAP_DB = 6.0           # per-lane total cut cap (summed across bands)
HARD_OVERLAP = 0.75        # overlap at/above which the required cut saturates to the full 6 dB cap
CONGESTED_BANDS = 3        # >= this many hard-contended bands on a core lane -> NEUTRAL/ESCALATE (untuned)
CORE_LANES = ("vocals", "bass", "drums")

# C.4 chop
CELL_QUALIFY = 40          # mean beat_content a cell must reach to qualify (0..100)
CONTENT_NONE = 12          # <= this is dropped as filler (mirrors timing.CONTENT_NONE)
SWING_CELL_2BEAT = 0.1     # swing >= this -> 2-beat cells
BODY_BAND = (150, 1500)    # beats.body_env snare/clap band
HAT_HZ = 6000              # > this is hats/cymbals

H_NEUTRAL = 0.75           # pairs.H_NEUTRAL (harmfit neutral-credit threshold); imported-value mirror
COMPAT_GATE = 0.60         # chroma compat gate for the optional other-overlay


# ================================================================== C.1 role selection

def energy_corr(a, b):
    """Pearson correlation of two energy envelopes (per-section means), 0 for degenerate input."""
    a = np.asarray(a, float).ravel()
    b = np.asarray(b, float).ravel()
    n = min(len(a), len(b))
    if n < 2:
        return 0.0
    a, b = a[:n], b[:n]
    sa, sb = a.std(), b.std()
    if sa < 1e-9 or sb < 1e-9:
        return 0.0
    return float(((a - a.mean()) * (b - b.mean())).mean() / (sa * sb))


def drum_fit(groove_L, groove_bed, env_L, env_bed):
    """Drum role-fit: groove compatibility (closeness of s8/s16 swing) + energy-envelope correlation. Never
    pitched. Returns a 0..1-ish score (higher = better)."""
    gL = groove_L or {}
    gB = groove_bed or {}
    ds8 = abs(float(gL.get("s8", 0.0)) - float(gB.get("s8", 0.0)))
    ds16 = abs(float(gL.get("s16", 0.0)) - float(gB.get("s16", 0.0)))
    groove_compat = 1.0 - np.clip(0.5 * (ds8 + ds16), 0.0, 1.0)
    corr = max(0.0, energy_corr(env_L, env_bed))
    return float(0.5 * groove_compat + 0.5 * corr)


def bass_fit(tonality_L, shift_scores_bass, compat_fallback):
    """Bass role-fit (C.1), harmfit-based with a chroma-compat neutral fallback.

    tonality_L         : automash.tonality(L, "bass") in 0..1 — gates whether key matters at all.
    shift_scores_bass  : harmfit.shift_scores(X_bass) best-over-keys value (max_k [0,k]) vs the bed chords.
    compat_fallback    : chroma automash.compat score (neutral fallback where harmfit relevance is low).

    Low tonality -> key doesn't matter -> use the chroma compat fallback. Otherwise blend the harmfit score
    (how well the bass locks to the actual chords) with the chroma fallback by tonality weight.
    Returns {"fit", "mode"} where mode is "chroma" (low tonality) or "harmfit".
    """
    t = float(np.clip(tonality_L, 0.0, 1.0))
    best = float(shift_scores_bass)
    comp = float(compat_fallback)
    if t < 0.5:
        return {"fit": comp, "mode": "chroma"}
    fit = t * best + (1 - t) * comp
    return {"fit": float(fit), "mode": "harmfit"}


def overlay_allowed(compat_bed_overlay, shift_scores_other_best, core_cut_within_cap):
    """Optional other-overlay gate (C.1, default OFF). ALL three must hold:
        1. chroma compat automash.compat(bed, overlay) >= 0.60
        2. harmfit max_k shift_scores(X_other)[0,k] >= H_NEUTRAL (0.75)
        3. the C.2 carve leaves no core lane past the 6 dB cap (core_cut_within_cap True)
    A missing/failing gate fails safe (nothing added)."""
    return bool(float(compat_bed_overlay) >= COMPAT_GATE
                and float(shift_scores_other_best) >= H_NEUTRAL
                and bool(core_cut_within_cap))


# ================================================================== C.2 frequency-slotting (EQ carving)

def band_energies(freqs, psd):
    """Integrate a Welch PSD into the 9 fixed octave bands. freqs/psd 1-D arrays. Returns (9,) band energies."""
    freqs = np.asarray(freqs, float)
    psd = np.asarray(psd, float)
    out = np.zeros(len(BAND_EDGES) - 1)
    for i in range(len(BAND_EDGES) - 1):
        lo, hi = BAND_EDGES[i], BAND_EDGES[i + 1]
        m = (freqs >= lo) & (freqs < hi)
        if m.any():
            out[i] = float(np.trapezoid(psd[m], freqs[m])) if m.sum() > 1 else float(psd[m].sum())
    return out


def lane_band_energy(lane_audio, sr=SR):
    """Welch PSD -> 9-band energies for one lane's post-warp, project-grid audio sliced to [t0,t1). A lane
    with no audio (empty/all-zero) contributes zero in every band."""
    from scipy.signal import welch
    y = np.asarray(lane_audio, float)
    if y.ndim > 1:
        y = y.mean(0)
    if not len(y) or not np.any(y):
        return np.zeros(len(BAND_EDGES) - 1)
    nper = min(len(y), 4096)
    f, pxx = welch(y, fs=sr, nperseg=nper)
    return band_energies(f, pxx)


def band_shares(lane_bands):
    """lane_bands: {lane: (9,) energies}. Returns per-band normalized shares {lane: (9,)} summing to 1 per
    band (bands with zero total mass yield zero shares)."""
    lanes = list(lane_bands)
    E = np.array([lane_bands[l] for l in lanes], float)          # (n_lanes, 9)
    total = E.sum(0)                                             # (9,)
    shares = np.zeros_like(E)
    nz = total > 1e-18
    shares[:, nz] = E[:, nz] / total[nz]
    return {l: shares[i] for i, l in enumerate(lanes)}


def _priority_rank(lane):
    return LANE_PRIORITY.index(lane) if lane in LANE_PRIORITY else len(LANE_PRIORITY)


def _geomean(a, b):
    return float(np.sqrt(float(a) * float(b)))


def carve_band(shares, band_index):
    """Decide the carve for one band. `shares` = {lane: (9,) share}. A band is CONTENDED when >=2 lanes each
    hold >= CONTENDED_SHARE. overlap = the 2nd-largest lane share. The lower-priority of the two leading lanes
    is cut:
        gain_db = -clip(2 + 6*(overlap-0.25), 2, 5);  Q = 0.5 + 1.5*overlap;  center = geomean(band edges)
    Returns None (not contended) or a dict with the carve. 'hard' is True when overlap >= 0.75 (full-cap cut).
    """
    lanes = list(shares)
    vals = {l: float(shares[l][band_index]) for l in lanes}
    occupying = [l for l in lanes if vals[l] >= CONTENDED_SHARE]
    if len(occupying) < 2:
        return None
    order = sorted(lanes, key=lambda l: -vals[l])
    top, second = order[0], order[1]
    overlap = vals[second]                                       # 2nd-largest lane share
    # cut the lower-priority of the two leading lanes
    lower = top if _priority_rank(top) > _priority_rank(second) else second
    gain_db = -float(np.clip(2 + 6 * (overlap - 0.25), 2, 5))
    Q = 0.5 + 1.5 * overlap
    center = _geomean(BAND_EDGES[band_index], BAND_EDGES[band_index + 1])
    is_bass_kick = (band_index in (0, 1)                        # 30-60 / 60-120 Hz
                    and {top, second} == {"bass", "drums"})
    return {
        "band": band_index, "lane": lower, "overlap": float(overlap),
        "gain_db": gain_db, "Q": float(Q), "center": center,
        "hard": overlap >= HARD_OVERLAP, "bass_kick": is_bass_kick,
    }


def frequency_slot(lane_bands):
    """C.2 full carve for one section window.

    lane_bands: {lane: (9,) band energies} for every lane over the SAME [t0,t1) window (lanes absent in the
    window contribute zeros). Returns:
        {"carves": [...], "ducks": [...], "congested": bool, "congested_lane": lane|None, "escalate": lane|None}

    - carves: PeakFilter cuts {band, lane, gain_db, Q, center, overlap, hard} with the per-lane total cut
      capped at CUT_CAP_DB (lowest-overlap bands dropped first once the cap is hit).
    - ducks: bass-vs-kick bands (30-60 / 60-120 Hz where bass & drums lead) become a sidechain duck of bass
      under the kick, NOT an EQ cut (reuse F7 ducking).
    - HIGH-2 NEUTRAL/ESCALATE: if a CORE lane is hard-contended (overlap>=0.75) in >= CONGESTED_BANDS of 9
      bands, do NOT carve -> flag 'spectrally congested', escalate the lower-priority lane to C.4 chop.
    """
    shares = band_shares(lane_bands)
    raw = [carve_band(shares, b) for b in range(len(BAND_EDGES) - 1)]
    raw = [c for c in raw if c is not None]

    # HIGH-2: count hard-contended bands per CORE lane (the lane that would be carved)
    hard_by_lane = {}
    for c in raw:
        if c["hard"] and c["lane"] in CORE_LANES:
            hard_by_lane.setdefault(c["lane"], 0)
            hard_by_lane[c["lane"]] += 1
    congested_lane = next((l for l in CORE_LANES if hard_by_lane.get(l, 0) >= CONGESTED_BANDS), None)
    if congested_lane is not None:
        # legitimate co-occupation: do NOT carve. Flag + route the lower-priority lane to C.4 chop.
        return {"carves": [], "ducks": [], "congested": True,
                "congested_lane": congested_lane, "escalate": congested_lane}

    # split bass-vs-kick bands into ducks; keep the rest as carves
    carves, ducks = [], []
    for c in raw:
        (ducks if c["bass_kick"] else carves).append(c)

    # per-lane total cut cap (<= 6 dB): keep the deepest-overlap bands first, drop the rest once over cap
    capped = []
    by_lane = {}
    for c in sorted(carves, key=lambda c: -c["overlap"]):
        used = by_lane.get(c["lane"], 0.0)
        if used + abs(c["gain_db"]) <= CUT_CAP_DB + 1e-6:
            by_lane[c["lane"]] = used + abs(c["gain_db"])
            capped.append(c)
        # else: dropping over-cap carves (prefer not carving the core three past the cap)
    capped.sort(key=lambda c: c["band"])
    return {"carves": capped, "ducks": ducks, "congested": False,
            "congested_lane": None, "escalate": None}


# ================================================================== C.3 energy-envelope level match

def section_gain(bed_section_energy, layer_section_energy, base_gain_db, lo=-12.0, hi=12.0):
    """Per-section energy-envelope level match (extends the _replace_gain idea). A layer's loudness tracks the
    bed's section dynamics: louder bed section -> relatively more layer gain, quieter -> less. `base_gain_db`
    is the static replace-gain; the per-section delta rides on top, clipped to [lo, hi]."""
    be = float(np.clip(bed_section_energy, 0.0, 1.0))
    le = float(np.clip(layer_section_energy, 0.0, 1.0))
    # match the layer's own section energy toward the bed's (positive when the bed is louder than the layer)
    delta_db = 6.0 * (be - le)
    return float(np.clip(base_gain_db + delta_db, lo, hi))


# ================================================================== C.4 chop-and-recombine

def chop_triggered(groove_L, groove_bed, tonality_bass, shift_scores_bass_best, style_chop=False):
    """C.4 triggers (ANY, VERBATIM):
        |groove(L).s8 - groove(I).s8| > 0.15  (OR s16 > 0.15);
        OR tonality(L,"bass") >= 0.5 AND max_k shift_scores(X_bass)[0,k] < H_NEUTRAL (0.75);
        OR style chop:true.
    L = source lane, I = bed. Returns {"fire": bool, "reasons": [...]}.
    """
    reasons = []
    gL, gB = groove_L or {}, groove_bed or {}
    if abs(float(gL.get("s8", 0.0)) - float(gB.get("s8", 0.0))) > 0.15:
        reasons.append("swing_s8")
    if abs(float(gL.get("s16", 0.0)) - float(gB.get("s16", 0.0))) > 0.15:
        reasons.append("swing_s16")
    if float(tonality_bass) >= 0.5 and float(shift_scores_bass_best) < H_NEUTRAL:
        reasons.append("bass_unplaceable")
    if style_chop:
        reasons.append("style")
    return {"fire": len(reasons) > 0, "reasons": reasons}


def cell_size_beats(swing):
    """Cell = 1 beat by default; 2 beats when the resolved style sets swing >= 0.1 (D.2)."""
    return 2 if float(swing) >= SWING_CELL_2BEAT else 1


def qualify_cells(cells):
    """A source cell qualifies when its mean beat_content >= 40; <= 12 is dropped as filler. `cells` is a list
    of dicts each with at least {"beat_content": mean(0..100), "beat": source_beat_index, ...}. Returns the
    qualifying cells, each annotated with its strength score (= mean beat_content), in source order."""
    out = []
    for c in cells:
        bc = float(c.get("beat_content", 0.0))
        if bc <= CONTENT_NONE:
            continue
        if bc >= CELL_QUALIFY:
            d = dict(c)
            d["strength"] = bc
            out.append(d)
    return out


def classify_drum_cell(body_energy, hat_energy, bands_separable=True):
    """Drum body-vs-hat via the band-energy split (beats.body_env 150-1500 Hz vs > 6 kHz). Returns "body"
    (kick/snare dominant — keep on strong beats), "hat" (predominantly > 6 kHz), or "coarse" when the bands
    can't be separated (bus-processed/distorted stems) — the coarse-chop fallback."""
    if not bands_separable:
        return "coarse"
    return "body" if float(body_energy) >= float(hat_energy) else "hat"


def chop_sequence(bed_bar_spans, pool, swing=0.0, bands_separable=True):
    """C.4 re-sequencing loop (VERBATIM). Tile the bed's bar span (structure.bar_spans) beat by beat:
    downbeat = highest-beat_content qualifying body cell; backbeat/offbeat = next-best; preserve the source
    cells' intra-bar order where multiple qualify; wrap/reuse the source cell pool when the bed is longer.

    bed_bar_spans : list of (beat_start, beat_end) per bed bar (structure.bar_spans output).
    pool          : qualifying source cells (output of qualify_cells); each may carry "kind" (body/hat/coarse)
                    and "src_order" (its original intra-bar position) and "strength".
    Returns a list of placements {"bar", "beat_in_bar", "cell", "role"} and a note for the coarse fallback.
    """
    cell_beats = cell_size_beats(swing)
    qualifying = list(pool)
    note = None
    if not qualifying:
        return {"placements": [], "note": "no qualifying cells"}

    if not bands_separable:
        note = "coarse chop (drum bands not separable)"
        # coarse: keep/drop whole cells by beat_content, no hat-selective dropping
        body_cells = sorted(qualifying, key=lambda c: -c["strength"])
        hat_cells = []
    else:
        body_cells = sorted([c for c in qualifying if c.get("kind", "body") == "body"],
                            key=lambda c: -c["strength"])
        hat_cells = sorted([c for c in qualifying if c.get("kind") == "hat"], key=lambda c: -c["strength"])
    # the "next-best" pool for backbeats: hats first (deprioritized on contended beats), then spare bodies
    backbeat_pool = hat_cells + body_cells[1:] if body_cells else hat_cells

    placements = []
    bi_pool = 0
    bb_pool = 0
    for bar_i, (a, b) in enumerate(bed_bar_spans):
        span = max(1, int(b) - int(a))
        for beat_in_bar in range(0, span, cell_beats):
            is_downbeat = (beat_in_bar == 0)
            if is_downbeat and body_cells:
                cell = body_cells[bi_pool % len(body_cells)]
                bi_pool += 1
                role = "downbeat"
            elif backbeat_pool:
                cell = backbeat_pool[bb_pool % len(backbeat_pool)]
                bb_pool += 1
                role = "backbeat"
            elif body_cells:
                cell = body_cells[bi_pool % len(body_cells)]
                bi_pool += 1
                role = "downbeat"
            else:
                continue
            placements.append({"bar": bar_i, "beat_in_bar": beat_in_bar, "cell": cell, "role": role})
    return {"placements": placements, "note": note}


def bass_root_on_downbeat(cell_root_pc, bed_chord_root_pc):
    """Bass chop qualification: a cell qualifies for a downbeat when its dominant pitch class matches the bed
    chord root at that beat (via the same harmfit/chroma machinery C.1 uses)."""
    return int(cell_root_pc) % 12 == int(bed_chord_root_pc) % 12
