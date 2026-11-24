"""Zero-phase full-band transient audit and beat-grid verifier.

Design guarantees:
  1. Zero-phase Butterworth filters (scipy.signal.filtfilt -> 0.0 ms group delay) in two bands:
       - kick/sub: 30-160 Hz (never blind to electronic / metal kicks below 150 Hz)
       - snare/clap: 250-3500 Hz
     sampled at 500 Hz (2 ms frame resolution).
  2. Always verifies beat phase at k = 1 (the actual beat), NEVER letting k = 4 (16ths) or
     k = 3 (triplets) lock 1/4 or 1/3 of a beat off while reporting low subdivision RMS.
  3. Continuity-constrained Viterbi phase tracking over overlapping 16-beat active windows
     eliminates both constant detector phase offsets and live tempo drift without letting
     isolated syncopated fills yank the grid.
  4. Steady-first IRLS straight-line lock: studio recordings that hold one tempo stay dead-steady
     with zero beat-to-beat speed whiplash; takes that truly drift get smooth tracking, evenly
     spaced drumless breaks, and at most +-5 ms gentle hit alignment.
"""
import numpy as np
from scipy.signal import butter, filtfilt

from .config import SR

ENV_FPS = 500.0            # 2 ms frame resolution for onset envelopes
ACTIVE_SHARE = 0.22        # window mean onset >= 22 % of song's active median -> drums active
OFFBEAT_WARN_MS = 22.0     # > 22 ms off real drum hits triggers off-beat warning / fix
DRIFT_WARN_MS = 16.0       # > 16 ms cumulative drift from a straight line -> recommends lock

_DELTAS = np.linspace(-0.5, 0.5, 201)     # 0.005 beat step (~2-3 ms resolution)
_LOCAL_MASK = np.abs(_DELTAS) <= 0.10     # +-10 % of a beat around 0


# ------------------------------------------------------------------ zero-phase full-band drum onsets

def onset_curves(drums_mono, sr=SR, mix_mono=None):
    """Return (body, kick, snare, fps, source) at ~500 Hz with ZERO filter phase delay.

    `drums_mono`: 1D float array (or (2, n) stereo, averaged to mono). If drums are silent and
    `mix_mono` is supplied, explicitly uses `mix_mono` and sets source='mix'.
    """
    source = "drums"
    y = np.asarray(drums_mono, np.float32)
    if y.ndim == 2:
        y = y.mean(0)
    if y.size == 0 or float(np.sqrt(np.mean(y ** 2))) < 1e-4:
        if mix_mono is not None:
            m = np.asarray(mix_mono, np.float32)
            if m.ndim == 2:
                m = m.mean(0)
            if m.size > 0 and float(np.sqrt(np.mean(m ** 2))) >= 1e-4:
                y = m
                source = "mix"

    hop = max(1, int(round(sr / ENV_FPS)))
    fps = sr / hop
    n_fr = len(y) // hop
    if n_fr < 16 or len(y) < 64:
        z = np.zeros(max(16, n_fr), np.float32)
        return z, z, z, fps, source

    nyq = sr * 0.5
    b_k, a_k = butter(2, [30.0 / nyq, min(160.0 / nyq, 0.99)], btype="band")
    b_s, a_s = butter(2, [250.0 / nyq, min(3500.0 / nyq, 0.99)], btype="band")
    k_sig = filtfilt(b_k, a_k, y).astype(np.float32)
    s_sig = filtfilt(b_s, a_s, y).astype(np.float32)

    k_rms = np.sqrt(np.mean(k_sig[:n_fr * hop].reshape(n_fr, hop) ** 2, axis=1))
    s_rms = np.sqrt(np.mean(s_sig[:n_fr * hop].reshape(n_fr, hop) ** 2, axis=1))

    # Half-wave rectified first difference on log-compressed RMS (sharp attack peaks, zero lag)
    k_on = np.maximum(0.0, np.diff(np.log1p(25.0 * k_rms), prepend=0.0)).astype(np.float32)
    s_on = np.maximum(0.0, np.diff(np.log1p(25.0 * s_rms), prepend=0.0)).astype(np.float32)

    kp = float(np.percentile(k_on, 99.5)) or 1.0
    sp = float(np.percentile(s_on, 99.5)) or 1.0
    k_n = k_on / kp
    s_n = s_on / sp
    body = (k_n + 0.85 * s_n).astype(np.float32)
    return body, k_n.astype(np.float32), s_n.astype(np.float32), fps, source


# ------------------------------------------------------------------ k=1 beat-hit phase audit

def _window_phase_profile(times, body, fps):
    """Full k=1 onset clarity curve across delta in [-0.5, +0.5] beats.

    Returns (clarity_curve, scores, per, med_score).
    """
    t = np.asarray(times, float)
    if len(t) < 4:
        return np.ones(len(_DELTAS), np.float64), np.zeros(len(_DELTAS), np.float64), 0.5, 1.0
    per = float(np.median(np.diff(t)))
    if per <= 0.1 or per >= 2.5:
        return np.ones(len(_DELTAS), np.float64), np.zeros(len(_DELTAS), np.float64), 0.5, 1.0
    n = len(body)
    ker = np.array([0.25, 0.5, 1.0, 0.5, 0.25], np.float32)
    sm = np.convolve(body, ker / ker.sum(), mode="same")

    scores = np.empty(len(_DELTAS), np.float64)
    for idx_d, d in enumerate(_DELTAS):
        fr = np.round((t + d * per) * fps).astype(np.int64)
        ok = (fr >= 0) & (fr < n)
        scores[idx_d] = float(sm[fr[ok]].mean()) if ok.any() else 0.0

    med_score = float(np.median(scores)) + 1e-9
    clarity_curve = scores / med_score
    return clarity_curve, scores, per, med_score


def _window_phase_offset(times, body, fps):
    """Honest k=1 beat phase offset (seconds) of `times` against `body` onset peaks.

    Positive offset means real drum hits occur AFTER `times` (`times` are early).
    Returns (offset_s, peak_score, zero_score, clarity).
    """
    clarity_curve, scores, per, _ = _window_phase_profile(times, body, fps)
    zero_score = float(scores[100])
    glob_i = int(np.argmax(scores))
    glob_score = float(scores[glob_i])

    local_indices = np.where(_LOCAL_MASK)[0]
    loc_rel = int(np.argmax(scores[local_indices]))
    loc_i = int(local_indices[loc_rel])
    loc_score = float(scores[loc_i])
    loc_cl = float(clarity_curve[loc_i])
    glob_cl = float(clarity_curve[glob_i])

    # Genuine drum hit near delta=0 (not overwhelmed by an off-beat peak) -> syncopation, not off-beat
    if (loc_cl >= 1.35 and (glob_score / max(loc_score, 1e-9)) < 2.1) or (glob_score / max(zero_score, 1e-9) < 1.35):
        return float(_DELTAS[loc_i] * per), loc_score, zero_score, loc_cl
    return float(_DELTAS[glob_i] * per), glob_score, zero_score, glob_cl


def viterbi_phase_track(curves, win_centers, p_med):
    """Globally smooth k=1 beat-phase offset trajectory (seconds) across active windows."""
    W, D = curves.shape
    if W == 0:
        return np.zeros(0, np.float64)
    # Soft per-window cap to <= 10.0 that preserves the exact peak shape (never flattens peaks into plateaus
    # on ultra-clean electronic kicks where the background median is near zero)
    prior = 0.22 * np.exp(-(_DELTAS / 0.065) ** 2)
    w_max = np.maximum(curves.max(axis=1, keepdims=True), 1e-9)
    c_soft = curves * (np.minimum(w_max, 10.0) / w_max)
    emit = c_soft + prior[None, :]
    if W == 1:
        return np.array([_DELTAS[int(np.argmax(emit[0]))] * p_med], np.float64)

    diff = (_DELTAS[None, :] - _DELTAS[:, None] + 0.5) % 1.0 - 0.5
    diff_sq = diff * diff

    dp = np.empty((W, D), np.float64)
    back = np.zeros((W, D), np.int32)
    dp[0] = emit[0]
    for w in range(1, W):
        gap_beats = max(4.0, float(win_centers[w] - win_centers[w - 1]))
        lam = 220.0 / (gap_beats / 4.0) ** 0.75
        trans = dp[w - 1][:, None] - lam * diff_sq
        idx = np.argmax(trans, axis=0)
        back[w] = idx
        dp[w] = emit[w] + trans[idx, np.arange(D)]

    path = np.empty(W, np.int32)
    path[W - 1] = int(np.argmax(dp[W - 1]))
    for w in range(W - 2, -1, -1):
        path[w] = back[w + 1, path[w + 1]]

    deltas_unwrapped = np.empty(W, np.float64)
    deltas_unwrapped[0] = _DELTAS[path[0]]
    for w in range(1, W):
        step_d = (_DELTAS[path[w]] - _DELTAS[path[w - 1]] + 0.5) % 1.0 - 0.5
        deltas_unwrapped[w] = deltas_unwrapped[w - 1] + step_d
    med_d = float(np.median(deltas_unwrapped))
    deltas_unwrapped -= round(med_d)
    return deltas_unwrapped * p_med


def audit_beats(beats, body, fps, win_beats=16, step_beats=8):
    """Audit a beat list against actual zero-phase drum onsets across the song.

    Returns dict with per-window measurements, median/max/signed offset in ms,
    active/bad window counts, straight-line drift stats, and status ('ok' | 'warn' | 'off_beat').
    """
    b = np.asarray(beats, float)
    n = len(b)
    if n < 8 or len(body) < 16 or float(np.max(body)) <= 1e-6:
        return dict(status="ok" if n >= 8 else "warn", median_offset_ms=0.0, max_offset_ms=0.0,
                    signed_median_ms=0.0, active_windows=0, bad_windows=0,
                    drift_pct=0.0, max_straight_err_ms=0.0, needs_quantize=False, sections=[])

    win_e = []
    for i in range(0, max(1, n - win_beats + 1), step_beats):
        seg = b[i:i + win_beats]
        f0, f1 = int(max(0, seg[0] * fps)), int(min(len(body), seg[-1] * fps))
        win_e.append(float(body[f0:f1].mean()) if f1 > f0 else 0.0)
    win_e = np.asarray(win_e, float)
    nz = win_e[win_e > 0.25 * (win_e.max() if len(win_e) else 0.0)]
    usual = float(np.median(nz)) if len(nz) else 0.0

    sections = []
    offs_ms = []
    for idx_w, i in enumerate(range(0, max(1, n - win_beats + 1), step_beats)):
        seg = b[i:min(n, i + win_beats)]
        if len(seg) < 8:
            continue
        active = bool(win_e[idx_w] >= ACTIVE_SHARE * usual and usual > 1e-4)
        per = float(np.median(np.diff(seg)))
        bpm_loc = 60.0 / per if per > 0 else 0.0
        if not active:
            sections.append(dict(i0=int(i), i1=int(i + len(seg)), t0=round(float(seg[0]), 2),
                                 t1=round(float(seg[-1]), 2), bpm=round(bpm_loc, 2),
                                 active=False, offset_ms=0.0, clarity=0.0, ratio=1.0))
            continue
        off_s, pk, z0, clarity = _window_phase_offset(seg, body, fps)
        off_ms = off_s * 1000.0
        ratio = pk / max(z0, 1e-6)
        offs_ms.append(off_ms)
        sections.append(dict(i0=int(i), i1=int(i + len(seg)), t0=round(float(seg[0]), 2),
                             t1=round(float(seg[-1]), 2), bpm=round(bpm_loc, 2),
                             active=True, offset_ms=round(off_ms, 1),
                             clarity=round(clarity, 2), ratio=round(ratio, 2)))

    j = np.arange(n, dtype=float)
    slope, intercept = np.polyfit(j, b, 1)
    straight_err_ms = np.abs(b - (intercept + slope * j)) * 1000.0
    max_straight_err_ms = float(np.percentile(straight_err_ms, 95))
    bpms_active = [s["bpm"] for s in sections if s["active"] and s["bpm"] > 0]
    drift_pct = (max(bpms_active) - min(bpms_active)) / max(1e-6, float(np.median(bpms_active))) * 100.0 if len(bpms_active) >= 2 else 0.0

    abs_offs = [abs(x) for x in offs_ms]
    med_abs = float(np.median(abs_offs)) if abs_offs else 0.0
    max_abs = float(np.max(abs_offs)) if abs_offs else 0.0
    signed_med = float(np.median(offs_ms)) if offs_ms else 0.0
    bad_wins = sum(1 for s in sections if s["active"] and abs(s["offset_ms"]) > OFFBEAT_WARN_MS and s["clarity"] >= 1.8)
    bad_thresh = max(2, int(round(0.10 * max(1, len(offs_ms)))))

    status = "ok"
    if med_abs > OFFBEAT_WARN_MS or bad_wins >= bad_thresh:
        status = "off_beat"
    elif bad_wins > 0 or max_abs > 18.0:
        status = "warn"

    needs_quantize = bool(max_straight_err_ms > DRIFT_WARN_MS or drift_pct > 0.35)
    return dict(
        status=status,
        median_offset_ms=round(med_abs, 1),
        max_offset_ms=round(max_abs, 1),
        signed_median_ms=round(signed_med, 1),
        active_windows=len(offs_ms),
        bad_windows=bad_wins,
        drift_pct=round(drift_pct, 2),
        max_straight_err_ms=round(max_straight_err_ms, 1),
        needs_quantize=needs_quantize,
        sections=sections,
    )


# ------------------------------------------------------------------ slip/glitch recounter & transient alignment

def recount_slips(beats):
    """Fix 2x / 0.5x detector tempo switches, missed-beat gaps, and isolated glitch sub-beats using local tempo."""
    b = np.asarray(beats, float)
    if len(b) < 12:
        return b
    for _ in range(2):
        iv = np.diff(b)
        p_glob = float(np.median(iv))
        if p_glob <= 0.05 or len(b) < 12:
            return b
        loc_p = np.empty(len(b), float)
        for k in range(len(b)):
            lo = max(0, k - 8)
            hi = min(len(iv), k + 8)
            loc_p[k] = float(np.median(iv[lo:hi])) if hi > lo else p_glob
        out = [float(b[0])]
        i = 1
        while i < len(b):
            d = float(b[i]) - out[-1]
            p_loc = float(loc_p[i])
            if p_loc < 0.62 * p_glob:
                p_loc = p_glob
            p_loc = max(0.18, p_loc)
            if d < 0.62 * p_loc:
                if i + 1 < len(b):
                    d2 = float(b[i + 1]) - float(b[i])
                    if 0.72 * p_loc <= (d + d2) <= 1.38 * p_loc:
                        out.append(float(b[i + 1]))
                        i += 2
                        continue
                if d < 0.60 * p_loc:
                    i += 1
                    continue
            if d >= 1.58 * p_loc:
                n_sub = max(2, int(round(d / p_loc)))
                prev = out[-1]
                for k in range(1, n_sub):
                    out.append(prev + d * k / n_sub)
                out.append(float(b[i]))
                i += 1
                continue
            out.append(float(b[i]))
            i += 1
        b = np.asarray(out, float)
    return b


def _even_inactive_gaps(beats, active_mask):
    """Interpolate drumless intros/breaks/outros linearly between their active boundary beats."""
    b = beats.copy()
    n = len(b)
    i = 0
    while i < n:
        if active_mask[i]:
            i += 1
            continue
        j = i
        while j < n and not active_mask[j]:
            j += 1
        if i > 0 and j < n:
            b[i:j] = np.linspace(b[i - 1], b[j], j - i + 2)[1:-1]
        elif i == 0 and j < n:
            per = float(np.median(np.diff(b[j:min(n, j + 16)]))) if j + 2 < n else float(np.median(np.diff(b)))
            b[0:j] = b[j] - np.arange(j, 0, -1) * per
        elif j == n and i > 0:
            per = float(np.median(np.diff(b[max(0, i - 16):i]))) if i >= 2 else float(np.median(np.diff(b)))
            b[i:n] = b[i - 1] + np.arange(1, n - i + 1) * per
        i = j
    return b


def _gentle_hit_snap(beats, active_mask, body, fps, max_shift_s=0.005):
    """Nudge active beats up to +-max_shift_s onto unambiguous transient peaks without breaking monotonicity."""
    b = beats.copy()
    w = max(1, int(round(max_shift_s * fps)))
    ctx_w = w * 6
    n = len(body)
    for k in range(1, len(b) - 1):
        if not active_mask[k]:
            continue
        i = int(round(b[k] * fps))
        lo, hi = max(0, i - w), min(n, i + w + 1)
        if hi - lo < 3:
            continue
        seg = body[lo:hi]
        p = int(np.argmax(seg))
        ctx = body[max(0, i - ctx_w):min(n, i + ctx_w + 1)]
        if seg[p] > 2.0 * (float(np.median(ctx)) + 1e-6) and p not in (0, len(seg) - 1):
            cand = (lo + p) / fps
            if b[k - 1] + 0.15 < cand < b[k + 1] - 0.15:
                b[k] = cand
    return b


def _pick_downbeat_offset(beats, kick, snare, fps, bpb, ref_bars_t=None):
    """Choose which beat index mod `bpb` is Beat 1 (downbeat)."""
    b = np.asarray(beats, float)
    n = len(b)
    if n < bpb * 2:
        return 0
    r = max(1, int(round(0.018 * fps)))
    kick_scores = np.zeros(bpb, float)
    for ph in range(bpb):
        idx = np.round(b[ph::bpb] * fps).astype(np.int64)
        vals = [float(kick[max(0, i - r):min(len(kick), i + r + 1)].max()) for i in idx if 0 <= i < len(kick)]
        kick_scores[ph] = float(np.mean(vals)) if vals else 0.0

    if ref_bars_t is not None and len(ref_bars_t) >= 2:
        votes = np.zeros(bpb, int)
        per = float(np.median(np.diff(b)))
        for bt in ref_bars_t:
            k = int(np.argmin(np.abs(b - bt)))
            if abs(b[k] - bt) < 0.25 * per:
                votes[k % bpb] += 1
        ref_ph = int(np.argmax(votes))
        if votes[ref_ph] >= 2 and kick_scores[ref_ph] >= 0.75 * float(kick_scores.max()):
            return ref_ph

    return int(np.argmax(kick_scores))


def _align_seed(seed, body, fps, max_shift_s=0.005):
    """Align a monotonic beat sequence `seed` onto zero-phase k=1 drum hits (preserving length).

    Returns (fixed_beats, mode, active_mask).
    """
    n = len(seed)
    if n < 8 or len(body) < 16 or float(np.max(body)) <= 1e-6:
        return seed.copy(), "verified", np.ones(n, bool)
    p_med = float(np.median(np.diff(seed)))
    if p_med <= 0.1 or p_med >= 2.5:
        return seed.copy(), "verified", np.ones(n, bool)

    win = 16
    step = 4
    all_e = []
    for i in range(0, max(1, n - win + 1), step):
        seg = seed[i:i + win]
        f0, f1 = int(max(0, seg[0] * fps)), int(min(len(body), seg[-1] * fps))
        all_e.append(float(body[f0:f1].mean()) if f1 > f0 else 0.0)
    all_e = np.asarray(all_e, float)
    nz = all_e[all_e > 0.25 * (all_e.max() if len(all_e) else 0.0)]
    usual = float(np.median(nz)) if len(nz) else 0.0

    win_centers = []
    win_curves = []
    for idx_w, i in enumerate(range(0, max(1, n - win + 1), step)):
        seg = seed[i:min(n, i + win)]
        if len(seg) < 8:
            continue
        if all_e[idx_w] < ACTIVE_SHARE * usual or usual <= 1e-4:
            continue
        c_curve, _, _, _ = _window_phase_profile(seg, body, fps)
        if float(c_curve.max()) >= 1.5:
            win_centers.append(i + 0.5 * (len(seg) - 1))
            win_curves.append(c_curve)

    active_mask = np.zeros(n, bool)
    for c_idx in win_centers:
        lo = max(0, int(c_idx - win // 2))
        hi = min(n, int(c_idx + win // 2))
        active_mask[lo:hi] = True

    if len(win_centers) >= 2:
        wc = np.asarray(win_centers, float)
        wo = viterbi_phase_track(np.asarray(win_curves, np.float64), wc, p_med)
        j_all = np.arange(n, dtype=float)
        off_curve = np.interp(j_all, wc, wo)
        ker_len = min(7, max(3, (n // 16) | 1))
        ker = np.hanning(ker_len + 2)[1:-1]
        off_smooth = np.convolve(np.pad(off_curve, ker_len // 2, mode="edge"), ker / ker.sum(), mode="valid")[:n]
        cand_follow = seed + off_smooth

        min_gap = 0.35 * p_med
        for k in range(1, len(cand_follow)):
            if cand_follow[k] < cand_follow[k - 1] + min_gap:
                cand_follow[k] = cand_follow[k - 1] + min_gap

        act_j = j_all[active_mask]
        act_t = cand_follow[active_mask]
        if len(act_j) >= 8:
            slope, intercept = np.polyfit(act_j, act_t, 1)
            res_s = np.abs(act_t - (intercept + slope * act_j))
            inliers = res_s <= max(0.018, float(np.percentile(res_s, 75)) * 1.8)
            if inliers.sum() >= 8:
                slope, intercept = np.polyfit(act_j[inliers], act_t[inliers], 1)
            cand_straight = intercept + slope * j_all
            straight_dev = float(np.percentile(np.abs(act_t - cand_straight[active_mask]) * 1000.0, 85))

            aud_follow = audit_beats(cand_follow, body, fps)
            aud_straight = audit_beats(cand_straight, body, fps)

            if straight_dev <= 14.0 and aud_straight["median_offset_ms"] <= max(10.0, aud_follow["median_offset_ms"] + 4.0):
                return cand_straight, "steady_locked", active_mask

        fixed = _even_inactive_gaps(cand_follow, active_mask)
        if max_shift_s > 0:
            fixed = _gentle_hit_snap(fixed, active_mask, body, fps, max_shift_s=max_shift_s)
        return fixed, "drift_tracked", active_mask

    if len(win_centers) == 1:
        off_s = float(_DELTAS[int(np.argmax(win_curves[0]))] * p_med)
        return seed + off_s, "phase_shifted", active_mask

    return seed.copy(), "verified", active_mask


def align_marks_to_hits(marks, drums_mono, sr=SR, mix_mono=None, max_shift_s=0.005):
    """Smoothly align an existing beat mark list onto zero-phase k=1 drum hits (exact same length).

    Returns (aligned_marks, moved_count, mode). Never introduces +-40 ms beat-to-beat speed whiplash
    or causal filter lag.
    """
    m = np.asarray(marks, float)
    if len(m) < 8:
        return m.copy(), 0, "verified"
    body, _, _, fps, _ = onset_curves(drums_mono, sr=sr, mix_mono=mix_mono)
    if float(np.max(body)) <= 1e-6:
        return m.copy(), 0, "verified"
    before = audit_beats(m, body, fps)
    if before["active_windows"] == 0:
        return m.copy(), 0, "verified"
    fixed, mode, _ = _align_seed(m, body, fps, max_shift_s=max_shift_s)
    # Keep first/last inside valid non-negative time and strictly rising
    if fixed[0] < 0.0 and m[0] >= 0.0:
        fixed = np.maximum(0.0, fixed)
    for k in range(1, len(fixed)):
        if fixed[k] <= fixed[k - 1] + 0.02:
            fixed[k] = fixed[k - 1] + 0.02
    after = audit_beats(fixed, body, fps)
    # Accept aligned grid unless it made hit accuracy worse
    if after["median_offset_ms"] > before["median_offset_ms"] + 5.0 and after["bad_windows"] > before["bad_windows"]:
        return m.copy(), 0, "kept"
    moved = int(np.sum(np.abs(fixed - m) > 0.001))
    return fixed, moved, mode


def verify_and_fix(raw_beats, dur, drums_mono, sr=SR, bpb=4, ref_bars_t=None, mix_mono=None, user_marks_t=None):
    """Build a transient-verified beat grid from raw detector beats + audio (no song.json mutation).

    Fixes 2x/0.5x slips and isolated glitch beats, runs k=1 Viterbi tracking against zero-phase
    kick+snare onsets, picks steady straight line when drums are steady, evens drumless breaks,
    picks downbeat phase, and returns full pre/post audit reports.
    """
    dur = float(dur or 0.0)
    bpb = int(bpb or 4)
    raw_all = np.asarray([] if raw_beats is None else raw_beats, float)
    raw = raw_all[(raw_all >= -0.05) & (raw_all <= dur + 0.05)] if dur > 0 else raw_all.copy()
    if len(raw) < 8:
        p = 0.5
        raw = np.arange(0.0, max(dur, p * 8), p)

    body, kick, snare, fps, env_src = onset_curves(drums_mono, sr=sr, mix_mono=mix_mono)
    raw_audit = audit_beats(raw, body, fps)
    um_audit = audit_beats(user_marks_t, body, fps) if user_marks_t is not None and len(user_marks_t) >= 8 else None

    seed = recount_slips(raw)
    fixed, mode, _ = _align_seed(seed, body, fps, max_shift_s=0.005)

    # Trim / extend cleanly to cover [0, dur]
    if len(fixed) >= 2 and dur > 0:
        p_start = float(np.median(np.diff(fixed[:min(len(fixed), 16)])))
        p_end = float(np.median(np.diff(fixed[max(0, len(fixed) - 16):])))
        if p_start > 0.05:
            while fixed[0] - p_start >= 0.0:
                fixed = np.r_[fixed[0] - p_start, fixed]
            while len(fixed) > 2 and fixed[0] < 0.0:
                fixed = fixed[1:]
        if p_end > 0.05:
            while fixed[-1] + p_end <= dur + 0.05:
                fixed = np.r_[fixed, fixed[-1] + p_end]

    down_off = _pick_downbeat_offset(fixed, kick, snare, fps, bpb, ref_bars_t=ref_bars_t)
    bars = list(range(down_off, len(fixed), bpb))
    post_audit = audit_beats(fixed, body, fps)
    bpm = round(60.0 / float(np.median(np.diff(fixed))), 4) if len(fixed) >= 2 else 120.0

    notes = []
    if um_audit and um_audit["median_offset_ms"] > OFFBEAT_WARN_MS:
        notes.append(f"existing marks off beat by {um_audit['signed_median_ms']:+.0f} ms (max {um_audit['max_offset_ms']:.0f} ms)")
    if raw_audit["max_offset_ms"] > OFFBEAT_WARN_MS:
        notes.append(f"detector offset/drift up to {raw_audit['max_offset_ms']:.0f} ms")
    if post_audit["needs_quantize"]:
        notes.append(f"drifts {post_audit['drift_pct']:.1f}% ({post_audit['max_straight_err_ms']:.0f} ms) - lock to make steady")
    if not notes:
        notes.append(f"hits verified clean ({post_audit['median_offset_ms']:.1f} ms median)")

    return dict(
        beats=[round(float(x), 5) for x in fixed],
        bars=bars,
        downbeat_offset=int(down_off),
        bpb=bpb,
        bpm=bpm,
        mode=mode,
        onset_source=env_src,
        raw_audit=raw_audit,
        user_marks_audit=um_audit,
        post_audit=post_audit,
        note="; ".join(notes),
    )


def load_song_drums_and_mix(sid, tspec=""):
    """Load drums mono (and mix mono only if drums are silent) for a library song."""
    from . import library
    drums = library.load_stem(sid, "drums", tspec).mean(0)
    mix = None
    if float(np.sqrt(np.mean(drums ** 2))) < 1e-4:
        parts = []
        for s in ("bass", "other", "vocals"):
            try:
                parts.append(library.load_stem(sid, s, tspec).mean(0))
            except Exception:
                pass
        if parts:
            n = min(len(p) for p in parts)
            mix = sum(p[:n] for p in parts)
    return drums, mix


def audit_song_marks(sid, marks_t=None, tspec=""):
    """Audit `marks_t` (or the song's active marks) against the song's drums stem (raw or locked via tspec)."""
    from . import library, lock
    info = library.read(sid)
    if not info:
        raise FileNotFoundError(f"no such song: {sid}")
    if marks_t is None:
        marks_t = lock.marks_of(info)
    drums, mix = load_song_drums_and_mix(sid, tspec=tspec)
    body, _, _, fps, src = onset_curves(drums, sr=SR, mix_mono=mix)
    aud = audit_beats(marks_t, body, fps)
    aud["onset_source"] = src
    return aud
