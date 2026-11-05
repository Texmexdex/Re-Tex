"""How a song's beats are laid onto the project grid.

Every song plays AS RECORDED (project.song_tspec): one fixed speed - project tempo / the song's tempo - with its
own timing kept (drifts, odd bars, tempo changes). positions(beats, {"m": "natural", "p": period, "fb": anchor})
gives W[j] = fb + (t_j - t_fb) / p: where beat j lands, counted in beats of the warped render (the renderer puts
beat j at W[j] * Tb). W is linear in time, so the warp is one constant stretch; nothing is fitted per beat.

The rest of this module (per-beat auto-align "grid" mode, crunch / pin / cut edits) is no longer used by the
app: the user asked for no automatic warping at all (2026-10-02).
"""
import hashlib
import json
import math

import numpy as np

MODES = ("grid", "natural")
XFADE_S = 0.006               # splice crossfade at every cut


# ---------------------------------------------------------------- spec: what a render needs to know

def spec_for(info, mode="grid"):
    """-> a compact, hashable JSON string describing this song's timing in this mode, or "" for plain
    auto-align (so every existing render and cache key stays valid)."""
    mode = mode if mode in MODES else "grid"
    edits = [e for e in (info.get("timing_edits") or []) if _applies(e, mode)]
    if mode == "grid" and not edits:
        return ""
    cuts = (info.get("timing_cuts") or {}).get(mode, [])
    d = dict(m=mode, p=round(60.0 / float(info.get("bpm") or 120.0), 6), fb=int(info.get("first_beat", 0)),
             e=sorted(edits, key=lambda e: e["b"]), c=[[round(a, 5), round(b, 5)] for a, b in cuts])
    return json.dumps(d, sort_keys=True, separators=(",", ":"))


def spec_key(tspec):
    return "" if not tspec else "_t" + hashlib.sha1(tspec.encode()).hexdigest()[:10]


def _applies(e, mode):
    return e.get("a") == "remove" or (e.get("a") == "crunch" and mode == "grid") or \
        (e.get("a") in ("pin", "pinc") and mode == "natural")


def parse(tspec):
    return json.loads(tspec) if tspec else None


# ---------------------------------------------------------------- cuts

def cut_map(cuts):
    """cuts [(t0, t1)] seconds -> f(t) = time after the cuts are taken out (inside a cut: its start)."""
    cuts = sorted((float(a), float(b)) for a, b in cuts if b > a)
    if not cuts:
        return lambda t: np.asarray(t, float)
    starts = np.array([a for a, _ in cuts])
    ends = np.array([b for _, b in cuts])
    before = np.concatenate([[0.0], np.cumsum(ends - starts)])

    def f(t):
        t = np.asarray(t, float)
        k = np.searchsorted(starts, t, side="right")          # cuts starting at/before t
        out = t - before[k]
        inside = (k > 0) & (t < ends[np.maximum(k - 1, 0)])
        return np.where(inside, starts[np.maximum(k - 1, 0)] - before[np.maximum(k - 1, 0)], out)
    return f


def apply_cuts(y, cuts, sr):
    """Remove cut spans from y (channels, n) with short equal-power crossfades. The result is exactly the
    total cut length shorter, so cut_map() timing holds to the sample."""
    n = y.shape[1]
    h = max(1, int(XFADE_S * sr / 2))
    iv = []
    for a, b in sorted((int(round(a * sr)), int(round(b * sr))) for a, b in cuts):
        a, b = max(0, a), min(n, b)
        if b <= a:
            continue
        if iv and a <= iv[-1][1] + 2 * h:                  # too close to splice separately: one cut
            iv[-1][1] = max(iv[-1][1], b)
        else:
            iv.append([a, b])
    if not iv:
        return y
    t = np.linspace(0, np.pi / 2, 2 * h, dtype=np.float32)
    fo, fi = np.cos(t), np.sin(t)
    out = y[:, :min(n, iv[0][0] + h)].astype(np.float32)
    for k, (a, b) in enumerate(iv):
        nxt = iv[k + 1][0] + h if k + 1 < len(iv) else n
        right = y[:, max(0, b - h):min(n, nxt)].astype(np.float32)
        if out.shape[1] >= 2 * h and right.shape[1] >= 2 * h:
            mid = out[:, -2 * h:] * fo + right[:, :2 * h] * fi
            out = np.concatenate([out[:, :-2 * h], mid, right[:, 2 * h:]], axis=1)
        else:                                               # a cut at the very start / end
            out = np.concatenate([out[:, :max(0, out.shape[1] - h)], right[:, h:]], axis=1)
    return np.ascontiguousarray(out)


def choose_cut(env_db, fps, t0, t1, amount, quiet_db=-38.0, min_gap=0.03, beats=None):
    """Where to take `amount` seconds out of [t0, t1]: squeezed evenly out of the quiet gaps if they hold
    enough (the notes stay whole), otherwise one cut at the quietest stretch - starting on a 16th of the
    beat when the beats are given, so what's left keeps its groove. env_db: the mix's level per frame
    relative to its loud parts. -> [(a, b)] seconds."""
    if amount <= 1.0 / fps or t1 <= t0:
        return []
    i0, i1 = max(0, int(t0 * fps)), min(len(env_db), int(math.ceil(t1 * fps)))
    seg = env_db[i0:i1]
    if len(seg) < 3:
        return [(t0, min(t1, t0 + amount))]
    quiet = seg < quiet_db
    gaps, k = [], 0
    while k < len(seg):
        if quiet[k]:
            m = k
            while m < len(seg) and quiet[m]:
                m += 1
            a, b = (i0 + k) / fps + XFADE_S, (i0 + m) / fps - XFADE_S
            if b - a >= min_gap:
                gaps.append((a, b))
            k = m
        else:
            k += 1
    room = sum(b - a for a, b in gaps)
    if room >= amount:
        out = []
        for a, b in gaps:
            take = amount * (b - a) / room
            mid = 0.5 * (a + b)
            out.append((mid - take / 2, mid + take / 2))
        return out
    # not enough silence: one contiguous cut where the mix is quietest (whole notes / beats if need be)
    n = max(1, int(round(amount * fps)))
    if n >= len(seg):
        return [(t0, t1)]
    c = np.convolve(10 ** (seg / 20), np.ones(n), mode="valid")
    if beats is not None:
        bt = np.asarray(beats, float)
        k = np.searchsorted(bt, [t0, t1])
        sub = [bt[i] + q * (bt[i + 1] - bt[i]) / 4 for i in range(max(0, k[0] - 1), min(len(bt) - 1, k[1] + 1))
               for q in range(4)]
        starts = [x for x in sub if t0 - 1e-6 <= x <= t1 - amount + 1e-6]
        if starts:
            idx = np.clip(np.round((np.array(starts) * fps) - i0).astype(int), 0, len(c) - 1)
            a = float(starts[int(np.argmin(c[idx]))])
            return [(a, a + amount)]
    j = int(np.argmin(c))
    a = (i0 + j) / fps
    return [(a, a + amount)]


# ---------------------------------------------------------------- positions

def local_period(beats, j, w=8):
    b = np.asarray(beats, float)
    a, c = max(0, j - w), min(len(b) - 1, j + w + 1)
    return float(np.median(np.diff(b[a:c + 1]))) if c - a >= 2 else float(np.median(np.diff(b)))


def odd_beats(beats, first=0, last=None, tol=0.12):
    """Beat intervals auto-align has to stretch or squeeze by more than tol: [(j, length in beats)]."""
    b = np.asarray(beats, float)
    last = len(b) - 1 if last is None else last
    out = []
    for j in range(max(0, first), min(last, len(b) - 1)):
        L = (b[j + 1] - b[j]) / local_period(b, j)
        if abs(math.log(max(L, 1e-6))) > math.log(1 + tol):
            out.append((j, round(float(L), 3)))
    return out


def crunch_target(L):
    """Whole beats an odd beat of length L keeps when crunched: 0.5 -> 0 (cut it), 1.5 -> 1, 1.8 -> 2."""
    return int(math.floor(L + 0.25))


def positions(beats, spec):
    """W for every beat (see module doc). spec: parsed dict or None (plain grid)."""
    b = np.asarray(beats, float)
    n = len(b)
    if not spec:
        return np.arange(n, dtype=float)
    cm = cut_map(spec.get("c", []))
    tb = cm(b)                                              # beat times with the cuts taken out
    if spec["m"] == "grid":
        d = np.ones(n - 1)
        cut_iv = set()
        for a, c in spec.get("c", []):
            lo, hi = np.searchsorted(b, a, side="right") - 1, np.searchsorted(b, c, side="left")
            cut_iv.update(range(max(0, lo), min(n - 1, hi)))
        for j in cut_iv:                                    # intervals the cuts shortened keep whole beats
            d[j] = max(0.0, round((tb[j + 1] - tb[j]) / local_period(b, j)))
        for j in spec.get("k", []):                         # un-stretched: this beat keeps its real length
            if 0 <= j < n - 1:
                d[j] = (b[j + 1] - b[j]) / local_period(b, j)
        return np.concatenate([[0.0], np.cumsum(d)])
    # natural: time since the first beat at the song's main tempo, stretched between pins
    p = float(spec["p"])
    fb = min(max(0, int(spec.get("fb", 0))), n - 1)
    W = fb + (tb - tb[fb]) / p
    anchors = sorted({int(e["b"]) for e in spec.get("e", []) if e["a"] in ("pin", "pinc") and 0 <= e["b"] < n})
    act = {int(e["b"]): e for e in spec.get("e", []) if e["a"] in ("pin", "pinc")}
    prev = fb
    for c in [a for a in anchors if a > fb]:
        L = (tb[c] - tb[prev]) / p
        e = act[c]
        D = e.get("d")
        if D is None:
            D = round(L) if e["a"] == "pin" else crunch_target(L - 0.15)
        D = max(0, int(D))
        span = tb[c] - tb[prev]
        seg = slice(prev, c + 1)
        W[seg] = W[prev] + (np.clip((tb[seg] - tb[prev]) / span, 0, 1) * D if span > 1e-9 else 0.0)
        W[c + 1:] = W[c] + (tb[c + 1:] - tb[c]) / p
        prev = c
    return W


def natural_cut_plan(beats, spec_nat, env_db, fps):
    """Cuts for every "pinc" edit in natural mode (each takes its segment's excess over whole beats)."""
    b = np.asarray(beats, float)
    p = float(spec_nat["p"])
    fb = int(spec_nat.get("fb", 0))
    out = []
    prev = fb
    for e in sorted(spec_nat.get("e", []), key=lambda e: e["b"]):
        c = int(e["b"])
        if e["a"] not in ("pin", "pinc") or c <= prev:
            continue
        if e["a"] == "pinc":
            L = (b[c] - b[prev]) / p
            D = e.get("d", crunch_target(L - 0.15))
            extra = (L - D) * p
            if extra > 0.01:
                out += choose_cut(env_db, fps, b[prev], b[c], extra, beats=b)
        prev = c
    return out


def resolve_cuts(info, env_db, fps):
    """All cuts for the song's edits, per mode -> {"grid": [...], "natural": [...]} (seconds)."""
    b = np.asarray(info["beats"], float)
    edits = info.get("timing_edits") or []
    common = []
    for e in edits:
        if e["a"] == "remove":
            j, k = int(e["b"]), int(e.get("n", 1))
            if 0 <= j < len(b) - 1:
                common.append((float(b[j]), float(b[min(len(b) - 1, j + k)])))
    grid = list(common)
    for e in edits:
        if e["a"] == "crunch":
            j = int(e["b"])
            if not 0 <= j < len(b) - 1:
                continue
            p = local_period(b, j)
            L = (b[j + 1] - b[j]) / p
            keep = crunch_target(L)
            extra = (L - keep) * p
            if keep == 0:
                grid.append((float(b[j]), float(b[j + 1])))
            elif extra > 0.01:
                grid += choose_cut(env_db, fps, b[j], b[j + 1], extra, beats=b)
    nat_spec = dict(m="natural", p=60.0 / float(info.get("bpm") or 120.0), fb=int(info.get("first_beat", 0)),
                    e=[e for e in edits if e["a"] in ("pin", "pinc")], c=[])
    natural = list(common) + natural_cut_plan(b, nat_spec, env_db, fps)
    tidy = lambda cs: [[round(a, 5), round(b_, 5)] for a, b_ in sorted(cs) if b_ - a > 1e-4]
    return {"grid": tidy(grid), "natural": tidy(natural)}


def mix_env_db(stems, sr, fps=100):
    """Level of the song's full mix per 1/fps frame, in dB relative to its 95th percentile."""
    m = sum(v.mean(0) for v in stems.values())
    hop = int(sr / fps)
    n = len(m) // hop
    r = np.sqrt(np.mean(m[:n * hop].reshape(n, hop).astype(np.float64) ** 2, axis=1)) + 1e-9
    db = 20 * np.log10(r)
    return db - np.quantile(db, 0.95), fps


# ---------------------------------------------------------------- the detector's marks (twin of web/js/timing.js)
CLEAR, SURE_SHARE = 50, 0.4          # a beat counts as sure when 40 % of its neighbours have confidence >= 50


def _smooth_conf(conf, lo, hi, w=4):
    n = len(conf)
    out = [0] * n
    for j in range(n):
        a, b = max(0, j - w), min(n, j + w + 1)
        clear = sum(1 for i in range(a, b) if conf[i] >= CLEAR)
        out[j] = 1 if lo <= j <= hi and clear / (b - a) >= SURE_SHARE else 0
    for j in range(1, n - 1):
        if out[j] and not out[j - 1] and not out[j + 1]:
            out[j] = 0
    return out


def detected_marks(info):
    """The detector's beat marks as an editable list {t, d, s, bpb} - exactly what the page starts from
    (timing.js detectedMarks): its beats inside the song, its bar lines, sure / unsure from its confidence and
    MuScriptor's second opinion."""
    b = list(info["beats"])
    n = len(b)
    d = [0] * n
    for j in info.get("bars") or []:
        if 0 <= j < n:
            d[j] = 1
    conf = info.get("beat_conf")
    last = info.get("last_beat")
    sure = _smooth_conf(conf, info.get("first_beat") or 0, n - 1 if last is None else last) if conf else [1] * n
    mv = (info.get("mus_check") or {}).get("v")
    if mv and len(mv) == n:
        sure = [0 if mv[j] == 2 else sure[j] for j in range(n)]
    end = (info.get("duration") or b[-1]) + 1e-6
    keep = [j for j in range(n) if -1e-6 <= b[j] <= end]
    return dict(t=[b[j] for j in keep], d=[d[j] for j in keep], s=[sure[j] for j in keep], bpb=info.get("bpb") or 4)
