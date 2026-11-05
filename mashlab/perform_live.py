"""Perform mode (web/js/perform.js): play chops and samples from the keyboard while the song plays.

While a key is held, the player plays the effect itself: the lane is read from a shifted position inside the audio
callback (render_live below follows Mixer.render line for line; mixer.py itself is untouched). When the key is let
go, the take is turned into ordinary clips (the ReadHead clip conversion in readhead.py) and the UI applies them as
one undo step. Once the project carrying them reaches the player (project["perform_seen"] >= the take's id), the
live version is dropped; the clips play the same thing from then on.

Effect pads (A-L) replace what the lane plays for as long as they're held, then the lane carries on where the song
is (slip). Sample pads (Z-M) add a highlighted span of a lane on top, looping while held; their clips go on a layer
lane of their own. Times are project beats (bar 1 = beat 0), as everywhere in the project.
"""
import math
import threading
import time

import numpy as np

from .config import SR
from .mixer import Mixer, MICRO_FADE

FADE_OUT = 0.03                      # beats: the fade_out readhead.py gives the clips a take becomes (fade_in 0 = MICRO_FADE)
ROLL = ((0.25, 4), (0.125, 8), (0.0625, 16))   # then 1/32 for as long as it's held
PADS = {
    "a": {"kind": "repeat", "len": 2.0},
    "s": {"kind": "repeat", "len": 1.0},
    "d": {"kind": "repeat", "len": 0.5},
    "f": {"kind": "repeat", "len": 0.25},
    "g": {"kind": "jump", "bars": -1},
    "h": {"kind": "jump", "bars": 1},
    "j": {"kind": "roll"},
    "k": {"kind": "chop", "len": 0.5},       # 1/16 on, 1/16 off, in place
    "l": {"kind": "mute"},
}
STALE_SEC = 15.0                     # a finished take the UI never applied is dropped after this


def sample_quant(length, setting, bpb):
    """Grid a sample starts on: auto = its own length up to a bar, so loops land with the bar."""
    if setting == "bar":
        return float(bpb)
    if isinstance(setting, (int, float)) and setting > 0:
        return float(setting)
    for q in (bpb, 2, 1, 0.5, 0.25, 0.125):
        if length >= q - 1e-9 and abs(length / q - round(length / q)) < 1e-6:
            return float(min(q, bpb))
    return 0.25


def _hits_even(T, L, lo, hi):
    k = max(0, math.floor((lo - T) / L + 1e-9))
    while True:
        h = T + k * L
        if h >= hi:
            return
        yield h, L
        k += 1


def _hits_roll(T, lo, hi):
    t = T
    for L, n in ROLL:
        for i in range(n):
            h = t + i * L
            if h >= hi:
                return
            if h + L > lo:
                yield h, L
        t += L * n
    yield from _hits_even(t, 1 / 32, lo, hi)


class Gesture:
    """One held key on one lane, from T until R (None while held)."""

    def __init__(self, gid, track, key, kind, T, opts, sample=None):
        self.id, self.track, self.key, self.kind, self.T = gid, track, key, kind, float(T)
        self.R = None
        self.t_fin = None
        p = PADS.get(key, {}) if sample is None else {}
        L = float(p.get("len", 0) or 0)
        if kind == "repeat" and opts.get("triplet"):
            L *= 2 / 3
        self.len = float(sample["len"]) if sample else L
        self.start = float(sample["start"]) if sample else 0.0
        self.bars = int(p.get("bars", 0) or 0)
        self.gate = float(np.clip(opts.get("gate", 1.0) or 1.0, 0.1, 1.0)) if kind in ("repeat", "roll") else 1.0
        self.additive = sample is not None

    def min_hold(self, q, bpb):
        if self.kind == "repeat":
            return min(self.len, 1.0)
        if self.kind in ("roll", "chop"):
            return 0.5
        if self.kind == "sample":
            return min(self.len, float(bpb))
        return q

    def pieces(self, a, b, bpb):
        """[(t0, t1, src0, gain)] over [a, b) inside [T, R): from t0 the lane plays from src0 (rate 1)."""
        lo = max(a, self.T)
        hi = b if self.R is None else min(b, self.R)
        if hi <= lo + 1e-12:
            return []
        out = []

        def add(x, y, s, g):
            t0, t1 = max(x, lo), min(y, hi)
            if t1 > t0 + 1e-12:
                out.append((t0, t1, s + (t0 - x), g))

        k = self.kind
        if k in ("repeat", "roll", "sample", "chop"):
            hits = _hits_roll(self.T, lo, hi) if k == "roll" else _hits_even(self.T, self.len, lo, hi)
            for h, L in hits:
                s0 = {"repeat": self.T - self.len, "roll": self.T, "sample": self.start}.get(k, h)
                on = L * (0.5 if k == "chop" else self.gate)
                add(h, h + on, s0, 1.0)
                if on < L - 1e-9:
                    add(h + on, h + L, s0 + on, 0.0)
        elif k == "jump":
            add(lo, hi, lo + self.bars * bpb, 1.0)
        elif k == "mute":
            add(lo, hi, lo, 0.0)
        return out


class _One:
    """Looks like Mixer's (tracks, 2, n) bus array to Mixer._mix_clip, but every track lands in one (2, n) buffer."""

    def __init__(self, n):
        self.a = np.zeros((2, n), np.float32)

    def __getitem__(self, k):
        return self.a[k[1:]]

    def __setitem__(self, k, v):
        self.a[k[1:]] = v


def _read_track(mixer, clips, s_from, n):
    """The lane as the mixer would play it over timeline samples [s_from, s_from + n)."""
    one = _One(n)
    end = s_from + n
    for c in clips:
        if c["s0"] >= end:
            break
        if c["s1"] <= s_from:
            continue
        try:
            Mixer._mix_clip(c, s_from, end, one)
        except Exception:
            pass
    return one.a


def _paint(dst, mixer, clips, pos, n, pieces, spb, lo_edge, hi_edge):
    """Add the pieces (beats) into dst (2, n) for timeline samples [pos, pos+n). Every cut gets the fades the
    take's clips will have in the mixer (in: MICRO_FADE from the cut; out: FADE_OUT beats up to it), so the take
    sounds the same live and once it's clips. Ends at the window edges carry on (no fade)."""
    fo = max(MICRO_FADE, int(FADE_OUT * spb))
    for t0, t1, src0, g in pieces:
        if g <= 0:
            continue
        s0, s1 = int(round(t0 * spb)), int(round(t1 * spb))
        a, b = max(pos, s0), min(pos + n, s1)
        if b <= a:
            continue
        seg = _read_track(mixer, clips, int(round(src0 * spb)) + (a - s0), b - a)
        idx = np.arange(a, b, dtype=np.float64)
        env = np.ones(b - a)
        if t0 > lo_edge + 1e-9:
            env *= np.sin(np.clip((idx - s0) / MICRO_FADE, 0, 1) * np.pi / 2)
        if t1 < hi_edge - 1e-9:
            env *= np.sin(np.clip((s1 - idx) / fo, 0, 1) * np.pi / 2)
        dst[:, a - pos:b - pos] += seg * (env * g).astype(np.float32)


def _merge(pieces):
    """Join neighbours that carry straight on (no cut, no fade)."""
    out = []
    for p in sorted(pieces):
        if out:
            q = out[-1]
            if abs(q[1] - p[0]) < 1e-9 and abs((q[2] + (q[1] - q[0])) - p[2]) < 1e-9 and q[3] == p[3]:
                out[-1] = (q[0], p[1], q[2], q[3])
                continue
        out.append(p)
    return out


def _margin(spb):
    """beats the piece window reaches past the block, so every fade that touches the block is seen whole"""
    return (max(MICRO_FADE, int(FADE_OUT * spb)) + MICRO_FADE + 64) / spb


def render_live(mixer, pos, n, over, add, spb):
    """Mixer.render with lanes replaced (over) or added to (add): {track index: [pieces]}. Follows Mixer.render."""
    tracks, clips = mixer.tracks, mixer.clips
    nt = len(tracks)
    buses = np.zeros((nt, 2, n), np.float32) if nt else np.zeros((0, 2, n), np.float32)
    used = np.zeros(nt, bool)
    end = pos + n
    for c in clips:
        if c["s0"] >= end:
            break
        if c["s1"] <= pos or c["ti"] in over:
            continue
        try:
            if mixer._mix_clip(c, pos, end, buses):
                used[c["ti"]] = True
        except Exception as e:
            if c.get("id") not in mixer._bad:
                mixer._bad.add(c.get("id"))
                print(f"clip {c.get('id')} skipped: {type(e).__name__}: {e}")
    x = _margin(spb)
    a_b, b_b = pos / spb - x, end / spb + x
    for ti in set(over) | set(add):
        lane = [c for c in clips if c["ti"] == ti]
        if ti in over:
            gp = sorted(over[ti])
            full, t = [], a_b
            for p in gp:                                   # the song where no key is held, the take where one is
                if p[0] > t + 1e-12:
                    full.append((t, p[0], t, 1.0))
                full.append(p)
                t = max(t, p[1])
            if b_b > t + 1e-12:
                full.append((t, b_b, t, 1.0))
            buses[ti] = 0
            _paint(buses[ti], mixer, lane, pos, n, _merge(full), spb, a_b, b_b)
            used[ti] = True
        if ti in add:
            _paint(buses[ti], mixer, lane, pos, n, add[ti], spb, a_b, b_b)
            used[ti] = True

    out = np.zeros((2, n), np.float32)
    reset = mixer._reset_next
    mixer._reset_next = False
    vox_level = 0.0
    if any(t["duck"] > 0 for t in tracks):
        vi = [i for i, t in enumerate(tracks) if t["role"] == "vocals" and used[i] and t["gain"] > 0]
        if vi:
            v = buses[vi].sum(0)
            vox_level = float(np.sqrt(np.mean(v ** 2)) + 1e-9)
    for i, t in enumerate(tracks):
        fx = mixer.fx.get(t["id"])
        if not used[i] and (fx is None or fx.board is None):
            continue
        if t["gain"] == 0:
            continue
        xb = buses[i]
        if fx is not None and fx.board is not None:
            xb = fx.process(xb, reset)
        g = t["gain"]
        if t["duck"] > 0 and fx is not None:
            db = 20 * np.log10(vox_level) if vox_level > 0 else -120
            target = float(np.clip((db + 42) / 24, 0, 1)) * t["duck"]
            k = 0.5 if target > fx.duck_env else 0.08
            fx.duck_env += (target - fx.duck_env) * k
            g *= 10 ** (-fx.duck_env / 20)
        out += xb * (t["lr"][:, None] * g)
    out *= mixer.master_gain
    if mixer.master is not None and len(mixer.master) > 0:
        out = mixer.master.process(out, SR, reset=reset)
    return out


class Live:
    def __init__(self):
        self.lock = threading.Lock()
        self.g = []                  # takes being played (held, or finished and not yet in the project)
        self.fin = {}                # id -> finished take, kept until the UI has turned it into clips
        self.last_end = None
        self.last_end_b = None       # the same, in beats (for a stop)
        self._last_id = 0
        self._err_t = 0.0

    def _new_id(self):
        # time-based so ids keep growing across restarts (project["perform_seen"] is saved with the project)
        self._last_id = max(self._last_id + 1, int(time.time() * 1000))
        return self._last_id

    def busy(self):
        return bool(self.g)

    # ---------------------------------------------------------------- from the UI
    def press(self, player, track, key, opts, sample=None):
        with player.lock:
            if not player.playing:
                raise ValueError("press play first")
            spb, bpb, pos_b = player._spb, player._bpb, player.pos / player._spb
        kind = "sample" if sample else PADS[key]["kind"]
        q = float(opts.get("quant") or 0.25)
        if sample:
            q = sample_quant(float(sample["len"]), sample.get("quant", "auto"), bpb)
        elif kind == "repeat":
            q = min(q, PADS[key]["len"] * (2 / 3 if opts.get("triplet") else 1))
        T = math.ceil(pos_b / q - 1e-9) * q
        g = Gesture(self._new_id(), sample["track"] if sample else track, key, kind, T, opts, sample)
        with self.lock:
            # on one lane, effect takes never overlap (a new one starts where the last one ends)
            for o in self.g:
                if o.track == g.track and not o.additive and not g.additive:
                    if o.R is None:
                        o.R, o.t_fin = max(g.T, o.T), time.time()
                        self._keep(o)
                    g.T = max(g.T, o.R)
                if g.additive and o.additive and o.key == g.key and o.R is not None:
                    g.T = max(g.T, o.R)
            self.g.append(g)
        return {"id": g.id, "T": g.T}

    def release(self, player, gid, quant):
        with player.lock:
            spb, bpb, pos_b = player._spb, player._bpb, player.pos / player._spb
        with self.lock:
            g = next((x for x in self.g if x.id == gid), None) or self.fin.get(gid)
            if g is None:
                return None
            if g.R is None:
                q = float(quant or 0.25)
                g.R = max(math.ceil(pos_b / q - 1e-9) * q, g.T + g.min_hold(q, bpb))
                g.t_fin = time.time()
            self._keep(g)
        return g

    def _keep(self, g):
        now = time.time()
        self.fin = {k: v for k, v in self.fin.items() if now - (v.t_fin or now) < 120}
        self.fin[g.id] = g

    def finished(self, gid):
        with self.lock:
            return self.fin.get(gid)

    # ---------------------------------------------------------------- from the player
    def on_project(self, proj):
        seen = proj.get("perform_seen") or 0
        with self.lock:
            self.g = [g for g in self.g if not (g.R is not None and g.id <= seen)]

    def stopped(self):
        # a key still held when the song stops ends where the playhead was; finished takes stay for the UI
        with self.lock:
            for g in self.g:
                if g.R is None:
                    g.R, g.t_fin = (max(g.T, self.last_end_b) if self.last_end_b is not None else g.T), time.time()
                self._keep(g)
            self.g = []
        self.last_end = None

    def render(self, mixer, pos, n, spb, bpb):
        """The block with the held / unapplied takes in it, or None when none touch it."""
        now = time.time()
        with self.lock:
            if self.last_end is not None and pos != self.last_end:
                # looped round or jumped: a held key ends where the playhead was
                for g in self.g:
                    if g.R is None:
                        g.R, g.t_fin = max(g.T, self.last_end / spb), now
                        self._keep(g)
            self.g = [g for g in self.g if g.t_fin is None or now - g.t_fin < STALE_SEC]
            gs = list(self.g)
        self.last_end = pos + n
        self.last_end_b = (pos + n) / spb
        if not gs:
            return None
        x = _margin(spb)
        a_b, b_b = pos / spb - x, (pos + n) / spb + x
        over, add = {}, {}
        for g in gs:
            ti = mixer.track_index.get(g.track)
            if ti is None:
                continue
            ps = g.pieces(a_b, b_b, bpb)
            if ps:
                (add if g.additive else over).setdefault(ti, []).extend(ps)
        if not over and not add:
            return None
        return render_live(mixer, pos, n, over, add, spb)

    def failed(self, e):
        if time.time() - self._err_t > 5:
            print(f"perform: live render failed, playing without it: {type(e).__name__}: {e}")
            self._err_t = time.time()


LIVE = Live()


# -------------------------------------------------------------------- takes -> clips
def take_clips(project, g, layer_track=None):
    """The finished take as clips: {clips, replace: {track, trims}} in the shape web/js/readhead.js applyResult uses."""
    from . import readhead as RH, project as P
    bpb = int(project.get("bpb") or 4)
    src_lane = [c for c in project.get("clips", []) if c.get("track") == g.track]
    keep = [c for c in src_lane if abs(float(c.get("rate", 1) or 1) - 1) <= 1e-6 and not c.get("warps")]
    ps = g.pieces(g.T, g.R, bpb)
    ev = [{"t": t0, "off": t0 - s0, "gain": gn, "rate": 1} for t0, t1, s0, gn in ps]
    # where the take carries straight on from the song (a chop / roll starting on the audio that was playing, or
    # ending on it) the clips butt-join with no fade, like the live version (Mixer._joins drops the 3 ms there)
    on_in = bool(ps) and not g.additive and abs(ps[0][0] - g.T) < 1e-9 and ps[0][3] > 0 and abs(ps[0][2] - g.T) < 1e-6
    on_out = bool(ps) and not g.additive and abs(ps[-1][1] - g.R) < 1e-9 and ps[-1][3] > 0 and abs(ps[-1][2] + ps[-1][1] - ps[-1][0] - g.R) < 1e-6
    if g.additive:
        dest = layer_track
        new = RH._timeline_clips(ev + [{"t": g.R, "off": 0, "gain": 0, "rate": 1}], keep, project, g.T, g.R)
        lane = [c for c in project.get("clips", []) if c.get("track") == dest]
        hit = lambda c: True
    else:
        dest = g.track
        ev.append({"t": g.R, "off": 0, "gain": 1, "rate": 1})
        new = RH._timeline_clips(ev, keep, project, g.T, g.R)
        lane = src_lane
        ids = {c["id"] for c in keep}
        hit = lambda c: c["id"] in ids
    lo, hi = g.T, g.R
    trims = []
    for c in lane:
        if not hit(c) or c["start"] + c["len"] <= lo + 1e-9 or c["start"] >= hi - 1e-9:
            continue
        left = lo - c["start"] if c["start"] < lo - 1e-9 else None
        right = None
        if c["start"] + c["len"] > hi + 1e-9 and not c.get("warps"):
            Wc, mc = RH._landing(project, c["song"]), P.song_mult(project, c["song"])
            k = float(c.get("rate", 1) or 1)
            w = RH._w_of(Wc, c["src"]) + (hi - c["start"]) * k / mc
            right = {**{kk: vv for kk, vv in c.items() if kk != "id"}, "start": hi, "len": c["start"] + c["len"] - hi,
                     "src": round(RH._src_of(Wc, w), 6), "fade_in": 0}
        trims.append({"id": c["id"], "left_len": left, "right": right, "left_fade": 0 if on_in else 0.03})
    if on_out:
        for c in new:
            if abs(c["start"] + c["len"] - hi) < 1e-6:
                c["fade_out"] = 0
    for c in new:
        c.pop("readhead", None)
        c["perform"] = True
    return {"clips": new, "replace": {"track": dest, "trims": trims, "start": lo, "end": hi, "end_inf": False}}


def register(app, player, get_project=None):
    from fastapi import HTTPException, Request

    @app.post("/api/perform/press")
    async def perform_press(request: Request):
        """Body: {track, key, opts: {quant, gate, triplet}, sample?: {track, start, len, quant}} -> {id, T}."""
        b = await request.json()
        try:
            return LIVE.press(player, b.get("track"), b["key"], b.get("opts") or {}, b.get("sample"))
        except (KeyError, ValueError) as e:
            raise HTTPException(400, str(e))

    @app.post("/api/perform/release")
    async def perform_release(request: Request):
        """Body: {id, quant} -> {id, T, R}. The key went up: the take's end is fixed now (on the grid)."""
        b = await request.json()
        g = LIVE.release(player, b["id"], b.get("quant"))
        if g is None:
            return {"id": b["id"], "gone": True}
        return {"id": g.id, "T": g.T, "R": g.R, "key": g.key, "kind": g.kind}

    @app.post("/api/perform/take")
    async def perform_take(request: Request):
        """Body: {id, project, layer_track?} -> {id, take: {clips, replace}} against the project the UI has now
        (the UI asks one take at a time and applies each before the next, so trims always match its clips)."""
        b = await request.json()
        g = LIVE.finished(b["id"])
        if g is None or g.R is None:
            return {"id": b["id"], "gone": True}
        try:
            take = take_clips(b["project"], g, b.get("layer_track"))
        except Exception as e:
            raise HTTPException(500, f"couldn't turn the take into clips: {type(e).__name__}: {e}")
        return {"id": g.id, "T": g.T, "R": g.R, "key": g.key, "kind": g.kind, "take": take}
