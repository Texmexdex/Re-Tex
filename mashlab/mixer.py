"""Block-based arrangement mixer. The same code drives live playback and export (what you hear = what you get)."""
import numba
import numpy as np
import pedalboard as pb

from . import clipfx, draft, library, project as P, timing, variant, warp, warps
from .config import SR

MICRO_FADE = int(0.003 * SR)       # every clip edge gets 3 ms so cuts never click
_WIN = (np.i0(8.0 * np.sqrt(np.clip(1 - np.linspace(0, 1, 4097) ** 2, 0, 1))) / np.i0(8.0)).astype(np.float64)


@numba.njit(cache=True, fastmath=True)
def _vari(x, p0, step, m, cutoff, win):
    """Read x at positions p0 + i*step (i < m) with a Kaiser-windowed sinc low-passed at `cutoff`
    (< 1 when speeding up, so varispeed never aliases)."""
    ch, n = x.shape
    H = int(np.ceil(16.0 / cutoff))
    nw = win.shape[0] - 1
    out = np.zeros((ch, m), np.float32)
    for i in range(m):
        p = p0 + i * step
        i0 = int(np.floor(p))
        for j in range(i0 - H + 1, i0 + H + 1):
            if j < 0 or j >= n:
                continue
            t = j - p
            a = abs(t) / H
            if a >= 1.0:
                continue
            u = t * cutoff
            s = 1.0 if abs(u) < 1e-9 else np.sin(np.pi * u) / (np.pi * u)
            k = cutoff * s * win[int(a * nw)]
            for c in range(ch):
                out[c, i] += x[c, j] * k
    return out


def varispeed_read(buf, pos0, step, m):
    """m samples of buf starting at fractional position pos0, advancing `step` per output sample."""
    cutoff = min(1.0, 1.0 / step) * 0.95
    H = int(np.ceil(16.0 / cutoff)) + 2
    lo = int(np.floor(pos0)) - H
    hi = int(np.ceil(pos0 + m * step)) + H
    la, lb = max(0, lo), min(buf.shape[1], hi)
    if lb <= la:
        return np.zeros((buf.shape[0], m), np.float32)
    seg = np.ascontiguousarray(buf[:, la:lb], dtype=np.float32)
    return _vari(seg, pos0 - la, step, m, cutoff, _WIN)


class _TrackFX:
    """Stateful per-track processing (filters + vocal pocket EQ), rebuilt only when settings change."""

    def __init__(self):
        self.sig = None
        self.board = None
        self.duck_env = 0.0

    def configure(self, t):
        sig = (round(t.get("hpf", 0)), round(t.get("lpf", 0)), round(t.get("pocket", 0), 1))
        if sig == self.sig:
            return
        self.sig = sig
        fx = []
        if sig[0] > 20:
            fx.append(pb.HighpassFilter(cutoff_frequency_hz=float(sig[0])))
        if 200 < sig[1] < 20000:
            fx.append(pb.LowpassFilter(cutoff_frequency_hz=float(sig[1])))
        if sig[2] > 0.05:
            fx.append(pb.PeakFilter(cutoff_frequency_hz=2800.0, gain_db=-float(sig[2]), q=0.7))
        self.board = pb.Pedalboard(fx) if fx else None

    def process(self, x, reset=False):
        if self.board is None:
            return x
        return self.board.process(x, SR, reset=reset)


class Mixer:
    def __init__(self):
        self.proj = None
        self.clips = []          # compiled clips sorted by start sample
        self.tracks = []
        self.track_index = {}
        self.fx = {}             # track id -> _TrackFX
        self.master = None
        self.master_sig = None
        self.master_gain = 1.0
        self.end_sample = 0
        self.missing = set()
        self.pending = []            # HQ clip slices (clipfx) the arrangement still needs
        self.clip_drafts = []        # ... and their quick R2 stand-ins
        self.drafts = []             # quick stand-ins for clips whose warp is still rendering (draft.py)
        self._bad = set()            # clip ids that failed to render (logged once each)
        self._reset_next = True

    # ------------------------------------------------------------------ compile
    def load(self, proj, allow_preview=True, swap_lock=None):
        """Compile the project. Clips whose HQ stretch/pitch render isn't ready play a varispeed preview
        (allow_preview) and are listed in self.pending for the caller to render."""
        bpm = float(proj["bpm"])
        T = 60.0 / bpm * SR                                   # samples per project beat
        tracks = proj["tracks"]
        solo = any(t.get("solo") for t in tracks)
        tinfo = []
        for t in tracks:
            audible = not t.get("mute") and (t.get("solo") or not solo)
            pan = float(np.clip(t.get("pan", 0.0), -1, 1))
            g = 10 ** (t.get("gain_db", 0.0) / 20) if audible else 0.0
            lr = np.array([np.cos((pan + 1) * np.pi / 4), np.sin((pan + 1) * np.pi / 4)]) * np.sqrt(2)
            tinfo.append(dict(id=t["id"], gain=g, lr=lr.astype(np.float32), duck=float(t.get("duck", 0.0)),
                              role=t.get("role", "any"), cfg=t))
            self.fx.setdefault(t["id"], _TrackFX()).configure(t)
        tindex = {t["id"]: i for i, t in enumerate(tinfo)}

        compiled, missing, pending, clip_drafts, drafts = [], set(), [], [], []
        for c in proj["clips"]:
            if c.get("mute") or c["track"] not in tindex or c["len"] <= 0:
                continue
            try:
                # a clip with stretched parts plays as consecutive plain pieces (warps.py)
                subs = warps.expand(c, P.song_mult(proj, c["song"]), bpm,
                                    self._landing(c["song"], P.song_tspec(proj, c["song"]))) if c.get("warps") else [c]
                for sc in subs:
                    self._compile_clip(sc, proj, bpm, T, tindex, compiled, missing, pending, clip_drafts, drafts,
                                       allow_preview)
            except Exception as e:                          # one broken clip must never stop playback / loading
                if c.get("id") not in self._bad:
                    self._bad.add(c.get("id"))
                    print(f"clip {c.get('id')} skipped: {type(e).__name__}: {e}")
        joins = self._joins(proj)
        for d in compiled:                                   # seamless joins: no 3 ms dip where the audio carries on
            if (d.get("id"), "in") in joins and d["fi"] == MICRO_FADE:
                d["fi"] = 0                                  # 0 = no fade at all
            if (d.get("id"), "out") in joins and d["fo"] == MICRO_FADE:
                d["fo"] = 0
        compiled.sort(key=lambda d: d["s0"])


        m = proj.get("master", {})
        sig = (bool(m.get("glue", True)), bool(m.get("limit", True)))
        if sig != self.master_sig:
            fx = []
            if sig[0]:
                fx.append(pb.Compressor(threshold_db=-16.0, ratio=2.0, attack_ms=20.0, release_ms=250.0))
            if sig[1]:
                fx.append(pb.Limiter(threshold_db=-1.2, release_ms=120.0))
            self.master = pb.Pedalboard(fx)
            self.master_sig = sig
        self.master_gain = 10 ** (m.get("gain_db", 0.0) / 20)
        # swap atomically for the audio thread
        import contextlib
        with swap_lock or contextlib.nullcontext():
            self.tracks, self.track_index, self.clips = tinfo, tindex, compiled
            self.end_sample = max([d["s1"] for d in compiled] + [0])
            self.missing = missing
            self.pending, self.clip_drafts, self.drafts = pending, clip_drafts, drafts
            self.proj = proj
        return missing

    def _compile_clip(self, c, proj, bpm, T, tindex, compiled, missing, pending, clip_drafts, drafts, allow_preview):
        """One clip -> its mixer entries (appended to compiled); lists what still has to render."""
        mult = P.song_mult(proj, c["song"])
        semis = P.song_semis(proj, c["song"], c["stem"])
        req = P.request(proj, c)
        Wm = self._landing(c["song"], req[6])            # None: beat j lands on beat j
        wsrc = float(c["src"]) if Wm is None else float(np.interp(c["src"], Wm[1], Wm[0]))
        r = warp.open_render(*req)
        if r is None:
            missing.add(req)
            if not allow_preview:
                return
            es = self._draft_entries(c, req, bpm, mult, semis, T, tindex, drafts, wsrc, Wm)
            if es is not None:
                compiled.extend(es)
                return
            srcbuf = library.peek_stem(c["song"], c["stem"], req[6])
            info = library.read(c["song"]) if srcbuf is not None else None
            if srcbuf is not None and info:
                # warp not rendered yet: read the original stem along its beat map (tempo-correct,
                # pitch follows speed) until the HQ render lands
                s0 = int(round(c["start"] * T))
                s1 = int(round((c["start"] + c["len"]) * T))
                compiled.append(dict(s0=s0, s1=s1, g=10 ** (c.get("gain_db", 0.0) / 20),
                                     fi=max(MICRO_FADE, int(c.get("fade_in", 0.0) * T)),
                                     fo=max(MICRO_FADE, int(c.get("fade_out", 0.0) * T)),
                                     ti=tindex[c["track"]], id=c["id"], mode="src", buf=srcbuf,
                                     beats=np.asarray(variant.beats_for(info, req[6]), float), src=wsrc,
                                     W=None if Wm is None else Wm[0], Widx=None if Wm is None else Wm[1],
                                     k=float(c.get("rate", 1.0) or 1.0) / mult, T=T, step=0.0, off=0))
            return
        buf, t0 = r
        s0 = int(round(c["start"] * T))
        s1 = int(round((c["start"] + c["len"]) * T))
        rate = float(c.get("rate", 1.0) or 1.0)
        off_f = wsrc * mult * T - t0 * SR                    # base-buffer sample at clip start
        e = dict(s0=s0, s1=s1, g=10 ** (c.get("gain_db", 0.0) / 20),
                 fi=max(MICRO_FADE, int(c.get("fade_in", 0.0) * T)),
                 fo=max(MICRO_FADE, int(c.get("fade_out", 0.0) * T)), ti=tindex[c["track"]])
        if clipfx.needs_hq(c):
            A, B = clipfx.segment(buf.shape[1], off_f, s1 - s0, rate, T)
            tp = clipfx.total_pitch(c)
            key = clipfx.key_for(req, A, B, rate, tp)
            d = clipfx.open_render(c["song"], key)
            if d is not None:
                e.update(buf=d, off=int(round((off_f - A) / rate)), step=1.0)
            else:
                pending.append((c["id"], req, A, B, rate, tp, c["stem"], key))
                if not allow_preview:
                    return
                # until the HQ render lands: quick R2 drafts, 16-beat pieces (the one at the playhead first)
                e.update(id=c["id"])

                def lookup(A, B, a, b, c=c, req=req, rate=rate, tp=tp, off_f=off_f, s0=s0):
                    dk = clipfx.key_for(req, A, B, rate, tp, draft=True)
                    dd = clipfx.open_render(c["song"], dk)
                    if dd is None:
                        clip_drafts.append((c["id"], req, A, B, rate, tp, c["stem"], dk, a, b))
                        return None
                    return dict(buf=dd, off=int(round((off_f + (a - s0) * rate - A) / rate)), step=1.0)

                def vari(a, b, e=e, buf=buf, off_f=off_f, rate=rate, s0=s0, s1=s1):
                    return dict(e, s0=a, s1=b, cs0=s0, cs1=s1, buf=buf, off=off_f + (a - s0) * rate, step=rate)

                compiled.extend(self._pieces(e, off_f, rate, T, buf.shape[1], lookup, vari))
                return
        else:
            # plain varispeed (pitch follows speed). Read position for timeline sample n: off_f + (n - start * T) *
            # rate, measured from the clip's exact start, not its rounded first sample - so clips that carry on the
            # same audio (song sections laid in one after another) share one sample-exact mapping
            e.update(buf=buf, off=off_f + (s0 - c["start"] * T) * rate, step=rate)
            if abs(rate - 1.0) < 1e-6:
                e.update(off=s0 + int(round(off_f - c["start"] * T)), step=1.0)
        if abs(e["step"] - 1.0) < 1e-6:
            # 1x reads are plain slices: whole-sample offset (pitch-only clips preview this way too)
            e.update(off=int(round(e["off"])), step=1.0)
        e["id"] = c["id"]
        compiled.append(e)

    def _joins(self, proj):
        """Clip edges where the next clip on the track simply carries on the same audio (a song laid in section by
        section, a clip split in two): same song + stem + speed + gain, butt-joined, and the second starts where
        the first one's audio ends. Both read one continuous stretch of the same warp, so the edge needs no
        anti-click fade (a 3 ms dip at every section line otherwise). -> {(clip id, "in" | "out")}."""
        out, by = set(), {}
        for c in proj["clips"]:
            if c.get("mute") or c.get("warps") or c["len"] <= 0 or clipfx.needs_hq(c):
                continue
            by.setdefault((c["track"], c["song"], c["stem"]), []).append(c)
        for (_, sid, _), cs in by.items():
            if len(cs) < 2:
                continue
            cs.sort(key=lambda c: c["start"])
            mult = P.song_mult(proj, sid)
            Wm = self._landing(sid, P.song_tspec(proj, sid))
            w = (lambda c: float(c["src"])) if Wm is None else (lambda c: float(np.interp(c["src"], Wm[1], Wm[0])))
            for a, b in zip(cs, cs[1:]):
                ra, rb = float(a.get("rate", 1.0) or 1.0), float(b.get("rate", 1.0) or 1.0)
                if (abs(b["start"] - (a["start"] + a["len"])) < 1e-6 and abs(ra - rb) < 1e-9
                        and float(a.get("gain_db", 0) or 0) == float(b.get("gain_db", 0) or 0)
                        and abs(w(b) - (w(a) + a["len"] * ra / mult)) < 1e-4):
                    out.add((a["id"], "out"))
                    out.add((b["id"], "in"))
        return out

    @staticmethod
    def _entry(c, T, tindex):
        s0 = int(round(c["start"] * T))
        s1 = int(round((c["start"] + c["len"]) * T))
        return dict(s0=s0, s1=s1, g=10 ** (c.get("gain_db", 0.0) / 20),
                    fi=max(MICRO_FADE, int(c.get("fade_in", 0.0) * T)),
                    fo=max(MICRO_FADE, int(c.get("fade_out", 0.0) * T)), ti=tindex[c["track"]], id=c["id"])

    def _landing(self, sid, tspec):
        """(W, beat index) arrays - where each beat of the song lands (timing.positions) - or None for
        plain auto-align. Cached per song + timing spec."""
        if not tspec:
            return None
        cache = self.__dict__.setdefault("_wcache", {})
        k = (sid, tspec)
        if k not in cache:
            info = library.read(sid) or {}
            W = timing.positions(variant.beats_for(info, tspec) if info.get("beats") else [0.0, 1.0], timing.parse(tspec))
            if len(cache) > 64:
                cache.clear()
            cache[k] = (W, np.arange(len(W), dtype=float))
        return cache[k]

    def _draft_entries(self, c, req, bpm, mult, semis, T, tindex, drafts, wsrc=None, Wm=None):
        """Clip whose warp is still rendering -> mixer entries playing its quick R2 drafts (draft.py), split
        at the 16-beat draft chunks with short crossfades; pieces whose draft hasn't landed yet play the
        plain varispeed preview meanwhile and are listed in `drafts`. None = no drafts needed (the plain
        preview is within ~1 cent anyway): the caller previews the whole clip."""
        g = draft.geometry(c["song"], c["stem"], bpm, mult, req[6])
        hq = clipfx.needs_hq(c)
        wsrc = float(c["src"]) if wsrc is None else wsrc
        if g is None or (draft.pitch_accurate_preview(g, semis) and not hq):
            return None
        t0, n_out = g[0], g[3]
        base = self._entry(c, T, tindex)
        s0, s1 = base["s0"], base["s1"]
        rate = float(c.get("rate", 1.0) or 1.0)
        off_f = wsrc * mult * T - t0 * SR                   # sample of the (future) HQ warp at clip start
        info = library.read(c["song"])
        srcbuf = library.peek_stem(c["song"], c["stem"], req[6])

        def src_piece(a, b):                                  # plain varispeed preview of timeline [a, b)
            if srcbuf is None or not info:
                return None
            k = rate / mult
            return dict(base, s0=a, s1=b, cs0=s0, cs1=s1, mode="src", buf=srcbuf,
                        beats=np.asarray(variant.beats_for(info, req[6]), float), src=wsrc + (a - s0) / T * k,
                        W=None if Wm is None else Wm[0], Widx=None if Wm is None else Wm[1],
                        k=k, T=T, step=0.0, off=0)

        tp = clipfx.total_pitch(c) if hq else 0.0
        r_d = rate if hq else 1.0                            # a draft with the clip's stretch built in reads at 1x

        def lookup(A, B, a, b):
            key = draft.key_for(req, A, B, r_d, tp)
            d = draft.open_render(c["song"], key)
            if d is None:
                drafts.append((c["id"], req, A, B, r_d, tp, key, a, b))
                return None
            off = off_f + (a - s0) * rate - A
            if hq:
                return dict(buf=d, off=int(round(off / rate)), step=1.0)
            if abs(rate - 1) < 1e-6:
                return dict(buf=d, off=int(round(off)), step=1.0)
            return dict(buf=d, off=off, step=rate)

        return self._pieces(base, off_f, rate, T, n_out, lookup, src_piece)

    @staticmethod
    def _pieces(base, off_f, rate, T, n_out, lookup, fallback):
        """Split a clip (base: its mixer entry; reading the HQ warp from sample off_f at `rate`) into pieces on
        the 16-beat draft-chunk grid, joined by short crossfades. lookup(A, B, a, b) -> buf/off/step of the
        draft covering warp samples [A, B) for timeline [a, b), or None if it hasn't landed (lookup lists it);
        fallback(a, b) -> a stand-in entry for that stretch meanwhile (or None: silent)."""
        s0, s1 = base["s0"], base["s1"]
        C = draft.CHUNK_BEATS * T
        tl = lambda x: s0 + (x - off_f) / rate               # HQ-warp sample -> timeline sample
        k0 = int(np.floor(off_f / C))
        k1 = int(np.floor((off_f + (s1 - s0) * rate - 1) / C))
        cuts = [int(np.ceil(tl(k * C))) for k in range(k0 + 1, k1 + 1)]
        # no piece shorter than a crossfade (a boundary that close to the clip edge stays in its neighbour)
        cuts = [s0] + [x for x in cuts if s0 + draft.XF < x < s1 - draft.XF] + [s1]
        out = []
        for i in range(len(cuts) - 1):
            a, b = cuts[i], cuts[i + 1]
            xi = draft.XF if i > 0 else 0
            xo = draft.XF if i < len(cuts) - 2 else 0
            k = int(np.floor((off_f + ((a + b) / 2 - s0) * rate) / C))   # the chunk this piece sits in
            A, B = draft.chunk_range(k, T, n_out)
            if B - A < 64:
                continue                          # past either end of the song: silence in the HQ warp too
            f = lookup(A, B, a, b + xo)
            p = dict(base, s0=a, s1=b + xo, cs0=s0, cs1=s1, **f) if f else fallback(a, b + xo)
            if p:
                p.update(xi=xi, xo=xo)
                out.append(p)
        return out

    def reset(self):
        self._reset_next = True
        for f in self.fx.values():
            f.duck_env = 0.0

    # ------------------------------------------------------------------ render
    @staticmethod
    def _mix_clip(c, pos, end, buses):
        a, b = max(pos, c["s0"]), min(end, c["s1"])
        buf = c["buf"]
        if c.get("mode") == "src":
            bt = c["beats"]
            def t_of(n):                       # timeline sample -> source seconds via the beat map
                sb = c["src"] + (n - c["s0"]) / c["T"] * c["k"]
                if c.get("W") is not None:     # sb is a landing position: back to a beat index
                    sb = float(np.interp(sb, c["W"], c["Widx"]))
                i = int(np.clip(np.floor(sb), 0, len(bt) - 2))
                return bt[i] + (sb - i) * (bt[i + 1] - bt[i])
            ta, tb = t_of(a), t_of(b)
            seg = varispeed_read(buf, ta * SR, max(1e-4, (tb - ta) * SR / (b - a)), b - a)
            oa = a - pos
            idx = np.arange(a, b)
        elif c["step"] != 1.0:
            seg = varispeed_read(buf, c["off"] + (a - c["s0"]) * c["step"], c["step"], b - a)
            oa = a - pos
            idx = np.arange(a, b)
        else:
            ba, bb = c["off"] + (a - c["s0"]), c["off"] + (b - c["s0"])
            la, lb = max(ba, 0), min(bb, buf.shape[1])
            if lb <= la:
                return False
            seg = np.asarray(buf[:, la:lb], np.float32)
            oa = a - pos + (la - ba)
            idx = np.arange(a + (la - ba), a + (la - ba) + seg.shape[1])
        env = np.full(seg.shape[1], c["g"], np.float32)
        if c["fi"]:                                          # clip fades (a split clip: relative to the whole clip;
            fi = (idx - c.get("cs0", c["s0"])) / c["fi"]     # 0 = none: a seamless join with the clip before)
            if fi.min() < 1:
                env *= np.sin(np.clip(fi, 0, 1) * np.pi / 2).astype(np.float32)
        if c["fo"]:
            fo = (c.get("cs1", c["s1"]) - idx) / c["fo"]
            if fo.min() < 1:
                env *= np.sin(np.clip(fo, 0, 1) * np.pi / 2).astype(np.float32)
        for k, t in (("xi", (idx - c["s0"])), ("xo", (c["s1"] - idx))):   # crossfades between pieces
            if c.get(k):
                t = t / c[k]
                if t.min() < 1:
                    env *= np.sin(np.clip(t, 0, 1) * np.pi / 2).astype(np.float32)
        buses[c["ti"], :, oa:oa + seg.shape[1]] += seg * env
        return True

    def render(self, pos, n, master=True):
        """Mix samples [pos, pos+n) -> (2, n) float32."""
        tracks, clips = self.tracks, self.clips
        nt = len(tracks)
        buses = np.zeros((nt, 2, n), np.float32) if nt else np.zeros((0, 2, n), np.float32)
        used = np.zeros(nt, bool)
        end = pos + n
        for c in clips:
            if c["s0"] >= end:
                break
            if c["s1"] <= pos:
                continue
            try:
                if self._mix_clip(c, pos, end, buses):
                    used[c["ti"]] = True
            except Exception as e:           # one broken clip must never silence the whole mix
                if c.get("id") not in self._bad:
                    self._bad.add(c.get("id"))
                    print(f"clip {c.get('id')} skipped: {type(e).__name__}: {e}")

        out = np.zeros((2, n), np.float32)
        reset = self._reset_next
        self._reset_next = False
        # sidechain source for ducking: everything on vocal-role tracks
        vox_level = 0.0
        if any(t["duck"] > 0 for t in tracks):
            vi = [i for i, t in enumerate(tracks) if t["role"] == "vocals" and used[i] and t["gain"] > 0]
            if vi:
                v = buses[vi].sum(0)
                vox_level = float(np.sqrt(np.mean(v ** 2)) + 1e-9)
        for i, t in enumerate(tracks):
            fx = self.fx.get(t["id"])
            if not used[i] and (fx is None or fx.board is None):
                continue
            if t["gain"] == 0:
                continue
            x = buses[i]
            if fx is not None and fx.board is not None:
                x = fx.process(x, reset)
            g = t["gain"]
            if t["duck"] > 0 and fx is not None:
                db = 20 * np.log10(vox_level) if vox_level > 0 else -120
                target = float(np.clip((db + 42) / 24, 0, 1)) * t["duck"]
                k = 0.5 if target > fx.duck_env else 0.08            # fast attack, slow release (per block)
                fx.duck_env += (target - fx.duck_env) * k
                g *= 10 ** (-fx.duck_env / 20)
            out += x * (t["lr"][:, None] * g)
        out *= self.master_gain
        if master and self.master is not None and len(self.master) > 0:
            out = self.master.process(out, SR, reset=reset)
        return out


@numba.njit(cache=True)
def _gain_curve(peak, ceiling, look, rel):
    n = peak.shape[0]
    need = np.empty(n, np.float32)
    for i in range(n):
        need[i] = ceiling / peak[i] if peak[i] > ceiling else 1.0
    # lookahead: gain at i must already be <= the requirement anywhere in [i, i+look]
    g = np.empty(n, np.float32)
    q = np.empty(n, np.int64)           # monotonic deque of indices (sliding-window minimum)
    h, t = 0, 0
    j = 0
    for i in range(n):
        while j < n and j <= i + look:
            while t > h and need[q[t - 1]] >= need[j]:
                t -= 1
            q[t] = j
            t += 1
            j += 1
        while q[h] < i:
            h += 1
        g[i] = need[q[h]]
    # smooth: instant attack (already looked ahead), exponential release
    out = np.empty(n, np.float32)
    cur = 1.0
    for i in range(n):
        if g[i] < cur:
            cur = g[i]
        else:
            cur = cur + (g[i] - cur) * rel
        out[i] = cur
    # short linear ramp into each gain drop so the attack itself doesn't click
    res = out.copy()
    for i in range(n - 2, -1, -1):
        step = 1.0 / look
        if res[i] > res[i + 1] + step:
            res[i] = res[i + 1] + step
    return res


def brickwall(x, ceiling=0.891, look_ms=3.0, release_ms=80.0):
    """Lookahead peak limiter: output never exceeds `ceiling` (linear)."""
    peak = np.abs(x).max(0).astype(np.float32)
    look = max(1, int(look_ms / 1000 * SR))
    rel = 1 - np.exp(-1 / (release_ms / 1000 * SR))
    g = _gain_curve(peak, np.float32(ceiling), look, rel)
    y = x * g[None, :]
    return np.clip(y, -ceiling, ceiling)


def bounce(proj, path, progress=lambda f, m="": None, target_lufs=None, block=16384):
    """Offline render of the whole arrangement to a WAV. Two passes when normalizing loudness."""
    import pyloudnorm as pyln
    from . import audio_io
    m = Mixer()
    missing = m.load(proj)
    if missing:
        raise RuntimeError(f"{len(missing)} stems are not rendered yet")
    if m.pending:                                   # HQ stretch/pitch slices must exist for export
        clipfx.render_blocking(m.pending)
        m = Mixer()
        m.load(proj, allow_preview=False)
    total = m.end_sample + int(0.05 * SR)
    gain_db = 0.0
    if target_lufs is not None:
        pre = np.zeros((2, total), np.float32)
        m.reset()
        for p in range(0, total, block):
            k = min(block, total - p)
            pre[:, p:p + k] = m.render(p, k, master=False)
            progress(0.45 * (p + k) / total, "measuring loudness")
        lufs = pyln.Meter(SR).integrated_loudness(pre.T.astype(np.float64))
        if np.isfinite(lufs):
            gain_db = float(np.clip(target_lufs - lufs, -24, 24))
        del pre
    m2 = Mixer()
    pj = dict(proj, master=dict(proj.get("master", {}), gain_db=proj.get("master", {}).get("gain_db", 0.0) + gain_db))
    m2.load(pj, allow_preview=False)
    out = np.zeros((2, total), np.float32)
    lo = 0.45 if target_lufs is not None else 0.0
    for p in range(0, total, block):
        k = min(block, total - p)
        out[:, p:p + k] = m2.render(p, k)
        progress(lo + (0.95 - lo) * (p + k) / total, "rendering")
    if target_lufs is not None:
        # final trim to the exact target, then a true brickwall at -1 dBFS
        lufs = pyln.Meter(SR).integrated_loudness(out.T.astype(np.float64))
        if np.isfinite(lufs):
            out *= 10 ** (np.clip(target_lufs - lufs, -6, 6) / 20)
            gain_db += float(np.clip(target_lufs - lufs, -6, 6))
    out = brickwall(out, 10 ** (-1.0 / 20))
    audio_io.save(path, out, SR, subtype="PCM_24")
    progress(1.0, "done")
    return dict(path=str(path), seconds=total / SR, gain_db=round(gain_db, 2))
