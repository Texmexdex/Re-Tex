"""Stretching / contracting part of a clip without slicing it.

clip["warps"] = [{"a": ..., "b": ..., "r": ..., "kp": ...}]: the song's source beats a..b inside this clip play at
r x the clip's own speed (r < 1 = that part takes longer), kp = keep pitch (False: pitch follows the speed,
like a turntable; None = whatever the clip does). Everything in the clip after a warp moves with it and the
clip's end moves too; nothing else on the timeline changes.

The mixer plays a warped clip as consecutive plain sub-clips (one per stretch), each overlapping the next by a
short crossfade. web/js/warps.js computes exactly the same pieces for drawing and editing.
"""
import numpy as np

from .config import SR

XF = 256                      # crossfade between consecutive pieces (samples)


def pieces(c, mult, landing=None):
    """-> [(u0, u1, w0, w1, r, kp)]: clip offsets u (project beats from the clip start) and landing positions w
    (song beats; = beat index unless beats were un-stretched) of each constant-speed piece."""
    k = float(c.get("rate", 1.0) or 1.0) / float(mult)
    if landing is None:
        wf = lambda b: float(b)
    else:
        W, idx = landing
        wf = lambda b: float(np.interp(b, idx, W))
    L = float(c["len"])
    regs = sorted(((wf(g["a"]), wf(g["b"]), float(g["r"]), g.get("kp")) for g in c.get("warps") or []
                   if g["b"] > g["a"] and g["r"] > 0), key=lambda g: g[0])
    out = []
    u, w = 0.0, wf(c["src"])

    def push(w1, r, kp):
        nonlocal u, w
        du = (w1 - w) / (k * r)
        if du <= 1e-9:
            return
        u1 = min(L, u + du)
        w1 = w + (u1 - u) * k * r
        out.append((u, u1, w, w1, r, kp))
        u, w = u1, w1

    for wa, wb, r, kp in regs:
        if u >= L:
            break
        if wb <= w:
            continue
        if wa > w:
            push(wa, 1.0, None)
        if u < L:
            push(wb, r, kp)
    if u < L:
        out.append((u, L, w, w + (L - u) * k, 1.0, None))
    return out


def expand(c, mult, bpm, landing=None):
    """A clip with warps -> plain sub-clips that play it piece by piece; [c] if it has none."""
    if not c.get("warps"):
        return [c]
    ps = pieces(c, mult, landing)
    if len(ps) <= 1 and (not ps or abs(ps[0][4] - 1) < 1e-9):
        return [c]
    if landing is None:
        binv = lambda w: float(w)
    else:
        W, idx = landing
        binv = lambda w: float(np.interp(w, W, idx))
    xf = XF / (60.0 / float(bpm) * SR)                    # crossfade in project beats
    out = []
    for i, (u0, u1, w0, w1, r, kp) in enumerate(ps):
        last = i == len(ps) - 1
        sc = dict(c)
        sc.pop("warps", None)
        sc.update(id=f"{c['id']}~{i}", start=float(c["start"]) + u0, len=(u1 - u0) + (0.0 if last else xf),
                  src=binv(w0), rate=float(c.get("rate", 1.0) or 1.0) * r,
                  fade_in=float(c.get("fade_in", 0.0)) if i == 0 else xf,
                  fade_out=float(c.get("fade_out", 0.0)) if last else xf)
        if kp is not None:
            sc["keep_pitch"] = bool(kp)
        out.append(sc)
    return out


def parent_id(clip_id):
    return str(clip_id).split("~")[0]
