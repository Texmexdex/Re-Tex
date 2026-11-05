"""Sample editor (the pop-out window, web/sample.html): one clip's audio on a grid you pick - halves down to 64ths,
triplets too - edited with chops, stutters, layers, tape stop / start, echo and tape delay, then baked into a sample
that replaces the clip in its own lane. Nothing else in the arrangement moves.

  open_clip(project, clip_id)  the clip rendered exactly as it plays (warp, stretch, pitch, screw - no master bus),
                               kept as the editing session's source (samples/_edit/<eid>/)
  render(eid, ops)             the source with the edit list applied, in order -> (2, n) float32
  apply(eid, ops)              ...baked into a sample (samples/smp_<id>/, song-shaped: one stem, a steady grid at the
                               project's tempo) - published as "sample_applied" so the arrangement swaps the clip

Every op has a range [a, b) in beats from the clip's start. Splices crossfade a few ms at both ends (no clicks).
"""
import copy
import json
import math
import shutil
import time
import uuid

import numba
import numpy as np

from . import audio_io, library, mixer, project as P, warp
from .config import SAMPLES, SR
from .events import bus

EDITS = SAMPLES / "_edit"
XF = 0.004                       # s: crossfade at every splice edge
PIECE_FADE = 0.002               # s: fade in / out on each stutter piece (a retrigger, not a seam)
MAX_TAIL_BEATS = 32              # an echo tail never runs longer than this past its source
KEEP_EDITS_S = 24 * 3600         # editing sessions older than this are cleared when a new one opens

_cache = {}                      # eid -> (meta, source) for the open sessions


# ---------------------------------------------------------------- sessions

def _render_raw(proj, n):
    """The project's audio without the master bus (one neutral track) - the clip exactly as it plays."""
    from . import clipfx
    reqs = [r for r in P.render_requests(proj) if not warp.is_cached(*r)]
    if reqs:
        warp.render_many(reqs)
    m = mixer.Mixer()
    m.load(proj)
    if m.pending:                                   # stretched / pitched parts render first
        clipfx.render_blocking(m.pending)
        m = mixer.Mixer()
        m.load(proj, allow_preview=False)
    out = np.zeros((2, n), np.float32)
    for p in range(0, n, 16384):
        k = min(16384, n - p)
        out[:, p:p + k] = m.render(p, k, master=False)
    return out


def open_clip(project, clip_id):
    """Start an editing session on one clip. -> meta {eid, title, stem, bpm, bpb, beats, ...}"""
    c = next((x for x in project.get("clips", []) if x.get("id") == clip_id), None)
    if c is None:
        raise KeyError("no such clip")
    _housekeeping()
    sub = copy.deepcopy(project)
    sub["tracks"] = [dict(id="t", name="edit", role="any", color="#888", gain_db=0.0, pan=0.0, mute=False, solo=False,
                          hpf=0, lpf=0, duck=0, pocket=0)]
    sub["clips"] = [dict(copy.deepcopy(c), id="c", track="t", start=0.0, gain_db=0.0, fade_in=0.0, fade_out=0.0, mute=False)]
    sub["master"] = dict(glue=False, limit=False, gain_db=0.0)
    sub.pop("loop", None)
    bpm = float(project["bpm"])
    n = int(round(float(c["len"]) * 60.0 / bpm * SR))
    y = _render_raw(sub, n)
    info = library.read(c["song"]) or {}
    eid = uuid.uuid4().hex[:10]
    d = EDITS / eid
    d.mkdir(parents=True)
    audio_io.save(d / "source.wav", y, SR, subtype="FLOAT")
    meta = dict(eid=eid, project=project.get("id"), clip=clip_id, song=c["song"], stem=c["stem"],
                title=(info.get("title") or c["song"]).replace("Sample: ", ""), bpm=bpm, bpb=int(project.get("bpb") or 4),
                beats=float(c["len"]), opened=time.time())
    (d / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    _cache[eid] = (meta, y)
    return meta


def session(eid):
    """(meta, source) of an editing session."""
    eid = str(eid)
    if eid not in _cache:
        d = EDITS / eid
        if not (d / "meta.json").exists():
            raise KeyError("no such editing session")
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        _cache[eid] = (meta, audio_io.load(d / "source.wav", SR).astype(np.float32))
        while len(_cache) > 6:
            _cache.pop(next(iter(_cache)))
    return _cache[eid]


def _housekeeping():
    if not EDITS.is_dir():
        return
    for d in EDITS.iterdir():
        try:
            if d.is_dir() and time.time() - d.stat().st_mtime > KEEP_EDITS_S:
                shutil.rmtree(d, ignore_errors=True)
                _cache.pop(d.name, None)
        except OSError:
            pass


# ---------------------------------------------------------------- building blocks

def _range(o, S):
    A = max(0, S(o["a"]))
    return A, max(A, S(o["b"]))


def _ramp(n):
    return np.linspace(0.0, 1.0, max(1, n), dtype=np.float32)


def _fit(y, n):
    """y with at least n samples (silence added)."""
    return y if y.shape[1] >= n else np.concatenate([y, np.zeros((y.shape[0], n - y.shape[1]), np.float32)], axis=1)


def _splice(y, A, seg):
    """seg replaces y[A:A+len(seg)], crossfading XF in at the start and back out at the end."""
    L = seg.shape[1]
    if L <= 0:
        return y
    y = _fit(y, A + L)
    f = min(int(XF * SR), L // 2)
    new = seg.copy()
    if f:
        r = _ramp(f)
        new[:, :f] = y[:, A:A + f] * (1 - r) + seg[:, :f] * r
        new[:, L - f:] = seg[:, L - f:] * (1 - r) + y[:, A + L - f:A + L] * r
    y[:, A:A + L] = new
    return y


def _edges(seg, fade=PIECE_FADE):
    """Short fades at both ends of a piece."""
    f = min(int(fade * SR), seg.shape[1] // 2)
    if f:
        r = _ramp(f)
        seg[:, :f] *= r
        seg[:, -f:] *= r[::-1]
    return seg


def _read(y, start, rates):
    """Varispeed read: output sample i plays y at start + sum(rates[:i]) (linear interpolation)."""
    pos = start + np.concatenate([[0.0], np.cumsum(rates[:-1], dtype=np.float64)])
    idx = np.arange(y.shape[1], dtype=np.float64)
    return np.stack([np.interp(pos, idx, ch, left=0.0, right=0.0) for ch in y]).astype(np.float32)


def _pitch(seg, semis):
    """Pitch shift, length kept (Rubber Band R3)."""
    import pedalboard
    if abs(semis) < 1e-3:
        return seg
    return pedalboard.time_stretch(seg, SR, stretch_factor=1.0, pitch_shift_in_semitones=float(semis),
                                   high_quality=True).astype(np.float32)[:, :seg.shape[1]]


@numba.njit(cache=True)
def _echo_kernel(x, D, fb, lp_a, wow, flutter, drive, sr):
    """Feedback delay: each repeat goes round a one-pole low-pass (darker every time), optional tape saturation (tanh)
    and a wandering read head (wow ~0.55 Hz, flutter ~6.5 Hz, in samples). x: (ch, n) send -> (ch, n) wet."""
    ch, n = x.shape
    size = D + int(wow + flutter) + 8
    buf = np.zeros((ch, size), np.float32)
    lp = np.zeros(ch, np.float32)
    out = np.zeros((ch, n), np.float32)
    w = 0
    for i in range(n):
        mod = wow * math.sin(2.0 * math.pi * 0.55 * i / sr) + flutter * math.sin(2.0 * math.pi * 6.5 * i / sr)
        rd = w - D - mod
        while rd < 0:
            rd += size
        i0 = int(rd)
        fr = rd - i0
        i0 = i0 % size
        i1 = (i0 + 1) % size
        for c in range(ch):
            d = buf[c, i0] * (1.0 - fr) + buf[c, i1] * fr
            lp[c] += lp_a * (d - lp[c])
            v = lp[c]
            if drive > 0:
                v = math.tanh(v * drive) / drive
            out[c, i] = v
            buf[c, w] = x[c, i] + fb * v
        w += 1
        if w >= size:
            w = 0
    return out


# ---------------------------------------------------------------- the edits

def _op_silence(y, o, S):
    A, B = _range(o, S)
    return _splice(y, A, np.zeros((y.shape[0], B - A), np.float32))


def _op_reverse(y, o, S):
    A, B = _range(o, S)
    y = _fit(y, B)
    return _splice(y, A, _edges(y[:, A:B][:, ::-1].copy()))


def _op_repeat(y, o, S):
    """Stutter: the first cell of the range, again and again to its end. roll: the pieces halve for each half of
    what's left (to 1/8 of the cell); rise: semitones up across the run (varispeed); level: even / swell in / fade."""
    A, B = _range(o, S)
    cell = float(o.get("cell", 0.25))
    y = _fit(y, B)
    total, lens, t = o["b"] - o["a"], [], 0.0
    while t < total - 1e-9:
        k = math.floor(-math.log2(max(1 - t / total, 1e-9)) + 1e-9) if o.get("roll") else 0
        L = min(max(cell / 2 ** k, cell / 8), total - t)
        lens.append(L)
        t += L
    seg, n = np.zeros((y.shape[0], B - A), np.float32), len(lens)
    t = 0.0
    for i, L in enumerate(lens):
        x = i / (n - 1) if n > 1 else 1.0
        r = 2 ** (float(o.get("rise", 0)) * x / 12)
        g = {"in": 10 ** (-18 * (1 - x) / 20), "out": 10 ** (-18 * x / 20)}.get(o.get("level"), 1.0)
        p0, p1 = S(o["a"] + t) - A, S(o["a"] + t + L) - A
        piece = _read(y, A, np.full(p1 - p0, r)) if abs(r - 1) > 1e-6 else y[:, A:A + (p1 - p0)].copy()
        seg[:, p0:p1] = _edges(piece) * g
        t += L
    return _splice(y, A, seg)


def _op_paste(y, o, S):
    """Copy [from_a, from_b) to a: replacing what's there, or layered on top (mode "mix")."""
    FA, A = max(0, S(o["from_a"])), max(0, S(o["a"]))
    FB = max(FA, S(o["from_b"]))
    y = _fit(y, max(FB, A + FB - FA))
    src = y[:, FA:FB].copy() * 10 ** (float(o.get("gain_db", 0)) / 20)
    if o.get("mode") == "mix":
        y[:, A:A + src.shape[1]] += _edges(src, XF)
        return y
    return _splice(y, A, src)


def _op_layer(y, o, S):
    """The range again on top of itself: shifted by `offset` beats, pitched by `semis`, at `gain_db`."""
    A, B = _range(o, S)
    y = _fit(y, B)
    seg = _pitch(y[:, A:B].copy(), float(o.get("semis", 0))) * 10 ** (float(o.get("gain_db", -3)) / 20)
    T = A + S(float(o.get("offset", 0)))
    if T < 0:
        seg, T = seg[:, -T:], 0
    y = _fit(y, T + seg.shape[1])
    y[:, T:T + seg.shape[1]] += _edges(seg, XF)
    return y


def _op_tapestop(y, o, S):
    """The speed ramps from full to nothing across the range (pitch with it), like a turntable switched off."""
    A, B = _range(o, S)
    y = _fit(y, B)
    N = B - A
    rates = (1.0 - np.arange(N) / N) ** float(o.get("curve", 1.6))
    seg = _read(y, A, rates)
    f = min(int(0.03 * SR), N // 3)
    seg[:, -f:] *= _ramp(f)[::-1]                 # the last of it fades as it stops
    out = y.copy()
    out[:, A:B] = seg                             # it starts exactly where the audio is: no seam at A
    g = min(int(XF * SR), out.shape[1] - B)
    if g > 0:                                     # ...and what follows comes back in without a click
        out[:, B:B + g] *= _ramp(g)
    return out


def _op_tapestart(y, o, S):
    """From nothing up to full speed across the range, arriving exactly where the audio carries on at its end."""
    A, B = _range(o, S)
    y = _fit(y, B)
    N = B - A
    rates = ((np.arange(N) + 0.5) / N) ** float(o.get("curve", 1.6))
    start = max(0.0, B - float(rates.sum()))
    seg = _read(y, start, rates)
    f = min(int(0.01 * SR), N // 3)
    seg[:, :f] *= _ramp(f)
    return _splice(y, A, seg)


def _op_pitch(y, o, S):
    """Pitch the range: "keep" its length (stretch), or "vari" (varispeed: faster and higher, like a turntable)."""
    A, B = _range(o, S)
    y = _fit(y, B)
    semis = float(o.get("semis", 0))
    if o.get("mode") == "vari":
        seg = _read(y, A, np.full(B - A, 2 ** (semis / 12)))
    else:
        seg = _pitch(y[:, A:B].copy(), semis)
    return _splice(y, A, seg)


def _op_gain(y, o, S):
    A, B = _range(o, S)
    y = _fit(y, B)
    g = 10 ** (float(o.get("db", 0)) / 20)
    shape = o.get("shape", "flat")
    env = np.full(B - A, g, np.float32)
    if shape == "in":
        env = _ramp(B - A) * g
    elif shape == "out":
        env = _ramp(B - A)[::-1] * g
    seg = y[:, A:B] * env
    return _splice(y, A, seg)


def _op_echo(y, o, S):
    """Echo / tape delay on the range: it feeds a delay of `time` beats (feedback, darker each repeat with `tone`,
    tape: wow / flutter and saturation); the repeats ring on past the range - over the clip's end too. out: the
    original stops at the end of the range and only the echoes carry on (an echo-out)."""
    A, B = _range(o, S)
    y = _fit(y, B)
    D = max(1, S(float(o.get("time", 0.75))))
    fb = float(np.clip(o.get("feedback", 0.45), 0.0, 0.95))
    reps = math.ceil(math.log(0.001) / math.log(max(fb, 0.05)))                 # until -60 dB
    tail = min(reps * D + D, S(MAX_TAIL_BEATS))
    x = np.zeros((y.shape[0], (B - A) + tail), np.float32)
    x[:, :B - A] = _edges(y[:, A:B].copy(), XF)
    tape = bool(o.get("tape"))
    tone = float(np.clip(o.get("tone", 0.45 if tape else 0.7), 0.0, 1.0))
    lp_a = 1.0 - math.exp(-2 * math.pi * (900 + tone * 14000) / SR)
    wet = _echo_kernel(x, D, fb, lp_a, 0.0015 * SR if tape else 0.0, 0.00012 * SR if tape else 0.0,
                       1.6 if tape else 0.0, float(SR))
    y = _fit(y, A + x.shape[1])
    if o.get("out"):                                           # echo-out: the original stops at the range's end
        y = _splice(y, B, np.zeros((y.shape[0], min(tail, y.shape[1] - B)), np.float32))
    y[:, A:A + x.shape[1]] += wet * float(np.clip(o.get("mix", 0.6), 0.0, 1.5))
    return y


OPS = {"silence": _op_silence, "reverse": _op_reverse, "repeat": _op_repeat, "paste": _op_paste, "layer": _op_layer,
       "tapestop": _op_tapestop, "tapestart": _op_tapestart, "pitch": _op_pitch, "gain": _op_gain, "echo": _op_echo}


def render(eid, ops):
    """The session's source with the edit list applied, in order. -> (meta, (2, n) float32)"""
    meta, src = session(eid)
    spb = 60.0 / float(meta["bpm"])
    S = lambda beats: int(round(float(beats) * spb * SR))      # beats -> samples (signed: offsets can be negative)
    y = src.copy()
    for o in ops or []:
        fn = OPS.get(o.get("op"))
        if fn is None:
            raise ValueError(f"unknown edit: {o.get('op')}")
        if float(o.get("b", o.get("a", 0))) < float(o.get("a", 0)):
            raise ValueError("a range must run forwards")
        y = fn(y, o, S)
    return meta, y


def preview(eid, ops):
    """Render to the session's preview.wav (16-bit, for the editor to play and draw). -> {seconds, beats, version}"""
    meta, y = render(eid, ops)
    d = EDITS / meta["eid"]
    v = int(time.time() * 1000)
    audio_io.save(d / "preview.wav", np.clip(y, -1, 1), SR, subtype="PCM_16")
    return dict(seconds=round(y.shape[1] / SR, 4), beats=round(y.shape[1] / SR / (60.0 / meta["bpm"]), 4), version=v)


# ---------------------------------------------------------------- baking it into a sample

def _trim_tail(y, n_min):
    """Drop silence after the clip's own length (an echo tail ends where it fades under -70 dB)."""
    if y.shape[1] <= n_min:
        return y
    loud = np.flatnonzero(np.max(np.abs(y[:, n_min:]), axis=0) > 10 ** (-70 / 20))
    end = n_min + (int(loud[-1]) + 1 if len(loud) else 0)
    return y[:, :end + int(0.01 * SR)] if end < y.shape[1] else y


def apply(eid, ops):
    """Bake the edits into a sample and tell the arrangement to swap the clip for it. -> {sid, beats, song}"""
    meta, y = render(eid, ops)
    spb = 60.0 / float(meta["bpm"])
    y = _trim_tail(y, int(round(meta["beats"] * spb * SR)))
    beats = y.shape[1] / SR / spb
    orig = library.read(meta["song"]) or {}
    sid = "smp_" + uuid.uuid4().hex[:10]
    d = SAMPLES / sid
    (d / "stems").mkdir(parents=True)
    (d / "peaks").mkdir()
    nb = int(math.ceil(beats)) + 2
    bpb = int(meta.get("bpb") or 4)
    info = dict(id=sid, title=f"Sample: {meta['title']} ({meta['stem']})", kind="sample", status="ready", error=None,
                source=None, created=time.time(), parent=dict(song=meta["song"], stem=meta["stem"], clip=meta["clip"]),
                duration=round(y.shape[1] / SR, 4), bpm=float(meta["bpm"]), bpb=bpb, stable_tempo=True,
                beats=[round(i * spb, 6) for i in range(nb)], bars=list(range(0, nb, bpb)), first_beat=0, last_beat=nb - 1,
                segments=[], sections=[], key=orig.get("key"), tuning=None, tuning_st=None, separation=None,
                has_vocals=meta["stem"] == "vocals", peaks_per_sec=100, grid_version=1, stems_rev=0, stems=[meta["stem"]])
    library.write(sid, info)
    library.save_stems(sid, {meta["stem"]: y})
    pk = audio_io.peaks(y, SR, 100)
    pk.tofile(d / "peaks" / f"{meta['stem']}.bin")
    pk.tofile(d / "peaks" / "mix.bin")
    summ = library.summary(library.read(sid))
    bus.publish("sample_applied", project=meta["project"], clip=meta["clip"], sid=sid, beats=round(beats, 6), song=summ,
                stem=meta["stem"])
    return dict(sid=sid, beats=round(beats, 6), song=summ)
