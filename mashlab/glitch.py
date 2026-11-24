"""Glitch scores: a MIDI part plays edits on project lanes instead of notes.

Every note of the part (a MIDI song in the library, midi.py) becomes one glitch effect on the chosen lanes, from the
note's start to its end, at the same place in the project - the same thing Perform mode's pads do (perform_live.py),
but written down, so it can be repeated, edited in the piano roll, and taken from any MIDI file (a famous song's snare
part as the stutter rhythm of yours). The result is ordinary clips (one undo step in the UI), read from the lane as it
was before - effects never feed on each other.

Which effect a note plays: "mix" (the default) is "by note" half the time and "by chance" the rest; "by note" gives each pitch of the part its own effect, the part's most used
pitches getting the most weighted effects - so a drum part's hats become the busy little stutters and its crash
something rare; "by chance" draws one per note by the weights. Rare effects (roll, drop out, bar back / ahead, tape
stop, slow down...) also wait out a cool-down of some bars before they can come again. Times are project beats
(bar 1 = beat 0); a beat is a quarter note, so a 1/16 is 0.25 beat.
"""
import math
import uuid

import numpy as np

# name -> (label, kind, params, default weight, rare)
EFFECTS = {
    "st16":   ("stutter 1/16", "repeat", {"len": 0.25}, 5, False),
    "st32":   ("stutter 1/32", "repeat", {"len": 0.125}, 4, False),
    "st8":    ("stutter 1/8", "repeat", {"len": 0.5}, 4, False),
    "st4":    ("stutter 1/4", "repeat", {"len": 1.0}, 2, False),
    "st8t":   ("stutter 1/8 triplet", "repeat", {"len": 1 / 3}, 2, False),
    "st16t":  ("stutter 1/16 triplet", "repeat", {"len": 1 / 6}, 2, False),
    "re16":   ("retrigger 1/16", "retrig", {"len": 0.25}, 4, False),
    "re8":    ("retrigger 1/8", "retrig", {"len": 0.5}, 3, False),
    "gate16": ("chop 1/16", "gate", {"len": 0.25}, 4, False),
    "gate32": ("chop 1/32", "gate", {"len": 0.125}, 3, False),
    "gate8":  ("chop 1/8", "gate", {"len": 0.5}, 2, False),
    "pat3":   ("gate x..x..x.", "pattern", {"steps": "x..x..x.", "len": 0.25}, 2, False),
    "pat2":   ("gate xx.x", "pattern", {"steps": "xx.x", "len": 0.25}, 2, False),
    "echo":   ("echo stutter", "echo", {"len": 0.5, "decay": 0.62}, 2, False),
    "dive":   ("pitch-dive stutter", "dive", {"len": 0.25, "step": 1.5}, 2, False),
    "rise":   ("pitch-rise stutter", "dive", {"len": 0.25, "step": -1.5}, 1, False),
    "scram":  ("scramble 1/16", "scramble", {"len": 0.25}, 2, False),
    "speed":  ("double speed", "rate", {"rate": 2.0}, 1, False),
    "roll":   ("roll (speeding up)", "roll", {}, 1, True),
    "tape":   ("tape stop", "tape", {}, 1, True),
    "slow":   ("half speed", "rate", {"rate": 0.5}, 1, True),
    "drop":   ("drop out", "mute", {}, 1, True),
    "back1":  ("beat back", "jump", {"beats": -1}, 1, True),
    "backbar": ("bar back", "jump", {"bars": -1}, 1, True),
    "fwdbar": ("bar ahead", "jump", {"bars": 1}, 1, True),
}
ROLL = ((0.25, 4), (0.125, 8), (0.0625, 16))


def catalog():
    return [{"id": k, "label": v[0], "weight": v[3], "rare": v[4]} for k, v in EFFECTS.items()]


# -------------------------------------------------------------------- what an effect plays
def pieces(eff, T, R, bpb, seed=0):
    """[(t0, t1, read, gain, rate)] covering [T, R): from t0 the lane plays from timeline position `read` at speed
    `rate` (1 = as is; other speeds are varispeed, the pitch moves with it)."""
    label, kind, p, _w, _r = EFFECTS[eff]
    out = []

    def add(t0, t1, read, gain=1.0, rate=1.0):
        t0, t1 = max(t0, T), min(t1, R)
        if t1 > t0 + 1e-9:
            out.append((t0, t1, read, gain, rate))

    L = float(p.get("len", 0.25))
    if kind in ("repeat", "retrig", "echo", "dive"):
        src = T - L if kind in ("repeat",) else T
        k = 0
        h = T
        while h < R - 1e-9:
            if kind == "echo":
                add(h, h + L, src, p["decay"] ** k)
            elif kind == "dive":
                add(h, h + L, src, 1.0, 2 ** (-p["step"] * k / 12))
            else:
                add(h, h + L, src)
            h += L
            k += 1
    elif kind == "gate":
        h = T
        while h < R - 1e-9:
            add(h, h + L / 2, h)
            add(h + L / 2, h + L, h + L / 2, 0.0)
            h += L
    elif kind == "pattern":
        steps, h, k = p["steps"], T, 0
        while h < R - 1e-9:
            on = steps[k % len(steps)] == "x"
            add(h, h + L, h, 1.0 if on else 0.0)
            h += L
            k += 1
    elif kind == "scramble":
        rng = np.random.default_rng(seed)
        n_src = int(round(bpb / L))                     # 1/16 slices of the bar before the note, shuffled
        base = T - bpb
        h, order = T, []
        while h < R - 1e-9:
            if not order:
                order = rng.permutation(n_src).tolist()
            add(h, h + L, base + order.pop() * L)
            h += L
    elif kind == "roll":
        t = T
        for Lr, n in ROLL:
            for i in range(n):
                add(t + i * Lr, t + (i + 1) * Lr, T)
            t += Lr * n
        while t < R - 1e-9:
            add(t, t + 1 / 32, T)
            t += 1 / 32
    elif kind == "tape":                                # slows to a stop over the note, in 1/32 steps
        n = max(1, int(math.ceil((R - T) / 0.125)))
        read = T
        for i in range(n):
            r = max(0.06, 1.0 - (i + 0.5) / n)
            add(T + i * 0.125, T + (i + 1) * 0.125, read, 1.0, r)
            read += 0.125 * r
    elif kind == "rate":
        add(T, R, T, 1.0, p["rate"])
    elif kind == "mute":
        add(T, R, T, 0.0)
    elif kind == "jump":
        add(T, R, T + p.get("beats", 0) + p.get("bars", 0) * bpb)
    return out


# -------------------------------------------------------------------- notes -> gestures
def plan(notes, opts, bpb):
    """notes [(start, end, pitch, vel)] in project beats -> [(T, R, effect id)], one at a time per lane."""
    weights = {k: float((opts.get("weights") or {}).get(k, EFFECTS[k][3])) for k in EFFECTS}
    pool = [k for k, w in weights.items() if w > 0]
    if not pool:
        return []
    rng = np.random.default_rng(int(opts.get("seed", 1) or 1))
    density = float(opts.get("density", 100)) / 100.0
    cool = float(opts.get("cooldown_bars", 2)) * bpb
    snap = float(opts.get("snap", 0.25) or 0)
    mode = opts.get("assign", "mix")
    notes = sorted(notes, key=lambda n: (n[0], -n[3]))
    # by note: pitches by how often they're played, effects by weight; the busiest pitch gets the heaviest effect
    count = {}
    for n in notes:
        count[n[2]] = count.get(n[2], 0) + 1
    pitches = sorted(count, key=lambda p: (-count[p], p))
    ranked = sorted(pool, key=lambda k: (-weights[k], list(EFFECTS).index(k)))
    by_pitch = {p: ranked[i % len(ranked)] for i, p in enumerate(pitches)}
    probs = np.array([weights[k] for k in pool], float)
    probs /= probs.sum()
    out, last_rare = [], -1e9
    for s, e, p, v in notes:
        if density < 1 and rng.random() > density:
            continue
        T = round(s / snap) * snap if snap else s
        R = max(T + (snap or 0.125), (round(e / snap) * snap) if snap else e)
        R = min(R, T + bpb)                            # one effect lasts a bar at most
        if mode == "note" or (mode == "mix" and rng.random() < 0.5):
            eff = by_pitch[p]                          # its own effect (by note; half the time with variety)
        else:
            eff = pool[int(rng.choice(len(pool), p=probs))]
        if EFFECTS[eff][4]:
            if T - last_rare < cool:                   # a rare one too soon: a common one instead
                common = [k for k in ranked if not EFFECTS[k][4]]
                if not common:
                    continue
                eff = common[hash((p, round(T, 3))) % len(common)] if mode == "note" else \
                    common[int(rng.integers(len(common)))]
            else:
                last_rare = T
        if out and T < out[-1][1] - 1e-9:              # overlapping notes: the earlier one gives way
            if T <= out[-1][0] + 1e-9:
                continue                               # same start (a chord): keep the first
            out[-1] = (out[-1][0], T, out[-1][2])
        out.append((T, R, eff))
    return out


def notes_in_project(part_notes, midi_from, midi_bars, start_beat, length, loop, bpb):
    """The part's notes from MIDI bar `midi_from` (beats of its song) for `midi_bars` bars, placed at project beat
    `start_beat`, repeated to fill `length` beats when `loop`."""
    a = float(midi_from)
    span = float(midi_bars) * bpb
    seg = [(s - a, s - a + l, p, v) for s, l, p, v in part_notes if a <= s < a + span]
    out, k = [], 0
    while True:
        off = start_beat + k * span
        if off >= start_beat + length - 1e-9:
            break
        for s, e, p, v in seg:
            if off + s < start_beat + length - 1e-9:
                out.append((off + s, min(off + e, start_beat + length), p, v))
        if not loop:
            break
        k += 1
    return out


# -------------------------------------------------------------------- gestures -> clips
def _uid():
    return "c" + uuid.uuid4().hex[:8]


def _clips_for(project, lane, eff_pieces, lo, hi):
    """New clips playing the pieces, read from the lane's original clips (any speed)."""
    from . import readhead as RH, project as P
    out = []
    lane = sorted(lane, key=lambda c: c["start"])
    land = {c["song"]: RH._landing(project, c["song"]) for c in lane}
    for t0, t1, read, gain, rate in eff_pieces:
        if gain <= 0:
            continue
        d = t1 - t0
        p0, p1 = read, read + d * rate                 # timeline beats being read
        for c in lane:
            a, b = max(p0, c["start"]), min(p1, c["start"] + c["len"])
            if b - a < 1e-4:
                continue
            mult = P.song_mult(project, c["song"])
            W = land[c["song"]]
            w = RH._w_of(W, c["src"]) + (a - c["start"]) * float(c.get("rate", 1) or 1) / mult
            start = t0 + (a - p0) / rate
            nc = RH._clip(None, c["song"], c["stem"], start, RH._src_of(W, w), (b - a) / rate, gain)
            nc["gain_db"] = round(nc["gain_db"] + float(c.get("gain_db", 0) or 0), 2)
            if abs(rate - 1) > 1e-6:
                nc["rate"] = round(rate * float(c.get("rate", 1) or 1), 6)
                nc["keep_pitch"] = False               # varispeed: dives, tape stop, half / double speed
            nc.pop("readhead", None)
            nc["glitch"] = True
            nc["id"] = _uid()
            nc["track"] = c["track"]
            out.append(nc)
    return out


def _cut(project, c, lo, hi):
    """(left part or None, right part or None) of clip c around [lo, hi)."""
    from . import readhead as RH, project as P
    left = right = None
    if c["start"] < lo - 1e-9:
        left = {**c, "len": round(lo - c["start"], 6), "fade_out": 0.03}
    if c["start"] + c["len"] > hi + 1e-9 and not c.get("warps"):
        Wc, mc = RH._landing(project, c["song"]), P.song_mult(project, c["song"])
        k = float(c.get("rate", 1) or 1)
        w = RH._w_of(Wc, c["src"]) + (hi - c["start"]) * k / mc
        right = {**c, "id": _uid(), "start": round(hi, 6), "len": round(c["start"] + c["len"] - hi, 6),
                 "src": round(RH._src_of(Wc, w), 6), "fade_in": 0}
    return left, right


def apply(project, tracks, gestures):
    """-> (the project's new clip list, number of effects). Each lane is cut where the effects go and the
    effects' clips put in; everything is read from the lanes as they were."""
    bpb = int(project.get("bpb") or 4)
    clips = list(project.get("clips", []))
    for tid in tracks:
        lane = [c for c in clips if c.get("track") == tid]
        if not lane:
            continue
        src = [c for c in lane if not c.get("warps")]
        new = []
        for k, (T, R, eff) in enumerate(gestures):
            ps = pieces(eff, T, R, bpb, seed=k)
            new += _clips_for(project, src, ps, T, R)
        # the lane with the effects' spans cut out
        cur = list(lane)
        for T, R, _ in gestures:
            nxt = []
            for c in cur:
                if c.get("warps") or c["start"] + c["len"] <= T + 1e-9 or c["start"] >= R - 1e-9:
                    nxt.append(c)
                    continue
                left, right = _cut(project, c, T, R)
                nxt += [x for x in (left, right) if x]
            cur = nxt
        clips = [c for c in clips if c.get("track") != tid] + cur + new
    return clips, len(gestures)


def register(app):
    from fastapi import HTTPException, Request
    import asyncio
    from . import midi

    @app.get("/api/glitch/effects")
    def glitch_effects():
        return {"effects": catalog()}

    @app.post("/api/glitch/apply")
    async def glitch_apply(request: Request):
        """Body: {project, midi, part, tracks, start_bar, bars, midi_bar, midi_bars, loop, opts} -> {clips, count,
        effects}: the project's whole new clip list (the UI applies it as one undo step)."""
        b = await request.json()
        proj = b.get("project") or {}
        try:
            mi = midi.read(b["midi"])
            part = midi.part_of(mi, b["part"])
        except (KeyError, TypeError):
            raise HTTPException(404, "no such MIDI part")
        bpb = int(proj.get("bpb") or 4)
        tracks = [t for t in (b.get("tracks") or []) if any(c.get("track") == t for c in proj.get("clips", []))]
        if not tracks:
            raise HTTPException(400, "pick at least one lane with clips on it")
        start = (max(1, int(b.get("start_bar", 1))) - 1) * bpb
        length = max(1, int(b.get("bars", 8))) * bpb
        notes = part.get("notes") or []
        if not notes:
            raise HTTPException(400, "that part has no notes")
        from . import library
        info = library.read(b["midi"]) or {}
        fb = int(info.get("first_beat", 0) or 0)
        m_from = b.get("midi_bar")
        m_from = fb + (int(m_from) - 1) * bpb if m_from else math.floor((min(n[0] for n in notes) - fb) / bpb) * bpb + fb
        m_bars = max(1, int(b.get("midi_bars") or b.get("bars", 8)))
        pn = notes_in_project(notes, m_from, m_bars, start, length, bool(b.get("loop", True)), bpb)
        ges = plan(pn, b.get("opts") or {}, bpb)
        clips, n = await asyncio.to_thread(apply, proj, tracks, ges)
        used = {}
        for _, _, e in ges:
            used[EFFECTS[e][0]] = used.get(EFFECTS[e][0], 0) + 1
        return {"clips": clips, "count": len(ges), "effects": used}
