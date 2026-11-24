"""Playing a MIDI part's instrument live from the computer keyboard (the piano roll's pad keys).

Rendering an instrument on every key press would be far too slow (FluidSynth and VST3 start up each time), so the
notes are rendered ahead into a bank - one held note per key, all keys of a few octaves in ONE render - and a key
press just starts that buffer in the player (Player._cb -> KEYS.render, next to the audition voice). Key up fades
the note out. Chop kits need no bank: a key press cuts its chop straight from the stem (chops.chop_buffer).
Notes held past HOLD seconds end on their own (their release is in the bank)."""
import hashlib
import json
import threading
from collections import OrderedDict

import numpy as np

from .config import SR

SLOT, HOLD = 4.5, 3.5                    # seconds per key in a bank render; how long a key is held in it
MAX_VOICES = 48


class Keys:
    def __init__(self):
        self.banks = OrderedDict()       # bank id -> {inst, notes: {pitch: (2, n)}, state, sid, pid}
        self.voices = []                 # {buf, pos, gain, key, rel, rel_n}
        self.lock = threading.Lock()
        self.render_lock = threading.Lock()

    @staticmethod
    def bank_id(sid, pid, inst):
        return hashlib.sha1(json.dumps([sid, pid, inst], sort_keys=True, default=str).encode()).hexdigest()[:14]

    # ---------------------------------------------------------------- banks
    def prepare(self, sid, pid, inst, lo, hi):
        """Make sure keys lo..hi of this instrument are ready to play. -> (bank id, keys ready)."""
        from . import instruments
        bid = self.bank_id(sid, pid, inst)
        b = self.banks.get(bid)
        if b is None:
            b = self.banks[bid] = {"inst": inst, "notes": {}, "state": {}, "sid": sid, "pid": pid}
            while len(self.banks) > 6:
                self.banks.popitem(last=False)
        self.banks.move_to_end(bid)
        if (inst or {}).get("type") == "chops":
            from . import chops
            chops.source(inst)                       # load the stem now, not on the first key
            if chops.tuned(inst) and len(inst.get("pitches") or []) != len(inst.get("chops") or []):
                b["inst"] = inst = dict(inst, pitches=chops.chop_pitches(inst))
            return bid, len(inst.get("chops") or [])
        need = [p for p in range(max(0, int(lo)), min(127, int(hi)) + 1) if p not in b["notes"]]
        with self.render_lock:
            for k in range(0, len(need), 24):        # a couple of octaves per render keeps each one short
                part = need[k:k + 24]
                notes = [(i * SLOT, i * SLOT + HOLD, p, 100) for i, p in enumerate(part)]
                y = instruments.render(inst, notes, [], int(len(part) * SLOT * SR))
                for i, p in enumerate(part):
                    seg = y[:, int(i * SLOT * SR):int((i + 1) * SLOT * SR)]
                    loud = np.flatnonzero(np.max(np.abs(seg), axis=0) > 1e-4)
                    end = int(loud[-1]) + 1 if len(loud) else 0
                    b["notes"][p] = np.ascontiguousarray(seg[:, :max(end, 1)], np.float32)
        return bid, len(b["notes"])

    def _buffer(self, b, pitch):
        """(buffer, loop cycle or None) for a key press."""
        inst = b["inst"] or {}
        if inst.get("type") == "chops":
            from . import chops
            mode = inst.get("play") or "keys"
            if mode == "flow":                       # the next stretch of the stem, as long as the key is held
                return chops.stream_take(inst, b["state"].get("cur", 0), int(12 * SR)), None
            i = chops.index_for(inst, pitch, b["state"])
            if i is None:
                return None, None
            tune = chops.tune_semis(inst, i, pitch) if chops.tuned(inst) else 0.0
            if mode == "cue":                        # the key's own chop start, on through the stem, till key up
                seg = chops.cue_take(inst, i, int((5 if tune else 12) * SR))     # (shorter when it's retuned live)
                return chops.shift(seg, tune), None
            if chops.fit_of(inst) in ("repeat", "stretch"):     # held: loop the chop, seamlessly, till key up
                raw = chops.shift(chops.chop_buffer(inst, i, chops.ratio_for(inst, pitch), fades=False), tune)
                first, cycle = chops.loop_cycle(raw)
                return chops._fade_in(first.copy(), 0.002), cycle
            return chops.shift(chops.chop_buffer(inst, i, chops.ratio_for(inst, pitch)), tune), None
        buf = b["notes"].get(pitch)
        if buf is None:                              # outside the prepared keys: render just this one
            self.prepare(b["sid"], b["pid"], inst, pitch, pitch)
            buf = b["notes"].get(pitch)
        return buf, None

    # ---------------------------------------------------------------- notes
    def on(self, bid, pitch, vel=100):
        b = self.banks.get(bid)
        if b is None:
            raise KeyError("that instrument isn't ready - press a key again")
        if (b["inst"] or {}).get("play") == "flow":       # legato: the key still down gives way first, so the
            with self.lock:                              # stream carries on exactly where it got to
                for v in self.voices:
                    if v["key"][0] == bid and v.get("flow") is not None:
                        self._release(v, 0.006)
                        self._flow_done(v)
        buf, cycle = self._buffer(b, int(pitch))
        if buf is None or buf.shape[1] == 0:
            return False
        inst = b["inst"] or {}
        chop = inst.get("type") == "chops"
        mode = inst.get("play") or "keys"
        mono = chop and inst.get("mono", mode not in ("drums", "cue"))
        if chop:
            from . import chops
            fit = "cut" if mode in ("flow", "cue") else chops.fit_of(inst)
            gain = max(1, min(127, int(vel))) / 127.0 if inst.get("velocity", mode == "drums") else 1.0
        else:
            fit, gain = "cut", max(1, min(127, int(vel))) / 100.0
        key = (bid, int(pitch))
        with self.lock:
            for v in self.voices:
                if v["key"] == key or (mono and v["key"][0] == bid):
                    self._release(v, 0.006)          # a retriggered key / a mono chop kit: the old one gives way
                    self._flow_done(v)
            self.voices.append({"buf": buf, "cycle": cycle, "pos": 0, "gain": gain, "key": key,
                                "rel": None, "rel_n": 1, "gate": fit != "whole",
                                "flow": b if chop and mode == "flow" else None,
                                "release": float(inst.get("release", 0.12) or 0.12)})
            if len(self.voices) > MAX_VOICES:
                for v in self.voices[:len(self.voices) - MAX_VOICES]:
                    self._release(v, 0.01)
        return True

    @staticmethod
    def _flow_done(v):
        """A "flow" key let go: the stem's stream carries on from where this key stopped."""
        b = v.get("flow")
        if b is not None:
            b["state"]["cur"] = b["state"].get("cur", 0) + v["pos"]
            v["flow"] = None

    @staticmethod
    def _release(v, sec):
        n = max(1, int(sec * SR))
        if v["rel"] is None or v["rel"] > n:
            v["rel"], v["rel_n"] = n, n

    def off(self, bid, pitch):
        key = (bid, int(pitch))
        with self.lock:
            for v in self.voices:
                if v["key"] == key and v["rel"] is None and v["gate"]:
                    self._release(v, min(1.0, max(0.03, v["release"])))
                    self._flow_done(v)

    def all_off(self):
        with self.lock:
            for v in self.voices:
                self._release(v, 0.03)

    # ---------------------------------------------------------------- the audio thread
    def render(self, frames):
        out = np.zeros((2, frames), np.float32)
        with self.lock:
            keep = []
            for v in self.voices:
                buf, cyc, pos = v["buf"], v["cycle"], v["pos"]
                k = frames if cyc is not None else min(frames, buf.shape[1] - pos)
                if v["rel"] is not None:
                    k = min(k, v["rel"])
                if k > 0:
                    if cyc is None:
                        seg = buf[:, pos:pos + k]
                    else:                              # a looping chop: once through, then the cycle round and round
                        idx = pos + np.arange(k)
                        lb = buf.shape[1]
                        seg = np.where(idx < lb, buf[:, np.minimum(idx, lb - 1)],
                                       cyc[:, (np.maximum(idx - lb, 0)) % cyc.shape[1]])
                    seg = seg * v["gain"]
                    if v["rel"] is not None:
                        r0 = v["rel"] / v["rel_n"]
                        seg = seg * np.linspace(r0, r0 - k / v["rel_n"], k, endpoint=False, dtype=np.float32)
                        v["rel"] -= k
                    out[:, :k] += seg
                    v["pos"] = pos + k
                alive = cyc is not None or v["pos"] < buf.shape[1]
                if alive and (v["rel"] is None or v["rel"] > 0):
                    keep.append(v)
                else:
                    self._flow_done(v)
            self.voices = keep
        return out


KEYS = Keys()
