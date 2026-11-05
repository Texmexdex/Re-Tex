"""Live audio output: plays the arrangement through the Mixer, plus a separate audition voice and a metronome on the
project's grid (added after the master, in the same callback - so it's sample-locked to the mix, never early or late
against what you hear)."""
import math
import threading
import time
import numpy as np
import sounddevice as sd

from .config import SR
from .events import bus
from .mixer import Mixer
from . import perform_live


def _click(freq, level, dur=0.05):
    """A hard, bright click: a 1.5 ms noise snap + a short high tone (a plain sine gets buried under distorted guitars) -
    the beat grid window's click, so both sound the same."""
    n = int(dur * SR)
    i = np.arange(n)
    rng = np.random.default_rng(7)
    snap = (rng.uniform(-1, 1, n) * np.exp(-i / (SR * 0.0015))).astype(np.float32) * 0.6
    tone = (np.sin(2 * np.pi * freq * i / SR) * np.exp(-i / (SR * 0.011))).astype(np.float32)
    return (snap + tone) * level


CLICKS = (_click(2000, 0.6), _click(2800, 0.85), _click(3600, 1.0))   # a beat, a bar's 1, a 4-bar phrase's 1


class Player:
    def __init__(self):
        self.mixer = Mixer()
        self.playing = False
        self.pos = 0                 # next sample to render
        self.loop = None             # (start, end) samples or None
        self.volume = 1.0
        self.aud = None              # audition: dict(buf, pos, end, gain, id)
        self.click = False           # metronome on the project's grid
        self.click_gain = 0.5
        self._spb = 60.0 / 120.0 * SR                      # samples per project beat
        self._bpb = 4
        self.stream = None
        self.lock = threading.RLock()          # audio thread + transport; held only for quick swaps
        self.load_lock = threading.Lock()      # one project compile at a time (can take a while)
        self.latency = 0.0
        self.error = None
        self.dev_lock = threading.RLock()      # opening / closing / re-scanning the output device
        from .config import read_settings
        self.device_name = read_settings().get("audio_output") or None   # None = the system default
        self.device_index = None
        try:
            self.device_index = self._resolve(self.device_name)
        except Exception:
            pass

    # ------------------------------------------------------------------ stream
    def _ensure_stream(self):
        with self.dev_lock:
            if self.stream is not None:
                return
            try:
                self._open(self.device_index)
            except Exception as e:
                if self.device_index is None:
                    self.stream = None
                    self.error = f"audio output failed: {e}"
                    raise
                self.error = f"couldn't open {self.device_name} ({e}) - using the system default"
                self.device_index = None
                self._open(None)

    def _open(self, index):
        self.stream = sd.OutputStream(samplerate=SR, channels=2, dtype="float32", blocksize=1024,
                                      latency="low", callback=self._cb, device=index)
        self.stream.start()
        self.latency = float(self.stream.latency or 0.0)
        self.error = None

    def _close(self):
        """Close the stream (never while holding self.lock: closing waits for the audio callback, which takes it)."""
        s, self.stream = self.stream, None
        if s is not None:
            try:
                s.abort()
                s.close()
            except Exception:
                pass

    # ------------------------------------------------------------------ output device
    @staticmethod
    def _outputs():
        """[(index, host api name, device name)] for every output PortAudio sees."""
        apis = sd.query_hostapis()
        return [(i, apis[d["hostapi"]]["name"], d["name"]) for i, d in enumerate(sd.query_devices())
                if d["max_output_channels"] >= 2]

    @classmethod
    def output_names(cls):
        """Output devices by their full names (DirectSound lists them uncut; MME cuts names at 31 characters), without
        the 'default device' aliases - 'System default' stands for those."""
        outs = cls._outputs()
        names = [n for _, api, n in outs if api == "Windows DirectSound" and n != "Primary Sound Driver"]
        if not names:                                      # not Windows: the default host API's devices
            api0 = sd.query_hostapis(sd.default.hostapi)["name"]
            names = [n for _, api, n in outs if api == api0]
        return list(dict.fromkeys(names))

    @classmethod
    def _resolve(cls, name):
        """PortAudio index to open for a device name: its MME entry (the driver playback has always used - MME names
        are the first 31 characters), else the exact name on any driver; None = the system default."""
        if not name:
            return None
        outs = cls._outputs()
        mme = [(i, n) for i, api, n in outs if api == "MME" and n != "Microsoft Sound Mapper - Output"]
        exact = [i for i, n in mme if n == name]
        if exact:
            return exact[0]
        pref = [(i, n) for i, n in mme if name.startswith(n)]
        if len(pref) == 1:
            return pref[0][0]
        other = [i for i, _, n in outs if n == name]
        return other[0] if other else None

    def set_output(self, name):
        """Switch the output device now; playback carries on from where it was. name None/"" = system default."""
        from .config import write_settings
        with self.dev_lock:
            busy = self.stream is not None
            self._close()
            self.device_name = name or None
            self.device_index = self._resolve(self.device_name)
            missing = self.device_name is not None and self.device_index is None
            if busy:
                self._ensure_stream()
            if missing:
                self.error = f"{self.device_name} isn't connected - using the system default"
        write_settings({"audio_output": self.device_name})
        bus.publish("transport", **self.state())
        return self.output_state()

    def refresh_outputs(self):
        """Re-scan the devices (a headset plugged in after start), re-opening the stream: 'System default' then
        follows whatever Windows has as default now."""
        with self.dev_lock:
            busy = self.stream is not None
            self._close()
            try:
                sd._terminate()
                sd._initialize()
            except Exception:
                pass
            self.device_index = self._resolve(self.device_name)
            if busy:
                self._ensure_stream()
        return self.output_state()

    def output_state(self):
        active = None
        try:
            idx = self.device_index if self.device_index is not None else sd.default.device[1]
            active = sd.query_devices(idx)["name"]
        except Exception:
            pass
        names = self.output_names()
        if active:                                         # MME's cut-off name -> the full one
            active = next((n for n in names if n.startswith(active)), active)
        return {"devices": names, "current": self.device_name, "active": active, "error": self.error}

    def _add_clicks(self, out, off, p0, n):
        """Metronome clicks for project samples [p0, p0 + n), into out[:, off:off + n] - every beat of the project's
        grid (beat 0 = bar 1), louder on each bar's 1 and loudest on every 4th bar's."""
        spb, L = self._spb, len(CLICKS[0])
        for b in range(max(0, math.floor((p0 - L) / spb) + 1), math.floor((p0 + n - 1) / spb) + 1):
            s = int(round(b * spb))
            a, e = max(s, p0), min(s + L, p0 + n)
            if e > a:
                c = CLICKS[2 if b % (4 * self._bpb) == 0 else 1 if b % self._bpb == 0 else 0]
                out[:, off + a - p0:off + e - p0] += c[a - s:e - s] * self.click_gain

    def _cb(self, outdata, frames, t, status):
        try:
            out = np.zeros((2, frames), np.float32)
            with self.lock:
                if self.playing:
                    if self.loop and self.pos >= self.loop[1]:
                        self.pos = self.loop[0]
                        self.mixer.reset()
                    n1 = frames
                    if self.loop and self.pos + frames > self.loop[1]:
                        n1 = self.loop[1] - self.pos
                    out[:, :n1] = self._mix(self.pos, n1)
                    if self.click:
                        self._add_clicks(out, 0, self.pos, n1)
                    self.pos += n1
                    if n1 < frames:
                        self.pos = self.loop[0]
                        self.mixer.reset()
                        out[:, n1:] = self._mix(self.pos, frames - n1)
                        if self.click:
                            self._add_clicks(out, n1, self.pos, frames - n1)
                        self.pos += frames - n1
                    elif not self.loop and self.pos > self.mixer.end_sample + SR:
                        self.playing = False
                        bus.publish("transport", **self.state())
                a = self.aud
                if a is not None:
                    k = min(frames, a["end"] - a["pos"])
                    if k > 0:
                        seg = np.asarray(a["buf"][:, a["pos"]:a["pos"] + k], np.float32) * a["gain"]
                        # 5 ms fades at the edges of the audition region
                        f = int(0.005 * SR)
                        idx = np.arange(a["pos"], a["pos"] + k)
                        env = np.clip(np.minimum(idx - a["start"], a["end"] - idx) / f, 0, 1)
                        out[:, :k] += seg * env
                        a["pos"] += k
                    if a["pos"] >= a["end"]:
                        if a.get("loop"):
                            a["pos"] = a["start"]
                        else:
                            self.aud = None
                            bus.publish("audition", playing=False, id=a.get("id"))
            outdata[:] = np.clip(out * self.volume, -1.0, 1.0).T
        except Exception as e:           # never let the audio thread die
            outdata[:] = 0
            # log once, then at most every 5 s with a count (the callback runs ~40x per second)
            now = time.time()
            self._errs = getattr(self, "_errs", 0) + 1
            if now - getattr(self, "_err_t", 0) > 5:
                extra = f" (x{self._errs})" if self._errs > 1 else ""
                print(f"audio callback error{extra}: {type(e).__name__}: {e}")
                self._err_t, self._errs = now, 0

    def _mix(self, pos, n):
        """The mix for [pos, pos+n), with any Perform-mode takes being played live (perform_live.py).
        Without takes this is exactly mixer.render; if the live part ever fails, the plain mix plays."""
        lv = perform_live.LIVE
        if lv.g:
            try:
                out = lv.render(self.mixer, pos, n, self._spb, self._bpb)
                if out is not None:
                    return out
            except Exception as e:
                lv.failed(e)
        lv.last_end = pos + n
        return self.mixer.render(pos, n)

    # ------------------------------------------------------------------ controls
    def set_project(self, proj):
        # compile outside the audio lock (the audio thread and play/stop never wait for it); only the final
        # swap takes the lock
        with self.load_lock:
            missing = self.mixer.load(proj, swap_lock=self.lock)
        perform_live.LIVE.on_project(proj)       # takes now in the project's clips stop playing live
        with self.lock:
            lp = proj.get("loop", {})
            T = 60.0 / float(proj["bpm"]) * SR
            self._spb, self._bpb = T, int(proj.get("bpb") or 4)
            if lp.get("on") and lp.get("end", 0) > lp.get("start", 0):
                self.loop = (int(lp["start"] * T), int(lp["end"] * T))
            else:
                self.loop = None
        return missing

    def play(self, from_sample=None):
        self._ensure_stream()
        with self.lock:
            if from_sample is not None:
                self.pos = max(0, int(from_sample))
            self.mixer.reset()
            self.playing = True
        perform_live.LIVE.stopped()
        bus.publish("transport", **self.state())

    def stop(self):
        with self.lock:
            self.playing = False
        perform_live.LIVE.stopped()
        bus.publish("transport", **self.state())

    def seek(self, sample):
        with self.lock:
            self.pos = max(0, int(sample))
            self.mixer.reset()
        bus.publish("transport", **self.state())

    def audition(self, buf, start, end, gain=1.0, ident=None, loop=False):
        self._ensure_stream()
        with self.lock:
            start = max(0, int(start))
            end = min(buf.shape[1], int(end))
            self.aud = dict(buf=buf, start=start, pos=start, end=end, gain=gain, id=ident, loop=loop)
        bus.publish("audition", playing=True, id=ident, start=start / SR, end=end / SR, t=time.time())

    def stop_audition(self):
        with self.lock:
            a, self.aud = self.aud, None
        if a:
            bus.publish("audition", playing=False, id=a.get("id"))

    def set_click(self, on=None, gain=None):
        with self.lock:
            if on is not None:
                self.click = bool(on)
            if gain is not None:
                self.click_gain = float(np.clip(gain, 0.0, 1.5))

    def state(self):
        return dict(playing=self.playing, pos=self.pos / SR, t=time.time(), latency=self.latency,
                    loop=[self.loop[0] / SR, self.loop[1] / SR] if self.loop else None, error=self.error,
                    click=self.click, click_gain=self.click_gain)


player = Player()
