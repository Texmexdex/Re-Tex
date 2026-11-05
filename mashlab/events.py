"""Thread-safe pub/sub: worker threads publish, each browser tab's SSE stream subscribes."""
import asyncio
import json
import threading


class EventBus:
    def __init__(self):
        self._subs = []
        self._lock = threading.Lock()

    def subscribe(self):
        loop = asyncio.get_running_loop()
        q = asyncio.Queue(maxsize=1000)
        with self._lock:
            self._subs.append((loop, q))
        return q

    def unsubscribe(self, q):
        with self._lock:
            self._subs = [(l, s) for l, s in self._subs if s is not q]

    def publish(self, kind, **data):
        msg = json.dumps({"type": kind, **data}, default=_np_default)
        with self._lock:
            subs = list(self._subs)
        for loop, q in subs:
            try:
                loop.call_soon_threadsafe(_put, q, msg)
            except RuntimeError:
                pass


def _np_default(o):
    import numpy as np
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serializable: {type(o).__name__}")


def _put(q, msg):
    if q.full():
        try:
            q.get_nowait()
        except asyncio.QueueEmpty:
            pass
    q.put_nowait(msg)


bus = EventBus()
