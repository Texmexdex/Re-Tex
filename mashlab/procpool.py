"""Worker processes for the CPU-heavy renders (warps, clip stretches), with real cancellation.

concurrent.futures.ProcessPoolExecutor can only cancel work that hasn't started yet: a warp nobody wants
any more (the tempo already moved on) would still run to the end and hold a CPU core for a minute.
Here every slot owns one worker process, so a running task is aborted by stopping just that process;
the slot starts a fresh one for its next task. Tasks run by priority (lower first), then in order.
"""
import itertools
import multiprocessing as mp
import os
import queue
import threading
import time
import traceback
from concurrent.futures import Future


class Aborted(Exception):
    """The task was stopped while it ran (see Pool.abort)."""


class WorkerError(Exception):
    """The task raised in the worker process (message carries the worker's traceback)."""


def watch_parent():
    """Call in any worker process: it quits as soon as the main app is gone, so closing the window (or a
    crash) never leaves orphaned workers holding CPU, RAM and cache files."""
    parent = mp.parent_process()

    def watch():
        while True:
            time.sleep(2)
            if parent is None or not parent.is_alive():
                os._exit(0)
    threading.Thread(target=watch, daemon=True).start()


def _preload(modules):
    """Warm-up task: import the heavy modules (and compile/load numba kernels) before real work arrives."""
    import importlib
    for m in modules:
        importlib.import_module(m)
    return True


def _serve(conn):
    """Worker process: run tasks from the pipe until it closes."""
    watch_parent()
    while True:
        try:
            msg = conn.recv()
        except (EOFError, OSError):
            return
        if msg is None:
            return
        fn, args = msg
        try:
            res = ("ok", fn(*args))
        except BaseException as e:
            res = ("err", f"{type(e).__name__}: {e}\n{traceback.format_exc()}")
        try:
            conn.send(res)
        except (EOFError, OSError):
            return


class Pool:
    def __init__(self, workers, preload=()):
        """preload: modules every worker imports as soon as it starts. Workers start right away and a killed
        one is replaced immediately, so the next task never waits ~2 s for a cold process."""
        self._preload = tuple(preload)
        self._ctx = mp.get_context("spawn")
        self._q = queue.PriorityQueue()
        self._seq = itertools.count()
        self._lock = threading.Lock()
        self._running = {}          # future -> worker process running it
        self._aborting = set()
        self._procs = set()
        self._closed = False
        self._slots = [threading.Thread(target=self._slot, daemon=True, name=f"render-slot-{i}")
                       for i in range(workers)]
        for t in self._slots:
            t.start()

    def submit(self, fn, *args, priority=1):
        if self._closed:
            raise RuntimeError("cannot schedule new futures after shutdown")
        fut = Future()
        self._q.put((priority, next(self._seq), (fut, fn, args)))
        return fut

    def abort(self, fut, kill=True):
        """Drop a queued task, or (kill=True) stop a running one. Its future ends cancelled / with Aborted.
        For tasks that finish in a second or two, kill=False is cheaper: a killed worker has to be replaced."""
        if fut.cancel():
            return True
        if not kill:
            return False
        with self._lock:
            proc = self._running.get(fut)
            if proc is None:
                return False                                    # already finished
            self._aborting.add(fut)
        try:
            proc.kill()
        except Exception:
            pass
        return True

    def busy(self):
        with self._lock:
            return len(self._running)

    def kill(self):
        """Stop everything now (used on shutdown)."""
        self._closed = True
        for _ in self._slots:
            self._q.put((float("inf"), next(self._seq), None))
        with self._lock:
            procs = list(self._procs)
        for p in procs:
            try:
                p.kill()
            except Exception:
                pass

    def _spawn(self):
        mine, theirs = self._ctx.Pipe()
        p = self._ctx.Process(target=_serve, args=(theirs,), daemon=True)
        p.start()
        theirs.close()
        with self._lock:
            self._procs.add(p)
        if self._preload:
            try:
                mine.send((_preload, (self._preload,)))
                mine.recv()
            except Exception:
                pass                                            # a real task will find out
        return p, mine

    def _retire(self, proc, conn):
        if proc is None:
            return
        try:
            conn.close()
        except Exception:
            pass
        try:
            proc.join(timeout=2)
        except Exception:
            pass
        with self._lock:
            self._procs.discard(proc)

    def _slot(self):
        proc = conn = None
        if self._preload and not self._closed:
            try:
                proc, conn = self._spawn()
            except Exception:
                proc = conn = None
        while True:
            item = self._q.get()[2]
            if item is None:
                break
            fut, fn, args = item
            if not fut.set_running_or_notify_cancel():          # dropped while queued
                continue
            status = val = None
            try:
                if proc is None or not proc.is_alive():
                    self._retire(proc, conn)
                    proc, conn = self._spawn()
                with self._lock:
                    self._running[fut] = proc
                conn.send((fn, args))
                status, val = conn.recv()
            except (EOFError, OSError) as e:                    # the worker died: aborted, shut down or crashed
                with self._lock:
                    aborted = fut in self._aborting
                self._retire(proc, conn)
                proc = conn = None
                status, val = "died", Aborted() if aborted or self._closed else WorkerError(f"worker process died ({e!r})")
                if self._preload and not self._closed:           # replace it now, not when the next task comes
                    try:
                        proc, conn = self._spawn()
                    except Exception:
                        proc = conn = None
            except Exception as e:                              # e.g. arguments that can't be pickled
                status, val = "died", e
            finally:
                with self._lock:
                    self._running.pop(fut, None)
                    self._aborting.discard(fut)
            if status == "ok":
                fut.set_result(val)
            elif status == "err":
                fut.set_exception(WorkerError(val))
            else:
                fut.set_exception(val)
        self._retire(proc, conn)


class Renders:
    """Background renders identified by a key (a cache file to produce). sync() is given everything that is
    wanted right now: keys no longer wanted are dropped (running ones killed if kill_running), new ones
    are submitted in the order given."""

    def __init__(self, get_pool, priority=1, kill_running=True, on_dropped=None):
        self._get_pool, self.priority, self.kill_running = get_pool, priority, kill_running
        self.on_dropped = on_dropped            # key -> None: clean up after a render stopped mid-write
        self._lock = threading.Lock()
        self._futs = {}
        self._failed = set()                    # keys that raised: not retried (the same input fails again)

    def sync(self, wanted, on_done, start=True):
        """wanted: list of (key, fn, args). on_done() runs after each render that completes."""
        pool = self._get_pool()
        keys = {k for k, _, _ in wanted}
        new = []
        with self._lock:
            stale = [(key, fut) for key, fut in self._futs.items() if key not in keys]
            if start:
                for key, fn, args in wanted:
                    if key not in self._futs and key not in self._failed:
                        self._futs[key] = pool.submit(fn, *args, priority=self.priority)
                        new.append((key, self._futs[key]))
        # abort OUTSIDE the lock: cancelling a queued future runs its done-callback (self._done, which takes
        # the lock) right here in this thread - under the lock that deadlocked the whole server
        for key, fut in stale:
            if pool.abort(fut, kill=self.kill_running) or fut.done():
                with self._lock:
                    if self._futs.get(key) is fut:
                        del self._futs[key]
        for key, fut in new:
            fut.add_done_callback(lambda f, key=key: self._done(f, key, on_done))

    def _done(self, f, key, on_done):
        with self._lock:
            if self._futs.get(key) is f:
                del self._futs[key]
        if f.cancelled():
            return
        e = f.exception()
        if isinstance(e, Aborted):
            if self.on_dropped:
                self.on_dropped(key)
        elif e is not None:
            with self._lock:
                self._failed.add(key)
            print("render failed:", key, str(e).splitlines()[0])
        else:
            try:
                on_done()
            except Exception as err:                        # e.g. a file still locked by Windows
                print("after render:", type(err).__name__, err)

    def busy(self):
        with self._lock:
            return len(self._futs)
