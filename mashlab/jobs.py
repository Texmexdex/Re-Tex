"""Background jobs with live progress. GPU work (separation/analysis) runs one at a time."""
import itertools
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor

from .events import bus

_ids = itertools.count(1)
_jobs = {}
_gpu = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gpu")
_misc = ThreadPoolExecutor(max_workers=4, thread_name_prefix="job")


class Job:
    def __init__(self, kind, title, ref=None):
        self.id = next(_ids)
        self.kind, self.title, self.ref = kind, title, ref
        self.progress, self.message, self.status = 0.0, "queued", "queued"
        self.error = None
        self.created = time.time()
        self.cancelled = False
        self._last = 0.0

    def to_dict(self):
        return dict(id=self.id, kind=self.kind, title=self.title, ref=self.ref, progress=round(self.progress, 3),
                    message=self.message, status=self.status, error=self.error)

    def update(self, progress=None, message=None, force=False):
        if self.cancelled:
            raise JobCancelled()
        if progress is not None:
            self.progress = max(0.0, min(1.0, progress))
        if message:
            self.message = message
        now = time.time()
        if force or now - self._last > 0.15:
            self._last = now
            bus.publish("job", job=self.to_dict())

    def sub(self, lo, hi, prefix=""):
        """Progress callback mapping 0..1 into [lo, hi] of this job."""
        def cb(f, msg=""):
            self.update(lo + (hi - lo) * f, (prefix + msg) if msg else (prefix or None))
        return cb


class JobCancelled(Exception):
    pass


# A GPU job that hangs (a stalled model download, a subprocess waiting on input, an ffmpeg that never
# returns) would otherwise block the single-worker _gpu pool forever and leave every later GPU job stuck at
# "queued"/"starting". A watchdog thread marks such a job failed and lets the queue drain. The underlying
# worker thread may still be alive, but no NEW GPU job is blocked behind a job already declared dead, and the
# user gets a clear error instead of a silent hang.
GPU_JOB_TIMEOUT = 1800        # 30 min: generous for a long separation+analysis, short of "forever"


def _watchdog(job, done_evt):
    """If a GPU job doesn't finish within GPU_JOB_TIMEOUT and hasn't made progress recently, fail it so the
    queue isn't wedged. Progress updates (job.update) reset the clock, so a slow-but-live job is never killed."""
    while not done_evt.wait(timeout=15):
        idle = time.time() - max(job._last, job.created)
        if idle > GPU_JOB_TIMEOUT and job.status == "running":
            job.cancelled = True      # cooperative: the next job.update() in fn() raises JobCancelled
            if job.status == "running":
                job.status, job.error, job.message = "error", \
                    f"timed out after {int(idle)}s with no progress (the GPU worker may be stuck; restart if needed)", "timed out"
                bus.publish("job", job=job.to_dict())
            return


def submit(kind, title, fn, ref=None, gpu=False):
    """fn(job) runs in the background; its return value is published with the 'done' event."""
    job = Job(kind, title, ref)
    _jobs[job.id] = job
    bus.publish("job", job=job.to_dict())

    def run():
        if job.cancelled:
            return
        job.status = "running"
        job.update(0.0, "starting", force=True)
        done_evt = threading.Event()
        if gpu:
            threading.Thread(target=_watchdog, args=(job, done_evt), daemon=True,
                             name=f"gpu-watchdog-{job.id}").start()
        try:
            res = fn(job)
            if job.status != "error":      # watchdog may have already failed it
                job.status, job.progress, job.message = "done", 1.0, "done"
                bus.publish("job", job=job.to_dict(), result=res if isinstance(res, (dict, list, str, int, float)) else None)
        except JobCancelled:
            if job.status != "error":
                job.status, job.message = "cancelled", "cancelled"
                bus.publish("job", job=job.to_dict())
        except Exception as e:
            traceback.print_exc()
            job.status, job.error, job.message = "error", f"{type(e).__name__}: {e}", "failed"
            bus.publish("job", job=job.to_dict())
        finally:
            done_evt.set()

    (_gpu if gpu else _misc).submit(run)
    return job


def cancel(job_id):
    j = _jobs.get(job_id)
    if j and j.status in ("queued", "running"):
        j.cancelled = True
        if j.status == "queued":
            j.status = "cancelled"
            bus.publish("job", job=j.to_dict())
        return True
    return False


def active():
    cutoff = time.time() - 30
    return [j.to_dict() for j in _jobs.values() if j.status in ("queued", "running") or j.created > cutoff]


def start(kind, title, ref=None):
    """A job the caller drives itself (no worker thread): report with progress(), end with finish()."""
    job = Job(kind, title, ref)
    job.status, job.message = "running", "starting"
    _jobs[job.id] = job
    bus.publish("job", job=job.to_dict())
    return job


def progress(job, value, message=None):
    job.progress = max(0.0, min(1.0, value))
    if message:
        job.message = message
    bus.publish("job", job=job.to_dict())


def finish(job, status="done", message=None, error=None):
    job.status, job.error = status, error
    if status == "done":
        job.progress = 1.0
    job.message = message or status
    bus.publish("job", job=job.to_dict())
