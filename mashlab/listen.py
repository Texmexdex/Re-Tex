"""Songs described by ear: NVIDIA's Music Flamingo listens to 30 s of each song and writes its genre, a sub-genre
and one line on how it sounds. In the Oct 15 blind test the user picked its descriptions over the text model's
(which only knows titles) 8 to 2: the text model called a metal band "techno" and a hardcore song "dream pop".

It needs ~18 GB of GPU memory and its own Python (torch + transformers >= 5.19 + xgrammar), so it runs as a
separate process (listen_worker.py), on request, as a GPU job. Configured in data/config.json:

    "listen": {"python": "<that venv's python.exe>", "model": "nvidia/music-flamingo-2601-hf",
               "hf_home": "<where the model is cached>"}

No "listen" block -> the feature is off and nothing shows. Results live in DATA/meta.json under each song's
"heard" (libmeta keeps them next to the tags); the heard genre is stored as the song's genre guess (the user's own
edits and file tags still win). The DJ brain's brief and the YuE2 style prompt quote the heard line.
"""
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

from . import appcfg, library

WORKER = Path(__file__).resolve().parent / "listen_worker.py"
CLIP_S = 30.0
FREE_NEEDED = 18e9          # bytes of GPU memory the model wants (16.5 GB weights + a 30 s clip)


def cfg():
    c = (appcfg.load_config() or {}).get("listen") or {}
    return c if c.get("python") and c.get("model") else None


def available():
    """-> (ok, reason)"""
    c = cfg()
    if not c:
        return False, "not set up (data/config.json \"listen\")"
    if not Path(c["python"]).is_file():
        return False, f"its Python is missing: {c['python']}"
    return True, ""


def _free_gpu():
    """Drop the app's own cached GPU models (they reload on next use) so the listener gets the card."""
    from . import align
    for mod, names in (("separate", ("_model", "_rof", "_sw")), ("beats", ("_model",))):
        try:
            m = __import__(f"mashlab.{mod}", fromlist=["_"])
            lock = getattr(m, "_lock", None)
            for n in names:
                if lock is not None:
                    with lock:
                        setattr(m, n, None)
                else:
                    setattr(m, n, None)
        except Exception:
            pass
    align.unload()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            return torch.cuda.mem_get_info()[0]
    except Exception:
        pass
    return None


def about(sid, info=None, folders=None):
    """What the model is told besides the audio - what the app knows about the song."""
    from . import libmeta
    info = info or library.read(sid)
    folders = folders or library.get_folders()
    fname = {f["id"]: f["name"] for f in folders["folders"]}
    k = (info.get("key") or {}).get("name", "")
    artist = libmeta.all_meta().get(sid, {}).get("artist") or fname.get(folders["assign"].get(sid), "-")
    return (f"title: {info['title']}; artist or library folder: {artist}; {round(info.get('bpm') or 0)} BPM"
            + (f"; key {k}" if k else ""))


def _clip(sid, path):
    import librosa
    import soundfile as sf
    from . import audio_io
    from .config import SR, resolve_source
    src = resolve_source(library.read(sid)["source"])
    if Path(src).is_file():
        y = audio_io.load(src, SR).mean(0)
    else:                       # imported from a folder of stems (no single file): the song's own stems, summed
        stems = [library.load_stem(sid, s) for s in ("drums", "bass", "other", "vocals")]
        n0 = min(s.shape[1] for s in stems)
        y = sum(s[:, :n0] for s in stems).mean(0)
    n = int(CLIP_S * SR)
    a = max(0, int(0.4 * (len(y) - n)))
    m = librosa.resample(y[a:a + n], orig_sr=SR, target_sr=16000)[:int(CLIP_S * 16000)]
    sf.write(path, m, 16000, subtype="PCM_16")


def describe(ids, progress=lambda f, m="": None, cancelled=lambda: False):
    """Listen to these songs (ready ones) and store what's heard. -> {sid: {genre, subgenre, sound} | {error}}"""
    from . import libmeta
    ok, why = available()
    if not ok:
        raise RuntimeError(f"Listening isn't available: {why}")
    c = cfg()
    ids = [s for s in ids if (library.read(s) or {}).get("status") == "ready"]
    if not ids:
        return {}
    out = {}
    with tempfile.TemporaryDirectory(prefix="listen-") as tmp:
        tmp = Path(tmp)
        folders = library.get_folders()
        items = []
        for k, sid in enumerate(ids):
            if cancelled():
                return out
            progress(0.2 * k / len(ids), f"cutting clips {k + 1}/{len(ids)}")
            f = tmp / f"{k}.wav"
            try:
                _clip(sid, f)
            except Exception as e:
                out[sid] = {"error": f"couldn't read the audio: {type(e).__name__}"}
                continue
            items.append({"id": sid, "clip": str(f), "about": about(sid, folders=folders)})
        (tmp / "job.json").write_text(json.dumps({"model": c["model"], "items": items}), encoding="utf-8")
        res = tmp / "out.jsonl"
        progress(0.2, "freeing the GPU")
        free = _free_gpu()
        if free is not None and free < FREE_NEEDED:
            print(f"listen: only {free / 1e9:.1f} GB of GPU memory free - it will run, slowly")
        env = dict(os.environ, HF_HUB_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1", PYTHONIOENCODING="utf-8")
        if c.get("hf_home"):
            env["HF_HOME"] = c["hf_home"]
        for k in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"):     # its own Python, not the app's
            env.pop(k, None)
        progress(0.22, "loading Music Flamingo")
        p = subprocess.Popen([c["python"], "-I", str(WORKER), str(tmp / "job.json"), str(res)], env=env,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                             errors="replace", creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        seen, log = 0, []
        try:
            while True:
                rows = [json.loads(x) for x in res.read_text(encoding="utf-8").splitlines() if x.strip()] if res.exists() else []
                for r in rows[seen:]:
                    if r.get("id"):
                        out[r["id"]] = {k: r[k] for k in ("genre", "subgenre", "sound", "error") if k in r}
                        if "error" not in r:
                            libmeta.set_heard(r["id"], out[r["id"]], c["model"])
                seen = len(rows)
                n = sum(1 for r in rows if r.get("id"))
                progress(0.25 + 0.75 * n / max(1, len(items)), f"listened to {n}/{len(items)}")
                if any(r.get("done") for r in rows) or p.poll() is not None:
                    break
                if cancelled():
                    p.kill()
                    break
                time.sleep(1.0)
        finally:
            if p.poll() is None:
                p.kill()
            log = (p.stdout.read() if p.stdout else "")[-2000:]
        if not any(r.get("done") for r in rows) and not cancelled():
            raise RuntimeError("Music Flamingo stopped before it finished: " + (log.strip().splitlines() or ["no output"])[-1])
    from .events import bus
    bus.publish("meta", meta=libmeta.all_meta())
    return out
