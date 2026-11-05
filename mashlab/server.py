"""HTTP API + live event stream for the web UI. Binds to localhost only."""
import asyncio
import itertools
import json
import os
import re
import shutil
import threading
import time
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form
from fastapi.responses import FileResponse, Response, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import automash, clipfx, draft, jobs, library, mixer, procpool, project as P, warp, warps
from .config import DATA, EXPORTS, SR, STEMS, WEB
from .events import bus
from .player import player

app = FastAPI(title="Re-Tex")
app.mount("/static", StaticFiles(directory=WEB), name="static")


# Bumped whenever the server API the UI talks to changes, so a browser can't keep running an old copy of the
# interface against newer endpoints even if no UI file's mtime changed. FEAT-004 added /api/mashup/auto,
# /api/songs/{id}/lyrics, /api/songs/{id}/backfill_beat_content, the capabilities detect() block, and
# lyrics_mode on the rate/table/generate bodies.
UI_TAG_SALT = "settings-1"


def _ui_tag():
    """Changes whenever any UI file changes OR UI_TAG_SALT bumps. The page loads its scripts from
    /ui/<tag>/..., so a browser can never keep running an old copy of the interface against a newer server."""
    import hashlib
    h = hashlib.sha1()
    h.update(UI_TAG_SALT.encode())
    for f in sorted(WEB.rglob("*")):
        if f.is_file():
            h.update(f"{f.relative_to(WEB)}:{f.stat().st_mtime_ns}".encode())
    return h.hexdigest()[:10]


UI_TAG = _ui_tag()
app.mount(f"/ui/{UI_TAG}", StaticFiles(directory=WEB), name="ui")


@app.middleware("http")
async def _revalidate_ui(request, call_next):
    """UI files: always check with the server before using a cached copy (a cheap 304 when unchanged)."""
    resp = await call_next(request)
    if request.url.path.startswith(("/static/", "/ui/")):
        resp.headers["Cache-Control"] = "no-cache"
    return resp

UPLOADS = DATA / "uploads"
UPLOADS.mkdir(exist_ok=True)

_active = {"project": None, "interactive": False}   # the project the player is playing / editing
_render_lock = threading.Lock()
_renders = {}                        # warp request -> future (queued or running) for the active project
_settle = {"timer": None}
RENDER_SETTLE = 0.5                  # s without edits before new renders start


# ------------------------------------------------------------------ basics

@app.get("/")
def index():
    html = (WEB / "index.html").read_text(encoding="utf-8").replace('"/static/', f'"/ui/{UI_TAG}/')
    return Response(html, media_type="text/html", headers={"Cache-Control": "no-store"})


@app.get("/api/events")
async def events(request: Request):
    q = bus.subscribe()

    async def stream():
        try:
            yield f"data: {json.dumps({'type': 'hello', 'transport': player.state(), 'jobs': jobs.active()})}\n\n"
            while True:
                if await request.is_disconnected():
                    break
                try:
                    msg = await asyncio.wait_for(q.get(), timeout=15)
                    yield f"data: {msg}\n\n"
                except asyncio.TimeoutError:
                    yield ": ping\n\n"
        finally:
            bus.unsubscribe(q)

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/api/jobs")
def list_jobs():
    return jobs.active()


@app.post("/api/jobs/{jid}/cancel")
def cancel_job(jid: int):
    return {"ok": jobs.cancel(jid)}


# ------------------------------------------------------------------ songs

@app.get("/api/songs")
def songs():
    return library.list_songs() + library.list_samples()      # samples: kind "sample", for the arrangement only


@app.get("/api/folders")
def folders():
    return library.get_folders()


@app.put("/api/folders")
async def put_folders(request: Request):
    return library.set_folders(await request.json())


# ------------------------------------------------------------------ library grouping (libmeta.py)

@app.get("/api/library/meta")
def library_meta():
    """Artist / album / genre / year per song (file tags, file name, guesses; the user's edits win)."""
    from . import libmeta
    from . import listen
    ok, why = listen.available()
    return {"meta": libmeta.all_meta(), "genres": libmeta.GENRES, "listen": {"ok": ok, "why": why}}


@app.put("/api/library/meta")
async def put_library_meta(request: Request):
    """Body {ids: [...], set: {artist?, album?, genre?, year?}} - "" clears the user's value (back to the tags)."""
    from . import libmeta
    b = await request.json()
    ids = [str(i) for i in (b.get("ids") or []) if (library.SONGS / str(i)).exists()]
    if not ids:
        raise HTTPException(400, "no songs")
    return {"meta": libmeta.set_user(ids, b.get("set") or {})}


@app.post("/api/library/genres/guess")
async def guess_genres(request: Request):
    """The DJ's language model suggests a genre for songs that have none."""
    from . import libmeta
    b = await request.json()
    ids = b.get("ids") or None

    def work(job):
        n = libmeta.guess_genres(ids, progress=job.sub(0, 1), cancelled=lambda: job.cancelled)
        return {"guessed": n}
    return jobs.submit("genres", "Suggesting genres", work, gpu=True).to_dict()


@app.post("/api/library/listen")
async def listen_songs(request: Request):
    """Music Flamingo listens to songs and describes them (listen.py): ids, or null = every ready song it hasn't
    heard yet; redo: true hears them again."""
    from . import libmeta, listen
    ok, why = listen.available()
    if not ok:
        raise HTTPException(400, f"Listening isn't set up: {why}")
    b = await request.json()
    ready = [s["id"] for s in library.list_songs() if s.get("status") == "ready"]
    ids = [i for i in (b.get("ids") or ready) if i in set(ready)]
    if not b.get("redo"):
        ids = [i for i in ids if not libmeta.heard(i)]
    if not ids:
        return {"queued": 0}

    def work(job):
        r = listen.describe(ids, progress=job.sub(0, 1), cancelled=lambda: job.cancelled)
        return {"heard": sum(1 for v in r.values() if "error" not in v), "failed": sum(1 for v in r.values() if "error" in v)}
    n = len(ids)
    return jobs.submit("listen", f"Listening to {n} song{'s' if n != 1 else ''} (Music Flamingo)", work, gpu=True).to_dict()


@app.get("/api/library/compat")
def library_compat():
    from . import libmeta
    return libmeta.compat_groups()


@app.post("/api/library/compat")
def compare_library():
    """Compare every song not compared yet with all the others (about 1.5 s per song), for 'works together' groups."""
    from . import libmeta

    def work(job):
        n = libmeta.compare_all(progress=job.sub(0, 1), cancelled=lambda: job.cancelled)
        return {"compared": n}
    return jobs.submit("compat", "Comparing songs", work).to_dict()


@app.get("/api/songs/{sid}")
def song(sid: str):
    info = library.read(sid)
    if not info:
        raise HTTPException(404, "no such song")
    from . import songform, userlyrics
    try:
        info["form"] = songform.form(sid, info)      # intro / verse / chorus ... for the song view (songform.py)
    except Exception:
        info["form"] = None
    segs, src = userlyrics.segments(sid)             # the words, timed (yours, else Whisper's) - the lyrics overlay
    info["lyrics"] = {"segments": segs, "source": src} if segs else None
    return info


@app.get("/api/songs/{sid}/lyrics_user")
def get_user_lyrics(sid: str):
    """Your typed lyrics for the song (userlyrics.py) - or, when there are none yet, Whisper's words to start from."""
    from . import userlyrics
    if not library.read(sid):
        raise HTTPException(404, "no such song")
    u = userlyrics.load(sid)
    if u:
        return {"text": u["text"], "yours": True, "stale": u["stale"], "lines": len(u.get("lines") or []),
                "low": sum(1 for l in u.get("lines") or [] if l.get("conf", 1) < 0.15)}
    segs, src = userlyrics.segments(sid)
    return {"text": "\n".join(s.get("text", "").strip() for s in segs if s.get("text", "").strip()), "yours": False,
            "stale": False, "lines": 0, "low": 0, "from": src}


@app.put("/api/songs/{sid}/lyrics_user")
def put_user_lyrics(sid: str, b: dict):
    """Keep your lyrics for the song and line them up with its vocal stem (a GPU job). Body: {text} (empty = forget)."""
    from . import userlyrics
    if not library.read(sid):
        raise HTTPException(404, "no such song")
    text = str(b.get("text") or "")

    def work(job):
        d = userlyrics.save(sid, text, job.update)
        return {"id": sid, "lines": len(d["lines"]) if d else 0, "cleared": d is None}

    # kind "lyrics": the page drops its copy of the song when it's done, so the overlay / form re-read the words
    return jobs.submit("lyrics", f"Your lyrics: {(library.read(sid) or {}).get('title', sid)}", work,
                       ref=sid, gpu=True).to_dict()


@app.put("/api/songs/{sid}/form")
async def put_form(sid: str, request: Request):
    """Your own song form (renamed / split / merged sections) - or {reset: true} for the detected one again."""
    from . import songform
    if not library.read(sid):
        raise HTTPException(404, "no such song")
    b = await request.json()
    try:
        return {"form": songform.set_user(sid, None if b.get("reset") else b.get("sections"))}
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.put("/api/songs/{sid}/marks")
async def put_marks(sid: str, request: Request):
    """The song's own grid: your beat marks for it (seconds + downbeat flags; null = the detector's again) - what
    projects from now on use. Saved projects that used the old grid keep it (frozen into them, see below). Marks
    only: they move nothing in the audio."""
    if not library.read(sid):
        raise HTTPException(404, "no such song")
    body = await request.json()
    m = body.get("user_marks")
    if m is not None:
        t, d = m.get("t") or [], m.get("d") or []
        s = m.get("s") or [1] * len(t)
        if len(t) != len(d) or len(s) != len(t) or any(not isinstance(x, (int, float)) for x in t):
            raise HTTPException(400, "bad marks")
        p0 = m.get("p0")                  # a downbeat a 4-bar phrase starts on (the beat grid window)
        c = m.get("c")                    # beats you placed yourself (Ctrl+drag): their bars keep their shape
        m = dict(t=[round(float(x), 5) for x in t], d=[int(x) for x in d], s=[1 if x else 0 for x in s],
                 bpb=int(m.get("bpb") or 0) or None, **({"p0": int(p0)} if isinstance(p0, (int, float)) and 0 <= p0 < len(t) else {}),
                 **({"c": [1 if x else 0 for x in c]} if isinstance(c, list) and len(c) == len(t) and any(c) else {}))
    frozen, old = P.set_song_grid(sid, m, from_project=body.get("project"))
    # the open page applies the same copy to a frozen project it has open (else its next autosave would drop it)
    return {"ok": True, "frozen": frozen, "old": old if frozen else None}


@app.delete("/api/songs/{sid}")
def delete_song(sid: str):
    library.delete(sid)
    return {"ok": True}


@app.post("/api/songs/{sid}/reseparate")
async def reseparate(sid: str, request: Request):
    b = await request.json() if (await request.body()) else {}
    if not library.read(sid):
        raise HTTPException(404)
    return library.reseparate(sid, b.get("quality", library.default_quality())).to_dict()


@app.get("/api/capabilities")
def capabilities():
    from .separate import ultra_available, sw_available
    from . import backends
    return {"ultra": ultra_available(), "sw": sw_available(), "default_quality": library.default_quality(),
            "backends": backends.detect()}


@app.post("/api/songs/{sid}/reanalyze")
def reanalyze(sid: str):
    return library.reanalyze(sid).to_dict()


@app.post("/api/songs/{sid}/lock")
async def lock_song(sid: str, request: Request):
    """🔒 Make the song's locked-to-its-marks stems (once; a background job). Projects switch per song.
    Body (optional): {"force": true} locks without the wrong-speed check (the user heard the warning and chose
    to); {"repair": true} repairs the detector's marks first (lock.repair_and_make); {"marks": {...}} locks to a
    project's own timing for the song instead of the song's grid (kept next to the song's own lock)."""
    if not library.read(sid):
        raise HTTPException(404, "no such song")
    from . import lock
    b = await request.json() if (await request.body()) else {}
    if b.get("repair") and not b.get("marks"):
        try:
            return await asyncio.to_thread(lock.repair_and_make, sid)
        except ValueError as e:
            raise HTTPException(400, str(e))
    return lock.make(sid, check=not b.get("force"), marks=b.get("marks") or None)


@app.get("/api/songs/{sid}/lock/check")
def lock_check(sid: str):
    """Before locking: stretches the lock would play at the wrong speed, and whether Repair clears them."""
    if not library.read(sid):
        raise HTTPException(404, "no such song")
    from . import lock
    return lock.check(sid)


@app.post("/api/songs/{sid}/lock/check")
async def lock_check_marks(sid: str, request: Request):
    """lock_check for a project's own timing of the song. Body: {marks}."""
    if not library.read(sid):
        raise HTTPException(404, "no such song")
    from . import lock
    b = await request.json() if (await request.body()) else {}
    return await asyncio.to_thread(lock.check, sid, b.get("marks") or None)


@app.post("/api/songs/{sid}/muscheck")
def muscheck(sid: str):
    """MuScriptor second opinion on the song's beat marks (a background job, ~2 min)."""
    if not library.read(sid):
        raise HTTPException(404, "no such song")
    return library.mus_check(sid).to_dict()


@app.post("/api/songs/{sid}/lyrics")
async def song_lyrics(sid: str, request: Request):
    """Transcribe + profile a song's vocal (ASR -> lyrics.json, concept -> concept.json). A GPU job: it owns
    the single _gpu worker through its backend and unloads after (design A / D.1a)."""
    if not library.read(sid):
        raise HTTPException(404, "no such song")

    def work(job):
        from . import lyrics, concept
        job.update(0.05, "transcribing")
        ly = lyrics.ensure(sid)
        job.update(0.70, "reading the lyrics")
        prof = concept.profile(ly)
        concept.write(sid, prof)
        job.update(1.0, "done")
        q = (ly or {}).get("quality", {}) or {}
        return {"id": sid, "lang": ly.get("lang"), "asr_conf": q.get("asr_conf"),
                "usable": q.get("usable"), "judgeable": prof.get("judgeable"),
                "mood": (prof.get("tags") or {}).get("mood")}

    return jobs.submit("lyrics", f"Lyrics: {(library.read(sid) or {}).get('title', sid)}", work,
                       ref=sid, gpu=True).to_dict()


@app.post("/api/songs/{sid}/backfill_beat_content")
async def backfill_beat_content(sid: str, request: Request):
    """Back-fill a song's 0..100 beat_content list for older analyses that predate it (the helper itself lands
    in FEAT-005; this is the lightweight job placeholder that will call it). Returns whether the field is
    already present so the UI can decide whether to offer the back-fill."""
    info = library.read(sid)
    if not info:
        raise HTTPException(404, "no such song")

    def work(job):
        job.update(0.1, "checking beat content")
        from . import analyze
        helper = getattr(analyze, "ensure_beat_content", None)
        if helper is not None:
            helper(sid, progress=job.sub(0.1, 1.0))
            return {"id": sid, "backfilled": True}
        # FEAT-005 ships the real helper; until then this is a no-op placeholder that reports current state.
        job.update(1.0, "done")
        return {"id": sid, "backfilled": False,
                "has_beat_content": (library.read(sid) or {}).get("beat_content") is not None,
                "note": "beat-content back-fill helper lands in FEAT-005"}

    return jobs.submit("backfill", f"Beat content: {info.get('title', sid)}", work, ref=sid, gpu=True).to_dict()


@app.patch("/api/songs/{sid}")
async def patch_song(sid: str, request: Request):
    body = await request.json()
    info = library.read(sid)
    if not info:
        raise HTTPException(404)
    if "title" in body:
        info["title"] = str(body["title"])[:120]
    if "shift_downbeat" in body and info.get("bars"):
        # move every bar line by k beats (fixes a wrong downbeat guess)
        k = int(body["shift_downbeat"])
        n = len(info["beats"])
        info["bars"] = [b + k for b in info["bars"] if 0 <= b + k < n]
    library.write(sid, info)
    bus.publish("song", song=library.summary(info))
    return info


@app.get("/api/songs/{sid}/peaks/{stem}")
def peaks(sid: str, stem: str):
    if stem not in STEMS + ("mix",):
        raise HTTPException(400)
    p = library.song_dir(sid) / "peaks" / f"{stem}.bin"
    if not p.exists():
        raise HTTPException(404)
    return Response(p.read_bytes(), media_type="application/octet-stream",
                    headers={"Cache-Control": "max-age=3600"})


_hi_lock = threading.Lock()


@app.get("/api/songs/{sid}/peaks_hi/{stem}")
def peaks_hi(sid: str, stem: str):
    """Close-up waveform: 16-bit min/max at config.PEAKS_HI_PER_SEC, made from the stem the first time it's asked
    for (and again whenever the stem file is newer)."""
    from .audio_io import peaks16
    from .config import PEAKS_HI_PER_SEC
    if stem not in STEMS + ("mix",):
        raise HTTPException(400)
    d = library.song_dir(sid)
    srcs = [d / "stems" / f"{s}.flac" for s in (STEMS if stem == "mix" else (stem,))]
    srcs = [f for f in srcs if f.exists()]
    if not srcs:
        raise HTTPException(404)
    out = d / "peaks" / f"{stem}.hi.bin"
    with _hi_lock:
        if not out.exists() or out.stat().st_mtime < max(f.stat().st_mtime for f in srcs):
            info = library.read(sid)
            y = sum(library.read_stem(sid, f.stem, info) for f in srcs)
            out.parent.mkdir(exist_ok=True)
            tmp = out.with_suffix(".tmp")
            peaks16(y, SR, PEAKS_HI_PER_SEC).tofile(tmp)
            tmp.replace(out)
    return Response(out.read_bytes(), media_type="application/octet-stream",
                    headers={"Cache-Control": "no-cache", "X-Peaks-Per-Sec": str(PEAKS_HI_PER_SEC)})


@app.post("/api/songs/upload")
async def upload(files: list[UploadFile] = File(...), quality: str = Form("")):
    out = []
    for f in files:
        name = re.sub(r'[<>:"/\\|?*]', "_", Path(f.filename or "song").name)
        dst = UPLOADS / f"{int(time.time() * 1000)}_{name}"
        with open(dst, "wb") as fh:
            shutil.copyfileobj(f.file, fh)
        sid, job = library.import_audio(dst, library.clean_title(name), quality or None)
        out.append(dict(id=sid, job=job.id))
    return out


@app.post("/api/stems/upload")
async def upload_stems(files: list[UploadFile] = File(...), roles: str = Form(...), title: str = Form("")):
    """Already-split stems -> ONE song, not separated again. roles: a JSON list, one per file, of drums / bass /
    other / vocals ("" = leave the file out); files on the same stem are mixed together, a stem with no file is
    silent. The files are kept together in uploads/<time>_<title>/ (the song's source)."""
    try:
        rl = json.loads(roles)
    except ValueError:
        raise HTTPException(400, "roles must be a JSON list")
    if len(rl) != len(files):
        raise HTTPException(400, "one role per file")
    title = library.clean_title(title.strip() or Path(files[0].filename or "stems").stem)
    d = UPLOADS / f"{int(time.time() * 1000)}_{re.sub(r'[^A-Za-z0-9._-]+', '_', title)[:60]}"
    d.mkdir(parents=True, exist_ok=True)
    by = {}
    for f, r in zip(files, rl):
        name = re.sub(r'[<>:"/\\|?*]', "_", Path(f.filename or "stem").name)
        dst = d / name
        with open(dst, "wb") as fh:
            shutil.copyfileobj(f.file, fh)
        if r in STEMS:
            by.setdefault(r, []).append(str(dst))
    if not by:
        raise HTTPException(400, "give at least one file a stem")
    sid, job = library.import_stems(by, title, d)
    return [dict(id=sid, job=job.id)]


@app.get("/api/stems/folder")
def stem_folder(path: str):
    """The audio files in a folder on this PC, for the Import stems dialog to assign to stems."""
    p = Path(path.strip().strip('"'))
    if not p.is_dir():
        raise HTTPException(400, f"not a folder: {p}")
    fs = sorted(f for f in p.iterdir() if f.is_file() and f.suffix.lower() in library.AUDIO_EXT)
    if not fs:
        raise HTTPException(400, "no audio files in that folder")
    return dict(folder=str(p), name=p.name, files=[dict(path=str(f), name=f.name, size=f.stat().st_size) for f in fs])


@app.post("/api/stems/import")
async def import_stem_paths(request: Request):
    """Like upload_stems, for files already on this PC (read in place). Body: {title, files: [{path, role}]}."""
    b = await request.json()
    by = {}
    for it in b.get("files") or []:
        if it.get("role") in STEMS and Path(it.get("path", "")).is_file():
            by.setdefault(it["role"], []).append(it["path"])
    if not by:
        raise HTTPException(400, "give at least one file a stem")
    first = Path(next(iter(by.values()))[0])
    title = library.clean_title(str(b.get("title") or "").strip() or first.parent.name)
    sid, job = library.import_stems(by, title, first.parent)
    return [dict(id=sid, job=job.id)]


@app.post("/api/songs/import_path")
async def import_path(request: Request):
    """Import a local file, a folder of songs, or a folder of stems (drums/bass/other/vocals)."""
    body = await request.json()
    p = Path(str(body.get("path", "")).strip().strip('"'))
    quality = body.get("quality") or None
    if not p.exists():
        raise HTTPException(400, f"not found: {p}")
    out = []
    if p.is_file():
        sid, job = library.import_audio(p, library.clean_title(p.name), quality)
        out.append(dict(id=sid, job=job.id))
    else:
        stems = library.find_stem_files(p)
        if stems:
            title = body.get("title") or (p.parent.name if p.name.lower().startswith("htdemucs") else p.name)
            sid, job = library.import_stems({k: str(v) for k, v in stems.items()}, library.clean_title(title), p)
            out.append(dict(id=sid, job=job.id))
        else:
            for f in sorted(p.iterdir()):
                if f.suffix.lower() in library.AUDIO_EXT:
                    sid, job = library.import_audio(f, library.clean_title(f.name), quality)
                    out.append(dict(id=sid, job=job.id))
    if not out:
        raise HTTPException(400, "no audio files found there")
    return out


# ------------------------------------------------------------------ audition (source audio, original tempo)

@app.post("/api/audition")
async def audition(request: Request):
    b = await request.json()
    sid = b["song"]
    info = library.read(sid)
    if not info:
        raise HTTPException(404)
    stems = [s for s in b.get("stems", STEMS) if s in STEMS]
    key = (sid, tuple(stems))
    buf = _aud_cache.get(key)
    if buf is None:
        buf = sum(library.load_stem(sid, s) for s in stems)
        _aud_cache.clear()
        _aud_cache[key] = buf
    beats = info["beats"]

    def t_of(beat):
        i = int(np.clip(np.floor(beat), 0, len(beats) - 2))
        return beats[i] + (beat - i) * (beats[i + 1] - beats[i])

    if "start_beat" in b:
        start, end = t_of(b["start_beat"]), t_of(b["end_beat"])
    else:
        start, end = b.get("start", 0.0), b.get("end", info["duration"])
    try:
        player.audition(buf, start * SR, end * SR, gain=0.9, ident=b.get("id"), loop=bool(b.get("loop")))
    except Exception as e:
        raise HTTPException(500, str(e))
    return {"ok": True, "start": start, "end": end}


_aud_cache = {}


@app.post("/api/audition/stop")
def audition_stop():
    player.stop_audition()
    return {"ok": True}


# ------------------------------------------------------------------ projects

@app.get("/api/projects")
def projects():
    return P.list_projects()


@app.get("/api/projects/{pid}")
def get_project(pid: str):
    try:
        return P.load(pid)
    except FileNotFoundError:
        raise HTTPException(404)


@app.delete("/api/projects/{pid}")
def delete_project(pid: str):
    P.delete(pid)
    return {"ok": True}


@app.post("/api/projects/new")
async def new_project(request: Request):
    b = await request.json() if (await request.body()) else {}
    return P.empty(b.get("name", "Untitled mashup"), b.get("bpm", 120.0), b.get("bpb", 4))


@app.post("/api/projects/{pid}/branch")
async def branch_project(pid: str, request: Request):
    """Copy saved project pid to a new one (the original is left as it is). Body: {name?}. -> the new project."""
    b = await request.json() if (await request.body()) else {}
    try:
        return P.branch(pid, b.get("name"))
    except FileNotFoundError:
        raise HTTPException(404, "save the project first")


@app.put("/api/projects/{pid}")
async def save_project(pid: str, request: Request):
    proj = await request.json()
    proj["id"] = pid
    P.save(proj)
    return {"ok": True}


@app.post("/api/active")
async def set_active(request: Request):
    """The UI sends the whole project on every edit; the player switches to it instantly.
    Renders the project no longer needs are dropped right away (even mid-render); new ones start once the
    edits settle, so clicking the tempo up 20 times renders only the tempo you stop on.
    Runs in a worker thread: compiling a big project must never hold up play / stop / other requests."""
    proj = await request.json()
    interactive = request.query_params.get("interactive") == "1"
    return await asyncio.to_thread(_set_active, proj, interactive, next(_active_seq))


_active_lock = threading.Lock()
_active_seq = itertools.count()              # edits apply in the order they arrived, newest wins
_applied = {"seq": -1}


def _set_active(proj, interactive, seq):
    with _active_lock:
        if seq < _applied["seq"]:                # a newer edit already went through
            return {"missing": 0, "rendering": 0, "waiting": _waiting_clips(proj),
                    "preview": [warps.parent_id(p[0]) for p in player.mixer.pending]}
        _applied["seq"] = seq
        return _apply_active(proj, interactive)


def _apply_active(proj, interactive):
    _active["project"], _active["interactive"] = proj, interactive
    missing = player.set_project(proj)
    need = [r for r in P.render_requests(proj) if not warp.is_cached(*r)]
    _drop_stale_renders(need)
    _sync_slices(start=False)
    if need:
        _preload_previews(need)
    m = player.mixer
    if need or m.pending or m.drafts or m.clip_drafts:
        _schedule_renders()
    return {"missing": len(missing), "rendering": len(need), "waiting": _waiting_clips(proj),
            "preview": [warps.parent_id(p[0]) for p in player.mixer.pending]}


def _drop_stale_renders(need):
    need = set(need)
    pool = warp._get_pool()
    with _render_lock:
        stale = [(r, f) for r, f in _renders.items() if r not in need]
        for r, _ in stale:
            del _renders[r]                 # so going back to these settings starts a fresh render
    for _, f in stale:
        pool.abort(f)


def _schedule_renders():
    with _render_lock:
        if _settle["timer"] is not None:
            _settle["timer"].cancel()
        t = threading.Timer(RENDER_SETTLE, _start_renders)
        t.daemon = True
        _settle["timer"] = t
        t.start()


def _start_renders():
    proj = _active["project"]
    if proj is None:
        return
    need = [r for r in P.render_requests(proj) if not warp.is_cached(*r)]
    pool = warp._get_pool()
    if not _active["interactive"]:          # mid-drag the varispeed preview is enough
        _sync_slices(start=True)
    _sweep_drafts()
    with _render_lock:
        new = [(r, pool.submit(warp._worker, *r, priority=2)) for r in need if r not in _renders]
        _renders.update(new)
    if new:
        _track_warps(new)


def _sync_slices(start):
    """Clip-sized renders for the active project: quick R2 drafts (so every clip plays at the right pitch
    within a second or two, clips at the playhead first), then the HQ clip stretches. Drafts outrank HQ
    clip renders, which outrank whole-stem warps (pool priorities 0 / 1 / 2)."""
    m = player.mixer
    proj = _active["project"] or {}
    T = 60.0 / float(proj.get("bpm", 120) or 120) * SR
    span = {c["id"]: (c["start"] * T, (c["start"] + c["len"]) * T) for c in proj.get("clips", [])}
    draft.queue(sorted(m.drafts, key=lambda d: _urgency(d[7], d[8])), _clip_render_done, start)
    clip_drafts = sorted(m.clip_drafts, key=lambda d: _urgency(d[8], d[9]))
    clipfx.queue(sorted(m.pending, key=lambda d: _urgency(*span.get(warps.parent_id(d[0]), (0.0, 0.0)))), clip_drafts,
                 _clip_render_done, start)


def _urgency(s0, s1):
    """Sort key for something playing at timeline samples [s0, s1): under the playhead first, then what's
    coming up next, then what's behind it."""
    pos = player.pos
    if s0 <= pos < s1:
        return 0.0
    return s0 - pos if s0 >= pos else 1e12 + (pos - s1)


def _track_warps(batch):
    """One status-bar job per batch of warps; each stem goes live as soon as it lands."""
    names = {r[0] for r, _ in batch}
    title = ", ".join((library.read(s) or {}).get("title", s)[:18] for s in names)
    n = len(batch)
    job = jobs.start("render", f"Warping {n} stem{'s' if n > 1 else ''} · {title}")
    jobs.progress(job, 0.0, f"warping {n} stems")
    tally = {"left": n, "ok": 0, "errors": []}
    lock = threading.Lock()

    def done(f, r):
        with _render_lock:
            if _renders.get(r) is f:
                del _renders[r]
        dropped = f.cancelled() or isinstance(f.exception(), procpool.Aborted)
        err = None if dropped else f.exception()
        if dropped:
            warp.discard_partial(*r)
        elif err is not None:
            print("warp failed:", r, err)
        else:
            _warp_landed(r)
        with lock:
            tally["left"] -= 1
            if err is not None:
                tally["errors"].append(str(err).splitlines()[0])
            elif not dropped:
                tally["ok"] += 1
            left, ok, errors = tally["left"], tally["ok"], list(tally["errors"])
        if left:
            jobs.progress(job, (n - left) / n, f"warped {ok}/{n} stems")
        elif errors:
            jobs.finish(job, "error", "failed", errors[0])
        elif ok == 0:
            jobs.finish(job, "cancelled", "replaced by newer settings")
        else:
            jobs.finish(job, "done", f"warped {ok} stems")
            threading.Thread(target=_prune_cache, daemon=True).start()

    for r, f in batch:
        f.add_done_callback(lambda f, r=r: done(f, r))


def _prune_cache():
    try:
        warp.prune_cache(max_gb=45)
    except Exception as e:                # housekeeping must never fail a render
        print("cache cleanup skipped:", e)


def _warp_landed(req):
    proj = _active["project"]
    if proj is not None:
        player.set_project(proj)          # clips switch from their drafts to the HQ warp
        if not _active["interactive"]:    # clips stretched/pitched on top of this stem can render now
            _sync_slices(start=True)
        _sweep_drafts()
    bus.publish("renders", ready=True)


def _sweep_drafts(tries=6):
    """Delete drafts nothing will play again; files still in use are retried a few seconds later."""
    proj = _active["project"]
    if proj is None:
        return
    left = draft.sweep([r for r in P.render_requests(proj) if not warp.is_cached(*r)])
    if left and tries > 1:
        t = threading.Timer(3.0, _sweep_drafts, kwargs={"tries": tries - 1})
        t.daemon = True
        t.start()


def _preload_previews(reqs):
    """Pull the source stems into memory so clips play a preview while their warp renders."""
    todo = [(r[0], r[1], r[6]) for r in reqs if library.peek_stem(r[0], r[1], r[6]) is None]
    if not todo:
        return

    def run():
        for sid, stem, tspec in dict.fromkeys(todo):
            try:
                library.load_stem(sid, stem, tspec)
            except Exception as e:
                print("preview load failed:", e)
        proj = _active["project"]
        if proj is not None:
            player.set_project(proj)
            bus.publish("renders", preview=True)
    threading.Thread(target=run, daemon=True).start()


def _clip_render_done():
    proj = _active["project"]
    if proj is not None:
        player.set_project(proj)
    bus.publish("renders", clips=True)


def _waiting_clips(proj):
    """Ids of clips whose warped audio isn't rendered yet."""
    out = []
    bpm = round(float(proj["bpm"]), 4)
    for c in proj["clips"]:
        r = P.request(proj, c)
        if not warp.is_cached(*r):
            out.append(c["id"])
    return out


@app.post("/api/waiting")
async def waiting(request: Request):
    proj = await request.json()
    return await asyncio.to_thread(lambda: {"waiting": _waiting_clips(proj),
                                            "preview": [warps.parent_id(p[0]) for p in player.mixer.pending] if _active["project"] is not None else []})


# ------------------------------------------------------------------ transport

@app.post("/api/transport")
async def transport(request: Request):
    b = await request.json()
    return await asyncio.to_thread(_transport, b)


def _transport(b):
    act = b.get("action")
    try:
        if act == "play":
            player.play(b["pos"] * SR if "pos" in b else None)
        elif act == "stop":
            player.stop()
        elif act == "seek":
            player.seek(b["pos"] * SR)
        elif act == "volume":
            player.volume = float(np.clip(b.get("value", 1.0), 0, 2))
        elif act == "metronome":                  # clicks on the project's grid: {on?, gain?}
            player.set_click(b.get("on"), b.get("gain"))
    except Exception as e:
        raise HTTPException(500, str(e))
    return player.state()


@app.get("/api/transport")
def transport_state():
    return player.state()


# ------------------------------------------------------------------ auto mashup

@app.get("/api/match/{sid}")
def match(sid: str):
    try:
        return automash.match_list(sid)
    except ValueError as e:
        raise HTTPException(400, str(e))


_variants = {}


@app.get("/api/mashup/variant/{jid}/{idx}")
def mashup_variant(jid: int, idx: int):
    try:
        return _variants[jid][idx]
    except (KeyError, IndexError):
        raise HTTPException(404, "that result list is gone - run the search again")


@app.post("/api/mashup/rate")
async def mashup_rate(request: Request):
    """Rate every vocal-over-music pairing among the songs (slow the first time; results are kept)."""
    b = await request.json()
    ids = b.get("songs") or [s["id"] for s in library.list_songs() if s.get("status") == "ready"]
    density = float(b.get("density", 0.5))

    lyrics_mode = b.get("lyrics_mode", "off")

    def work(job):
        from . import pairs
        stats = pairs.rate_all(ids, density, b.get("vocals"), b.get("beds"), progress=job.sub(0, 1),
                               involving=b.get("involving"), lyrics_mode=lyrics_mode)
        return dict(stats, songs=len(ids))

    job = jobs.submit("rate", "Rating mashups", work)
    return job.to_dict()


@app.post("/api/mashup/table")
async def mashup_table(request: Request):
    b = await request.json()
    from . import pairs
    ids = b.get("songs") or [s["id"] for s in library.list_songs() if s.get("status") == "ready"]
    return JSONResponse(await asyncio.to_thread(pairs.table, ids, float(b.get("density", 0.5)),
                                                b.get("vocals"), b.get("beds"), b.get("lyrics_mode", "off")))


@app.post("/api/mashup/build")
async def mashup_build(request: Request):
    b = await request.json()
    try:
        proj, desc = await asyncio.to_thread(automash.build_pair, b["vocal"], b["bed"], b.get("shift_rel"),
                                             float(b.get("density", 0.5)))
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"project": proj, "desc": desc}


@app.post("/api/mashup/auto")
async def mashup_auto(request: Request):
    """Autonomous "2+ songs -> finished editable project" (design D). Body: {paths:[audio file paths] OR
    songs:[already-imported song ids], prompt:"<style>", density?}. Runs the orchestrator on the single-worker
    _gpu slot (gpu=True) so the D.1a VRAM/unload discipline holds; returns a job whose result is a normal,
    fully-editable project JSON the timeline opens."""
    b = await request.json()
    paths = [str(p).strip().strip('"') for p in (b.get("paths") or []) if str(p).strip()]
    song_ids = [s for s in (b.get("songs") or []) if s]
    prompt = b.get("prompt")
    density = b.get("density")
    if not paths and not song_ids:
        raise HTTPException(400, "give at least two input songs (paths or already-imported song ids)")

    # The Auto-Masher UI only ever supplies already-imported library songs, so validate them up front and
    # hand their ids straight to the orchestrator, which REUSES their existing stems/analysis (no re-ingest,
    # no re-separation of the library). `paths` stays as a defensive fallback for a raw file.
    if song_ids:
        for sid in song_ids:
            info = library.read(sid)
            if not info:
                raise HTTPException(404, f"no such song {sid}")
            if info.get("status") != "ready":
                raise HTTPException(409, f"song {sid} isn't ready yet (status {info.get('status')!r}) - "
                                         "wait for its import to finish before an auto-mashup")

    def work(job):
        from . import orchestrate
        if song_ids:
            proj = orchestrate.run([], prompt=prompt,
                                   density=None if density is None else float(density),
                                   job=job, progress=job.sub(0, 1), song_ids=song_ids)
        else:
            if len(paths) < 2:
                raise ValueError("auto-mashup needs at least two input songs")
            proj = orchestrate.run(paths, prompt=prompt,
                                   density=None if density is None else float(density),
                                   job=job, progress=job.sub(0, 1))
        return {"project": proj, "desc": proj.get("mashup")}

    return jobs.submit("auto", "Auto-mashup", work, gpu=True).to_dict()


@app.post("/api/dj/mix")
async def dj_mix(request: Request):
    """DJ Mix: two library songs + what you want -> a language model plans the mix, the arrangement engine plays it
    in time. style "remix" (djremix.run, the default): one song is the bed as recorded, the other's vocal on top line
    by line, optionally a third song's drums; "blend" (djbrain.run): stems swapped between the songs part by part.
    Body: {songs: [id, id], style?, drums?, prompt?, minutes?, seed?, planner?}. Runs on the GPU slot (the model, and
    Whisper for songs without words yet, load there); the result is a normal editable project plus the plan."""
    b = await request.json()
    songs = [s for s in (b.get("songs") or []) if s]
    if len(songs) != 2 or songs[0] == songs[1]:
        raise HTTPException(400, "pick two different songs")
    style = "blend" if b.get("style") == "blend" else "remix"
    drums = (b.get("drums") or None) if style == "remix" else None
    if drums in songs:
        drums = None
    for sid in songs + ([drums] if drums else []):
        info = library.read(sid)
        if not info:
            raise HTTPException(404, f"no such song {sid}")
        if info.get("status") != "ready":
            raise HTTPException(409, f"{info.get('title', sid)} isn't ready yet - wait for its import to finish")
    prompt = str(b.get("prompt") or "")[:400]
    minutes = float(np.clip(float(b.get("minutes") or 3.0), 1.0, 8.0))
    seed = None if b.get("seed") is None else int(b["seed"])
    planner = "claude" if b.get("planner") == "claude" else None

    def work(job):
        from . import djbrain, djremix, llm
        model = llm.client(planner) if planner else None
        if style == "remix":
            proj, desc = djremix.run(songs, prompt=prompt, minutes=minutes, seed=seed, progress=job.sub(0, 1),
                                     model=model, drums=drums)
        else:
            proj, desc = djbrain.run(songs, prompt=prompt, minutes=minutes, seed=seed, progress=job.sub(0, 1),
                                     model=model)
        return {"project": proj, "desc": desc}

    return jobs.submit("dj", "DJ mix", work, gpu=True).to_dict()


@app.get("/api/dj/planners")
def dj_planners():
    """Which models can plan a DJ mix: Claude (via Claude Code, the user's own login) and the local one."""
    from . import llm
    ok, why = llm.claude_status()
    return {"claude": {"ok": ok, "why": why}}


@app.post("/api/yue2/prepare")
async def yue2_prepare(request: Request):
    """Reimagine, stage 1 (reimagine.prepare): render the open mix, hear its lyrics, have YuE2 Studio read its
    melody and chords, write the style - for the user to review and edit. Body: {project, prompt?}."""
    b = await request.json()
    proj = b.get("project")
    if not proj or not proj.get("clips"):
        raise HTTPException(400, "open a mix first")
    prompt = str(b.get("prompt") or "")[:300]

    def work(job):
        from . import reimagine
        return reimagine.prepare(proj, prompt, progress=job.sub(0, 1), cancelled=lambda: job.cancelled)

    return jobs.submit("reimagine-prep", "YuE2 reimagining: preparing", work, gpu=True).to_dict()


@app.post("/api/yue2/fit")
async def yue2_fit(request: Request):
    """The Studio's advisory lyric fit for edited lyrics. Body: {prep_id, lyrics}."""
    b = await request.json()
    from . import reimagine, yue2link
    try:
        return reimagine.fit(str(b.get("prep_id") or ""), str(b.get("lyrics") or ""))
    except (OSError, ValueError, yue2link.StudioError) as e:
        raise HTTPException(400, str(e))


@app.post("/api/yue2/place")
def yue2_place(b: dict):
    """Your lyrics placed where they're sung in the mix (reimagine.place): aligned to its vocals, sections re-cut at
    the bar lines where the singing starts / stops. Body: {prep_id, lyrics}."""
    from . import reimagine, yue2link
    try:
        return reimagine.place(str(b.get("prep_id") or ""), str(b.get("lyrics") or ""))
    except (OSError, ValueError, RuntimeError, yue2link.StudioError) as e:
        raise HTTPException(400, str(e))


@app.post("/api/yue2/perform")
async def yue2_perform(request: Request):
    """Reimagine, stage 2 (reimagine.perform): YuE2 re-performs the mix with the (edited) style and lyrics.
    Body: {prep_id, style, lyrics, takes?}. Takes minutes; holds the GPU slot while YuE2 Studio works."""
    b = await request.json()
    prep_id = str(b.get("prep_id") or "")
    if not (DATA / "yue2" / Path(prep_id).name / "prep.json").exists():
        raise HTTPException(400, "prepare the reimagining first")
    style = str(b.get("style") or "").strip()
    if not style:
        raise HTTPException(400, "the style prompt is empty")
    lyrics = str(b.get("lyrics") or "")
    takes = int(np.clip(int(b.get("takes") or 2), 1, 4))
    from . import reimagine
    mode = str(b.get("mode") or "cover_full")
    if mode not in reimagine.MODES:
        raise HTTPException(400, f"mode must be one of {', '.join(reimagine.MODES)}")
    cfg = b.get("cfg_scale")
    if cfg is not None:
        try:
            cfg = float(cfg)
        except (TypeError, ValueError):
            raise HTTPException(400, "style strength must be a number")
        if not 1.0 <= cfg <= 1.5:
            raise HTTPException(400, "style strength (CFG) must be between 1.0 and 1.5")
    flow = int(b.get("flow_steps") or 32)
    if flow not in (32, 48, 64):
        raise HTTPException(400, "flow steps must be 32, 48 or 64")
    voices = str(b.get("voices") or "both")
    seed = None if b.get("seed") in (None, "", -1) else int(b["seed"])
    cont = int(np.clip(int(b.get("continuation_seconds") or 30), 10, 120))

    def work(job):
        return reimagine.perform(prep_id, style, lyrics, takes, mode=mode, cfg_scale=cfg, flow_steps=flow, voices=voices,
                                 seed=seed, continuation_seconds=cont, progress=job.sub(0, 1),
                                 cancelled=lambda: job.cancelled)

    return jobs.submit("reimagine", f"YuE2 reimagining: {reimagine.MODES[mode]}", work, gpu=True).to_dict()


@app.get("/api/yue2/prep/{prep_id}")
def yue2_prep(prep_id: str):
    """A prepared reimagining (lyrics, style, sections, fit) - to go back to editing a reopened run."""
    from . import reimagine
    try:
        return reimagine._prep(prep_id)
    except (OSError, ValueError):
        raise HTTPException(404, "that preparation is gone")


@app.get("/api/yue2/renders")
def yue2_renders():
    """Recent reimaginings with their takes, newest first - reopen and save them any time."""
    from . import reimagine
    return reimagine.renders()


@app.get("/api/yue2/audio/{rid}/{name}")
def yue2_audio(rid: str, name: str):
    p = DATA / "yue2" / Path(rid).name / Path(name).name
    if p.suffix != ".flac" or not p.exists():
        raise HTTPException(404)
    return FileResponse(p, media_type="audio/flac")


# ---------------------------------------------------------------- beat grid window (web/grid.html)
# The song as recorded with a metronome on its marks, both through the page's own audio - so monitoring latency
# can't pull them apart - for lining the marks up by ear. Saves through PUT /api/songs/{sid}/marks.

@app.get("/grid")
def grid_page():
    html = (WEB / "grid.html").read_text(encoding="utf-8").replace('"/static/', f'"/ui/{UI_TAG}/')
    return Response(html, media_type="text/html", headers={"Cache-Control": "no-store"})


@app.post("/api/songs/{sid}/grid/repair")
def grid_repair(sid: str):
    """The detector's beats re-counted on one pulse (beatfix.py) - a proposal for the beat grid window to play and
    save; nothing is written here. -> {marks: {t, d, s, bpb}, report}"""
    from . import beatfix
    info = library.read(sid)
    if not info or not info.get("beats"):
        raise HTTPException(404, "no such song")
    drums = library.read_stem(sid, "drums").mean(0)
    marks, rep = beatfix.as_marks(info, drums)
    return {"marks": marks, "report": rep}


@app.get("/api/songs/{sid}/source")
def song_source(sid: str):
    """The song's original audio file (as imported), for the page to play."""
    from .config import resolve_source
    info = library.read(sid)
    if not info:
        raise HTTPException(404, "no such song")
    p = resolve_source(info.get("source"))
    if not p or not Path(p).is_file():
        raise HTTPException(404, "the original file isn't there any more")
    return FileResponse(p)


# ---------------------------------------------------------------- sample editor (sampler.py, web/sample.html)
# A pop-out window that edits one clip's audio on a grid and bakes it into a sample replacing the clip in its lane.

@app.get("/sample")
def sample_page():
    html = (WEB / "sample.html").read_text(encoding="utf-8").replace('"/static/', f'"/ui/{UI_TAG}/')
    return Response(html, media_type="text/html", headers={"Cache-Control": "no-store"})


@app.post("/api/sample/open")
def sample_open(b: dict):
    """Body: {project (the arrangement as it is now), clip_id} -> the editing session's meta."""
    from . import sampler
    try:
        return sampler.open_clip(b.get("project") or {}, str(b.get("clip_id") or ""))
    except KeyError as e:
        raise HTTPException(404, str(e))
    except (RuntimeError, ValueError) as e:
        raise HTTPException(400, str(e))


@app.get("/api/sample/{eid}")
def sample_meta(eid: str):
    from . import sampler
    try:
        return sampler.session(eid)[0]
    except KeyError:
        raise HTTPException(404, "no such editing session")


@app.post("/api/sample/{eid}/render")
def sample_render(eid: str, b: dict):
    """Body: {ops} -> the edited audio as preview.wav (GET it) + {seconds, beats, version}."""
    from . import sampler
    try:
        return sampler.preview(eid, b.get("ops") or [])
    except KeyError:
        raise HTTPException(404, "no such editing session")
    except (ValueError, TypeError) as e:
        raise HTTPException(400, str(e))


@app.get("/api/sample/{eid}/preview.wav")
def sample_preview(eid: str):
    from . import sampler
    p = sampler.EDITS / Path(eid).name / "preview.wav"
    if not p.exists():
        raise HTTPException(404)
    return FileResponse(p, media_type="audio/wav", headers={"Cache-Control": "no-store"})


@app.post("/api/sample/{eid}/apply")
def sample_apply(eid: str, b: dict):
    """Body: {ops} -> bake into a sample; the arrangement swaps the clip for it ("sample_applied")."""
    from . import sampler
    try:
        return sampler.apply(eid, b.get("ops") or [])
    except KeyError:
        raise HTTPException(404, "no such editing session")
    except (ValueError, TypeError) as e:
        raise HTTPException(400, str(e))


# ---------------------------------------------------------------- listening sessions (listening.py)
# Test results to hear and weigh in on: what was tried, what was measured and decided, and your verdict + notes.

@app.get("/api/listening")
def listening_list():
    from . import listening
    return {"enabled": listening.enabled(), "sessions": listening.sessions(), "verdicts": listening.VERDICTS}


@app.get("/api/listening/{sid}/audio/{name}")
def listening_audio(sid: str, name: str):
    from . import listening
    try:
        p = listening.audio_path(sid, name)
    except KeyError:
        raise HTTPException(404)
    return FileResponse(p, media_type="audio/flac")


@app.put("/api/listening/{sid}/feedback")
def listening_feedback(sid: str, b: dict):
    """Body: {item: item id | "overall", verdict?: right | close | wrong | "", notes?}."""
    from . import listening
    try:
        return listening.set_feedback(sid, str(b.get("item") or ""), b.get("verdict"), b.get("notes"))
    except KeyError:
        raise HTTPException(404, "no such session or item")
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.delete("/api/listening/{sid}")
def listening_remove(sid: str):
    """Done with a session: the audio is deleted, its notes kept in listening/_archive/."""
    from . import listening
    try:
        listening.remove(sid)
    except KeyError:
        raise HTTPException(404)
    return {"ok": True}


@app.post("/api/listening/{sid}/reveal")
def listening_reveal(sid: str):
    from . import listening
    try:
        listening.reveal(sid)
    except KeyError:
        raise HTTPException(404)
    return {"ok": True}


@app.post("/api/yue2/save")
async def yue2_save(request: Request):
    """Import chosen takes as library songs (stems separated) next to the originals, each then lined up with the
    mix bar for bar. Body: {render_id, takes: [n], add_to?: project id to drop the aligned stems into,
    vocals_only?: drop only the new vocals in, over the mix's own music, retime?: its vocals moved line by line to
    where the mix sings them}."""
    b = await request.json()
    from . import reimagine
    try:
        out = reimagine.save(str(b.get("render_id") or ""), [int(n) for n in (b.get("takes") or [])],
                             add_to=b.get("add_to") or None, vocals_only=bool(b.get("vocals_only")),
                             retime=bool(b.get("retime")))
    except (OSError, StopIteration, ValueError) as e:
        raise HTTPException(400, f"can't save those takes ({e})")
    return [dict(id=sid, job=job.id, align_job=aj.id) for sid, job, aj in out]


@app.post("/api/mashup/generate")
async def generate(request: Request):
    b = await request.json()
    if not b.get("roles"):
        # only an old copy of the page still asks for the retired 8-result quick search
        raise HTTPException(409, "This page is an old version of Re-Tex - press Ctrl+F5 to load the new one "
                                 "(Every pairing: all mashups rated, no 8-result limit).")

    lyrics_mode = b.get("lyrics_mode", "off")

    def work(job):
        if b.get("roles"):
            vs = automash.generate_roles(b["roles"], max_variants=int(b.get("max", 500)),
                                         density=float(b.get("density", 0.5)), progress=job.sub(0, 1),
                                         lyrics_mode=lyrics_mode)
            # keep the (large) projects here; the UI fetches one when it's opened
            _variants.clear()
            _variants[job.id] = [v["project"] for v in vs]
            return {"variants": [dict(desc=v["desc"], ref=[job.id, i]) for i, v in enumerate(vs)]}
        raise RuntimeError("stale page")

    job = jobs.submit("mashup", "Finding mashups", work)
    return job.to_dict()


# ------------------------------------------------------------------ export

@app.post("/api/export")
async def export(request: Request):
    b = await request.json()
    proj = b["project"]
    fmt = b.get("format", "wav")
    normalize = b.get("normalize", True)
    name = re.sub(r"[^A-Za-z0-9 _\-×x]+", "", proj.get("name", "mashup")).strip() or "mashup"
    name = name.replace("×", "x")
    stamp = time.strftime("%Y%m%d-%H%M%S")
    wav = EXPORTS / f"{name} {stamp}.wav"

    def work(job):
        reqs = [r for r in P.render_requests(proj) if not warp.is_cached(*r)]
        if reqs:
            warp.render_many(reqs, job.sub(0, 0.5, "warping: "))
        target = proj.get("master", {}).get("target_lufs", -12.0) if normalize else None
        res = mixer.bounce(proj, wav, job.sub(0.5 if reqs else 0, 0.97), target_lufs=target)
        out = wav
        if fmt == "mp3":
            from .audio_io import save_mp3
            out = wav.with_suffix(".mp3")
            save_mp3(wav, out)
            wav.unlink(missing_ok=True)
        return dict(res, file=out.name, path=str(out))

    return jobs.submit("export", f"Exporting {name}", work).to_dict()


@app.get("/api/exports/{name}")
def download(name: str):
    p = EXPORTS / Path(name).name
    if not p.exists():
        raise HTTPException(404)
    return FileResponse(p, filename=p.name)


@app.post("/api/reveal")
async def reveal(request: Request):
    b = await request.json()
    p = Path(b.get("path") or EXPORTS)
    target = p if p.is_dir() else p.parent
    if not any(str(target.resolve()).startswith(str(root.resolve())) for root in (DATA, EXPORTS)):
        raise HTTPException(400, "only folders inside the library or the exports folder")
    os.startfile(str(target))
    return {"ok": True}


# ------------------------------------------------------------------ settings (where the library and exports live)

def _clean_path(raw):
    """A pasted folder path: quotes ("Copy as path") and spaces trimmed, ~ and %VARS% expanded."""
    t = str(raw or "").strip().strip('"').strip("'").strip()
    return Path(os.path.expandvars(os.path.expanduser(t))) if t else None


def _folder_info(p):
    """What's in a folder the user is pointing the library at."""
    if p is None:
        return {"ok": False, "error": "empty"}
    if not p.is_absolute():
        return {"ok": False, "error": "use a full path, like D:\\Music\\My Library"}
    if p.exists() and not p.is_dir():
        return {"ok": False, "error": "that's a file, not a folder"}
    out = {"ok": True, "path": str(p), "exists": p.exists(), "songs": 0, "projects": 0}
    if p.exists():
        try:
            if (p / "songs").is_dir():
                out["songs"] = sum(1 for d in (p / "songs").iterdir() if (d / "song.json").exists())
            if (p / "projects").is_dir():
                out["projects"] = sum(1 for _ in (p / "projects").glob("*.json"))
        except OSError as e:
            return {"ok": False, "error": f"can't read it ({e.strerror or e})"}
    return out


def _writable(p):
    p.mkdir(parents=True, exist_ok=True)
    probe = p / f".mashlab-write-test-{os.getpid()}"
    probe.write_text("ok", encoding="utf-8")
    probe.unlink()


@app.get("/api/settings")
def get_settings():
    from . import config as C
    saved = C.read_settings()
    return {
        "data": str(DATA), "exports": str(EXPORTS),
        "saved": {"data": saved.get("data") or "", "exports": saved.get("exports") or ""},
        "pinned": {"data": bool(os.environ.get("MASHLAB_DATA")), "exports": bool(os.environ.get("MASHLAB_EXPORTS"))},
        "default_data": str(C.ROOT / "data"), "settings_file": str(C.SETTINGS_JSON),
        "release": os.environ.get("MASHLAB_RELEASE") or "", "code": str(C.ROOT), "models": str(C.MODELS),
        "library": _folder_info(DATA),
    }


@app.get("/api/audio/outputs")
def audio_outputs(refresh: int = 0):
    """Output devices + the one in use. refresh=1 re-scans (a headset connected after start)."""
    return player.refresh_outputs() if refresh else player.output_state()


@app.post("/api/audio/output")
async def set_audio_output(request: Request):
    """Body {name: device name | null (system default)} - switches now, playback carries on."""
    b = await request.json()
    try:
        return player.set_output(b.get("name") or None)
    except Exception as e:
        raise HTTPException(500, f"couldn't switch the output: {e}")


@app.post("/api/audio/test")
def audio_test():
    """A short, quiet two-note chime through the current output."""
    t = np.arange(int(0.22 * SR)) / SR
    env = np.minimum(1, t / 0.005) * np.exp(-t * 9)
    note = lambda f: (np.sin(2 * np.pi * f * t) + 0.3 * np.sin(4 * np.pi * f * t)) * env
    y = np.concatenate([note(659.25), note(987.77)]).astype(np.float32) * 0.22
    player.audition(np.stack([y, y]), 0, len(y), ident="test-tone")
    return player.output_state()


@app.post("/api/settings/check")
async def check_settings(request: Request):
    b = await request.json()
    return _folder_info(_clean_path(b.get("path")))


@app.post("/api/settings")
async def put_settings(request: Request):
    """Save where the library and exports go. Takes effect when Mashup Lab restarts (POST /api/restart)."""
    from . import config as C
    b = await request.json()
    data = _clean_path(b.get("data"))
    info = _folder_info(data)
    if not info["ok"]:
        raise HTTPException(400, f"Library folder: {info['error']}")
    exports = _clean_path(b.get("exports"))
    if exports is not None:
        ei = _folder_info(exports)
        if not ei["ok"]:
            raise HTTPException(400, f"Exports folder: {ei['error']}")
    try:
        _writable(data)
        if exports is not None:
            _writable(exports)
    except OSError as e:
        raise HTTPException(400, f"can't write there ({e.strerror or e})")
    C.write_settings({"data": str(data), "exports": str(exports) if exports is not None else None})
    same = data.resolve() == DATA.resolve() and (exports or data / "exports").resolve() == EXPORTS.resolve()
    return {"ok": True, "restart": not same, "library": info}


@app.post("/api/settings/browse")
async def browse_folder(request: Request):
    """A Windows folder picker (the server runs on this PC, so it can show one)."""
    b = await request.json()
    start = _clean_path(b.get("path"))

    def pick():
        import tkinter
        from tkinter import filedialog
        root = tkinter.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        try:
            return filedialog.askdirectory(parent=root, title=b.get("title") or "Choose a folder",
                                           initialdir=str(start) if start and start.is_dir() else None,
                                           mustexist=False)
        finally:
            root.destroy()
    try:
        got = await asyncio.to_thread(pick)
    except Exception as e:
        raise HTTPException(500, f"no folder picker here ({e}) - paste the path instead")
    return {"path": str(Path(got)) if got else ""}


@app.post("/api/restart")
def restart(request: Request):
    """Start a fresh copy of Mashup Lab on this port; it closes this one (see __main__._replace_stale)."""
    import subprocess
    import sys
    args = list(sys.orig_argv[1:])
    port = str(request.url.port or 8765)
    if "--port" in args:
        args[args.index("--port") + 1] = port
    else:
        args += ["--port", port]
    if "--no-browser" not in args:
        args.append("--no-browser")
    subprocess.Popen([sys.executable] + args, cwd=os.getcwd())
    return {"ok": True}


@app.get("/api/whoami")
def whoami():
    return {"app": "mashup_lab", "pid": os.getpid()}


# ReadHead: automatic beat-aligned chops / vocal arranging, returned as clips for the UI to apply (mashlab/readhead.py)
from . import readhead as _readhead  # noqa: E402
_readhead.register(app)
# Perform mode: chops / samples played live from the keyboard, then written as clips (mashlab/perform_live.py)
from . import perform_live as _perform_live  # noqa: E402
_perform_live.register(app, player)


@app.post("/api/shutdown")
def shutdown():
    """Quit this copy (a newly launched Mashup Lab calls this to replace a stale one)."""
    threading.Thread(target=_die, daemon=True).start()
    return {"ok": True}


def _die():
    time.sleep(0.3)                      # let the HTTP response go out first
    try:
        player.stop()
        if player.stream is not None:
            player.stream.close()
    except Exception:
        pass
    warp.kill_pool()
    os._exit(0)


@app.on_event("startup")
def _startup():
    library.resume_interrupted()

    def purge():
        warp._get_pool()                  # start the render workers now, so the first render doesn't wait
        draft.purge_old()
        freed = warp.purge_old_versions()
        if freed > 1e8:
            print(f"  (cleared {freed / 1e9:.1f} GB of outdated warp renders - they re-render on demand)", flush=True)
    threading.Thread(target=purge, daemon=True).start()
