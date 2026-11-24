"""HTTP endpoints for MIDI songs (midi.py) and their instruments (instruments.py). Registered from server.py."""
import asyncio
import re
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np

from . import instruments, midi
from .config import SR, UPLOADS


def register(app, player):
    from fastapi import File, HTTPException, Request, UploadFile
    from fastapi.responses import FileResponse
    from . import jobs, library

    # MIDI songs left half-rendered by a restart: play them in again (quick, and nothing else would)
    for s in midi.list_songs():
        if s.get("status") not in ("ready", "error"):
            sid = s["id"]
            jobs.submit("midi", f"MIDI: {s.get('title') or sid}", lambda job, sid=sid: midi._render_job(sid, None, job), ref=sid)

    def need(sid):
        if not midi.is_midi(sid) or midi.read(sid) is None:
            raise HTTPException(404, "no such MIDI song")

    @app.post("/api/midi/upload")
    async def midi_upload(files: list[UploadFile] = File(...)):
        out = []
        for f in files:
            name = re.sub(r'[<>:"/\\|?*]', "_", Path(f.filename or "song.mid").name)
            dst = UPLOADS / f"{int(time.time() * 1000)}_{name}"
            UPLOADS.mkdir(parents=True, exist_ok=True)
            with open(dst, "wb") as fh:
                shutil.copyfileobj(f.file, fh)
            try:
                sid, job = await asyncio.to_thread(midi.import_file, dst, library.clean_title(Path(name).stem))
            except Exception as e:
                raise HTTPException(400, f"{name}: {e}")
            out.append(dict(id=sid, job=job.id))
        return out

    @app.post("/api/midi/new")
    async def midi_new(request: Request):
        """Body: {title, bpm, bpb, bars, program, drum} -> {id}. An empty part to draw in."""
        b = await request.json()
        sid = await asyncio.to_thread(midi.blank, b.get("title") or "New MIDI", float(b.get("bpm") or 120),
                                      int(b.get("bpb") or 4), int(b.get("bars") or 8), int(b.get("program") or 0),
                                      bool(b.get("drum")))
        return {"id": sid}

    @app.get("/api/midi/{sid}")
    def midi_get(sid: str):
        need(sid)
        mi, info = midi.read(sid), library.read(sid)
        for p in mi["parts"]:
            p["instrument_label"] = instruments.describe(p.get("instrument"))
        return {"id": sid, "title": info["title"], "bpm": info["bpm"], "bpb": info["bpb"], "beats": info["beats"],
                "first_beat": info.get("first_beat", 0), "parts": mi["parts"], "status": info.get("status"),
                "part_errors": info.get("part_errors"), "timing_from": info.get("timing_from"), "parent": info.get("parent")}

    @app.put("/api/midi/{sid}/parts/{pid}")
    async def midi_save_part(sid: str, pid: str, request: Request):
        """Body: any of {notes, cc, name, instrument, drum}. Saves, then re-renders the part. -> {job}"""
        need(sid)
        b = await request.json()
        try:
            job = await asyncio.to_thread(midi.save_part, sid, pid, b.get("notes"), b.get("cc"), b.get("name"),
                                          b.get("instrument"), b.get("drum"))
        except KeyError as e:
            raise HTTPException(404, str(e))
        except ValueError as e:
            raise HTTPException(400, str(e))
        return {"job": job.id}

    @app.post("/api/midi/{sid}/parts")
    async def midi_add_part(sid: str, request: Request):
        """Body: {name, program, drum, copy_of} -> {pid, job}"""
        need(sid)
        b = await request.json()
        pid = await asyncio.to_thread(midi.add_part, sid, b.get("name") or "New part", int(b.get("program") or 0),
                                      bool(b.get("drum")), b.get("copy_of"))
        job = jobs.submit("midi", "MIDI: new part", lambda job: midi._render_job(sid, [pid], job), ref=sid)
        return {"pid": pid, "job": job.id}

    @app.delete("/api/midi/{sid}/parts/{pid}")
    async def midi_delete_part(sid: str, pid: str):
        need(sid)
        try:
            await asyncio.to_thread(midi.delete_part, sid, pid)
        except ValueError as e:
            raise HTTPException(400, str(e))
        job = jobs.submit("midi", "MIDI: part removed", lambda job: midi._render_job(sid, [], job), ref=sid)
        return {"job": job.id}

    @app.post("/api/midi/{sid}/timing")
    async def midi_timing(sid: str, request: Request):
        """Body: {song} -> play this MIDI song on that song's beat grid. -> {job}"""
        need(sid)
        b = await request.json()
        try:
            job = await asyncio.to_thread(midi.take_timing, sid, b["song"])
        except ValueError as e:
            raise HTTPException(400, str(e))
        return {"job": job.id}

    @app.get("/api/midi/{sid}/export.mid")
    def midi_export(sid: str, parts: str = ""):
        need(sid)
        info = library.read(sid)
        out = Path(tempfile.gettempdir()) / f"retex_{sid}.mid"
        midi.to_midi_file(sid, out, [p for p in parts.split(",") if p] or None)
        safe = re.sub(r'[<>:"/\\|?*]', "_", info["title"])
        return FileResponse(out, media_type="audio/midi", filename=f"{safe}.mid")

    @app.post("/api/midi/transcribe")
    async def midi_transcribe(request: Request):
        """Body: {song, stem} -> {job}; the job's result is {sid} of the new MIDI song."""
        b = await request.json()
        if not library.read(b.get("song", "")):
            raise HTTPException(404, "no such song")
        return {"job": midi.transcribe(b["song"], b.get("stem") or "other").id}

    @app.post("/api/midi/audition")
    async def midi_audition(request: Request):
        """Body: {sid, pid, notes?, instrument?, from, to, bpm} -> plays that stretch of the part (at the project's
        tempo) through the audition voice. notes / instrument: unsaved edits from the piano roll."""
        b = await request.json()
        need(b["sid"])
        mi = midi.read(b["sid"])
        part = dict(midi.part_of(mi, b["pid"]))
        if b.get("notes") is not None:
            part["notes"] = b["notes"]
        if b.get("cc") is not None:
            part["cc"] = b["cc"]
        if b.get("instrument"):
            part["instrument"] = b["instrument"]
        a, z = float(b.get("from", 0)), float(b.get("to", 16))
        spb = 60.0 / float(b.get("bpm") or library.read(b["sid"])["bpm"])
        part["notes"] = [[s - a, ln, p, v] for s, ln, p, v in part["notes"] if a <= s < z]
        part["cc"] = {k: [[x - a, v] for x, v in pts if a <= x < z] for k, pts in (part.get("cc") or {}).items()}
        n = int(((z - a) * spb + midi.TAIL) * SR)

        others = []
        if b.get("all"):              # the piano roll's "with the other parts": their saved notes and instruments
            for q in mi["parts"]:
                if q["id"] != b["pid"]:
                    q = dict(q)
                    q["notes"] = [[s - a, ln, p, v] for s, ln, p, v in q["notes"] if a <= s < z]
                    q["cc"] = {k: [[x - a, v] for x, v in pts if a <= x < z] for k, pts in (q.get("cc") or {}).items()}
                    if q["notes"]:
                        others.append(q)

        def go():
            y = None
            for q in [part] + others:
                ev, cc = midi._events_at(q, lambda bb: np.asarray(bb, float) * spb)
                try:
                    r = instruments.render(q.get("instrument"), ev, cc, n)
                except Exception:
                    if q is part:
                        raise
                    continue
                y = r if y is None else y + r
            pk = float(np.max(np.abs(y))) if y is not None and y.size else 0.0
            return y / pk * 0.98 if pk > 0.98 else y
        try:
            y = await asyncio.to_thread(go)
        except Exception as e:
            raise HTTPException(400, f"couldn't play it: {e}")
        player.audition(y, 0, y.shape[1], gain=0.9, ident=f"midi:{b['sid']}:{b['pid']}")
        return {"ok": True, "seconds": round(y.shape[1] / SR, 2)}

    @app.post("/api/midi/sample_from_highlight")
    async def midi_sample_from_highlight(request: Request):
        """Body: {project, clip_id, t0, t1} (timeline beats) -> {path, seconds}: that bit of the clip exactly as it
        plays, saved as a WAV under data/midi/_samples for the sampler instrument."""
        b = await request.json()
        proj, cid = b.get("project") or {}, str(b.get("clip_id") or "")
        c = next((x for x in proj.get("clips", []) if x.get("id") == cid), None)
        if c is None:
            raise HTTPException(404, "no such clip")

        def go():
            import copy
            import hashlib
            import json as _json
            from . import audio_io, sampler
            from .config import MIDI_DIR
            bpm = float(proj["bpm"])
            spb = 60.0 / bpm
            a = max(0.0, float(b["t0"]) - float(c["start"]))
            z = min(float(c["len"]), float(b["t1"]) - float(c["start"]))
            if z - a < 1e-3:
                raise ValueError("the highlight doesn't cover that clip")
            sub = copy.deepcopy(proj)
            sub["tracks"] = [dict(id="t", name="s", role="any", color="#888", gain_db=0.0, pan=0.0, mute=False,
                                  solo=False, hpf=0, lpf=0, duck=0, pocket=0)]
            sub["clips"] = [dict(copy.deepcopy(c), id="c", track="t", start=0.0, gain_db=0.0, fade_in=0.0,
                                 fade_out=0.0, mute=False)]
            sub["master"] = dict(glue=False, limit=False, gain_db=0.0)
            sub.pop("loop", None)
            y = sampler._render_raw(sub, int(round(z * spb * SR)) + 1)
            y = np.ascontiguousarray(y[:, int(round(a * spb * SR)):int(round(z * spb * SR))], dtype=np.float32)
            key = hashlib.sha1(_json.dumps([c, a, z, bpm], sort_keys=True, default=str).encode()).hexdigest()[:12]
            out = MIDI_DIR / "_samples" / f"{c['song']}_{c['stem']}_{key}.wav"
            out.parent.mkdir(parents=True, exist_ok=True)
            audio_io.save(out, y, SR, subtype="FLOAT")
            return {"path": str(out), "seconds": round(y.shape[1] / SR, 3), "song": c["song"], "stem": c["stem"]}
        try:
            return await asyncio.to_thread(go)
        except (KeyError, ValueError, RuntimeError) as e:
            raise HTTPException(400, str(e))

    @app.get("/pianoroll")
    def pianoroll_page():
        from fastapi.responses import Response
        from .server import WEB, UI_TAG
        html = (WEB / "pianoroll.html").read_text(encoding="utf-8").replace('"/static/', f'"/ui/{UI_TAG}/')
        return Response(html, media_type="text/html", headers={"Cache-Control": "no-store"})

    @app.post("/api/midi/audition_stop")
    def midi_audition_stop():
        player.stop_audition()
        return {"ok": True}

    # ---------------------------------------------------------------- instruments
    @app.get("/api/instruments/status")
    def inst_status():
        return {"fluidsynth": instruments.fluidsynth_exe(), "default_sf2": instruments.default_sf2(quiet=True),
                "vst3": instruments.known_vst3() is not None, "gm": instruments.GM}

    @app.get("/api/instruments/sf2")
    def inst_sf2(path: str):
        try:
            return {"presets": instruments.sf2_presets(path)}
        except (OSError, ValueError) as e:
            raise HTTPException(400, str(e))

    @app.get("/api/instruments/soundfonts")
    def inst_soundfonts():
        """Every soundfont under the soundfont folders (subfolders too): [{path, name, folder, mb}]."""
        return {"soundfonts": instruments.list_soundfonts(), "default": instruments.default_sf2(quiet=True),
                "folders": [str(d) for d in instruments.soundfont_dirs()]}

    @app.get("/api/instruments/vst3")
    def inst_vst3():
        return {"plugins": instruments.known_vst3(), "folders": instruments.vst3_dirs()}

    @app.get("/api/instruments/folders")
    def inst_folders():
        return instruments.folders()

    @app.put("/api/instruments/folders")
    async def inst_set_folders(request: Request):
        """Body: {sf2: [...], vst3: [...]} - the folders to look in besides the built-in ones."""
        b = await request.json()
        return instruments.set_folders(b.get("sf2"), b.get("vst3"))

    # ---------------------------------------------------------------- chop kits (chops.py)
    @app.post("/api/midi/chops")
    async def midi_chops(request: Request):
        """Body: {song, stem, slice, from, to} -> {chops: [[t0, t1, class, peak_db(, word)]], seconds}."""
        from . import chops
        b = await request.json()
        try:
            return await asyncio.to_thread(chops.make, b["song"], b["stem"], b.get("slice") or "hits",
                                           b.get("from"), b.get("to"))
        except (ValueError, OSError, KeyError) as e:
            raise HTTPException(400, str(e))

    @app.post("/api/midi/chop_pitches")
    async def midi_chop_pitches(request: Request):
        """Body: {song, stem, chops} -> {pitches: [MIDI note or None]}: each chop's sung / played pitch."""
        from . import chops
        b = await request.json()
        try:
            return {"pitches": await asyncio.to_thread(chops.chop_pitches, {"song": b["song"], "stem": b["stem"],
                                                                              "chops": b.get("chops") or []})}
        except (OSError, KeyError, ValueError) as e:
            raise HTTPException(400, str(e))

    @app.post("/api/midi/say")
    async def midi_say(request: Request):
        """Body: {chops, text} -> {sequence, found, missing, words}: the word chops that say the sentence."""
        from . import chops
        b = await request.json()
        return chops.say(b.get("chops") or [], b.get("text") or "")

    @app.get("/api/midi/words_status")
    def midi_words_status(song: str):
        """{words: n} - how many timed words the song's lyrics have (0 = its lyrics haven't been found yet)."""
        from . import chops
        if not library.read(song):
            raise HTTPException(404, "no such song")
        return {"words": len(chops._lyrics_words(song) or [])}

    @app.post("/api/midi/chop_audition")
    async def midi_chop_audition(request: Request):
        """Body: {song, stem, a, b} -> plays that bit of the stem (seconds as recorded)."""
        from . import chops
        b = await request.json()
        inst = {"song": b["song"], "stem": b["stem"], "chops": [[float(b["a"]), float(b["b"]), "", 0]]}
        try:
            y = await asyncio.to_thread(chops.chop_buffer, inst, 0)
        except (OSError, KeyError) as e:
            raise HTTPException(400, str(e))
        player.audition(y, 0, y.shape[1], gain=0.9, ident="chop")
        return {"ok": True}

    @app.post("/api/midi/clip_span")
    async def midi_clip_span(request: Request):
        """Body: {project, clip_id, t0, t1} (timeline beats) -> {song, stem, from, to}: the highlighted bit of a clip
        as seconds of its song as recorded (for a chop kit's range)."""
        from . import project as P, warps
        b = await request.json()
        proj = b.get("project") or {}
        c = next((x for x in proj.get("clips", []) if x.get("id") == b.get("clip_id")), None)
        if c is None:
            raise HTTPException(404, "no such clip")
        info = library.read(c["song"])
        mult = P.song_mult(proj, c["song"])
        ps = warps.pieces(c, mult, None)
        beats = np.asarray(info["beats"], float)

        def sec(t):
            u = min(max(float(t) - float(c["start"]), 0.0), float(c["len"]))
            w = float(c["src"])
            for u0, u1, w0, w1, _r, _k in ps:
                if u0 <= u <= u1:
                    w = w0 + (w1 - w0) * (u - u0) / max(u1 - u0, 1e-9)
                    break
            else:
                if ps:
                    w = ps[-1][3]
            return float(midi.beat_time(beats, w))
        a, z = sec(b["t0"]), sec(b["t1"])
        return {"song": c["song"], "stem": c["stem"], "from": round(min(a, z), 3), "to": round(max(a, z), 3)}

    # ---------------------------------------------------------------- the piano roll's keyboard pad (midi_live.py)
    from . import midi_live

    @app.post("/api/midi/keys/prepare")
    async def keys_prepare(request: Request):
        """Body: {sid, pid, lo, hi} -> {bank, ready}: renders keys lo..hi of the part's instrument ahead."""
        b = await request.json()
        need(b["sid"])
        try:
            inst = midi.part_of(midi.read(b["sid"]), b["pid"]).get("instrument")
        except KeyError as e:
            raise HTTPException(404, str(e))
        try:
            bid, ready = await asyncio.to_thread(midi_live.KEYS.prepare, b["sid"], b["pid"], inst,
                                                 int(b.get("lo", 48)), int(b.get("hi", 72)))
        except Exception as e:
            raise HTTPException(400, f"couldn't get the instrument ready: {e}")
        player._ensure_stream()
        return {"bank": bid, "ready": ready}

    @app.post("/api/midi/keys/on")
    async def keys_on(request: Request):
        b = await request.json()
        try:
            ok = await asyncio.to_thread(midi_live.KEYS.on, b["bank"], int(b["pitch"]), int(b.get("vel", 100)))
        except KeyError as e:
            raise HTTPException(409, str(e))
        return {"ok": ok}

    @app.post("/api/midi/keys/off")
    async def keys_off(request: Request):
        b = await request.json()
        midi_live.KEYS.off(b["bank"], int(b["pitch"]))
        return {"ok": True}

    @app.post("/api/midi/keys/all_off")
    def keys_all_off():
        midi_live.KEYS.all_off()
        return {"ok": True}

    @app.post("/api/instruments/vst3/scan")
    def inst_vst3_scan():
        job = jobs.submit("vst3", "Scanning VST3 plugins", lambda job: {"plugins": instruments.scan_vst3(job.sub(0, 1, ""))})
        return {"job": job.id}

    @app.post("/api/instruments/vst3/editor")
    async def inst_vst3_editor(request: Request):
        """Body: the instrument. Opens the plugin's own window on this PC; returns {state} when it's closed."""
        inst = await request.json()
        try:
            state = await asyncio.to_thread(instruments.open_editor, inst)
        except Exception as e:
            raise HTTPException(400, f"couldn't open the plugin window: {e}")
        return {"state": state}

    @app.post("/api/pickfile")
    async def pickfile(request: Request):
        """Body: {title, types: [[label, "*.sf2 *.sf3"], ...], start} -> {path} ("" if cancelled). A file dialog on
        this PC (as /api/pickfolder)."""
        b = await request.json()
        start = Path(b["start"]).parent if b.get("start") else None

        def pick():
            import tkinter
            from tkinter import filedialog
            root = tkinter.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            try:
                types = [tuple(t) for t in (b.get("types") or [])] + [("All files", "*.*")]
                return filedialog.askopenfilename(parent=root, title=b.get("title") or "Choose a file", filetypes=types,
                                                  initialdir=str(start) if start and start.is_dir() else None)
            finally:
                root.destroy()
        try:
            got = await asyncio.to_thread(pick)
        except Exception as e:
            raise HTTPException(500, f"no file picker here ({e}) - paste the path instead")
        return {"path": str(Path(got)) if got else ""}
