"""Song store: data/songs/<id>/ holds stems (FLAC), song.json, features.npz and waveform peaks."""
import json
import re
import shutil
import threading
import time
import hashlib
from collections import OrderedDict
from pathlib import Path

import numpy as np

from . import audio_io, analyze, jobs
from .config import SAMPLES, SONGS, SR, STEMS, rel_source, resolve_source, song_path
from .events import bus

AUDIO_EXT = {".wav", ".flac", ".mp3", ".m4a", ".aac", ".ogg", ".opus", ".aiff", ".aif", ".mp4", ".webm", ".wma"}

_lock = threading.Lock()
_stem_cache = OrderedDict()      # (song_id, stem) -> (2, n) float32
_CACHE_MAX = 48


def _slug(s):
    return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")[:40] or "song"


def clean_title(name):
    t = Path(name).stem
    t = re.sub(r"\s*[\[(][A-Za-z0-9_-]{8,14}[\])]\s*$", "", t)   # youtube ids
    return t.replace("_", " ").strip() or "song"


def song_dir(sid):
    return song_path(sid)


def read(sid):
    p = song_dir(sid) / "song.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def write(sid, info):
    p = song_dir(sid) / "song.json"
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(info), encoding="utf-8")
    tmp.replace(p)


SUMMARY_KEYS = ("id", "title", "status", "error", "duration", "bpm", "bpb", "key", "tuning", "tuning_st", "created",
                "stable_tempo", "loudness", "source", "has_vocals", "separation", "kind")


def summary(info):
    out = {k: info.get(k) for k in SUMMARY_KEYS}
    if info.get("lock"):
        from . import lock
        out["lock_ok"] = lock.current(info) is not None          # a 🔒 locked version matching its marks exists
    return out


def list_samples():
    """Baked samples (sampler.py) - summaries, kind "sample"."""
    out = []
    if SAMPLES.is_dir():
        for d in SAMPLES.iterdir():
            if d.is_dir() and d.name.startswith("smp_"):
                info = read(d.name)
                if info:
                    out.append(summary(info))
    return out


def list_songs():
    out = []
    for d in SONGS.iterdir():
        if d.is_dir():
            info = read(d.name)
            if info:
                out.append(summary(info))
    return sorted(out, key=lambda s: s.get("created") or 0, reverse=True)


def _new(title, source):
    h = hashlib.sha1(f"{title}{source}{time.time()}".encode()).hexdigest()[:6]
    sid = f"{_slug(title)}_{h}"
    d = song_dir(sid)
    (d / "stems").mkdir(parents=True, exist_ok=True)
    info = dict(id=sid, title=title, source=rel_source(source), status="queued", created=time.time())
    write(sid, info)
    bus.publish("song", song=summary(info))
    return sid, info


def _set(sid, **kw):
    info = read(sid) or {}
    info.update(kw)
    write(sid, info)
    bus.publish("song", song=summary(info))
    return info


def delete(sid):
    d = song_dir(sid)
    with _lock:
        for k in [k for k in _stem_cache if k[0] == sid]:
            del _stem_cache[k]
    from . import warp
    warp.drop_song(sid)
    if d.exists():
        shutil.rmtree(d, ignore_errors=True)
    bus.publish("song_deleted", id=sid)


STEM_PEAK = 0.98


def stem_scale(info):
    """How much the song's stem files were turned down when saved (save_stems); 1 for older songs."""
    return float((info or {}).get("stem_scale", 1.0) or 1.0)


def save_stems(sid, stems):
    """Write a song's four stems. 24-bit FLAC can't hold anything above full scale, and stems of loud masters
    often go over (decoded mp3s, parts that partly cancel in the mix) - saving them clipped flattened drum hits in
    125 of the first 178 songs. So when needed all four are turned down by the same factor (song.json
    "stem_scale") and every reader turns them back up (read_stem). -> the factor."""
    d = song_dir(sid)
    peak = max(float(np.abs(v).max()) for v in stems.values())
    scale = round(min(1.0, STEM_PEAK / peak), 6) if peak > 0 else 1.0
    for s, y in stems.items():
        audio_io.save(d / "stems" / f"{s}.flac", np.asarray(y, np.float32) * scale, SR, subtype="PCM_24")
    _set(sid, stem_scale=scale)
    return scale


def read_stem(sid, stem, info=None):
    """A stem file at its true level (undoing save_stems' headroom)."""
    return audio_io.load(song_dir(sid) / "stems" / f"{stem}.flac", SR) / stem_scale(info or read(sid))


def load_stem(sid, stem, tspec=""):
    """A stem at its true level - as recorded, or locked to its marks when the timing spec says so (variant.py)."""
    from . import variant
    key = (sid, stem, variant.lock_rev(tspec))
    with _lock:
        if key in _stem_cache:
            _stem_cache.move_to_end(key)
            return _stem_cache[key]
    if key[2]:
        info = read(sid)
        y = audio_io.load(variant.stem_file(sid, stem, tspec), SR) / variant.scale_for(info, tspec)
    else:
        y = read_stem(sid, stem)
    with _lock:
        _stem_cache[key] = y
        while len(_stem_cache) > _CACHE_MAX:
            _stem_cache.popitem(last=False)
    return y


def peek_stem(sid, stem, tspec=""):
    """The stem if it's already in memory, else None (never blocks on disk)."""
    from . import variant
    with _lock:
        return _stem_cache.get((sid, stem, variant.lock_rev(tspec)))


def find_stem_files(folder):
    """drums/bass/other/vocals audio in a folder: exact names or names containing the stem (UVR style)."""
    folder = Path(folder)
    files = [f for f in folder.iterdir() if f.suffix.lower() in AUDIO_EXT]
    out = {}
    for s in STEMS:
        cand = [f for f in files if f.stem.lower() == s] or \
               [f for f in files if s in f.name.lower() and f"no_{s}" not in f.name.lower()
                and f"no {s}" not in f.name.lower() and "instrumental" not in f.name.lower()]
        if not cand:
            return None
        out[s] = sorted(cand, key=lambda f: len(f.name))[0]
    return out


# ---------------------------------------------------------------- import pipelines (run as jobs)

def _finish(sid, job, stems):
    d = song_dir(sid)
    job.update(0.72, "saving stems")
    save_stems(sid, stems)
    _set(sid, status="analyzing")
    info = analyze.analyze(d, stems, job.sub(0.75, 1.0, "analyzing: "))
    out = _set(sid, status="ready", error=None, **info)
    schedule_melody(sid)
    return out


def schedule_melody(sid):
    """Track the vocal melody (for mashup rating) in a separate process, so it never stalls playback.
    Runs as its own background job after the song is ready."""
    from . import mashfeat
    p = song_dir(sid) / "mash.npz"
    if p.exists():
        p.unlink()
    info = read(sid) or {}

    def work(job):
        from . import pairs
        pool = pairs._pool(1)
        try:
            fut = pool.submit(pairs._w_features, sid)
            t0 = time.time()
            while not fut.done():
                job.update(min(0.95, (time.time() - t0) / max(30.0, 0.35 * float(info.get("duration", 200)))),
                           "tracking the vocal melody (for mashup ratings)")
                time.sleep(0.5)
            fut.result()
        finally:
            for pr in list(getattr(pool, "_processes", {}).values()):
                try:
                    pr.terminate()
                except Exception:
                    pass
            pool.shutdown(wait=False, cancel_futures=True)

    return jobs.submit("melody", f"Melody: {info.get('title', sid)}", work, ref=sid)


def import_audio(path, title=None, quality=None):
    quality = quality or default_quality()
    path = Path(path)
    title = title or clean_title(path.name)
    sid, _ = _new(title, path)

    def work(job):
        try:
            job.update(0.01, "decoding")
            y = audio_io.load(path, SR)
            _set(sid, status="separating")
            stems = run_separation(y, quality, job.sub(0.03, 0.70, "separating stems "))
            _set(sid, separation=quality)
            return summary(_finish(sid, job, stems))
        except Exception as e:
            _set(sid, status="error", error=str(e))
            raise

    return sid, jobs.submit("import", title, work, ref=sid, gpu=True)


def ingest_inline(path, job, quality=None):
    """Decode + separate + analyze a NEW audio file SYNCHRONOUSLY, inside the caller's already-held `_gpu`
    slot — NO new gpu job is submitted. Used by the orchestrator (auto-mashup), which already owns the single
    _gpu worker: submitting another gpu=True job and polling for it would deadlock (the second job can never be
    scheduled while this one holds the worker). Running the GPU work directly in the held slot cannot deadlock
    and still can't run concurrently with other GPU jobs, so the single-heavy-model-at-a-time / OOM invariant
    holds. Returns the ready song's id (raises on failure, mirroring import_audio's error path).

    This is the defensive fallback for an input that is NOT yet a library song; the common UI path (two
    already-'ready' library songs) never reaches here."""
    quality = quality or default_quality()
    path = Path(path)
    title = clean_title(path.name)
    sid, _ = _new(title, path)
    try:
        job.update(0.01, "decoding")
        y = audio_io.load(path, SR)
        _set(sid, status="separating")
        stems = run_separation(y, quality, job.sub(0.03, 0.70, "separating stems "))
        _set(sid, separation=quality)
        _finish(sid, job, stems)
        return sid
    except Exception as e:
        _set(sid, status="error", error=str(e))
        raise


def stems_from_files(files, sr=SR):
    """{stem: path or [paths]} -> the song's four stems, (2, n) each. Several files for one stem (kick + snare +
    hats -> drums, two guitars -> other) are mixed together; a stem with no file is silence; shorter files are
    padded at the end (stems exported from one session all start at its zero)."""
    loaded = {s: [audio_io.load(p, sr) for p in ([ps] if isinstance(ps, (str, Path)) else ps)]
              for s, ps in files.items() if ps}
    if not loaded:
        raise ValueError("no stem files")
    n = max(y.shape[1] for ys in loaded.values() for y in ys)
    pad = lambda y: np.pad(y, ((0, 0), (0, n - y.shape[1])))
    return {s: (np.sum([pad(y) for y in loaded[s]], axis=0) if s in loaded else np.zeros((2, n))).astype(np.float32)
            for s in STEMS}


def import_stems(files, title, source=""):
    """Already-split stems -> one song, NOT separated again. files: {stem: path or [paths]} (stems_from_files);
    a song can have only some stems (an acapella, a drum loop)."""
    files = {s: ([p] if isinstance(p, (str, Path)) else list(p)) for s, p in files.items() if p}
    if not files:
        raise ValueError("no stem files")
    sid, _ = _new(title, source or Path(next(iter(files.values()))[0]).parent)

    def work(job):
        try:
            job.update(0.02, "loading stems")
            stems = stems_from_files(files)
            _set(sid, separation="imported", stem_files={s: [Path(p).name for p in ps] for s, ps in files.items()})
            return summary(_finish(sid, job, stems))
        except Exception as e:
            _set(sid, status="error", error=str(e))
            raise

    return sid, jobs.submit("import", title, work, ref=sid, gpu=True)


def default_quality():
    """How new songs are separated unless you pick otherwise: the SW + Ultra blend (best in the Oct 2026 listening
    test), else Ultra, else plain Demucs - whichever models are here."""
    from .separate import sw_available, ultra_available
    if ultra_available():
        return "blend" if sw_available() else "ultra"
    return "best"


def run_separation(y, quality, progress):
    from . import separate as sep
    if quality in ("sw", "blend"):         # BS-Roformer SW alone, or averaged with Ultra (right-click a song)
        if not sep.sw_available():
            raise RuntimeError(f"the BS-Roformer SW model isn't in mashup_lab/models ({sep.SW_CKPT})")
        if quality == "blend":
            if not sep.ultra_available():
                raise RuntimeError("the SW + Ultra blend needs the Ultra (BS-Roformer vocal) model too")
            return sep.separate_blend(y, progress)
        return sep.separate_sw(y, progress)
    if quality == "ultra" and sep.ultra_available():
        return sep.separate_ultra(y, progress)
    # plain Demucs: its four stems don't add up to the song exactly either - share the leftover out
    return sep.spread_leftover(y, sep.separate(y, "best" if quality == "ultra" else quality, progress))


def reseparate(sid, quality="ultra"):
    """Redo a song's stems with a better separator. Keeps its beat grid, so projects stay lined up."""
    info = read(sid)

    def work(job):
        d = song_dir(sid)
        src = resolve_source(info.get("source"))
        job.update(0.01, "loading the original")
        if src.is_file() and src.suffix.lower() in AUDIO_EXT:
            y = audio_io.load(src, SR)
        else:                              # imported as stems: their sum is the original mix
            y = sum(read_stem(sid, s, info) for s in STEMS)
        _set(sid, status="separating")
        try:
            stems = run_separation(y, quality, job.sub(0.03, 0.72, "re-separating "))
            n = min(v.shape[1] for v in stems.values())
            job.update(0.73, "saving stems")
            stems = {s_: v[:, :n] for s_, v in stems.items()}
            save_stems(sid, stems)
            with _lock:
                for k in [k for k in _stem_cache if k[0] == sid]:
                    del _stem_cache[k]
            grid = dict(beats=info["beats"], bars=info["bars"], bpb=info["bpb"], bpm=info["bpm"],
                        segments=info["segments"], stable=info.get("stable_tempo", False),
                        first_beat=info["first_beat"], last_beat=info["last_beat"])
            _set(sid, status="analyzing")
            new = analyze.analyze(d, stems, job.sub(0.75, 1.0, "analyzing: "), grid=grid)
        except Exception as e:
            _set(sid, status="error", error=str(e))
            raise
        from . import warp
        warp.drop_song(sid)
        out = summary(_set(sid, status="ready", error=None, separation=quality,
                           stems_rev=int(info.get("stems_rev", 0)) + 1, **new))
        schedule_melody(sid)           # new vocal stem -> new melody track
        return out

    how = {"sw": " with BS-Roformer SW", "blend": " (SW + Ultra blend)"}.get(quality, "")
    return jobs.submit("separate", f"Re-separating {info['title']}{how}", work, ref=sid, gpu=True)


def restore_missing(sid, backup_dir=None):
    """Put back what the separation dropped: whatever of the original mix is in no stem is shared out between
    drums / bass / other, each getting the time/frequency cells it already owns (separate.spread_leftover - the
    same as a fresh separation does now). Vocals stay as they are. Up to Oct 2026 Ultra separation threw away the
    part of the instrumental Demucs called "vocals" (synth leads, pads, chops...). Keeps the beat grid, so
    projects stay lined up. -> (dB of the mix that was missing in its worst 4 s, peak level) or None if the song
    has no original mix to compare with."""
    from . import separate as sep
    info = read(sid)
    d = song_dir(sid)
    src = resolve_source(info.get("source"))
    if not (src.is_file() and src.suffix.lower() in AUDIO_EXT):
        return None
    mix = audio_io.load(src, SR)
    stems = {s: read_stem(sid, s, info) for s in STEMS}
    n = min([mix.shape[1]] + [v.shape[1] for v in stems.values()])
    mix = mix[:, :n]
    stems = {k: v[:, :n] for k, v in stems.items()}
    missing = mix - sum(stems.values())
    w = 4 * SR
    worst = max((20 * np.log10(np.sqrt(np.mean(missing[:, i:i + w] ** 2)) / (np.sqrt(np.mean(mix[:, i:i + w] ** 2)) + 1e-9) + 1e-12)
                 for i in range(0, n - w, w) if np.sqrt(np.mean(mix[:, i:i + w] ** 2)) > 0.01), default=-200.0)
    if backup_dir:
        bk = Path(backup_dir) / sid
        bk.mkdir(parents=True, exist_ok=True)
        for s_ in STEMS:
            shutil.copy2(d / "stems" / f"{s_}.flac", bk / f"{s_}.flac")
        shutil.copy2(d / "song.json", bk / "song.json")
        if (d / "features.npz").exists():
            shutil.copy2(d / "features.npz", bk / "features.npz")
    inst = mix - stems["vocals"]
    stems.update(sep.spread_leftover(inst, {k: stems[k] for k in ("drums", "bass", "other")}))
    peak = max(float(np.abs(v).max()) for v in stems.values())
    save_stems(sid, stems)
    with _lock:
        for k in [k for k in _stem_cache if k[0] == sid]:
            del _stem_cache[k]
    grid = dict(beats=info["beats"], bars=info["bars"], bpb=info["bpb"], bpm=info["bpm"],
                segments=info["segments"], stable=info.get("stable_tempo", False),
                first_beat=info["first_beat"], last_beat=info["last_beat"], grid_version=info.get("grid_version", 1))
    new = analyze.analyze(d, stems, grid=grid)
    from . import warp
    warp.drop_song(sid)
    _set(sid, status="ready", error=None, stems_rev=int(info.get("stems_rev", 0)) + 1, **new)
    return round(float(worst), 1), round(peak, 3)


def reanalyze(sid):
    """Redo the beat grid and all analysis. Saved projects that use the song are moved onto the new grid
    (clips keep their moment in the audio), and melody / warp caches are rebuilt."""
    info = read(sid)

    def work(job):
        stems = {s: load_stem(sid, s) for s in STEMS}
        _set(sid, status="analyzing")
        try:
            new = analyze.analyze(song_dir(sid), stems, job.sub(0, 1, "analyzing: "))
        except Exception as e:
            _set(sid, status="error", error=str(e))
            raise
        return summary(apply_new_grid(sid, info, new))

    return jobs.submit("analyze", info["title"], work, ref=sid, gpu=True)


def mus_check(sid):
    """MuScriptor second opinion on a song's beat marks (muscheck.py): transcribe the song (~2 min on the GPU), keep the
    notes (muscriptor_notes.json) and store, per beat, whether the notes agree with the marks (song.json mus_check)."""
    from . import muscheck as M
    info = read(sid)

    def work(job):
        import soundfile as sf
        job.update(0.0, "loading the MuScriptor model", force=True)
        stems = [load_stem(sid, s) for s in STEMS]
        n = min(v.shape[1] for v in stems)
        mix = sum(v[:, :n] for v in stems).astype(np.float32)
        dur = n / SR
        wav = song_dir(sid) / "muscriptor_in.tmp.wav"
        sf.write(str(wav), mix.T, SR, subtype="FLOAT")
        try:
            notes = M.transcribe(wav, lambda t: job.update(min(0.98, t / max(dur, 1e-6)), "transcribing notes"))
        finally:
            wav.unlink(missing_ok=True)
        notes = M.join_seams(notes)
        (song_dir(sid) / "muscriptor_notes.json").write_text(json.dumps(notes), encoding="utf-8")
        return summary(_set(sid, mus_check=_mus_check(read(sid), notes)))

    return jobs.submit("muscheck", f"MuScriptor second opinion: {info['title']}", work, ref=sid, gpu=True)


def _mus_check(info, notes):
    from . import muscheck as M
    v = M.verdicts(info["beats"], notes)
    n = max(1, len(v))
    return dict(v=v, agree=round(v.count(M.AGREE) / n * 100, 1), disagree=round(v.count(M.DISAGREE) / n * 100, 1),
                meter=M.meter_check(info["beats"], notes, info.get("bpb", 4), info["first_beat"], info["last_beat"]),
                notes=len(notes), t=time.time())


def apply_new_grid(sid, old_info, new_info):
    """Store re-analysis results; if the beats themselves moved, carry projects / caches over."""
    ob, nb = old_info.get("beats") or [], new_info["beats"]
    moved = len(ob) != len(nb) or (len(ob) and float(np.max(np.abs(np.asarray(ob) - np.asarray(nb)))) > 0.004)
    extra = {}
    if moved and ob:
        from . import project as P, warp
        P.remap_song_beats(sid, ob, nb, old_info.get("bpm"), new_info.get("bpm"),
                           old_info.get("bars"), new_info.get("bars"))
        warp.drop_song(sid)
        extra["stems_rev"] = int(old_info.get("stems_rev", 0)) + 1      # never reuse a render of the old grid
    notes_f = song_dir(sid) / "muscriptor_notes.json"
    if moved and notes_f.exists():                                     # the second opinion follows the new beats
        try:
            extra["mus_check"] = _mus_check(new_info, json.loads(notes_f.read_text(encoding="utf-8")))
        except Exception:
            extra["mus_check"] = None
    out = _set(sid, status="ready", error=None, **new_info, **extra)
    if moved:
        schedule_melody(sid)                                             # melody features are per beat
    return out


def resume_interrupted():
    """Songs left mid-import when the app closed. CAREFUL at startup: do NOT blindly resubmit GPU jobs for
    every non-ready song, because one stuck/hung re-analysis blocks the single-worker _gpu pool and leaves
    every later GPU job (including a freshly dropped song) stuck at "queued"/"starting" forever (the recurring
    wedge). Rule:
      - complete stems present AND status was 'analyzing' (separation finished, analysis didn't)  -> re-analyze
        (the only genuinely-safe resume; separation is the expensive part and it's already done);
      - a 'queued' song (separation never started) or any state without complete stems  -> mark 'error' with a
        clear retry message. It is NOT auto-resubmitted, so a stale/duplicate queued record from a previous run
        can never jam the queue on startup. The user re-adds or retries it explicitly.
    """
    for d in SONGS.iterdir():
        if not d.is_dir():
            continue
        info = read(d.name)
        if not info or info.get("status") in ("ready", "error"):
            continue
        have_stems = all((d / "stems" / f"{s}.flac").exists() for s in STEMS)
        if have_stems and info.get("status") == "analyzing":
            reanalyze(d.name)                       # safe resume: stems done, only analysis to redo
        else:
            _set(d.name, status="error",
                 error="import didn't finish (interrupted or still queued at shutdown) - add the song again or retry")


# ---------------------------------------------------------------- folders (user-named groups of songs)

FOLDERS = SONGS.parent / "folders.json"


def get_folders():
    """{"folders": [{"id", "name", "collapsed"}], "assign": {song_id: folder_id}}"""
    try:
        d = json.loads(FOLDERS.read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError):
        d = {}
    d.setdefault("folders", [])
    d.setdefault("assign", {})
    ids = {f["id"] for f in d["folders"]}
    d["assign"] = {s: f for s, f in d["assign"].items() if f in ids and (SONGS / s).exists()}
    return d


def set_folders(d):
    clean = {"folders": [{"id": str(f["id"]), "name": str(f.get("name", "Folder"))[:80],
                          "collapsed": bool(f.get("collapsed", False))} for f in d.get("folders", [])],
             "assign": {str(k): str(v) for k, v in d.get("assign", {}).items()}}
    tmp = FOLDERS.with_suffix(".tmp")
    tmp.write_text(json.dumps(clean, indent=1), encoding="utf-8")
    tmp.replace(FOLDERS)
    bus.publish("folders", folders=get_folders())
    return clean
