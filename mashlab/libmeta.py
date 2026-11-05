"""Library metadata for grouping the song list.

  meta    artist / album / genre / year: read from the source file's tags (ffprobe), else guessed from its file
          name ("Artist - Title"); a genre can also be suggested by the DJ's language model. The user's own edits
          always win. Stored in DATA/meta.json: {"songs": {sid: {"tags": {..}, "guess": {..}, "user": {..}}}}.
  compat  "songs that work together": every ready song scored against every other with automash.match_list
          (tempo stretch + the best harmonic combination), then grouped by average-linkage clustering. Scores are
          kept in DATA/compat.json, so only new songs are compared later.

Both live next to the songs, never inside song.json, so they can't clash with the analysis jobs that rewrite it.
"""
import json
import re
import subprocess
import threading
import time
from pathlib import Path

import numpy as np

from . import library
from .config import DATA, resolve_source
from .events import bus

FIELDS = ("artist", "album", "genre", "year")
META = DATA / "meta.json"
COMPAT = DATA / "compat.json"
COMPAT_VERSION = 1
GROUP_MIN = 62          # average match score a group must keep (match_list scores are 0-100)
GROUP_MAX = 14          # songs per group, so a group stays a usable crate
GENRES = ["Alternative", "Ambient", "Blues", "Classical", "Country", "Dance", "Drum & Bass", "Dubstep", "Electronic",
          "Experimental", "Folk", "Funk", "Hardcore", "Hip-Hop", "House", "Indie", "Jazz", "Latin", "Metal", "Pop",
          "Post-Rock", "Progressive", "Punk", "R&B", "Reggae", "Rock", "Soul", "Soundtrack", "Techno", "Trap"]
_lock = threading.Lock()


def _load(path):
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else {}
    except (FileNotFoundError, ValueError):
        return {}


def _save(path, d):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(d, indent=1, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


# ---------------------------------------------------------------- tags

def _clean(v):
    v = re.sub(r"\s+", " ", str(v or "")).strip()
    return v[:120]


def from_filename(name):
    """{artist?, title} from an upload name like "1790318015146_Puscifer  - Agostina [rWQwjAu752c].wav"."""
    stem = Path(name).stem
    stem = re.sub(r"^\d{10,}_", "", stem)                 # upload timestamp
    stem = re.sub(r"\s*\[[\w-]{6,}\]$", "", stem)         # [youtube id]
    stem = re.sub(r"^\d{1,3}[\s._-]+", "", stem)           # track number
    if " - " in stem:
        a, t = stem.split(" - ", 1)
        if a.strip() and t.strip():
            return {"artist": _clean(a), "title": _clean(t)}
    return {"title": _clean(stem)}


def read_tags(path):
    """{artist, album, genre, year} found in the file's tags (missing ones left out)."""
    try:
        r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format_tags:stream_tags", "-of", "json",
                            str(path)], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=20)
        j = json.loads(r.stdout or "{}")
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return {}
    tags = {}
    for src in [j.get("format", {}).get("tags", {})] + [s.get("tags", {}) for s in j.get("streams", [])]:
        for k, v in (src or {}).items():
            tags.setdefault(k.lower(), v)
    out = {"artist": _clean(tags.get("artist") or tags.get("album_artist") or tags.get("albumartist")),
           "album": _clean(tags.get("album")),
           "genre": _clean((tags.get("genre") or "").split(";")[0].split("/")[0]),
           "year": (re.search(r"(1[89]|20)\d\d", str(tags.get("date") or tags.get("year") or "")) or [""])[0]}
    return {k: v for k, v in out.items() if v}


PLACEHOLDER_GENRES = {"other", "unknown", "misc", "miscellaneous", "genre", "none", "default", "music"}


def _effective(e):
    out, src = {}, {}
    for f in FIELDS:
        for layer in ("user", "tags", "file", "guess"):
            v = (e.get(layer) or {}).get(f)
            if f == "genre" and layer in ("tags", "file") and str(v or "").strip().lower() in PLACEHOLDER_GENRES:
                continue                                   # a file tag of "Other" says nothing: a guess may fill in
            if v:
                out[f], src[f] = v, layer
                break
    out["src"] = src
    h = e.get("heard")
    if h and h.get("sound"):
        out["heard"] = {k: h[k] for k in ("genre", "subgenre", "sound") if h.get(k)}
    return out


def all_meta():
    """{sid: {artist?, album?, genre?, year?, src: {field: "user"|"tags"|"file"|"guess"}}} for every song. Reads the
    tags of songs it hasn't seen yet (~50 ms each)."""
    with _lock:
        d = _load(META)
        songs = d.setdefault("songs", {})
        changed = False
        live = {s["id"]: s for s in library.list_songs()}
        for sid, s in live.items():
            e = songs.get(sid)
            if e is not None and "tags" in e:
                continue
            e = songs.setdefault(sid, {})
            src = resolve_source(s.get("source"))
            e["tags"] = read_tags(src) if Path(src).is_file() else {}
            fn = from_filename(Path(str(s.get("source") or "")).name or s.get("title") or "")
            e["file"] = {k: v for k, v in fn.items() if k == "artist"}
            changed = True
        for sid in [k for k in songs if k not in live]:
            del songs[sid]
            changed = True
        if changed:
            _save(META, d)
        return {sid: _effective(e) for sid, e in songs.items()}


def set_user(ids, changes):
    """Set (or with "" clear) the user's own values for these songs. -> their effective meta."""
    with _lock:
        d = _load(META)
        songs = d.setdefault("songs", {})
        for sid in ids:
            u = songs.setdefault(sid, {}).setdefault("user", {})
            for f, v in changes.items():
                if f not in FIELDS:
                    continue
                v = _clean(v)
                if f == "year":
                    v = (re.search(r"\d{4}", v) or [""])[0]
                if v:
                    u[f] = v
                else:
                    u.pop(f, None)
        _save(META, d)
        out = {sid: _effective(songs[sid]) for sid in ids if sid in songs}
    bus.publish("meta", meta=out)
    return out


def set_heard(sid, heard, model=""):
    """Store what an audio model heard in a song ({genre, subgenre, sound}, listen.py); its genre becomes the song's
    genre guess (the user's edits and file tags still win)."""
    with _lock:
        d = _load(META)
        e = d.setdefault("songs", {}).setdefault(sid, {})
        e["heard"] = dict({k: heard[k] for k in ("genre", "subgenre", "sound") if heard.get(k)}, model=model,
                          at=time.strftime("%Y-%m-%d"))
        if heard.get("genre") in GENRES:
            e.setdefault("guess", {})["genre"] = heard["genre"]
        _save(META, d)


def heard(sid):
    """{genre, subgenre, sound} an audio model wrote for this song, or None."""
    e = _load(META).get("songs", {}).get(sid) or {}
    h = e.get("heard") or {}
    return {k: h[k] for k in ("genre", "subgenre", "sound") if h.get(k)} or None


def guess_genres(ids=None, progress=lambda f, m="": None, cancelled=lambda: False, batch=12):
    """Ask the DJ's language model for a genre (from GENRES) for songs that have none. Stored as a guess, which tags
    and the user's edits override. -> number of songs guessed."""
    from . import llm
    meta = all_meta()
    titles = {s["id"]: s for s in library.list_songs()}
    folders = library.get_folders()
    fname = {f["id"]: f["name"] for f in folders["folders"]}
    todo = [sid for sid in (ids or meta) if sid in titles and not meta.get(sid, {}).get("genre")]
    if not todo:
        return 0
    client = llm.client()
    schema = {"type": "object", "properties": {"items": {"type": "array", "items": {
        "type": "object", "properties": {"n": {"type": "integer"}, "genre": {"type": "string", "enum": GENRES + ["Unknown"]}},
        "required": ["n", "genre"]}}}, "required": ["items"]}
    done = 0
    for i in range(0, len(todo), batch):
        if cancelled():
            break
        chunk = todo[i:i + batch]
        lines = []
        for n, sid in enumerate(chunk):
            m, s = meta.get(sid, {}), titles[sid]
            hint = fname.get(folders["assign"].get(sid), "")
            lines.append(f"{n}. title: {s['title']}" + (f"; artist: {m['artist']}" if m.get("artist") else "")
                         + (f"; album: {m['album']}" if m.get("album") else "") + (f"; library folder: {hint}" if hint else "")
                         + f"; {round(s.get('bpm') or 0)} BPM")
        msgs = [{"role": "system", "content": "You label songs with their music genre. Use what you know about the artist "
                 "and title; the library folder is often the artist's name. Answer Unknown when you can't tell."},
                {"role": "user", "content": "Pick one genre for each song:\n" + "\n".join(lines)}]
        try:
            res = client.chat_json(msgs, schema, max_tokens=900, temperature=0.2, seed=1)
        except Exception:
            continue
        with _lock:
            d = _load(META)
            songs = d.setdefault("songs", {})
            for it in res.get("items", []):
                n, g = it.get("n"), it.get("genre")
                if isinstance(n, int) and 0 <= n < len(chunk) and g in GENRES:
                    songs.setdefault(chunk[n], {}).setdefault("guess", {})["genre"] = g
                    done += 1
            _save(META, d)
        progress(min(1.0, (i + batch) / len(todo)), f"genres: {min(i + batch, len(todo))}/{len(todo)}")
    bus.publish("meta", meta=all_meta())
    return done


# ---------------------------------------------------------------- songs that work together

def _ready_ids():
    return [s["id"] for s in library.list_songs() if s.get("status") == "ready"]


def compat_state():
    d = _load(COMPAT)
    if d.get("version") != COMPAT_VERSION:
        d = {"version": COMPAT_VERSION, "done": [], "scores": {}}
    return d


def compare_all(progress=lambda f, m="": None, cancelled=lambda: False):
    """Score every ready song not compared yet against all the others. -> how many songs were compared."""
    from . import automash
    with _lock:
        d = compat_state()
    ids = _ready_ids()
    todo = [sid for sid in ids if sid not in set(d["done"])]
    for n, sid in enumerate(todo):
        if cancelled():
            break
        progress(n / max(1, len(todo)), f"comparing songs: {n + 1}/{len(todo)}")
        try:
            r = automash.match_list(sid)
        except Exception:
            continue
        with _lock:
            d = compat_state()
            for m in r.get("matches", []):
                a, b = sorted((sid, m["id"]))
                prev = d["scores"].get(f"{a}|{b}")
                s = 0 if m.get("same_song") else int(m.get("score") or 0)
                d["scores"][f"{a}|{b}"] = s if prev is None else round((prev + s) / 2)
            if sid not in d["done"]:
                d["done"].append(sid)
            _save(COMPAT, d)
    return len(todo)


def compat_groups():
    """{"groups": [{ids, fit, keys, bpm}], "loose": [ids], "missing": [ids not compared yet]}"""
    d = compat_state()
    songs = {s["id"]: s for s in library.list_songs() if s.get("status") == "ready"}
    ids = [i for i in d["done"] if i in songs]
    missing = [i for i in songs if i not in set(d["done"])]
    n = len(ids)
    if n < 2:
        return {"groups": [], "loose": ids, "missing": missing}
    ix = {sid: i for i, sid in enumerate(ids)}
    M = np.zeros((n, n), np.float32)
    for k, v in d["scores"].items():
        a, _, b = k.partition("|")
        if a in ix and b in ix:
            M[ix[a], ix[b]] = M[ix[b], ix[a]] = v
    groups = cluster(M)
    out, loose = [], []
    for g in groups:
        members = [ids[i] for i in g]
        if len(members) < 2:
            loose += members
            continue
        sub = M[np.ix_(g, g)]
        fit = float(sub[np.triu_indices(len(g), 1)].mean())
        hub = members[int(np.argmax(sub.sum(axis=1)))]             # the song that fits the rest best
        keys = {}
        for sid in members:
            k = (songs[sid].get("key") or {}).get("camelot")
            if k:
                keys[k] = keys.get(k, 0) + 1
        bpms = [songs[sid].get("bpm") or 0 for sid in members]
        rows = [ix[o] for o in members]
        members.sort(key=lambda sid: -M[ix[sid], rows].sum())               # best-connected first
        out.append({"ids": members, "hub": hub, "fit": round(fit), "keys": sorted(keys, key=lambda k: -keys[k])[:3],
                    "bpm": [round(min(bpms)), round(max(bpms))]})
    out.sort(key=lambda g: (-g["fit"], -len(g["ids"])))
    return {"groups": out, "loose": loose, "missing": missing}


def cluster(M, min_fit=GROUP_MIN, max_size=GROUP_MAX):
    """Average-linkage agglomerative clustering on a similarity matrix: keep merging the two groups whose songs fit
    each other best on average, while that average stays >= min_fit. -> [[indices]]"""
    groups = [[i] for i in range(len(M))]
    S = M.astype(np.float64).copy()
    np.fill_diagonal(S, -1)
    sizes = np.ones(len(M))
    alive = np.ones(len(M), bool)
    while True:
        A = np.where(np.outer(alive, alive), S, -1)
        i, j = np.unravel_index(int(np.argmax(A)), A.shape)
        if A[i, j] < min_fit:
            break
        if sizes[i] + sizes[j] > max_size:
            S[i, j] = S[j, i] = -1                       # too big together: never merge these two
            continue
        # average linkage: the merged group's similarity to every other group, weighted by size
        S[i] = (S[i] * sizes[i] + S[j] * sizes[j]) / (sizes[i] + sizes[j])
        S[:, i] = S[i]
        S[i, i] = -1
        groups[i] += groups[j]
        sizes[i] += sizes[j]
        alive[j] = False
        S[j] = S[:, j] = -1
    return [groups[i] for i in range(len(groups)) if alive[i]]
