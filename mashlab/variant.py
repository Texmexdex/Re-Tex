"""Which version of a song's audio a render / preview uses: as recorded ("stems/"), or locked to its marks
("stems_locked/", made once by lock.py). A render's timing spec carries "lock": <rev> when the project has the song
locked and that lock is current; everything that reads a song's stems or beat times for a render goes through here.
Kept free of heavy imports: render worker processes use it."""
import json

from .config import song_path


def lock_rev(tspec):
    """The lock revision a timing spec asks for ('' = the song as recorded)."""
    if not tspec:
        return ""
    try:
        return str(json.loads(tspec).get("lock") or "")
    except (ValueError, AttributeError):
        return ""


def lock_entry(info, rev):
    """The lock with this revision: the song's current one ("lock"), or one kept for projects that still use an
    older grid of the song, or one made from a project's own timing ("locks", by marks signature). None if gone."""
    if not rev:
        return None
    for lk in [(info or {}).get("lock") or {}, *((info or {}).get("locks") or {}).values()]:
        if lk.get("rev") == rev:
            return lk
    return None


def stem_file(sid, stem, tspec=""):
    """The current lock's stems live in stems_locked/ (as they always have - the frozen releases read them there);
    a kept or project lock's in stems_locked/<rev>/."""
    rev = lock_rev(tspec)
    if not rev:
        return song_path(sid) / "stems" / f"{stem}.flac"
    kept = song_path(sid) / "stems_locked" / rev / f"{stem}.flac"
    return kept if kept.exists() else song_path(sid) / "stems_locked" / f"{stem}.flac"


def beats_for(info, tspec=""):
    """The song's beat times (indices = the song's beats) in the version the spec asks for."""
    lk = lock_entry(info, lock_rev(tspec))
    return lk["beats"] if lk else info["beats"]


def scale_for(info, tspec=""):
    """How far the stem files of that version were turned down when saved (turn them back up by dividing)."""
    if lock_rev(tspec):
        return float((lock_entry(info, lock_rev(tspec)) or {}).get("scale", 1.0) or 1.0)
    return float((info or {}).get("stem_scale", 1.0) or 1.0)
