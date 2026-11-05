"""Full analysis of a separated song -> song.json + features.npz + waveform peaks."""
import json
from pathlib import Path

import numpy as np
import soxr
import pyloudnorm as pyln

from . import beats as B, harmony as H, structure as S
from .audio_io import peaks
from .config import SR, STEMS, PITCHED, PEAKS_PER_SEC

LSR = 22050     # analysis rate for chroma / rms


def _lufs(y):
    try:
        v = pyln.Meter(SR).integrated_loudness(y.T.astype(np.float64))
        return float(v) if np.isfinite(v) else -70.0
    except Exception:
        return -70.0


def analyze(song_dir, stems, progress=lambda f, msg="": None, grid=None):
    """stems: dict stem -> (2, n) float32 at SR. Writes files into song_dir; returns the song info dict.
    grid: reuse an existing beat grid (re-separation keeps every project's clips exactly where they were)."""
    n = min(v.shape[1] for v in stems.values())
    stems = {k: v[:, :n] for k, v in stems.items()}
    mix = sum(stems.values())
    dur = n / SR

    progress(0.02, "finding beats")
    mono = lambda y: y.mean(0)
    if grid is None:
        grid = B.build_grid(mono(mix), mono(stems["drums"]), SR, dur)
    bt = np.array(grid["beats"])
    bt_clip = np.clip(bt, 0, dur)

    progress(0.30, "tuning & harmony")
    low = {k: soxr.resample(mono(v), SR, LSR, quality="HQ").astype(np.float32) for k, v in stems.items()}
    harm = low["bass"] + low["other"] + low["vocals"]
    tuning = H.estimate_tuning(harm, LSR)
    tuning_st = H.tuning_semitones(harm, LSR)
    chroma = {}
    for i, s in enumerate(PITCHED):
        chroma[s] = H.beat_chroma(low[s], LSR, bt_clip, tuning)
        progress(0.30 + 0.35 * (i + 1) / len(PITCHED), "harmony")
    rms = {s: H.beat_rms_db(low[s], LSR, bt_clip) for s in STEMS}
    rms_mix = H.beat_rms_db(sum(low.values()), LSR, bt_clip)

    progress(0.70, "structure")
    spans = S.bar_spans(grid["bars"], len(bt), grid["bpb"])
    bar_rms = {s: S.per_bar(rms[s][:, None], spans)[:, 0] for s in STEMS}
    peak_db = {s: float(np.quantile(rms[s][grid["first_beat"]:grid["last_beat"] + 1], 0.95)) for s in STEMS}
    bar_mix = S.per_bar(rms_mix[:, None], spans)[:, 0]
    pres = {s: S.presence(bar_rms[s], peak_db[s], bar_mix) for s in STEMS}
    top = np.quantile(bar_mix, 0.95)
    energy = np.clip((bar_mix - (top - 30)) / 30, 0, 1)
    active = np.where(np.any(np.vstack([pres[s] for s in STEMS]), axis=0))[0]
    content = (int(active[0]), int(active[-1]) + 1) if len(active) else (0, len(spans))

    # key: bass defines the root, the rest the colour; only count beats where the stem plays
    ksum = np.zeros(12)
    for s, w in (("bass", 1.3), ("other", 1.0), ("vocals", 0.6)):
        on = S.presence(rms[s], peak_db[s], rms_mix)
        c = H.normalize_rows(chroma[s][on])
        ksum += w * c.sum(0)
    tonic, mode, kconf = H.detect_key(ksum)

    # bar features for segmentation: harmony + per-stem energy + overall energy
    bar_ch = S.per_bar(H.normalize_rows(chroma["other"] + chroma["bass"] + 0.5 * chroma["vocals"]), spans)
    en = np.vstack([np.clip((bar_rms[s] - peak_db[s] + 30) / 30, 0, 1) for s in STEMS]).T
    feats = np.hstack([bar_ch * 1.0, en * 0.8, energy[:, None]])
    feats = (feats - feats.mean(0)) / (feats.std(0) + 1e-6)
    beat_to_bar = {b: i for i, b in enumerate(grid["bars"])}
    seg_breaks = [beat_to_bar[a] for a, _ in grid["segments"][1:] if a in beat_to_bar]
    secs = S.sections(feats, energy, pres, seg_breaks, content)

    progress(0.85, "loudness & waveforms")
    loud = {s: _lufs(v) for s, v in stems.items()}
    loud["mix"] = _lufs(mix)
    loud["inst"] = _lufs(stems["drums"] + stems["bass"] + stems["other"])
    pk_dir = song_dir / "peaks"
    pk_dir.mkdir(exist_ok=True)
    for s, v in stems.items():
        peaks(v, SR, PEAKS_PER_SEC).tofile(pk_dir / f"{s}.bin")
    peaks(mix, SR, PEAKS_PER_SEC).tofile(pk_dir / "mix.bin")

    np.savez_compressed(song_dir / "features.npz",
                        **{f"chroma_{s}": chroma[s].astype(np.float32) for s in PITCHED},
                        **{f"rms_{s}": rms[s].astype(np.float32) for s in STEMS},
                        rms_mix=rms_mix.astype(np.float32),
                        energy=energy.astype(np.float32),
                        **{f"pres_{s}": pres[s] for s in STEMS},
                        **{f"barrms_{s}": bar_rms[s].astype(np.float32) for s in STEMS})

    a, b = grid["first_beat"], grid["last_beat"] + 1
    vpres = S.presence(rms["vocals"], peak_db["vocals"], rms_mix)[a:b]
    has_vocals = bool(loud["vocals"] - loud["mix"] > -13 and vpres.mean() > 0.1)
    info = dict(duration=dur, has_vocals=has_vocals, vocal_fraction=round(float(vpres.mean()), 3), bpm=grid["bpm"], bpb=grid["bpb"], stable_tempo=grid["stable"],
                beats=[round(t, 5) for t in grid["beats"]], bars=grid["bars"], segments=grid["segments"],
                first_beat=grid["first_beat"], last_beat=grid["last_beat"], content_bars=list(content),
                tuning=round(tuning, 3), tuning_st=round(tuning_st, 3), key=dict(tonic=tonic, mode=mode, conf=round(kconf, 2),
                                                  name=H.key_name(tonic, mode), camelot=H.camelot(tonic, mode)),
                loudness={k: round(v, 2) for k, v in loud.items()}, stem_peak_db={k: round(v, 1) for k, v in peak_db.items()},
                sections=secs, peaks_per_sec=PEAKS_PER_SEC, grid_version=int(grid.get("grid_version", 1)))
    if grid.get("beat_conf") is not None:
        info["beat_conf"] = grid["beat_conf"]
    if grid.get("beat_content") is not None:
        info["beat_content"] = grid["beat_content"]
    progress(1.0, "done")
    return info


def load_features(song_dir):
    with np.load(song_dir / "features.npz") as z:
        return {k: z[k] for k in z.files}


# ------------------------------------------------------------------ beat-content back-fill (FEAT-005)

def _resolve_song(song):
    """Accept a song id (str), a song_dir (Path), or an info dict and return (sid, song_dir).

    The FEAT-005 contract describes the helper over (song_dir, info); the already-shipped
    /api/songs/{sid}/backfill_beat_content endpoint calls it with a bare song id. Both are supported so the
    live server path and the design-stated signature agree."""
    from .config import SONGS
    if isinstance(song, dict):
        sid = song.get("id")
        return sid, (SONGS / sid if sid else None)
    p = Path(song)
    # a real directory (or something that lives under SONGS) is a song_dir; otherwise treat it as a song id
    if p.exists() and p.is_dir():
        return p.name, p
    if len(p.parts) > 1:
        return p.name, p
    return str(song), SONGS / str(song)


def ensure_beat_content(song, info=None, *, progress=lambda f, m="": None, force=False):
    """Back-fill a song's per-beat 0..100 `beat_content` list for analyses made before it existed, WITHOUT
    re-separating or re-detecting anything. It reuses the stored beat grid and only re-measures onset energy
    on each beat (the same cheap computation analyze() already does at the end of a full run).

    `song` may be a song id, a song_dir Path, or an info dict (the design states the (song_dir, info) form;
    the server calls it with a bare id). Returns a small status dict:
      {"backfilled": True,  "id": sid, "n": len(beat_content)}                        on success
      {"backfilled": False, "id": sid, "defer": True, "reason": <why>}                when a FULL re-analyze
                                                                                       is needed (writes NOTHING)

    DEFER (write nothing, signal 'needs full re-analyze') when the stored info lacks beats / first_beat /
    last_beat, or the drums stem can't be loaded AND there's no grid to fall back on. Mirrors build_grid's
    mix->drums fallback only in the sense that a present grid is enough; here we never have a mix stem to fall
    back to, so a missing drums stem with a present grid still defers (nothing to measure onsets from)."""
    from . import library
    sid, song_dir = _resolve_song(song)
    if info is None:
        info = (library.read(sid) if sid else None) or {}

    def defer(reason):
        return {"backfilled": False, "id": sid, "defer": True, "reason": reason}

    progress(0.05, "checking beat grid")
    if not (info.get("beats") and info.get("first_beat") is not None and info.get("last_beat") is not None):
        return defer("no stored beat grid - needs a full re-analyze")
    if info.get("beat_content") is not None and not force:
        return {"backfilled": False, "id": sid, "already": True, "n": len(info["beat_content"])}

    progress(0.25, "loading drums stem")
    try:
        drums = library.load_stem(sid, "drums")        # (2, n) at SR, true level
    except Exception as e:
        return defer(f"drums stem unloadable ({type(e).__name__}) - needs a full re-analyze")
    if drums is None or getattr(drums, "size", 0) == 0:
        return defer("drums stem empty - needs a full re-analyze")

    progress(0.55, "measuring onset energy")
    drums_mono = np.asarray(drums, np.float32).mean(0)
    env, hop = B.body_env(drums_mono, SR)
    content = B.beat_content(info["beats"], env, hop, SR, int(info["first_beat"]), int(info["last_beat"]))

    # write ONLY beat_content back into song.json (write-then-rename), leaving every other field untouched
    progress(0.85, "saving")
    fresh = library.read(sid) or info
    fresh["beat_content"] = content
    library.write(sid, fresh)
    progress(1.0, "done")
    return {"backfilled": True, "id": sid, "n": len(content)}
