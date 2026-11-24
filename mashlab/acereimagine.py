"""ACE Reimagine: drive ACE Studio from Re-Tex to reimagine mashups, swap genres/instruments/vocals, inject
context-aware layers, or reuse past ACE cloud generations at 0 credits - then import, bar-align, grid-lock, and
drop the stems straight onto the open Re-Tex timeline.

Mirrors mashlab/reimagine.py (YuE2) while adding ACE Studio's strengths:
  - enhancer   : Full Reimagine / AI Cover (music-enhancer) with continuous Adherence <-> Drift (--influence 0..1),
                 source stem/range scoping, multi-axis style builder, and lyrics override. [credits]
  - layer      : Instrument / Stem Swap & Layer Injection (add-a-layer) with optional stem masking (mute drums/bass/
                 other in the reference bounce so ACE generates a replacement locked to your harmony & groove). [credits]
  - voice_swap : Multi-model Voice & Instrument Timbre Swap (voice-changer convert) across 33 voice + 14 instrument
                 models in one render, with transpose, pitch-correction, random-offset drift, and key/scale lock. [FREE]
  - cloud_reuse: Pull any prior ACE Studio cloud generation (music-enhancer / inspire-me) into Re-Tex. [FREE]
"""
import copy
import json
import re
import shutil
import time
from pathlib import Path

import numpy as np

from . import acelink, analyze, audio_io, components, jobs, library, mixer, project as P, reimagine as RI, takealign, warp
from .config import DATA, SR, STEMS
from .events import bus

ACE_DIR = DATA / "ace"

MODES = {
    "enhancer": "Full Reimagine (AI Cover)",
    "layer": "Instrument / Layer Swap",
    "voice_swap": "Voice & Timbre Swap (Free)",
    "cloud_reuse": "Cloud History & Vault (Free)",
}

STT_ENGINES = {
    "whisper_vad": dict(
        id="whisper_vad",
        label="Whisper large-v3 (Default · VAD on)",
        note="Re-Tex's proven default on GPU. Best general accuracy; Silero VAD can drop soft or breathy singing.",
        free=True, gpu=True, needs_language=False, component="whisper",
    ),
    "whisper_novad": dict(
        id="whisper_novad",
        label="Whisper large-v3 (VAD off · breathy/soft vocals)",
        note="Disables Silero VAD and normalizes each vocal phrase separately to catch quiet, whispered, or reverb-drenched lines.",
        free=True, gpu=True, needs_language=False, component="whisper",
    ),
    "ace_v2m": dict(
        id="ace_v2m",
        label="ACE Vocal-to-MIDI V2 (Free · singing-trained)",
        note="ACE Studio's 0-credit singing transcriber. Strong on melodic/reverberant vocals; requires choosing the sung language.",
        free=True, gpu=False, needs_language=True, component="ace_studio",
    ),
    "parakeet_tdt": dict(
        id="parakeet_tdt",
        label="NVIDIA Parakeet-TDT 0.6B v3 (Local ONNX)",
        note="Fast local ONNX recognizer on CPU (zero CUDA conflict). Strong on crisp/dry pop, rock, and rap vocals.",
        free=True, gpu=False, needs_language=False, component="parakeet_tdt",
    ),
}


def stt_engines_info():
    """List selectable vocal STT engines with live component availability."""
    out = []
    for k, meta in STT_ENGINES.items():
        d = dict(meta)
        ck = meta.get("component")
        if ck and ck in components.C:
            ci = components.info(ck)
            d["ok"] = bool(ci["ok"])
            d["problem"] = ci["problem"]
        else:
            d["ok"] = True
            d["problem"] = ""
        out.append(d)
    return out

LAYER_SOUND_TYPES = (
    "instrumental", "drums", "bass", "guitar", "keyboard", "percussion",
    "strings", "synth", "brass", "woodwinds", "fx",
    "vocals", "backing-vocals", "song-track", "custom",
)

# Map ACE add-a-layer sound_type -> Re-Tex stem lane role when importing as an isolated single stem
SOUND_TYPE_TO_STEM = {
    "drums": "drums",
    "percussion": "drums",
    "bass": "bass",
    "vocals": "vocals",
    "backing-vocals": "vocals",
    "song-track": "vocals",
}

KEY_NAMES = ("C", "C#", "Db", "D", "D#", "Eb", "E", "F", "F#", "Gb", "G", "G#", "Ab", "A", "A#", "Bb", "B")
SCALE_NAMES = ("Major", "Minor")


def _slug(text):
    s = re.sub(r"[^a-z0-9]+", "_", str(text or "ace").lower()).strip("_")
    return (s[:44].strip("_")) or "ace"


def _project_key(proj, song_ids):
    """Best-effort musical key & scale guess from the open project / main song (for Voice Changer default)."""
    k = proj.get("key")
    if not k and song_ids:
        info = library.read(song_ids[0]) or {}
        k = info.get("key")
    if not k:
        return "", "Major"
    s = str(k).strip()
    m = re.match(r"^([A-Ga-g][#b]?)\s*(min|minor|m|maj|major)?", s)
    if not m:
        return "", "Major"
    root = m.group(1)[0].upper() + m.group(1)[1:]
    qual = (m.group(2) or "").lower()
    scale = "Minor" if qual in ("m", "min", "minor") else "Major"
    return (root if root in KEY_NAMES else ""), scale


def _project_metrics(proj):
    """(bpm, bpb, total_beats, total_bars, mix_seconds, stems_present)."""
    bpm = float(proj.get("bpm") or 120.0)
    bpb = int(proj.get("bpb") or 4)
    stems_present = sorted({c.get("stem") for c in (proj.get("clips") or []) if not c.get("mute") and c.get("stem")})
    end_beat = max((float(c.get("start", 0)) + float(c.get("len", 0))
                    for c in (proj.get("clips") or []) if not c.get("mute")), default=0.0)
    total_bars = max(1, int(np.ceil(end_beat / max(1, bpb))))
    mix_seconds = round(end_beat * 60.0 / bpm, 1) if bpm > 0 else 0.0
    return bpm, bpb, end_beat, total_bars, mix_seconds, stems_present


def _prep(prep_id):
    return json.loads((ACE_DIR / Path(prep_id).name / "prep.json").read_text(encoding="utf-8"))


def _bounce_subset(work, proj, source_scope="full", mute_stems=None, start_bar=None, end_bar=None, tag="input"):
    """Bounce the requested stem subset and optional 1-based bar range of `proj` to a WAV at -14 LUFS.
    -> (wav_path, start_beat, n_beats, dur_sec)."""
    sub = copy.deepcopy(proj)
    muted_track_ids = {t["id"] for t in (sub.get("tracks") or []) if t.get("mute")}
    scope = str(source_scope or "full").lower()
    mute_set = {str(s).lower() for s in (mute_stems or []) if s}

    active_clips = 0
    for c in sub.get("clips") or []:
        if c.get("mute") or c.get("track") in muted_track_ids:
            c["mute"] = True
            continue
        stem = str(c.get("stem") or "").lower()
        keep = True
        if scope == "vocals":
            keep = (stem == "vocals")
        elif scope == "instrumental":
            keep = (stem != "vocals")
        elif scope in ("drums", "bass", "other"):
            keep = (stem == scope)
        if stem in mute_set:
            keep = False
        c["mute"] = not keep
        if keep:
            active_clips += 1

    if active_clips == 0:
        raise ValueError(f"No unmuted clips match source scope '{scope}'"
                         + (f" (excluding {', '.join(sorted(mute_set))})" if mute_set else ""))

    # Ensure required warped stem renders exist
    reqs = [r for r in P.render_requests(sub) if not warp.is_cached(*r)]
    if reqs:
        warp.render_many(reqs)

    full_wav = work / f"bounce_{tag}_full.wav"
    mixer.bounce(sub, full_wav, target_lufs=-14.0)

    bpm = float(proj.get("bpm") or 120.0)
    bpb = int(proj.get("bpb") or 4)
    spb = 60.0 / bpm
    y = audio_io.load(full_wav, SR)
    total_sec = y.shape[1] / SR

    sb = int(start_bar) if start_bar not in (None, "", 0) else 1
    eb = int(end_bar) if end_bar not in (None, "", 0) else None
    start_beat = max(0.0, float((sb - 1) * bpb))
    t0 = start_beat * spb
    if eb is not None:
        if eb < sb:
            raise ValueError(f"end bar ({eb}) must be >= start bar ({sb})")
        end_beat = float(eb * bpb)
        t1 = min(total_sec, end_beat * spb)
    else:
        t1 = total_sec
        end_beat = t1 / spb

    if t1 - t0 < 1.5:
        raise ValueError(f"Selected range is too short ({t1 - t0:.1f}s). Pick a wider bar range.")

    s0 = max(0, int(round(t0 * SR)))
    s1 = min(y.shape[1], int(round(t1 * SR)))
    out_wav = work / f"bounce_{tag}.wav"
    if s0 > 0 or s1 < y.shape[1]:
        audio_io.save(out_wav, y[:, s0:s1], SR)
    else:
        out_wav = full_wav

    dur_sec = round((s1 - s0) / SR, 3)
    n_beats = round(max(1.0, (s1 - s0) / SR / spb), 4)
    return out_wav, start_beat, n_beats, dur_sec


# ------------------------------------------------------------------ Vocal STT helpers (non-destructive multi-engine)

def _ensure_mix_vocals(work, proj):
    """Ensure work/mix_vocals.wav exists for the open project (bounces vocal clips only on demand)."""
    vpath = work / "mix_vocals.wav"
    if vpath.exists():
        return vpath
    vp = copy.deepcopy(proj)
    muted_tracks = {t["id"] for t in (vp.get("tracks") or []) if t.get("mute")}
    active = 0
    for c in vp.get("clips") or []:
        keep = (not c.get("mute")) and (c.get("track") not in muted_tracks) and (str(c.get("stem") or "").lower() == "vocals")
        c["mute"] = not keep
        if keep:
            active += 1
    if active == 0:
        raise ValueError("This mix has no unmuted vocal clips to transcribe.")
    reqs = [r for r in P.render_requests(vp) if not warp.is_cached(*r)]
    if reqs:
        warp.render_many(reqs)
    mixer.bounce(vp, vpath)
    return vpath


def _format_lines_with_sections(proj, lines, bpm, mix_seconds):
    if not lines:
        return ""
    marks = sorted(proj.get("sections") or [], key=lambda m: m["beat"])
    if marks:
        spb = 60.0 / float(bpm or 120.0)
        secs = [dict(label=m["name"], t0=round(m["beat"] * spb, 3),
                     t1=round((marks[k + 1]["beat"] * spb) if k + 1 < len(marks) else mix_seconds, 3))
                for k, m in enumerate(marks)]
        if marks[0]["beat"] > 0:
            secs.insert(0, dict(label="intro", t0=0.0, t1=secs[0]["t0"]))
        return RI.tag_lyrics(secs, lines)
    return "\n".join(l["text"] for l in lines)


def _transcribe_vocal_wav(wav_path, engine="whisper_vad", bpm=120.0, language="english", time_offset=0.0,
                          progress=lambda f, m="": None, cancelled=lambda: False):
    """Run the chosen opt-in vocal STT engine on `wav_path` -> list of {t0, t1, text, words} shifted by time_offset."""
    eng = str(engine or "whisper_vad").strip().lower()
    if eng not in STT_ENGINES:
        raise ValueError(f"Unsupported STT engine '{eng}'. Choose one of: {', '.join(STT_ENGINES)}")

    if eng == "ace_v2m":
        res = acelink.vocal_to_midi_lyrics(
            wav_path, bpm=bpm, language=language, progress=progress, cancelled=cancelled,
        )
        raw_lines = res.get("lines") or []
        return [
            dict(
                t0=round(float(l["t0"]) + time_offset, 3),
                t1=round(float(l["t1"]) + time_offset, 3),
                text=str(l["text"]).strip(),
                words=list(l.get("words") or []),
            )
            for l in raw_lines if str(l.get("text") or "").strip()
        ]

    from .backends.asr import FasterWhisperLarge, FasterWhisperNoVad, ParakeetTdtAsr
    if eng == "whisper_vad":
        progress(0.15, "transcribing vocals with Whisper large-v3 (VAD on)")
        asr = FasterWhisperLarge()
    elif eng == "whisper_novad":
        progress(0.15, "transcribing vocals with Whisper large-v3 (VAD off)")
        asr = FasterWhisperNoVad()
    elif eng == "parakeet_tdt":
        progress(0.15, "transcribing vocals with NVIDIA Parakeet-TDT 0.6B v3")
        asr = ParakeetTdtAsr()
    else:
        raise ValueError(f"Unsupported STT engine '{eng}'")

    try:
        y = audio_io.load(wav_path, SR).mean(0).astype(np.float32)
        if cancelled():
            raise RuntimeError("cancelled")
        raw = asr.transcribe(y, SR)
    finally:
        asr.unload()

    segs = raw.get("segments", raw) if isinstance(raw, dict) else raw
    out = []
    for s in segs or []:
        txt = str(s.get("text") or "").strip()
        if not txt:
            continue
        t0 = round(float(s.get("t0", s.get("start", 0.0))) + time_offset, 3)
        t1 = round(float(s.get("t1", s.get("end", 0.0))) + time_offset, 3)
        words = [dict(t=round(float(w["t"]) + time_offset, 3), w=w["w"]) for w in (s.get("words") or [])]
        out.append(dict(t0=t0, t1=t1, text=txt, words=words))
    progress(1.0, f"done ({len(out)} lines)")
    return out


# ------------------------------------------------------------------ Stage 1: prepare

def prepare(proj, prompt="", skip_lyrics=False, stt_engine="whisper_vad", stt_language="english",
            progress=lambda f, m="": None, cancelled=lambda: False):
    """Stage 1: snapshot the open Re-Tex project, bounce full mix, extract lyrics + heard song traits, and
    check ACE Studio status so the user can configure any of the 4 modes without re-preparing."""
    eng = str(stt_engine or "whisper_vad").strip().lower()
    if eng not in STT_ENGINES:
        raise ValueError(f"Unsupported STT engine '{eng}'")
    lang = str(stt_language or "english").strip().lower()

    songs = RI._songs_of(proj)
    bpm, bpb, total_beats, total_bars, mix_seconds, stems_present = _project_metrics(proj)
    key_root, key_scale = _project_key(proj, songs)

    prep_id = f"aceprep-{time.strftime('%Y%m%d-%H%M%S')}"
    work = ACE_DIR / prep_id
    work.mkdir(parents=True, exist_ok=True)

    # Save exact open project snapshot so perform() can bounce any stem-mask or bar-range on demand
    (work / "project.json").write_text(json.dumps(proj, indent=1), encoding="utf-8")

    progress(0.05, "rendering the mix")
    reqs = [r for r in P.render_requests(proj) if not warp.is_cached(*r)]
    if reqs:
        warp.render_many(reqs)
    mix = work / "mix.wav"
    mixer.bounce(proj, mix, target_lufs=-14.0)
    mix_seconds = round(audio_io.load(mix, SR).shape[1] / SR, 1)

    lines, lyrics_from, lyrics_text = [], "none", ""
    if not skip_lyrics and "vocals" in stems_present:
        progress(0.25, f"hearing the lyrics ({STT_ENGINES[eng]['label']})")
        if eng == "whisper_vad":
            # Exact original proven path (untouched default)
            lines = RI.sung_lines(proj, work)
            from . import userlyrics
            typed, untyped = userlyrics.mix_lines(proj)
            lyrics_from = ("mixed" if untyped else "yours") if typed else ("whisper_vad" if lines else "none")
        else:
            vpath = _ensure_mix_vocals(work, proj)
            from . import userlyrics
            typed, untyped = userlyrics.mix_lines(proj)
            if typed and not untyped:
                lines = [{k: l[k] for k in ("t0", "t1", "text", "words")} for l in typed]
                lyrics_from = "yours"
            else:
                heard = _transcribe_vocal_wav(
                    vpath, engine=eng, bpm=bpm, language=lang, time_offset=0.0,
                    progress=lambda f, m="": progress(0.25 + 0.42 * f, m), cancelled=cancelled,
                )
                if typed:
                    def _cov(s):
                        return sum(max(0.0, min(s["t1"], l["t1"]) - max(s["t0"], l["t0"])) for l in typed) > 0.5 * max(s["t1"] - s["t0"], 1e-3)
                    lines = sorted(
                        [{k: l[k] for k in ("t0", "t1", "text", "words")} for l in typed] + [s for s in heard if not _cov(s)],
                        key=lambda l: l["t0"],
                    )
                    lyrics_from = "mixed"
                else:
                    lines = heard
                    lyrics_from = eng if lines else "none"
        lyrics_text = _format_lines_with_sections(proj, lines, bpm, mix_seconds)
    elif skip_lyrics:
        lyrics_from = "skipped"

    progress(0.70, "reading what the songs are")
    sung = bool(lines or ("vocals" in stems_present and not skip_lyrics))
    style = RI.style_prompt(songs, prompt, sung=sung)
    heard_info = RI.heard(songs, sung=sung)

    progress(0.85, "checking ACE Studio connection")
    ace_st = acelink.status()
    marks = sorted(proj.get("sections") or [], key=lambda m: m["beat"])
    sections = []
    for k, m in enumerate(marks):
        b0 = int(round(float(m["beat"]) / max(1, bpb))) + 1
        b1 = int(round(float(marks[k + 1]["beat"]) / max(1, bpb))) if k + 1 < len(marks) else total_bars
        sections.append(dict(label=m["name"], start_bar=b0, end_bar=max(b0, b1)))

    prep = dict(
        prep_id=prep_id,
        engine="ace",
        title=proj.get("name", "mix"),
        bpm=bpm,
        bpb=bpb,
        key=key_root,
        scale=key_scale,
        total_bars=total_bars,
        mix_seconds=mix_seconds,
        stems_present=stems_present,
        songs=songs,
        style=style,
        lyrics=lyrics_text,
        lyrics_from=lyrics_from,
        stt_engine=eng,
        stt_language=lang,
        heard=heard_info,
        sung=sung,
        lines=lines,
        sections=sections,
        prompt=prompt,
        ace_status=ace_st,
        created=time.time(),
    )
    (work / "prep.json").write_text(json.dumps(prep, indent=1), encoding="utf-8")
    progress(1.0, "ready to review")
    return prep


def transcribe_prep(prep_id, engine="whisper_vad", language="english", start_bar=None, end_bar=None,
                    progress=lambda f, m="": None, cancelled=lambda: False):
    """Re-transcribe a prepared mix's vocal stem (or selected bar slice) using any of the 4 opt-in STT engines
    without re-preparing the mix or spending ACE credits. Updates prep.json in place."""
    eng = str(engine or "whisper_vad").strip().lower()
    if eng not in STT_ENGINES:
        raise ValueError(f"Unsupported STT engine '{eng}'. Choose one of: {', '.join(STT_ENGINES)}")
    lang = str(language or "english").strip().lower()
    if eng == "ace_v2m" and lang not in acelink.ACE_V2M_LANGUAGES:
        raise ValueError(f"ACE Vocal-to-MIDI language must be one of {', '.join(acelink.ACE_V2M_LANGUAGES)}")

    prep = _prep(prep_id)
    work = ACE_DIR / prep["prep_id"]
    proj = json.loads((work / "project.json").read_text(encoding="utf-8"))
    bpm = float(prep.get("bpm") or 120.0)
    bpb = int(prep.get("bpb") or 4)
    total_bars = int(prep.get("total_bars") or 1)
    mix_seconds = float(prep.get("mix_seconds") or 0.0)

    t_start = time.time()
    progress(0.05, "loading mix vocal stem")
    vpath = _ensure_mix_vocals(work, proj)

    sb = int(start_bar) if start_bar not in (None, "", 0) else 1
    eb = int(end_bar) if end_bar not in (None, "", 0) else total_bars
    if eb < sb:
        raise ValueError(f"end bar ({eb}) must be >= start bar ({sb})")
    scoped = (sb > 1 or eb < total_bars)
    time_offset = 0.0
    target_wav = vpath

    if scoped:
        spb = 60.0 / bpm
        t0 = max(0.0, (sb - 1) * bpb * spb)
        y = audio_io.load(vpath, SR)
        total_sec = y.shape[1] / SR
        t1 = min(total_sec, eb * bpb * spb)
        if t1 - t0 < 1.5:
            raise ValueError(f"Selected bar range is too short ({t1 - t0:.1f}s) to transcribe.")
        s0 = max(0, int(round(t0 * SR)))
        s1 = min(y.shape[1], int(round(t1 * SR)))
        target_wav = work / f"mix_vocals_bars_{sb}_{eb}.wav"
        audio_io.save(target_wav, y[:, s0:s1], SR)
        time_offset = round(s0 / SR, 3)

    lines = _transcribe_vocal_wav(
        target_wav, engine=eng, bpm=bpm, language=lang, time_offset=time_offset,
        progress=lambda f, m="": progress(0.08 + 0.90 * f, m), cancelled=cancelled,
    )
    elapsed_s = round(time.time() - t_start, 2)
    lyrics_text = "\n".join(l["text"] for l in lines) if scoped else _format_lines_with_sections(proj, lines, bpm, mix_seconds)

    prep["lines"] = lines
    prep["lyrics"] = lyrics_text
    prep["lyrics_from"] = eng if lines else "none"
    prep["stt_engine"] = eng
    prep["stt_language"] = lang
    prep["stt_scoped_bars"] = [sb, eb] if scoped else None
    (work / "prep.json").write_text(json.dumps(prep, indent=1), encoding="utf-8")
    progress(1.0, f"done ({len(lines)} lines in {elapsed_s:.1f}s)")

    return dict(
        prep_id=prep["prep_id"],
        engine=eng,
        engine_label=STT_ENGINES[eng]["label"],
        language=lang,
        lines=lines,
        lyrics=lyrics_text,
        lyrics_from=prep["lyrics_from"],
        line_count=len(lines),
        elapsed_s=elapsed_s,
        scoped_bars=([sb, eb] if scoped else None),
    )


# ------------------------------------------------------------------ Stage 2: perform

def perform(prep_id, mode="enhancer", style="", lyrics=None, lyrics_mode="mix",
            influence=0.50, source_scope="full", mute_stems=None,
            start_bar=None, end_bar=None,
            # add-a-layer options
            sound_type="instrumental", instrument="",
            # voice-changer options (0 credits!)
            model_ids=None, semitones=0, pitch_correction=20, random_offset=0,
            correct_to_key="", correct_to_scale="Major", remove_instrument=False, remove_reverb=False,
            progress=lambda f, m="": None, cancelled=lambda: False):
    """Stage 2: run ACE Studio on the prepared mix in the selected mode and pull finished takes into data/ace/."""
    if mode not in ("enhancer", "layer", "voice_swap"):
        raise ValueError(f"unsupported ACE perform mode: {mode}")

    prep = _prep(prep_id)
    work = ACE_DIR / prep["prep_id"]
    proj = json.loads((work / "project.json").read_text(encoding="utf-8"))
    bpm = float(prep["bpm"])

    # 1. Fast duration check for Music Enhancer (3s..240s) before bouncing or opening ACE Studio
    bpb = int(prep.get("bpb") or 4)
    spb = 60.0 / bpm
    sb_num = int(start_bar) if start_bar not in (None, "", 0) else 1
    eb_num = int(end_bar) if end_bar not in (None, "", 0) else int(prep.get("total_bars") or 1)
    est_sec = max(0.0, (eb_num - sb_num + 1) * bpb * spb)
    if mode == "enhancer" and est_sec > 240.5:
        max_bars = max(1, int(235.0 // (bpb * spb)))
        raise ValueError(
            f"Selected range is ~{est_sec:.1f}s (bars {sb_num}–{eb_num}), which exceeds ACE Studio Music Enhancer's "
            f"240s (4:00) maximum. Set Bars to at most {max_bars} bars (e.g. {sb_num}–{sb_num + max_bars - 1})."
        )

    # 2. Bounce exact source scope + stem mask + bar range
    progress(0.04, "preparing source audio for ACE Studio")
    effective_scope = source_scope or ("vocals" if mode == "voice_swap" and "vocals" in prep.get("stems_present", []) else "full")
    effective_mutes = list(mute_stems or [])
    # True instrumental enhancer: mute vocals from the reference bounce so ACE has no vocals to auto-transcribe
    if mode == "enhancer" and lyrics_mode == "instrumental" and effective_scope == "full":
        if "vocals" not in [m.lower() for m in effective_mutes]:
            effective_mutes.append("vocals")
    tag = f"{mode}_{int(time.time()) % 100000}"
    wav_path, start_beat, n_beats, dur_sec = _bounce_subset(
        work, proj, source_scope=effective_scope, mute_stems=effective_mutes,
        start_bar=start_bar, end_bar=end_bar, tag=tag,
    )
    t0_sec = round(start_beat * spb, 3)
    t1_sec = round(t0_sec + dur_sec, 3)
    if mode == "enhancer" and not (3.0 <= dur_sec <= 240.0):
        max_bars = max(1, int(235.0 // (bpb * spb)))
        raise ValueError(
            f"Bounced range is {dur_sec:.1f}s, outside ACE Music Enhancer's 3s–240s (4:00) window. "
            f"Narrow Bars to <= {max_bars} bars (e.g. {sb_num}–{min(eb_num, sb_num + max_bars - 1)})."
        )

    # Slice timed lyric lines to the bounced window (relative to t0_sec=0) for accurate ACE singing & post-retime
    sliced_lines = []
    for l in (prep.get("lines") or []):
        mid = 0.5 * (float(l.get("t0", 0)) + float(l.get("t1", 0)))
        if t0_sec <= mid <= t1_sec and str(l.get("text", "")).strip():
            sliced_lines.append(dict(
                t0=round(max(0.0, float(l["t0"]) - t0_sec), 3),
                t1=round(min(dur_sec, float(l["t1"]) - t0_sec), 3),
                text=str(l["text"]).strip(),
            ))

    # 3. Safely open a clean staging project in ACE Studio at Re-Tex's exact BPM
    progress(0.10, "connecting to ACE Studio")
    acelink.ensure_clean_stage(bpm=bpm)
    progress(0.16, "importing mix into ACE Studio")
    imp = acelink.import_audio(wav_path, track_index=0, pos="0t")
    src_uuid = imp.get("trackUuid") or imp.get("uuid")
    geom = imp.get("geometry") or {}
    end_tick = int(geom.get("end") or imp.get("naturalDur") or 0)

    rid = f"acerender-{time.strftime('%Y%m%d-%H%M%S')}-{mode}"
    out = ACE_DIR / rid
    takes_out = []
    credit_billed = False
    target_stem = None
    summary_prompt = " ".join(str(style or "").split())

    # 3. Execute mode-specific ACE Studio operation
    if mode == "enhancer":
        credit_billed = True
        inf = round(max(0.0, min(1.0, float(influence))), 2)
        if end_tick <= 0:
            raise acelink.AceError("Imported mix had zero tick length in ACE Studio", code="ZERO_LENGTH")
        acelink.set_arrangement_selection(0, end_tick, begin_index=0, end_index=0)

        # Lyrics handling:
        # - "instrumental": vocals were already muted from the bounce above; strip contradictory vocal descriptors
        #   from the prompt AND pass "[Instrumental]" so ACE does NOT fall back to auto-transcribing lyrics!
        # - "mix": if user kept prep lyrics unmodified and sliced a bar range, send only the sliced window's lines.
        # - "custom": send user's edited lyrics verbatim.
        # - "auto": omit --lyrics so ACE transcribes from the audio itself.
        send_lyrics = None
        send_prompt = summary_prompt or ""
        if lyrics_mode == "instrumental":
            send_prompt = RI._no_vocals(send_prompt) if send_prompt else ""
            send_prompt = f"Instrumental, {send_prompt}, no vocals, no singing" if send_prompt else "Instrumental, no vocals, no singing"
            send_lyrics = "[Instrumental]"
        elif lyrics_mode == "mix":
            raw_lyr = str(lyrics if lyrics is not None else prep.get("lyrics") or "").strip()
            unchanged = (raw_lyr == str(prep.get("lyrics") or "").strip())
            if unchanged and sliced_lines and (t0_sec > 0.5 or t1_sec < float(prep.get("mix_seconds") or 0) - 1.0):
                send_lyrics = "\n".join(l["text"] for l in sliced_lines)
            else:
                send_lyrics = raw_lyr or None
        elif lyrics_mode == "custom":
            send_lyrics = str(lyrics or "").strip() or None
        elif lyrics_mode == "auto":
            send_lyrics = None

        # Anchor project tempo in the style prompt when not already specified so fast genre tags don't speed up the track
        if send_prompt and not re.search(r"\b\d{2,3}\s*bpm\b", send_prompt, re.I) and inf <= 0.70:
            send_prompt = f"{send_prompt}, steady {int(round(bpm))} BPM"
        send_prompt = send_prompt.strip() or None

        run_title = f"{prep['title'][:50]} (ACE {int(round(inf * 100))}%)"
        progress(0.22, f"ACE Studio: launching Full Reimagine (drift {int(round(inf * 100))}%)")
        launch = acelink.music_enhancer(prompt=send_prompt, lyrics=send_lyrics, influence=inf, title=run_title)
        job_id = launch["jobId"]
        acelink.wait_job(job_id, progress=progress, lo=0.25, hi=0.80, cancelled=cancelled,
                         label="ACE Music Enhancer", poll_s=3.0, timeout_s=900)

        progress(0.81, "ACE Studio: settling durable take IDs")
        result_ids = acelink.wait_settled_result_ids(job_id, kit="music-enhancer", title=run_title, timeout_s=35.0)

        for k, res_id in enumerate(result_ids, 1):
            if not res_id:
                continue
            out.mkdir(parents=True, exist_ok=True)
            dst = out / f"take-{k}.flac"
            progress(0.82 + 0.14 * ((k - 1) / max(1, len(result_ids))), f"ACE Studio: fetching take {k} of {len(result_ids)}")
            acelink.fetch_staged_result(res_id, dst, kit="music-enhancer", progress=progress,
                                        lo=0.82, hi=0.96, cancelled=cancelled)
            sec = round(audio_io.load(dst, SR).shape[1] / SR, 1)
            takes_out.append(dict(
                n=k, file=dst.name, label=f"Take {k}", result_id=str(res_id),
                seconds=sec, top=(k == 1),
            ))
            acelink.vault_register(
                kind="cover", source="music-enhancer", title=f"{run_title} (Take {k})",
                prompt=send_prompt or "", tags=[t.strip() for t in (send_prompt or "").split(",") if t.strip()][:10],
                bpm=bpm, key=prep.get("key") or None, duration_sec=sec,
                result_id=str(res_id), local_path=str(dst),
                notes=f"Re-Tex ACE Full Reimagine (influence={inf}, scope={effective_scope})",
            )

    elif mode == "layer":
        credit_billed = True
        st = str(sound_type or "instrumental").strip().lower()
        if st not in LAYER_SOUND_TYPES:
            raise ValueError(f"sound_type must be one of {', '.join(LAYER_SOUND_TYPES)}")
        target_stem = SOUND_TYPE_TO_STEM.get(st, "other")
        tr = acelink.create_audio_track(name=f"Re-Tex {st}")
        target_uuid = tr.get("trackUuid") or tr.get("uuid")
        if not target_uuid:
            raise acelink.AceError("ACE Studio didn't return a trackUuid for the new layer track", code="NO_TRACK")

        send_lyrics = None
        if st in ("song-track", "vocals", "backing-vocals"):
            txt = str(lyrics if lyrics is not None else prep.get("lyrics") or "").strip()
            send_lyrics = txt or None

        progress(0.22, f"ACE Studio: generating {st} layer")
        launch = acelink.add_a_layer(
            target_uuid, "0t", f"{dur_sec:.3f}s", sound_type=st,
            instrument=(instrument if st == "custom" else None),
            prompt=summary_prompt or None, lyrics=send_lyrics,
        )
        acelink.wait_job(launch["jobId"], progress=progress, lo=0.25, hi=0.85, cancelled=cancelled,
                         label=f"ACE Add A Layer ({st})", poll_s=2.5, timeout_s=600)

        progress(0.88, "ACE Studio: exporting generated layer")
        out.mkdir(parents=True, exist_ok=True)
        dst = out / "take-1.flac"
        acelink.export_track(target_uuid, dst, from_t="0t", to_t=f"{dur_sec:.3f}s")
        sec = round(audio_io.load(dst, SR).shape[1] / SR, 1)

        # Archive durable WAV copy into user's ACE Studio Library\layers\
        _, vault = acelink._cfg()
        arch_dir = Path(vault) / "layers"
        arch_dir.mkdir(parents=True, exist_ok=True)
        arch_wav = arch_dir / f"{int(round(bpm))}bpm_{_slug(prep['title'])}_{_slug(st)}_{time.strftime('%Y%m%d_%H%M%S')}.wav"
        try:
            audio_io.save(arch_wav, audio_io.load(dst, SR), SR)
        except Exception:
            arch_wav = dst

        label = f"{instrument or st} layer"
        takes_out.append(dict(n=1, file=dst.name, label=label, sound_type=st, target_stem=target_stem, seconds=sec, top=True))
        acelink.vault_register(
            kind="layer", source="add-a-layer", title=f"{prep['title'][:45]} - {label}",
            prompt=summary_prompt, instrument=(instrument or st),
            tags=[st] + [t.strip() for t in summary_prompt.split(",") if t.strip()][:8],
            bpm=bpm, key=prep.get("key") or None, duration_sec=sec, local_path=str(arch_wav),
            notes=f"Re-Tex ACE Layer Swap (muted={mute_stems or []}, bars={start_bar or 1}-{end_bar or prep['total_bars']})",
        )

    elif mode == "voice_swap":
        credit_billed = False
        mids = [int(x) for x in (model_ids or []) if str(x).strip()]
        if not mids:
            raise ValueError("Pick at least one Voice Changer model (1 to 6 models).")
        mids = list(dict.fromkeys(mids))[:6]

        roster = {int(m["id"]): m for m in acelink.voice_changer_models()}
        # Decide target stem: vocals unless every selected model is an instrument model
        all_inst = all((roster.get(mid) or {}).get("kind") == "instrument" for mid in mids)
        target_stem = "other" if all_inst else "vocals"

        progress(0.22, f"ACE Studio (Free): converting across {len(mids)} model{'s' if len(mids) > 1 else ''}")
        launch = acelink.voice_changer_convert(
            track_uuids=[src_uuid], from_t="0t", to_t=f"{dur_sec:.3f}s", model_ids=mids,
            semitones=int(semitones or 0), pitch_correction=int(pitch_correction if pitch_correction is not None else 20),
            random_offset=int(random_offset or 0),
            correct_to_key=(str(correct_to_key).strip() or None),
            correct_to_scale=(str(correct_to_scale).strip() or "Major"),
            remove_instrument=bool(remove_instrument), remove_reverb=bool(remove_reverb),
        )
        acelink.wait_job(launch["jobId"], progress=progress, lo=0.25, hi=0.80, cancelled=cancelled,
                         label="ACE Voice Changer (Free)", poll_s=2.0, timeout_s=600)

        tuuids = launch.get("trackUuids") or []
        for k, (mid, tuuid) in enumerate(zip(mids, tuuids), 1):
            minfo = roster.get(mid) or {"name": f"Model {mid}", "kind": "voice", "tags": []}
            out.mkdir(parents=True, exist_ok=True)
            dst = out / f"take-{k}.flac"
            progress(0.80 + 0.16 * ((k - 1) / max(1, len(mids))), f"ACE Studio: exporting {minfo['name']}")
            acelink.export_track(tuuid, dst, from_t="0t", to_t=f"{dur_sec:.3f}s")
            sec = round(audio_io.load(dst, SR).shape[1] / SR, 1)
            stem_for_take = "other" if minfo.get("kind") == "instrument" else "vocals"
            takes_out.append(dict(
                n=k, file=dst.name, label=f"{minfo['name']} ({minfo.get('kind', 'voice')})",
                model_id=mid, model_name=minfo["name"], model_kind=minfo.get("kind", "voice"),
                target_stem=stem_for_take, seconds=sec, top=(k == 1),
            ))
            acelink.vault_register(
                kind=("layer" if stem_for_take == "other" else "vocal"),
                source="voice-changer",
                title=f"{prep['title'][:45]} - {minfo['name']}",
                prompt=f"voice-changer semitones={semitones} pitch={pitch_correction} drift={random_offset}",
                instrument=(minfo["name"] if stem_for_take == "other" else None),
                tags=[minfo["name"].lower(), minfo.get("kind", "voice"), "free"] + list(minfo.get("tags") or [])[:5],
                bpm=bpm, key=(correct_to_key or prep.get("key") or None), duration_sec=sec,
                local_path=str(dst), credit_cost=0,
                notes="Re-Tex ACE Voice & Timbre Swap (0 credits)",
            )

    if not takes_out:
        raise RuntimeError("ACE Studio produced no usable audio takes")

    meta = dict(
        render_id=rid,
        engine="ace",
        prep_id=prep["prep_id"],
        mode=mode,
        mode_label=MODES[mode],
        credit_billed=credit_billed,
        influence=round(float(influence), 2) if mode == "enhancer" else None,
        source_scope=effective_scope,
        mute_stems=effective_mutes,
        bounce_file=wav_path.name,
        t0_sec=t0_sec,
        sliced_lines=sliced_lines,
        start_bar=int(start_bar) if start_bar not in (None, "", 0) else 1,
        end_bar=int(end_bar) if end_bar not in (None, "", 0) else prep["total_bars"],
        start_beat=round(float(start_beat), 4),
        n_beats=round(float(n_beats), 4),
        sound_type=sound_type if mode == "layer" else None,
        instrument=instrument if mode == "layer" else None,
        target_stem=target_stem,
        lyrics_mode=lyrics_mode if mode == "enhancer" else None,
        style=send_prompt if mode == "enhancer" else summary_prompt,
        lyrics=str(send_lyrics if mode == "enhancer" and send_lyrics is not None else (lyrics if lyrics is not None else prep.get("lyrics") or "")),
        bpm=bpm,
        songs=prep["songs"],
        title=prep["title"],
        takes=takes_out,
        created=time.time(),
        saved={},
    )
    (out / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    progress(1.0, "done")
    return meta


def import_cloud_take(result_id, title="", kit="music-enhancer", prep_id=None, bpm=120.0,
                      progress=lambda f, m="": None, cancelled=lambda: False):
    """Fetch a previously generated ACE cloud result (0 credits!) into data/ace/ so it can be auditioned,
    stem-separated, bar-aligned, and dropped onto the open Re-Tex mix."""
    acelink.ensure_running()
    rid = f"acerender-{time.strftime('%Y%m%d-%H%M%S')}-cloud"
    out = ACE_DIR / rid
    out.mkdir(parents=True, exist_ok=True)
    dst = out / "take-1.flac"
    acelink.fetch_staged_result(result_id, dst, kit=kit, progress=progress, lo=0.05, hi=0.92, cancelled=cancelled)
    sec = round(audio_io.load(dst, SR).shape[1] / SR, 1)

    prep_bpm = float(bpm or 120.0)
    songs = []
    if prep_id:
        try:
            p = _prep(prep_id)
            prep_bpm = float(p.get("bpm") or prep_bpm)
            songs = list(p.get("songs") or [])
        except (OSError, ValueError):
            prep_id = None

    meta = dict(
        render_id=rid,
        engine="ace",
        prep_id=prep_id,
        mode="cloud_reuse",
        mode_label=MODES["cloud_reuse"],
        credit_billed=False,
        influence=None,
        source_scope="full",
        mute_stems=[],
        start_bar=1,
        end_bar=None,
        start_beat=0.0,
        n_beats=round(sec / (60.0 / prep_bpm), 4),
        target_stem=None,
        style=f"Reused from ACE Studio {kit} history ({result_id})",
        lyrics="",
        bpm=prep_bpm,
        songs=songs,
        title=title or f"ACE Cloud ({result_id})",
        takes=[dict(n=1, file=dst.name, label=title or "Cloud Take", result_id=str(result_id), seconds=sec, top=True)],
        created=time.time(),
        saved={},
    )
    (out / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    progress(1.0, "ready")
    return meta


def renders(limit=25):
    """Recent ACE Studio reimaginings, newest first."""
    if not ACE_DIR.exists():
        return []
    out = []
    for d in sorted(ACE_DIR.glob("acerender-*"), key=lambda d: d.stat().st_mtime, reverse=True):
        try:
            m = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        m.setdefault("created", (d / "meta.json").stat().st_mtime)
        m.setdefault("mode_label", MODES.get(m.get("mode", "enhancer"), "ACE Reimagine"))
        m["saved"] = {n: sid for n, sid in (m.get("saved") or {}).items() if library.read(sid)}
        out.append(m)
        if len(out) >= limit:
            break
    return out


# ------------------------------------------------------------------ Stage 3: save + align + drop into Re-Tex mix

def save(render_id, takes, add_to=None, drop_mode="all", retime=False):
    """Import chosen ACE takes into Re-Tex's library, bar-align + lock to the mix, and emit 'reimagine' event.
    - For `layer` and `voice_swap`, imports as a clean isolated single stem (no unnecessary 4-stem re-separation!).
    - For `enhancer` and `cloud_reuse`, runs full Blend separation + bar-for-bar alignment + grid lock.
    drop_mode: 'vocals' | 'instrumental' | 'all' | 'single'."""
    d = ACE_DIR / Path(render_id).name
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    prep = _prep(meta["prep_id"]) if meta.get("prep_id") else None
    is_single = meta.get("mode") in ("layer", "voice_swap")
    out = []

    for n in takes:
        t = next(x for x in meta["takes"] if x["n"] == int(n))
        when = re.search(r"-(\d{2})(\d{2})\d{2}-", meta["render_id"])
        stamp = f", {when.group(1)}:{when.group(2)}" if when else ""
        short_label = t.get("label") or f"take {t['n']}"
        title = f"{meta['title'][:52]} (ACE {short_label}{stamp})"

        if is_single:
            stem_role = t.get("target_stem") or meta.get("target_stem") or "other"
            sid, job = library.import_stems({stem_role: d / t["file"]}, title, source=str(d / t["file"]))
            aj = _single_stem_align_job(sid, title, meta, stem_role, add_to)
        else:
            sid, job = library.import_audio(d / t["file"], title)
            aj = _full_align_job(sid, title, meta, prep, add_to, drop_mode=drop_mode, retime=retime)
        out.append((sid, job, aj))
        meta.setdefault("saved", {})[str(t["n"])] = sid

    (d / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")

    # File in the first source song's library folder (matches reimagine.py)
    first = meta["songs"][0] if meta.get("songs") else None
    if first:
        f = library.get_folders()
        folder = f["assign"].get(first)
        if folder:
            for sid, _, _ in out:
                f["assign"][sid] = folder
            library.set_folders(f)
    return out


def _single_stem_align_job(sid, title, meta, stem_role, add_to):
    """Wait for single-stem import, apply exact project BPM grid, and publish 'reimagine' to drop cleanly at start_beat."""
    def work(job):
        while True:
            info = library.read(sid) or {}
            if info.get("status") == "ready":
                break
            if info.get("status") == "error" or not info:
                raise RuntimeError(f"{title} didn't import: {info.get('error')}")
            if job.cancelled:
                return None
            time.sleep(1.5)
        bpm = float(meta["bpm"])
        stems = {s: library.load_stem(sid, s) for s in STEMS}
        dur_s = min(v.shape[1] for v in stems.values()) / SR
        n_beats = float(meta.get("n_beats") or (dur_s / (60.0 / bpm)))
        spb = 60.0 / bpm
        # Build exact steady grid at project BPM starting at t=0 (ACE generated directly against our bounced BPM!)
        nb = max(2, int(np.floor(dur_s / spb)) + 1)
        beats = [round(i * spb, 5) for i in range(nb)]
        bars = [beats[i] for i in range(0, nb, 4)]
        g = dict(bpm=bpm, beats=beats, bars=bars, bpb=4, first_beat=0.0)
        job.update(0.35, "applying mix grid")
        new = analyze.analyze(library.song_dir(sid), stems, job.sub(0.35, 0.90, "analysing on the mix's grid: "), grid=g)
        library.apply_new_grid(sid, info, new)
        out = library.read(sid)
        bus.publish(
            "reimagine", engine="ace", sid=sid, title=title, aligned=True, project=add_to,
            first_beat=0.0, start_beat=float(meta.get("start_beat") or 0.0),
            n_beats=n_beats, bpm=bpm, lock=False,
            drop_mode="single", target_stem=stem_role, vocals_only=(stem_role == "vocals"), retime=[],
        )
        return library.summary(out)
    return jobs.submit("align", f"Placing {title} on the mix", work, ref=sid, gpu=False)


def _full_align_job(sid, title, meta, prep, add_to, drop_mode="all", retime=False):
    """After Blend stem separation of a Full Reimagine take: line up bar-for-bar with the mix, lock to the grid,
    optionally re-time vocals line-by-line, and publish 'reimagine' with drop_mode ('vocals'|'instrumental'|'all')."""
    def work(job):
        while True:
            info = library.read(sid) or {}
            if info.get("status") == "ready":
                break
            if info.get("status") == "error" or not info:
                raise RuntimeError(f"{title} didn't import: {info.get('error')}")
            if job.cancelled:
                return None
            time.sleep(2)

        bpm = float(meta["bpm"])
        stems = {s: library.load_stem(sid, s) for s in STEMS}
        n = min(v.shape[1] for v in stems.values())
        start_beat = float(meta.get("start_beat") or 0.0)
        first_beat = 0.0
        n_beats = float(meta.get("n_beats") or int((n / SR) / (60.0 / bpm)))
        locked = False

        # Align against the exact bounced WAV slice sent to ACE (works for ANY bar range, not just bar 1!)
        ref_wav = None
        if prep:
            bf = meta.get("bounce_file")
            if bf and (ACE_DIR / prep["prep_id"] / Path(bf).name).exists():
                ref_wav = ACE_DIR / prep["prep_id"] / Path(bf).name
            elif (ACE_DIR / prep["prep_id"] / "mix.wav").exists() and start_beat == 0.0:
                ref_wav = ACE_DIR / prep["prep_id"] / "mix.wav"

        if ref_wav and ref_wav.exists():
            job.update(0.08, "lining up bar-for-bar with the mix")
            mix = audio_io.load(ref_wav, SR).mean(0)
            n_beats = int(round(len(mix) / SR / (60.0 / bpm)))
            take = sum(v[:, :n] for v in stems.values()).mean(0)
            tau = takealign.align_bars(mix, take, SR, bpm, n_beats,
                                       dict(beats=info["beats"], bars=info["bars"], bpb=info.get("bpb", 4)))
            g = takealign.grid(tau, n / SR, bpm)
            first_beat = float(g["first_beat"])
            new = analyze.analyze(library.song_dir(sid), stems, job.sub(0.20, 0.85, "analysing on the mix's grid: "), grid=g)
            library.apply_new_grid(sid, info, new)
            from . import lock
            job.update(0.88, "locking it to the mix's grid")
            lock.make(sid, check=False)
            t = time.time()
            while not lock.current(library.read(sid) or {}):
                locking = any(j.get("kind") == "lock" and j.get("ref") == sid and j.get("status") in ("queued", "running")
                              for j in jobs.active())
                if not locking and time.time() - t > 5 and not lock.current(library.read(sid) or {}):
                    raise RuntimeError(f"{title} couldn't be locked to the mix's grid")
                if job.cancelled or time.time() - t > 1800:
                    return None
                time.sleep(2)
            locked = True

        plan = []
        if retime and prep and drop_mode in ("vocals", "all"):
            job.update(0.94, "moving the new vocals line by line to where the mix sings them")
            try:
                retime_prep = dict(prep)
                if meta.get("sliced_lines"):
                    retime_prep["lines"] = meta["sliced_lines"]
                    retime_prep.pop("placed", None)
                plan = RI.retime_clips(retime_prep, sid, bpm)
            except Exception as e:
                print(f"ACE re-timing {title} skipped: {type(e).__name__}: {e}")

        out = library.read(sid)
        bus.publish(
            "reimagine", engine="ace", sid=sid, title=title, aligned=True, project=add_to,
            first_beat=first_beat, start_beat=start_beat, n_beats=n_beats, bpm=bpm,
            lock=locked, drop_mode=drop_mode, vocals_only=(drop_mode == "vocals"), retime=plan,
        )
        return library.summary(out)
    return jobs.submit("align", f"Lining up {title} with the mix", work, ref=sid, gpu=True)
