"""Reimagine: a mix's aligned blend of two songs, re-sung and re-instrumented by YuE2 as one new production - a
cover of the mashup that keeps recognisable parts of both songs. In stages, so the user can edit in between:

prepare  The mix is bounced as it plays (both songs on one tempo and key, stems lined up measure on measure,
         vocals in whole lines). Its vocal tracks alone are transcribed (Whisper): the lyrics in the order they're
         sung, both singers, with their times. YuE2 Studio transcribes the mix itself, melody and chords
         (SheetSage2 "full"), so the plan YuE2 performs is the mashup's own: both songs' tunes over the blend's
         harmony, bar for bar. Each sung line is filed under the score section it falls in (by time), so the words
         go where they're sung - not all into the first section. The DJ model writes the style prompt.
         -> the user reviews / edits the lyrics (with section tags) and the style; fit() checks a draft.
perform  YuE2 re-performs it all (cover_full, both voices) at the mix's tempo and key; N takes, ranked by the
         Studio. Takes are copied to data/yue2/<render id>/.
save     A chosen take is imported as a library song (stems separated, analysed) in the folder of the mix's first
         song, then lined up with the mix bar for bar (takealign: YuE2 keeps its own clock and sometimes adds or
         drops a bar) and re-analysed on that grid - so take bar N plays on mix bar N. A "reimagine" event tells
         the page, which can drop the take's stems into the open mix on muted tracks.
"""
import copy
import json
import re
import shutil
import time
from pathlib import Path

import numpy as np

from . import analyze, audio_io, automash, djbrain, jobs, library, llm as llm_mod, mixer, project as P, takealign, warp, yue2link
from .config import DATA, SR, STEMS
from .events import bus

YUE2 = DATA / "yue2"
STYLE_WORDS = 45


# ------------------------------------------------------------------ helpers

def _folder_names(song_ids):
    f = library.get_folders()
    names = {x["id"]: x["name"] for x in f["folders"]}
    return [names.get(f["assign"].get(s)) for s in song_ids]


def presets_for(song_ids):
    """The YuE2 Studio style presets whose artist matches the songs' library folders: [(name, style)]."""
    _, studio = yue2link._cfg()
    try:
        presets = json.loads(Path(studio, "ui", "presets.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    key = lambda s: re.sub(r"[^a-z0-9]", "", (s or "").lower())
    out = []
    for folder in dict.fromkeys(_folder_names(song_ids)):
        for p in presets:
            if folder and key(p.get("name", "").split("—")[0]) == key(folder):
                out.append((p["name"], p["style"]))
    return out


MAX_STYLE_SONGS = 6        # songs whose presets / traits feed the style prompt (most-heard first)


def _songs_of(proj):
    """Every song in the mix, most-heard first (by the beats of it that play). A sample from the sample editor
    counts as the song it was cut from (it has no analysis of its own)."""
    heard = {}
    for c in proj["clips"]:
        info = library.read(c["song"]) if not c.get("mute") else None
        if not info:
            continue
        sid = c["song"]
        if info.get("kind") == "sample":
            sid = (info.get("parent") or {}).get("song")
            if not sid or not library.read(sid):
                continue
        heard[sid] = heard.get(sid, 0.0) + float(c.get("len", 0))
    return sorted(heard, key=heard.get, reverse=True)


def sung_lines(proj, work):
    """The words sung in the mix, in order, with their times (s). Songs with your typed lyrics (userlyrics.py) give
    theirs, placed through the clips; any other singing is transcribed from the mix's vocal tracks alone (kept as
    work/mix_vocals.wav either way). -> [{t0, t1, text, words: [{t, w}]}]"""
    from . import userlyrics
    vp = copy.deepcopy(proj)
    for c in vp["clips"]:
        c["mute"] = c["stem"] != "vocals"
    path = work / "mix_vocals.wav"
    mixer.bounce(vp, path)
    typed, untyped = userlyrics.mix_lines(proj)
    if typed and not untyped:
        return [{k: l[k] for k in ("t0", "t1", "text", "words")} for l in typed]
    from .backends.asr import FasterWhisperLarge
    asr = FasterWhisperLarge()
    try:
        segs = asr.transcribe(audio_io.load(path, SR).mean(0).astype(np.float32), SR)
    finally:
        asr.unload()
    segs = segs.get("segments", segs) if isinstance(segs, dict) else segs
    heard = [dict(t0=float(s.get("t0", s.get("start", 0))), t1=float(s.get("t1", s.get("end", 0))), text=s["text"].strip(),
                  words=[dict(t=float(w["t"]), w=w["w"]) for w in (s.get("words") or [])])
             for s in segs if s.get("text", "").strip()]
    if not typed:
        return heard

    def covered(s):                                       # Whisper's take on a line you typed: yours wins
        return sum(max(0.0, min(s["t1"], l["t1"]) - max(s["t0"], l["t0"])) for l in typed) > 0.5 * max(s["t1"] - s["t0"], 1e-3)
    out = [{k: l[k] for k in ("t0", "t1", "text", "words")} for l in typed] + [s for s in heard if not covered(s)]
    return sorted(out, key=lambda l: l["t0"])



def score_sections(abc):
    """Sections of a YuE2/SheetSage2 score: [{label, bar0, bars, t0, t1}] - from its "% label" markers and the bars
    of each group's Vocal line (a "Z<n>" rest is n bars), timed by its Q: tempo and M: meter."""
    q = re.search(r"^Q:\s*1/4\s*=\s*(\d+)", abc, re.M)
    bpm = float(q.group(1)) if q else 120.0
    meter = re.search(r"^M:\s*(\d+)/(\d+)", abc, re.M)
    beats_per_bar = (int(meter.group(1)) * 4 / int(meter.group(2))) if meter else 4.0
    secs, lines = [], abc.splitlines()
    for i, ln in enumerate(lines):
        s = ln.strip()
        if s.startswith("%"):
            secs.append(dict(label=s[1:].strip().lower() or "section", bars=0))
        elif re.fullmatch(r"V:\s*Vocal", s) and secs:
            j = i + 1
            while j < len(lines) and re.match(r"^[A-Z]:", lines[j].strip()):
                m = re.match(r"^M:\s*(\d+)/(\d+)", lines[j].strip())
                if m:
                    beats_per_bar = int(m.group(1)) * 4 / int(m.group(2))
                j += 1
            if j < len(lines):
                for tok in [t for t in lines[j].split("|") if t.strip()]:
                    z = re.fullmatch(r'\s*(?:"[^"]*")?\s*Z(\d*)\s*', tok)
                    secs[-1]["bars"] += int(z.group(1) or 1) if z else 1
                secs[-1]["beats_per_bar"] = beats_per_bar
    t, bar = 0.0, 0
    for s in secs:
        s["bar0"], s["t0"] = bar, round(t, 2)
        t += s["bars"] * s.get("beats_per_bar", 4.0) * 60.0 / bpm
        bar += s["bars"]
        s["t1"] = round(t, 2)
    return [s for s in secs if s["bars"] > 0]


def tag_lyrics(sections, lines):
    """Lyrics with the score's section tags, each sung line under the section its middle falls in; a section with
    no singing keeps an empty tag (YuE2 plays it instrumental)."""
    out = []
    for k, s in enumerate(sections):
        last = k == len(sections) - 1
        mine = [l["text"] for l in lines if s["t0"] <= 0.5 * (l["t0"] + l["t1"]) < (1e9 if last else s["t1"])]
        out.append(f"[{s['label'].title()}]" + ("\n" + "\n".join(mine) if mine else ""))
    return "\n\n".join(out) + "\n"


def measured_tags(song_ids):
    """Plain style tags from what the engine measured about the songs (so the prompt says something about them
    even when the model only echoes a preset)."""
    tags = []
    for sid in song_ids:
        S = automash.get_song(sid)
        sec, _ = djbrain.sections(S, "A")
        if not sec:
            continue
        d = np.mean([s["stems"]["drums"] for s in sec])
        b = np.mean([s["stems"]["bass"] for s in sec])
        e = np.mean([s["energy"] for s in sec])
        tags += (["hard-hitting drums"] if d > 0.6 else ["sparse percussion"] if d < 0.25 else [])
        tags += (["heavy bass"] if b > 0.6 else [])
        tags += (["high energy"] if e > 0.85 else ["moody, restrained"] if e < 0.6 else [])
        mood = ((S.concept or {}).get("tags") or {}).get("mood")
        if mood:
            tags.append(mood)
        tags.append("dense rhythmic vocal delivery" if S.stability < 0.4 else "melodic sung vocals")
    return list(dict.fromkeys(tags))


def style_prompt(song_ids, user_words="", model=None):
    """The YuE2 style prompt for a reimagining: the artist presets + measured tags + the user's words, phrased by
    the DJ model as one line of tags (and built without it if its answer doesn't use them)."""
    song_ids = list(song_ids)[:MAX_STYLE_SONGS]
    presets = presets_for(song_ids)
    tags = measured_tags(song_ids)
    base = ", ".join([p for _, p in presets[:2]] + tags + ([user_words] if user_words else []))
    heard = []
    for sid in song_ids:                                 # an audio model's description of each song (listen.py)
        h = djbrain._heard(sid)
        if h:
            heard.append(f"\n- heard in one of the songs: {' / '.join(x for x in (h.get('genre'), h.get('subgenre')) if x)}"
                         f" - {h['sound']}")
    model = model or llm_mod.client()
    try:
        r = model.chat_json([
            {"role": "system", "content": "You write style prompts for the YuE2 music generator: one line of comma-"
             "separated tags - genre, sub-genre, production, drums, bass, synths or instruments, vocal delivery, "
             f"mood. No song or artist names, no tempo or key. At most {STYLE_WORDS} words."},
            {"role": "user", "content": "Write the style prompt for a reimagining - a cover - of a mashup of "
             f"{len(song_ids)} songs. "
             "It must combine ALL of these into one coherent new production:\n"
             + "\n".join(f"- artist style: {s}" for _, s in presets) + "\n- measured about the songs: " + ", ".join(tags)
             + "".join(heard)
             + (f"\n- the user wants: {user_words}" if user_words else "") + "\n\nWrite the style prompt."}],
            {"type": "object", "properties": {"style": {"type": "string"}}, "required": ["style"]},
            max_tokens=300, temperature=0.4)
        style = " ".join(r["style"].split()).strip().rstrip(".")
    except llm_mod.LlmError:
        style = ""
    finally:
        model.unload()
    used = sum(t.split()[0].lower() in style.lower() for t in tags) if tags else 1
    if not style or used == 0 or (user_words and user_words.split()[0].lower() not in style.lower()):
        style = base                                       # the model ignored the measured traits / the user
    return style[:1400]


# ------------------------------------------------------------------ stages

def prepare(proj, prompt="", progress=lambda f, m="": None, cancelled=lambda: False):
    """Stage 1 -> {prep_id, title, bpm, style, lyrics, sections, fit}: everything YuE2 will get, for review."""
    yue2link.ensure_running()
    songs = _songs_of(proj)
    prep_id = f"prep-{time.strftime('%Y%m%d-%H%M%S')}"
    work = YUE2 / prep_id
    work.mkdir(parents=True, exist_ok=True)
    progress(0.02, "rendering the mix")
    reqs = [r for r in P.render_requests(proj) if not warp.is_cached(*r)]
    if reqs:
        warp.render_many(reqs)
    mix = work / "mix.wav"
    mixer.bounce(proj, mix, target_lufs=-14.0)
    progress(0.15, "hearing the lyrics")
    lines = sung_lines(proj, work)
    from . import userlyrics
    typed, untyped = userlyrics.mix_lines(proj)
    lyrics_from = ("mixed" if untyped else "yours") if typed else "whisper"
    progress(0.35, "the DJ is writing the style")
    style = style_prompt(songs, prompt)
    progress(0.45, "YuE2: reading the melody and chords")
    upload_id = yue2link.upload(mix)
    res = yue2link.wait(yue2link.analyze(upload_id, task="full"), progress, 0.45, 0.9, cancelled)
    aid = res["analysis_id"]
    nominal = float(res.get("nominal_seconds") or (res.get("report") or {}).get("nominal_duration_seconds") or 0)
    rel = str(res["score_path"]).replace("\\", "/").split("/runs/", 1)[-1]
    sections = score_sections(yue2link.read_text(rel))
    lyrics = tag_lyrics(sections, lines) if lines else "\n\n".join(f"[{s['label'].title()}]" for s in sections) + "\n"
    progress(0.95, "checking the lyric fit")
    fit = yue2link.lyrics_fit(aid, lyrics, "Vocal")       # the sung line is what the words go on
    mix_seconds = round(audio_io.load(mix, SR).shape[1] / SR, 1)
    prep = dict(prep_id=prep_id, analysis_id=aid, upload_id=upload_id, title=proj.get("name", "mix"), bpm=proj["bpm"],
                songs=songs, mix_seconds=mix_seconds, nominal_seconds=nominal or mix_seconds, style=style, lyrics=lyrics,
                sections=[dict(label=s["label"], bars=s["bars"], t0=s["t0"], t1=s["t1"]) for s in sections],
                lines=lines, fit=dict(sections=fit.get("sections", []), warnings=fit.get("warnings", [])), prompt=prompt,
                score_rel=rel, lyrics_from=lyrics_from)
    (work / "prep.json").write_text(json.dumps(prep, indent=1), encoding="utf-8")
    marks = sorted(proj.get("sections") or [], key=lambda m: m["beat"])
    if marks:                                             # your sections (timeline markers) replace the Studio's
        spb = 60.0 / float(proj["bpm"])
        prep["user_sections"] = [dict(label=m["name"], t0=round(m["beat"] * spb, 3),
                                      t1=round((marks[k + 1]["beat"] * spb) if k + 1 < len(marks) else mix_seconds, 3))
                                 for k, m in enumerate(marks)]
        if marks[0]["beat"] > 0:
            prep["user_sections"].insert(0, dict(label="intro", t0=0.0, t1=prep["user_sections"][0]["t0"]))
        progress(0.97, "using your sections")
        r = _placement(prep, lines)
        prep = _prep(prep_id)
        prep["lyrics"], prep["fit"] = r["lyrics"], r["fit"]
        (work / "prep.json").write_text(json.dumps(prep, indent=1), encoding="utf-8")
    progress(1.0, "ready to review")
    return prep


def _prep(prep_id):
    return json.loads((YUE2 / Path(prep_id).name / "prep.json").read_text(encoding="utf-8"))


# ------------------------------------------------------------------ your lyrics, placed where they're sung
# YuE2 sings each section's lyrics starting where the section starts - it doesn't follow word timing inside a section
# (measured Oct 13: a pickup line filed under [Intro] was sung at 0:00 instead of 0:40, and the rest of the intro got
# filler). So placing lyrics = finding where each typed line is sung (align.py, forced alignment), then cutting the
# score's sections at bar lines where the singing really starts / stops, and filing each line under the section it
# starts in (a line up to a quarter bar early is a pickup into the next bar).
LABELS = {"intro": "intro", "verse": "verse", "pre-chorus": "pre-chorus", "chorus": "chorus", "post-chorus": "chorus",
          "bridge": "bridge", "break": "interlude", "interlude": "interlude", "drop": "instrumental",
          "instrumental": "instrumental", "outro": "outro", "silence": "instrumental", "hook": "chorus"}
PICKUP = 0.25            # a line starting this much of a bar before a bar line belongs to that bar
LEAD_IN = 1              # bars of no singing at a section's start that become their own instrumental section
GAP = 2                  # bars of no singing inside a section that become their own instrumental section


def _groups(abc):
    """A native two-voice score as header lines + one entry per bar: {vocal, ins, label (if a section starts here),
    fields (M:/K: lines starting a group)}."""
    lines = abc.splitlines()
    head, bars, i, label = lines[:8], [], 8, None
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("%"):
            label = ln[1:].strip()
            i += 1
            continue
        if ln != "V: Vocal":
            i += 1
            continue
        i += 1
        vf = []
        while i < len(lines) and re.match(r"^[A-Z]:", lines[i]) and not lines[i].startswith("V:"):
            vf.append(lines[i]); i += 1
        vbars = _expand(lines[i]); i += 1
        assert lines[i] == "V: Ins", "unexpected score layout"
        i += 1
        inf = []
        while i < len(lines) and re.match(r"^[A-Z]:", lines[i]) and not lines[i].startswith("V:"):
            inf.append(lines[i]); i += 1
        ibars = _expand(lines[i]); i += 1
        for k, (vb, ib) in enumerate(zip(vbars, ibars)):
            bars.append(dict(vocal=vb, ins=ib, label=label if k == 0 else None,
                             fields=(vf, inf) if k == 0 and (vf or inf) else None))
        label = None
    return head, bars


def _expand(line):
    out = []
    for b in [x.strip() for x in line.split("|")][:-1]:
        z = re.fullmatch(r"Z([2-4])?", b)
        out += ["Z"] * int(z.group(1) or 1) if z else [b]
    return out


def resection(abc, cuts):
    """The score with its section comments replaced: cuts = [(bar, label)] (bar 0 first). Bars are regrouped (1-4 per
    group, a group never spans a section start), so the Studio's parser and YuE2 read the new sections."""
    head, bars = _groups(abc)
    starts = dict(cuts)
    out, k = list(head), 0
    while k < len(bars):
        j = k + 1
        while j < len(bars) and j - k < 4 and j not in starts and not bars[j]["fields"]:
            j += 1
        if k in starts:
            out.append(f"% {starts[k]}")
        vf, inf = bars[k]["fields"] or ([], [])
        out += ["V: Vocal", *vf, "|".join(b["vocal"] for b in bars[k:j]) + "|",
                "V: Ins", *inf, "|".join(b["ins"] for b in bars[k:j]) + "|"]
        k = j
    return "\n".join(out) + "\n"


def plan_sections(sections, lines, bar_s, t0=0.0):
    """Sections cut where the singing starts and stops. sections: [{label, t0, t1}] (the score's or yours); lines:
    aligned [{t0, t1, text}]. -> ([(bar, label)], {line index: section index})"""
    bar_of = lambda t: int(np.floor((t - t0) / bar_s + PICKUP))
    nbar_end = int(round((sections[-1]["t1"] - t0) / bar_s)) if sections else 0
    sung = sorted((max(0, bar_of(l["t0"])), max(0, int(np.ceil((l["t1"] - t0) / bar_s))), k) for k, l in enumerate(lines))
    cuts = []
    for si, s in enumerate(sections):
        b0 = int(round((s["t0"] - t0) / bar_s))
        b1 = int(round((s["t1"] - t0) / bar_s)) if si + 1 < len(sections) else nbar_end
        label = LABELS.get(str(s["label"]).lower(), "verse")      # the Studio only reads its own names
        mine = [(a, b) for a, b, _ in sung if b0 <= a < b1]
        if not mine:
            cuts.append((b0, label if label in ("intro", "outro", "instrumental", "interlude") else
                         ("intro" if si == 0 else "outro" if si == len(sections) - 1 else "interlude")))
            continue
        first = mine[0][0]
        if first - b0 >= LEAD_IN:                       # singing starts late in this section: lead-in of its own
            cuts.append((b0, "intro" if si == 0 else "interlude"))
            cuts.append((first, label if label not in ("intro", "interlude", "instrumental") else "verse"))
        else:
            cuts.append((b0, label if label not in ("intro", "interlude", "instrumental") or not mine else "verse"))
        end = max(b for _, b in mine)
        for (a1, e1), (a2, _) in zip(mine, mine[1:]):  # a long gap inside: an interlude between
            if a2 - max(e1, a1 + 1) >= GAP:
                cuts.append((max(e1, a1 + 1), "interlude"))
                cuts.append((a2, cuts[-2][1] if cuts[-2][1] not in ("intro", "interlude") else label))
        if b1 - end >= GAP and si + 1 < len(sections):   # singing stops well before the section ends
            cuts.append((end, "interlude"))
    cuts = sorted({b: l for b, l in sorted(cuts)}.items())
    cuts = [c for k, c in enumerate(cuts) if k == 0 or c[1] != cuts[k - 1][1] or c[1] in ("verse", "chorus")]
    if not cuts or cuts[0][0] != 0:
        cuts.insert(0, (0, "intro"))
    owner = {}
    for a, _, k in sung:
        owner[k] = max(i for i, (b, _) in enumerate(cuts) if b <= max(a, 0))
    return cuts, owner


def tagged_lyrics(cuts, lines, owner):
    """Lyrics text with one tag per section (in score order), each line under the section it starts in."""
    out = []
    for i, (_, label) in enumerate(cuts):
        mine = [lines[k]["text"] for k in sorted(owner) if owner[k] == i]
        tag = "[Pre-Chorus]" if label == "pre-chorus" else f"[{label.title()}]"
        out.append(tag + ("\n" + "\n".join(mine) if mine else ""))
    return "\n\n".join(out) + "\n"


def place(prep_id, lyrics, progress=lambda f, m="": None):
    """Your lyrics placed where they're sung in the mix: aligned to its vocals (align.py), sections re-cut at the bar
    lines where singing starts / stops, lines filed under the section they start in. Saves the re-cut score for
    perform(). -> {lyrics (tagged), lines: [{t0, t1, text, conf}], sections, fit}"""
    from . import align
    prep = _prep(prep_id)
    work = YUE2 / prep["prep_id"]
    vox = work / "mix_vocals.wav"
    if not vox.exists():
        raise RuntimeError("this preparation is from before lyrics could be placed - prepare the mix again")
    if not align.text_lines(lyrics):
        raise ValueError("no lyrics to place")
    progress(0.1, "lining your words up with the vocals")
    words = align.align_file(vox, lyrics)                # its own process (see align.align_file)
    lines = align.lines(words, lyrics)
    progress(0.6, "cutting the sections where the singing starts")
    return _placement(prep, lines)


def _placement(prep, lines):
    """Re-cut the score's sections around sung lines (yours, aligned - or Whisper's), file the lines, save the score
    for perform(). Sections come from the project's own markers when it has them, else the Studio's score."""
    work = YUE2 / prep["prep_id"]
    rel = prep.get("score_rel") or f"analysis/{prep['analysis_id']}/transcription/score.abc"
    abc = yue2link.read_text(rel)
    m = re.search(r"^M:\s*(\d+)/(\d+)", abc, re.M)
    bar_s = (int(m.group(1)) * 4 / int(m.group(2)) if m else 4) * 60.0 / float(prep["bpm"])
    t0 = 0.0
    try:
        db = yue2link.read_text(rel.rsplit("/", 1)[0] + "/downbeat.lab").split()
        t0 = float(db[0]) if db else 0.0
    except Exception:
        pass
    cuts, owner = plan_sections(prep.get("user_sections") or prep["sections"], lines, bar_s, t0)
    score = resection(abc, cuts)
    tagged = tagged_lyrics(cuts, lines, owner)
    (work / "score_placed.abc").write_text(score, encoding="utf-8")
    r = yue2link.lyrics_fit(prep["analysis_id"], tagged, "Vocal", abc=score)
    nbars = len(_groups(abc)[1])
    secs = [dict(label=l, bar=b, bars=(cuts[i + 1][0] if i + 1 < len(cuts) else nbars) - b, t0=round(t0 + b * bar_s, 2),
                 t1=round(t0 + (cuts[i + 1][0] if i + 1 < len(cuts) else nbars) * bar_s, 2))
            for i, (b, l) in enumerate(cuts)]
    prep.update(placed=dict(lyrics=tagged, lines=[dict(t0=l["t0"], t1=l["t1"], text=l["text"], conf=l.get("conf", 1.0))
                                                  for l in lines], sections=secs))
    (work / "prep.json").write_text(json.dumps(prep, indent=1), encoding="utf-8")
    return dict(lyrics=tagged, lines=prep["placed"]["lines"], sections=secs,
                fit=dict(sections=r.get("sections", []), warnings=r.get("warnings", [])))


def _placed_score(prep):
    p = YUE2 / prep["prep_id"] / "score_placed.abc"
    return p.read_text(encoding="utf-8") if prep.get("placed") and p.exists() else None


def fit(prep_id, lyrics):
    """The Studio's advisory lyric fit for edited lyrics: per section, its melody notes vs the words' syllables."""
    prep = _prep(prep_id)
    r = yue2link.lyrics_fit(prep["analysis_id"], lyrics, "Vocal", abc=_placed_score(prep))
    return dict(sections=r.get("sections", []), warnings=r.get("warnings", []),
                leftover=r.get("leftover_lyric_sections", []))


# how YuE2 covers the mix - the YuE2 Studio's cover modes (its cover page explains them the same way)
MODES = {"cover_full": "melody + harmony", "cover_melody": "melody cover", "cover_instrumental": "instrumental cover",
         "rerender": "real-audio re-render", "continue": "real-audio continuation"}
REALAUDIO = ("rerender", "continue")


def _ra_analysis(prep, progress, cancelled):
    """The mix read as real-audio codes (what the re-render / continuation modes perform from); made once per prep."""
    if prep.get("ra_analysis_id"):
        return prep["ra_analysis_id"]
    progress(0.02, "YuE2: reading the mix as audio")
    try:
        job = yue2link.analyze(prep["upload_id"], kind="realaudio")
    except (KeyError, yue2link.StudioError):          # an older prep, or the Studio let the upload go: upload again
        prep["upload_id"] = yue2link.upload(YUE2 / prep["prep_id"] / "mix.wav")
        job = yue2link.analyze(prep["upload_id"], kind="realaudio")
    try:
        prep["ra_analysis_id"] = yue2link.wait(job, progress, 0.02, 0.12, cancelled)["analysis_id"]
    except yue2link.StudioError as e:
        if "No module named 'studio'" in str(e):
            raise RuntimeError("YuE2 Studio can't run its real-audio reader: it starts it from its root folder, where "
                               "Python can't find its 'studio' package (<YuE2 folder>/ui/server.py, the analyze job's cwd). "
                               "The score modes - melody + harmony, melody cover, instrumental - work.") from e
        raise
    (YUE2 / prep["prep_id"] / "prep.json").write_text(json.dumps(prep, indent=1), encoding="utf-8")
    return prep["ra_analysis_id"]


def perform(prep_id, style, lyrics, takes=2, mode="cover_full", cfg_scale=None, flow_steps=32, voices="both",
            seed=None, continuation_seconds=30, progress=lambda f, m="": None, cancelled=lambda: False):
    """Stage 2 -> {render_id, prep_id, mode, cfg_scale, style, lyrics, bpm, title, takes: [{n, file, seed, melody,
    chroma, metrics, top, seconds}]}. mode: a MODES key; cfg_scale: None (YuE2's default) or 1.0-1.5 - how hard it
    follows the style prompt, i.e. how far it drifts from the mix's own sound."""
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode}")
    yue2link.ensure_running()
    prep = _prep(prep_id)
    style = " ".join(str(style or prep["style"]).split())[:1400]
    lyrics = str(lyrics if lyrics is not None else prep["lyrics"])[:10000]
    seed = int(seed) if seed is not None else int(time.time()) % 100000
    body = dict(mode=mode, style=style, lyrics=lyrics, candidates=int(takes), seed=seed, flow_steps=int(flow_steps))
    if cfg_scale is not None:
        body["cfg_scale"] = round(float(cfg_scale), 2)
    lo = 0.02
    if mode in REALAUDIO:
        body.update(analysis_id=_ra_analysis(prep, progress, cancelled), use_adapter=True)
        expect = None
        if mode == "continue":
            body["continuation_seconds"] = int(np.clip(int(continuation_seconds), 10, 120))
            expect = body["continuation_seconds"] * 25
        lo = 0.12
    else:
        body.update(analysis_id=prep["analysis_id"], transpose=0, bpm=int(round(prep["bpm"])))
        placed = _placed_score(prep)
        if placed:
            body["score_override"] = placed            # sections re-cut where your lyrics are sung (place())
        if mode != "cover_instrumental":
            body["voices"] = voices if voices in ("both", "Vocal", "Ins") else "both"
        expect = int(prep.get("nominal_seconds") or prep["mix_seconds"]) * 25
    progress(lo, f"YuE2: {MODES[mode]}, {takes} take{'s' if takes > 1 else ''}")
    res = yue2link.wait(yue2link.render(**body), progress, lo, 0.97, cancelled, takes=takes, expect_tokens=expect)
    rid = res["render_id"]
    out = YUE2 / rid
    got = []
    ranked = sorted(res["candidates"], key=lambda c: (c.get("failed", False), (c.get("metrics") or {}).get("rank", 99)))
    for k, c in enumerate(ranked):
        if c.get("failed") or not c.get("audio"):
            continue
        dst = yue2link.fetch(c["audio"], out / f"take-{k + 1}.flac")
        m = c.get("metrics") or {}
        got.append(dict(n=k + 1, file=f"take-{k + 1}.flac", seed=c.get("seed"), melody=m.get("melody"),
                        chroma=m.get("chroma"), top=bool(m.get("top")), metrics=m,
                        seconds=round(audio_io.load(dst, SR).shape[1] / SR, 1)))
    if not got:
        raise RuntimeError("YuE2 made no usable take")
    meta = dict(render_id=rid, prep_id=prep["prep_id"], mode=mode, mode_label=MODES[mode], cfg_scale=body.get("cfg_scale"),
                flow_steps=int(flow_steps), seed=seed, style=style, lyrics=lyrics, bpm=prep["bpm"], songs=prep["songs"],
                title=prep["title"], takes=got, created=time.time(), saved={})
    (out / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    progress(1.0, "done")
    return meta


def renders(limit=20):
    """Recent reimaginings, newest first (so takes can be reopened and saved after a reload or restart)."""
    out = []
    for d in sorted(YUE2.glob("render-*"), key=lambda d: d.stat().st_mtime, reverse=True):
        try:
            m = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        m.setdefault("created", (d / "meta.json").stat().st_mtime)
        m.setdefault("mode_label", MODES.get(m.get("mode", "cover_full"), "melody + harmony"))
        m["saved"] = {n: sid for n, sid in (m.get("saved") or {}).items() if library.read(sid)}
        out.append(m)
        if len(out) >= limit:
            break
    return out


# ------------------------------------------------------------------ the new vocals, line by line where yours are sung
# A cover (YuE2 re-performing the mix) sings the lyrics roughly in place but keeps its own clock for the words: on a rap
# mix (Oct 13) the lines came 1-2 s late and the first one at 0:00 - with the sections re-cut around the sung lines or
# not, because a score with no sung notes gives YuE2 nothing to time the words by. So the timing is fixed after the fact:
# the same lyrics are lined up with the take's own vocals (align.py), and each line of the take is moved (and gently
# stretched) to where the mix sings it - one clip per line, so any of them can still be dragged or trimmed by hand.
RETIME_PRE, RETIME_POST = 0.08, 0.20       # s kept before a line's first word / after its last
RETIME_RATIO = (0.7, 1.45)                 # a take line this much shorter / longer than the mix's is a mis-alignment
RETIME_CONF = 0.12                         # ... and so is one the aligner hardly hears
RETIME_SAME = 0.04                         # within 4 % of the mix's length: moved, not stretched


def retime_plan(mix_lines, take_lines, take_beats, bpm):
    """Clips that play each line of the take where the mix sings it. mix_lines: [{t0, t1}] (mix seconds, mix beat 0 =
    0 s); take_lines: align.lines() of the same text on the take's vocals ({line} = index into mix_lines); take_beats:
    the take's beat times (its analysed grid). A line the aligner couldn't find in the take borrows the offset its
    neighbours have. -> [{start, len, src, rate, line}] (project beats; src = the take's beat index)"""
    from .project import _time_to_beat
    tb = np.asarray(take_beats, float)
    beat_of = lambda t: float(_time_to_beat(t, tb))
    spb = 60.0 / float(bpm)
    found = {}
    for t in take_lines:
        m = mix_lines[t["line"]]
        r = (t["t1"] - t["t0"]) / max(m["t1"] - m["t0"], 1e-3)
        if t.get("conf", 1) >= RETIME_CONF and RETIME_RATIO[0] <= r <= RETIME_RATIO[1]:
            found[t["line"]] = (t["t0"], t["t1"])
    if not found:
        return []
    order = sorted(found)
    plan = []
    for i, m in enumerate(mix_lines):
        if m["t1"] <= m["t0"]:
            continue
        if i in found:
            s0, s1 = found[i]
        else:                                             # the local median offset of the lines around it
            near = sorted(order, key=lambda j: abs(j - i))[:5]
            d = float(np.median([found[j][0] - mix_lines[j]["t0"] for j in near]))
            s0, s1 = m["t0"] + d, m["t1"] + d
        r = (s1 - s0) / (m["t1"] - m["t0"])
        if abs(r - 1) < RETIME_SAME:
            r = 1.0
        T0 = max(0.0, m["t0"] - RETIME_PRE)
        T1 = m["t1"] + RETIME_POST
        nxt = next((x["t0"] for x in mix_lines[i + 1:] if x["t1"] > x["t0"]), None)
        if nxt is not None and nxt >= m["t1"] - 0.1:      # the next line follows this one (isn't sung over it):
            T1 = min(T1, max(m["t1"], nxt - RETIME_PRE))  # no overlap with its clip
        a, b = s0 - (m["t0"] - T0) * r, s0 + (T1 - m["t0"]) * r
        if a < tb[0]:                                     # sung before the take's first beat: a clip can't reach
            if b <= tb[0] + 0.3:                          # back there (the mixer holds it at beat 0), so the line
                continue                                  # loses its head instead of playing the wrong audio
            T0 += (tb[0] - a) / r
            a = float(tb[0])
        start, ln = T0 / spb, (T1 - T0) / spb
        src0, src1 = beat_of(a), beat_of(b)
        plan.append(dict(start=round(start, 4), len=round(ln, 4), src=round(src0, 4),
                         rate=round((src1 - src0) / ln, 5) if ln > 0 else 1.0, line=i))
    return plan


def retime_clips(prep, sid, bpm):
    """retime_plan() for a saved take (song sid, on its mix-aligned grid) - [] when the mix has no lyric lines."""
    from . import align
    lines = [l for l in ((prep.get("placed") or {}).get("lines") or prep.get("lines") or [])
             if str(l.get("text", "")).strip() and l["t1"] > l["t0"]]
    if not lines:
        return []
    text = "\n".join(" ".join(l["text"].split()) for l in lines)       # one line of text per sung line
    words = align.align_file(library.song_dir(sid) / "stems" / "vocals.flac", text)
    return retime_plan(lines, align.lines(words, text), (library.read(sid) or {})["beats"], bpm)


def save(render_id, takes, add_to=None, vocals_only=False, retime=False):
    """Import chosen takes as library songs (stems separated + analysed) in the folder of the mix's first song, then
    line each up with the mix (a follow-up job). add_to: a project id the page should drop the aligned stems into;
    vocals_only: drop only the new vocal stem in, playing, over the mix's own music (its vocal tracks muted).
    -> [(song id, import job, align job)]"""
    d = YUE2 / Path(render_id).name
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    prep = _prep(meta["prep_id"])
    out = []
    for n in takes:
        t = next(x for x in meta["takes"] if x["n"] == int(n))
        when = re.search(r"-(\d{2})(\d{2})\d{2}-", meta["render_id"])
        title = f"{meta['title'][:60]} (YuE2 take {t['n']}" + (f", {when.group(1)}:{when.group(2)})" if when else ")")
        sid, job = library.import_audio(d / t["file"], title)
        out.append((sid, job, _align_job(sid, title, prep, add_to, vocals_only, retime)))
        meta.setdefault("saved", {})[str(t["n"])] = sid
    (d / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    first = meta["songs"][0] if meta["songs"] else None
    f = library.get_folders()
    folder = f["assign"].get(first)
    if folder:
        for sid, _, _ in out:
            f["assign"][sid] = folder
        library.set_folders(f)
    return out


def _align_job(sid, title, prep, add_to, vocals_only=False, retime=False):
    """After the take's import: line it up with the mix bar for bar and re-analyse it on that grid; retime: then its
    vocals line by line where the mix sings them (retime_clips)."""
    def work(job):
        while True:                                       # the import (separation + analysis) comes first
            info = library.read(sid) or {}
            if info.get("status") == "ready":
                break
            if info.get("status") == "error" or not info:
                raise RuntimeError(f"{title} didn't import: {info.get('error')}")
            if job.cancelled:
                return None
            time.sleep(2)
        job.update(0.05, "lining up with the mix")
        mix = audio_io.load(YUE2 / prep["prep_id"] / "mix.wav", SR).mean(0)
        bpm = float(prep["bpm"])
        n_beats = int(len(mix) / SR / (60.0 / bpm))
        stems = {s: library.load_stem(sid, s) for s in STEMS}
        n = min(v.shape[1] for v in stems.values())
        take = sum(v[:, :n] for v in stems.values()).mean(0)
        tau = takealign.align_bars(mix, take, SR, bpm, n_beats, dict(beats=info["beats"], bars=info["bars"],
                                                                    bpb=info.get("bpb", 4)))
        g = takealign.grid(tau, n / SR, bpm)
        new = analyze.analyze(library.song_dir(sid), stems, job.sub(0.2, 0.85, "analysing on the mix's grid: "), grid=g)
        library.apply_new_grid(sid, info, new)
        # songs play "as recorded" (one fixed speed, project.song_tspec), so the aligned beats only take effect once
        # the stems are locked to them: 🔒 re-renders them with take beat k on steady beat k (lock.py)
        from . import lock
        job.update(0.88, "locking it to the mix's grid")
        lock.make(sid, check=False)                     # the take's grid is the mix's: no wrong-speed check
        t = time.time()
        while not lock.current(library.read(sid) or {}):
            locking = any(j.get("kind") == "lock" and j.get("ref") == sid and j.get("status") in ("queued", "running")
                          for j in jobs.active())
            if not locking and time.time() - t > 5 and not lock.current(library.read(sid) or {}):
                raise RuntimeError(f"{title} couldn't be locked to the mix's grid")
            if job.cancelled or time.time() - t > 1800:
                return None
            time.sleep(2)
        plan = []
        if retime:
            job.update(0.94, "moving the new vocals line by line to where the mix sings them")
            try:
                plan = retime_clips(prep, sid, bpm)
            except Exception as e:                        # the whole take, bar-aligned, is still fine
                print(f"re-timing {title} skipped: {type(e).__name__}: {e}")
        out = library.read(sid)
        bus.publish("reimagine", sid=sid, title=title, aligned=True, project=add_to, first_beat=g["first_beat"],
                    n_beats=n_beats, bpm=bpm, lock=True, vocals_only=bool(vocals_only), retime=plan)
        return library.summary(out)
    return jobs.submit("align", f"Lining up {title} with the mix", work, ref=sid, gpu=True)
