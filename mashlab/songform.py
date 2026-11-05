"""Song form for display: intro / verse / pre-chorus / chorus / bridge / break / drop / outro, from what was already
measured about a song (no new audio analysis).

structure.sections() (made at import, used by the arrangement engine) is a generic change detector: it cuts every
4 bars or so and gives almost every piece its own letter, so the song view showed 17-27 sections with arbitrary
names. This finds the form the way a listener hears it:

  boundaries  only on 2-bar measure starts (SongData.measure_starts), chosen for the whole song at once (dynamic
              programming): sections that are internally alike, 8 / 16 bars preferred, cut where the vocal comes in or
              stops, the drums enter or drop, the energy jumps, the harmony turns, or a lyric line starts
  repeats     sections compared bar against bar on harmony (and on their words, when both have lyrics): parts that
              come back share a letter
  names       intro = before the singing starts, outro = after it stops; chorus = the repeated vocal part whose words
              repeat most / that's loudest; other vocal parts are verses; a one-off vocal part after the choruses is a
              bridge; a short part leading into each chorus is a pre-chorus; parts without vocals are a break (no
              drums), a drop (loud, repeated) or an instrumental

Kept in songs/<id>/form.json (the engine's song.json sections are left exactly as they are).
"""
import json
import re

import numpy as np

from . import library, userlyrics

FORM_VERSION = 1
PREFERRED = {8: 0.0, 16: 0.15, 4: 0.6, 12: 0.5, 6: 0.9, 10: 0.9, 14: 1.0, 2: 2.0}
SECTION_COST = 1.0           # per section: keeps the form from splitting into slivers
SPREAD_W = 0.5               # weight of how unlike each other a section's bars are


def _norm(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)


def _bar_feats(S):
    """Per bar: harmony (chroma, song mean removed so differences show), stems, energy, vocal presence."""
    f = S.f
    nbar = len(S.bars)
    beats_of = [(S.bars[i], S.bars[i + 1] if i + 1 < nbar else min(S.nb, S.bars[i] + S.bpb)) for i in range(nbar)]
    ch = _norm(f["chroma_other"]) + _norm(f["chroma_bass"]) + 0.5 * _norm(f["chroma_vocals"])
    C = np.array([ch[a:b].mean(0) if b > a else np.zeros(12) for a, b in beats_of])
    C = _norm(C - C.mean(0))
    pres = {s: np.asarray(f[f"pres_{s}"], float)[:nbar] for s in ("drums", "bass", "other", "vocals")}
    energy = np.asarray(f["energy"], float)[:nbar]
    return C, pres, energy


def _lyric_bars(sid, S):
    """Words (lowercase) per bar from the song's words (yours, else the transcript), or None."""
    segs, _ = userlyrics.segments(sid)
    if not segs:
        return None
    words = [[] for _ in S.bars]
    beat_t = np.asarray(S.beats)
    bar_t = beat_t[np.minimum(np.asarray(S.bars), len(beat_t) - 1)]
    for seg in segs:
        for w in seg.get("words") or []:
            tok = re.sub(r"[^a-z']", "", str(w.get("w", "")).lower())
            if tok:
                i = int(np.searchsorted(bar_t, float(w["t"]), side="right")) - 1
                if 0 <= i < len(words):
                    words[i].append(tok)
    return words if any(words) else None


def _evidence(C, pres, energy, words):
    """How strongly a section should start at each bar (0..~3)."""
    n = len(C)
    ev = np.zeros(n)
    sm = lambda x: np.convolve(x, np.ones(3) / 3, mode="same")
    v = sm(pres["vocals"])
    d = sm(pres["drums"])
    ev[1:] += 1.2 * np.abs(np.diff(v)) + 0.8 * np.abs(np.diff(d))
    ev[1:] += 1.5 * np.abs(np.diff(sm(energy)))
    for k in (4, 8):                                     # harmony turning: the k bars after vs before
        for i in range(k, n - k):
            ev[i] += 0.5 * max(0.0, 1 - float(_norm(C[i - k:i].mean(0)) @ _norm(C[i:i + k].mean(0))))
    if words:
        for i in range(1, n):
            if words[i] and not words[i - 1]:
                ev[i] += 0.6                             # singing (re)starts here
    return ev


def _segment(C, pres, energy, ev, starts, c0, c1):
    """Best boundaries (DP over candidate bars). -> [(a, b)]"""
    X = np.hstack([C, 0.9 * pres["vocals"][:, None], 0.6 * pres["drums"][:, None], 0.4 * pres["bass"][:, None],
                   0.4 * pres["other"][:, None], 0.8 * energy[:, None]])
    cs = np.vstack([np.zeros(X.shape[1]), np.cumsum(X, 0)])
    cs2 = np.r_[0, np.cumsum((X ** 2).sum(1))]

    def within(a, b):                                    # spread of the bars around the section's mean (grows with
        n = b - a                                        # length, so a long section must really be one part)
        s = cs[b] - cs[a]
        return SPREAD_W * (cs2[b] - cs2[a] - (s @ s) / n)

    cand = sorted(set([c0, c1] + [i for i in starts if c0 < i < c1]))
    best = {c0: (0.0, None)}
    for j in cand[1:]:
        options = []
        for i in cand:
            if i >= j or i not in best:
                continue
            L = j - i
            lp = PREFERRED.get(L, 1.2 + 0.08 * max(0, L - 16))
            if L < 2:
                continue
            cost = best[i][0] + within(i, j) + SECTION_COST + lp - (ev[i] if i > c0 else 0)
            options.append((cost, i))
        if options:
            best[j] = min(options)
    cuts, j = [], c1
    while j is not None and j != c0:
        cuts.append(j)
        j = best[j][1]
    cuts = sorted([c0] + cuts)
    return list(zip(cuts[:-1], cuts[1:]))


def _lyric_lines(sid, S):
    """The song's sung lines: [(bar of its middle, set of words)], or []."""
    segs, _ = userlyrics.segments(sid)
    beat_t = np.asarray(S.beats)
    bar_t = beat_t[np.minimum(np.asarray(S.bars), len(beat_t) - 1)]
    out = []
    for seg in segs:
        toks = {re.sub(r"[^a-z']", "", w) for w in str(seg.get("text", "")).lower().split()} - {""}
        if len(toks) >= 2:
            mid = 0.5 * (float(seg.get("t0", 0)) + float(seg.get("t1", 0)))
            out.append((int(np.searchsorted(bar_t, mid, side="right")) - 1, toks))
    return out


def _line_groups(lines):
    """Lines that come back (fuzzy: Whisper hears a repeat a little differently): group id per line."""
    parent = list(range(len(lines)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i in range(len(lines)):
        for j in range(i):
            a, b = lines[i][1], lines[j][1]
            if len(a & b) / len(a | b) >= 0.5:
                parent[root(i)] = root(j)
    return [root(i) for i in range(len(lines))]


def _music_sim(C, T, A, B):
    """How alike two sections sound: harmony and instrumentation compared bar against bar (the shorter slid along
    the longer by whole 2-bar measures). 0..1"""
    (a0, a1), (b0, b1) = A, B
    if a1 - a0 > b1 - b0:
        (a0, a1), (b0, b1) = (b0, b1), (a0, a1)
    n = a1 - a0
    best = 0.0
    for off in range(0, (b1 - b0) - n + 1, 2):
        harm = (float(np.mean((C[a0:a1] * C[b0 + off:b0 + off + n]).sum(1))) + 1) / 2
        tex = 1 - float(np.mean(np.abs(T[a0:a1] - T[b0 + off:b0 + off + n])))
        best = max(best, 0.6 * harm + 0.4 * tex)
    return best


def analyze(sid):
    """-> [{start, end, label, name, n, energy, stems}] (bars)."""
    from .automash import get_song
    S = get_song(sid)
    C, pres, energy = _bar_feats(S)
    nbar = len(C)
    c0, c1 = (S.info.get("content_bars") or (0, nbar))
    c0, c1 = max(0, int(c0)), min(nbar, int(c1))
    words = _lyric_bars(sid, S)
    ms = S.measure_starts()
    starts = [i for i in range(nbar) if ms[i]]
    ev = _evidence(C, pres, energy, words)
    spans = _segment(C, pres, energy, ev, starts, c0, c1)
    T = np.c_[pres["vocals"], pres["drums"], pres["bass"], pres["other"], energy]

    # sung lines that come back in another section = the hook
    lines = _lyric_lines(sid, S)
    groups = _line_groups(lines)
    sec_of = [next((k for k, (a, b) in enumerate(spans) if a <= bar < b), None) for bar, _ in lines]
    where = {}
    for g, k in zip(groups, sec_of):
        where.setdefault(g, set()).add(k)
    hook = np.zeros(len(spans))
    nlines = np.zeros(len(spans))
    for g, k in zip(groups, sec_of):
        if k is not None:
            nlines[k] += 1
            hook[k] += len(where[g] - {k, None}) > 0
    hook = np.where(nlines > 0, hook / np.maximum(nlines, 1), 0)
    lyrics_ok = len(lines) >= 8 and hook.max() >= 0.35

    # letters: the same words come back, or the same music (song-relative: most parts of a looped song sound alike)
    n = len(spans)
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(i):
            M[i, j] = M[j, i] = _music_sim(C, T, spans[i], spans[j])
    off = M[np.triu_indices(n, 1)]
    thr = max(0.86, float(np.quantile(off, 0.8))) if len(off) else 1.0
    shared = lambda i, j: (sum(1 for g, k in zip(groups, sec_of) if k == i and j in where[g]) / max(1, nlines[i]))
    labels = []
    for k in range(n):
        lab = None
        for j in range(k):
            same_words = lyrics_ok and nlines[k] >= 2 and nlines[j] >= 2 and max(shared(k, j), shared(j, k)) >= 0.4
            if same_words or (M[k, j] >= thr and not (lyrics_ok and (hook[k] >= 0.35) != (hook[j] >= 0.35))):
                lab = labels[j]
                break
        labels.append(lab if lab is not None else max(labels, default=-1) + 1)

    secs = []
    for (a, b), lab, h in zip(spans, labels, hook):
        st = {s: round(float(pres[s][a:b].mean()), 2) for s in pres}
        k = len(secs)
        secs.append(dict(start=int(a), end=int(b), label=chr(65 + lab % 26), energy=round(float(energy[a:b].mean()), 3),
                         stems=st, hook=round(float(h), 2),
                         vocal=bool(st["vocals"] >= 0.4 or (nlines[k] >= 2 and st["vocals"] >= 0.2))))
    _names(secs, labels, lyrics_ok)
    if c0 > 0:
        secs.insert(0, dict(start=0, end=c0, label="-", name="silence", n=1, energy=0.0, stems={}, vocal=False))
    if c1 < nbar:
        secs.append(dict(start=c1, end=nbar, label="-", name="silence", n=1, energy=0.0, stems={}, vocal=False))
    return secs


def _names(secs, labels, lyrics_ok):
    voc = [i for i, s in enumerate(secs) if s["vocal"]]
    first_v, last_v = (voc[0], voc[-1]) if voc else (len(secs), -1)
    count = {l: labels.count(l) for l in set(labels)}
    hi = np.quantile([s["energy"] for s in secs], 0.6) if secs else 0
    if lyrics_ok:                                        # the chorus is where the repeated lines are
        is_chorus = [s["vocal"] and s["hook"] >= 0.35 for s in secs]
    else:                                                # no usable words: the loudest repeated vocal part - unless
        score = {}                                       # it's most of the song (rap over one loop: all verse)
        vbars = sum(secs[i]["end"] - secs[i]["start"] for i in voc) or 1
        for i in voc:
            if count[labels[i]] >= 2:
                score[labels[i]] = max(score.get(labels[i], 0.0), secs[i]["energy"] + 0.05 * count[labels[i]])
        ch = max(score, key=score.get) if score else None
        if ch is not None and sum(s["end"] - s["start"] for i, s in enumerate(secs) if labels[i] == ch and s["vocal"]) > 0.45 * vbars:
            ch = None
        is_chorus = [s["vocal"] and labels[i] == ch for i, s in enumerate(secs)]
    first_ch = next((i for i, c in enumerate(is_chorus) if c), None)
    last_ch = max((i for i, c in enumerate(is_chorus) if c), default=None)
    for i, s in enumerate(secs):
        if i < first_v and not s["vocal"]:
            nm = "intro"
        elif i > last_v and not s["vocal"]:
            nm = "outro"
        elif s["vocal"]:
            if s["stems"].get("drums", 0) < 0.3 and first_v < i < last_v and not is_chorus[i]:
                nm = "break"                             # the singer over a stripped-down beat
            elif is_chorus[i]:
                nm = "chorus"
            elif i + 1 < len(secs) and is_chorus[i + 1] and s["end"] - s["start"] <= 4:
                nm = "pre-chorus"
            elif (first_ch is not None and last_ch is not None and first_ch < i < last_ch and count[labels[i]] == 1
                  and i > first_ch + 1):
                nm = "bridge"
            else:
                nm = "verse"
        elif s["stems"].get("drums", 0) < 0.3:
            nm = "break"
        elif s["energy"] >= hi and count[labels[i]] >= 2:
            nm = "drop"
        else:
            nm = "instrumental"
        s["name"] = nm
    seen = {}
    for s in secs:
        seen[s["name"]] = seen.get(s["name"], 0) + 1
        s["n"] = seen[s["name"]]


def _key(sid, info):
    lt = userlyrics.mtime(sid)
    bars = info.get("bars") or []
    return f"{FORM_VERSION}|{len(bars)}|{bars[-1] if bars else 0}|{info.get('grid_version')}|{info.get('stems_rev')}|{lt}"


def form(sid, info=None):
    """The song's form, cached in songs/<id>/form.json (remade when the song or its lyrics change). None if the song
    isn't ready or the form can't be worked out."""
    info = info or library.read(sid)
    if not info or info.get("status") != "ready":
        return None
    p = library.song_dir(sid) / "form.json"
    key = _key(sid, info)
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
        if d.get("user") and d.get("user_bars") == len(info.get("bars") or []):
            return d["user"]                             # your own edits win (while the bar grid is the same)
        if d.get("key") == key:
            return d["sections"]
    except (OSError, ValueError):
        pass
    try:
        secs = analyze(sid)
    except Exception:
        return None
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps({"key": key, "sections": secs}), encoding="utf-8")
    tmp.replace(p)
    return secs


NAMES = ("intro", "verse", "pre-chorus", "chorus", "post-chorus", "bridge", "break", "drop", "instrumental", "outro",
         "silence")


def set_user(sid, sections):
    """Save your own form (None = back to the detected one). sections: [{start, end, name}] covering the bars in
    order. -> the form now in use."""
    info = library.read(sid)
    p = library.song_dir(sid) / "form.json"
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        d = {}
    if sections is None:
        d.pop("user", None)
    else:
        nbar = len(info.get("bars") or [])
        out, pos = [], 0
        for s in sections:
            a, b, nm = int(s["start"]), int(s["end"]), str(s.get("name", "verse"))
            if a != pos or b <= a or b > nbar or nm not in NAMES:
                raise ValueError("sections must cover the bars in order, with known names")
            out.append(dict(start=a, end=b, name=nm, label=str(s.get("label", "")), energy=s.get("energy", 0),
                            stems=s.get("stems", {}), user=True))
            pos = b
        seen = {}
        for s in out:
            seen[s["name"]] = seen.get(s["name"], 0) + 1
            s["n"] = seen[s["name"]]
        d["user"], d["user_bars"] = out, nbar
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(d), encoding="utf-8")
    tmp.replace(p)
    return form(sid, info)
