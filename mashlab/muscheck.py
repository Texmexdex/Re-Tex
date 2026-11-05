"""A second opinion on a song's beat marks from MuScriptor (Kyutai / Mirelo): transcribe the song into notes, then
check, mark by mark, whether the notes land on the marks.

Only the notes are used. MuScriptor's own beat grid / tempo comes from Beat This! (the detector the marks already
come from), so it would be no second opinion.

Per mark: the kick / snare / bass / chord onsets within +-4 beats are placed on the beat grid the marks make (where
in its beat each onset falls, 0..1). If they bunch on the marks clearly better than on the grid shifted by a quarter,
half or three quarters of a beat, the notes AGREE. If they bunch clearly better on a shifted grid (the marks half a
beat off, say), they DISAGREE. If they show no clear bunching (ambient, rubato, a triplet feel against the marks,
too few notes), the notes give NO opinion. Only disagreement changes anything: those marks show as unsure.

MuScriptor works in 5-second chunks and re-strikes held notes at each chunk's start; those are joined back first.
"""
import threading

import numpy as np

CHUNK = 5.0                  # MuScriptor's chunk length (s)
SEAM = 0.03                  # a note ending / starting within 30 ms of a chunk seam is the same held note
WINDOW = 4                   # beats either side of a mark that decide its verdict
ON = 0.08                    # an onset within 8 % of a beat of a grid line is "on" it
MARGIN = 1.3                 # the winner must beat the runner-up by 30 %
MIN_WEIGHT = 3.0             # less note weight than this in a window = no opinion
SHIFTS = (0.0, 0.25, 0.5, 0.75)
AGREE, NONE, DISAGREE = 1, 0, 2

_model = None
_lock = threading.Lock()


def weight(inst, pitch):
    """How much an onset says about where the beat is: kick / snare most, bass and chords next, hats least."""
    inst = str(inst).lower()
    if "drum" in inst:
        p = int(pitch)
        if p in (35, 36):
            return 1.0                                   # kick
        if p in (37, 38, 39, 40):
            return 1.0                                   # snare / clap
        if p in (42, 44, 46):
            return 0.25                                  # hats
        return 0.4                                       # toms / cymbals
    if "bass" in inst:
        return 0.6
    return 0.3


def join_seams(notes):
    """notes: [[onset, offset, pitch, instrument]] -> the same without the re-strikes at chunk seams."""
    notes = sorted(notes, key=lambda n: n[0])
    ended = {}
    out = []
    for on, off, pitch, inst in notes:
        k = (pitch, inst)
        seam = round(on / CHUNK) * CHUNK
        prev = ended.get(k)
        if prev is not None and abs(on - seam) < SEAM and abs(out[prev][1] - seam) < SEAM:
            out[prev][1] = off                           # the held note carries on
            continue
        out.append([on, off, pitch, inst])
        ended[k] = len(out) - 1
    return out


def verdicts(beats, notes):
    """beats: the song's beat times; notes: [[onset, offset, pitch, instrument]] -> per beat AGREE / NONE / DISAGREE,
    plus the share of each."""
    b = np.asarray(beats, float)
    n = len(b)
    if n < 3 or not notes:
        return [NONE] * n
    ons = np.array([x[0] for x in notes], float)
    w = np.array([weight(x[3], x[2]) for x in notes], float)
    # where in its beat each onset falls (0..1), on the grid the marks make
    k = np.clip(np.searchsorted(b, ons, side="right") - 1, 0, n - 2)
    ph = (ons - b[k]) / np.maximum(b[k + 1] - b[k], 1e-6)
    inside = (ph >= 0) & (ph < 1)
    out = []
    for j in range(n):
        sel = inside & (k >= j - WINDOW) & (k < j + WINDOW)
        tot = w[sel].sum()
        if tot < MIN_WEIGHT:
            out.append(NONE)
            continue
        p, ww = ph[sel], w[sel]
        score = []
        for s in SHIFTS:
            d = np.abs(((p - s) + 0.5) % 1.0 - 0.5)
            score.append(float(ww[d < ON].sum()))
        best = int(np.argmax(score))
        rest = max(score[i] for i in range(len(score)) if i != best)
        if score[best] < MARGIN * max(rest, 1e-9) or score[best] < 0.2 * tot:
            out.append(NONE)
        else:
            out.append(AGREE if best == 0 else DISAGREE)
    return out


METERS = (3, 4, 6)
METER_GAIN = 0.15            # the drums must repeat this much more alike at the other bar length ...
METER_MIN = 0.5              # ... and clearly repeat there at all (cosine similarity of kick / snare patterns)


def meter_check(beats, notes, bpb, first, last):
    """Do MuScriptor's drums repeat at another bar length than the marks' (3, 4 or 6 beats)? Beat This!'s downbeat
    probability can't always tell 3 from 6 (in 6/8 it calls the snare on 4 a downbeat too: Formant). The kick / snare
    pattern can: compare it with itself 3, 4 and 6 beats later. -> {bpb, start (s): a beat with the kick that starts
    a bar, sims} when the drums clearly say another meter, else None."""
    b = np.asarray(beats, float)
    if len(b) < 24 or not notes:
        return None
    p = float(np.median(np.diff(b)))
    kick, snare = np.zeros(len(b)), np.zeros(len(b))
    for on, off, pitch, inst in notes:
        if "drum" not in str(inst).lower():
            continue
        j = int(np.clip(np.searchsorted(b, on), 1, len(b) - 1))
        j = j if abs(b[j] - on) < abs(b[j - 1] - on) else j - 1
        if abs(b[j] - on) > 0.12 * p:
            continue
        if int(pitch) in (35, 36):
            kick[j] += 1
        elif int(pitch) in (37, 38, 39, 40):
            snare[j] += 1
    v = np.stack([kick, snare], 1)[first:last + 1]
    if v.sum() < 16:
        return None

    def sim(m):
        a, c = v[:-m], v[m:]
        return float((a * c).sum() / (np.sqrt((a * a).sum() * (c * c).sum()) + 1e-9))
    sims = {m: round(sim(m), 3) for m in METERS}
    best = max(METERS, key=lambda m: sims[m])
    if best == 6 and sims[6] - sims[3] < METER_GAIN:
        best = 3                                    # repeats every 3 just as well: that's the simpler meter
    cur = int(bpb) if int(bpb) in METERS else 4
    if best == cur or sims[best] < METER_MIN or sims[best] - sims[cur] < METER_GAIN:
        return None
    idx = np.arange(first, last + 1)
    phase = max(range(best), key=lambda ph: kick[idx[(idx - first) % best == ph]].sum())
    return dict(bpb=int(best), start=round(float(b[first + phase]), 4), sims=sims)


def transcribe(audio_path, progress=lambda f: None, size="large"):
    """-> [[onset, offset, pitch, instrument]] for an audio file (MuScriptor; GPU if there is one)."""
    global _model
    from muscriptor import TranscriptionModel, NoteStartEvent, NoteEndEvent
    with _lock:
        if _model is None:
            _model = TranscriptionModel.load_model(size)
        notes, open_ = [], {}
        for ev in _model.transcribe(str(audio_path)):
            if isinstance(ev, NoteStartEvent):
                notes.append([round(ev.start_time, 4), None, int(ev.pitch), str(ev.instrument)])
                open_[ev.index] = len(notes) - 1
                progress(ev.start_time)
            elif isinstance(ev, NoteEndEvent):
                i = open_.pop(ev.start_event.index, None)
                if i is not None:
                    notes[i][1] = round(ev.end_time, 4)
        for i in open_.values():
            notes[i][1] = notes[i][0] + 0.25
    return notes
