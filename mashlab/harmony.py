"""Tuning, key, chroma and consonance.

Rather than trusting key *labels* (often wrong, and two songs "in the same key" can still clash), the
mashup engine scores actual pitch content against each other: consonance(p, q) weighs every pair of
pitch classes sounding together by how consonant that interval is.
"""
import numpy as np
import librosa

NOTES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
# Temperley (Kostka-Payne) key profiles - better than Krumhansl on pop/rock
MAJ = np.array([0.748, 0.060, 0.488, 0.082, 0.670, 0.460, 0.096, 0.715, 0.104, 0.366, 0.057, 0.400])
MIN = np.array([0.712, 0.084, 0.474, 0.618, 0.049, 0.460, 0.105, 0.747, 0.404, 0.067, 0.133, 0.330])
# consonance of an interval class 0..6 (unison, m2, M2, m3, M3, P4, tritone); mirrored for 7..11
_IC = np.array([1.0, -1.0, -0.35, 0.5, 0.55, 0.7, -0.6])
IW = np.array([_IC[min(k, 12 - k)] for k in range(12)])
CONS = np.array([[IW[(i - j) % 12] for j in range(12)] for i in range(12)])

# Camelot wheel: major keys "nB", minor keys "nA"
_CAM_MAJ = {0: 8, 7: 9, 2: 10, 9: 11, 4: 12, 11: 1, 6: 2, 1: 3, 8: 4, 3: 5, 10: 6, 5: 7}


def camelot(tonic, mode):
    if mode == "major":
        return f"{_CAM_MAJ[tonic]}B"
    return f"{_CAM_MAJ[(tonic + 3) % 12]}A"          # relative major shares the number


def key_name(tonic, mode):
    return f"{NOTES[tonic]} {'major' if mode == 'major' else 'minor'}"


def estimate_tuning(y_mono, sr):
    """Deviation from A440 in fractions of a 36-per-octave chroma bin (1/3 semitone), wrapped to -0.5..0.5 -
    the form chroma_cqt(bins_per_octave=36, tuning=...) takes. NOT semitones: see tuning_semitones."""
    return float(librosa.estimate_tuning(y=y_mono, sr=sr, bins_per_octave=36))


def tuning_semitones(y_mono, sr):
    """Deviation from A440 in semitones (-0.5..0.5), 1-cent resolution: what a pitch shift to A440 needs.
    (The song's stored "tuning" is in 1/3-semitone bins and wraps around; using it as semitones detuned songs
    by up to 50 cents, often in the wrong direction.)"""
    return float(librosa.estimate_tuning(y=y_mono, sr=sr, bins_per_octave=12, resolution=0.01))


def beat_chroma(y_mono, sr, beat_times, tuning=0.0):
    """Unnormalized CQT chroma averaged per beat -> (n_beats, 12)."""
    hop = 512
    C = librosa.feature.chroma_cqt(y=y_mono, sr=sr, hop_length=hop, tuning=tuning, norm=None,
                                   bins_per_octave=36, n_octaves=6, fmin=librosa.note_to_hz("C1"))
    return _per_beat(C.T, beat_times, sr, hop)


def beat_rms_db(y_mono, sr, beat_times):
    hop = 512
    r = librosa.feature.rms(y=y_mono, frame_length=2048, hop_length=hop)[0]
    v = _per_beat(r[:, None] ** 2, beat_times, sr, hop)[:, 0]
    return 10 * np.log10(v + 1e-12)


def _per_beat(F, beat_times, sr, hop):
    """Average frame features F (n_frames, d) over each beat interval [b_j, b_j+1)."""
    nf = len(F)
    fr = np.clip(np.round(np.asarray(beat_times) * sr / hop).astype(int), 0, nf)
    out = np.zeros((len(beat_times), F.shape[1]))
    for j in range(len(fr)):
        a = fr[j]
        b = fr[j + 1] if j + 1 < len(fr) else min(nf, a + max(1, a - fr[j - 1] if j else 1))
        if b > a:
            out[j] = F[a:b].mean(0)
    return out


def detect_key(chroma_sum):
    """chroma_sum: 12-vector of pitch-class energy. Returns (tonic, mode, confidence 0..1)."""
    c = np.asarray(chroma_sum, float)
    c = (c - c.mean()) / (c.std() + 1e-9)
    scores = []
    for mode, prof in (("major", MAJ), ("minor", MIN)):
        p = (prof - prof.mean()) / prof.std()
        for t in range(12):
            scores.append((float(np.dot(c, np.roll(p, t)) / 12), t, mode))
    scores.sort(reverse=True)
    best, second = scores[0], scores[1]
    conf = float(np.clip((best[0] - second[0]) * 4 + best[0] * 0.5, 0, 1))
    return best[1], best[2], conf


def normalize_rows(X):
    X = np.asarray(X, float)
    s = X.sum(-1, keepdims=True)
    return np.divide(X, s, out=np.zeros_like(X), where=s > 1e-12)


def consonance(p, q, shift=0):
    """Consonance of pitch-class distributions p (sounding) and q (shifted by `shift` semitones). ~[-1, 1]."""
    p = np.asarray(p, float)
    q = np.roll(np.asarray(q, float), shift, axis=-1)
    p = p / (p.sum(-1, keepdims=True) + 1e-12)
    q = q / (q.sum(-1, keepdims=True) + 1e-12)
    return np.einsum("...i,ij,...j->...", p, CONS, q)


def sharpen(X, power=2.0):
    """Emphasise the dominant pitch classes (chroma is smeared by overtones)."""
    X = np.maximum(np.asarray(X, float), 0) ** power
    return X
