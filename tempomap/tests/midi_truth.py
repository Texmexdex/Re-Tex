"""The truth of a .mid as Re-Tex renders it: bar lines (s), meters, quarter-note beats, drum note times.
Re-Tex plays a file's first tempo from tick 0 even when the file sets it later (the standard would be 120 BPM until
then), so the truth does the same - it must match the render, not the standard."""
import mido


def midi_truth(path):
    mf = mido.MidiFile(str(path), clip=True)
    ppq = mf.ticks_per_beat
    tempos, ts, drums, end = {}, [], [], 0
    for tr in mf.tracks:
        t = 0
        for msg in tr:
            t += msg.time
            if msg.type == "set_tempo":
                tempos[t] = msg.tempo
            elif msg.type == "time_signature":
                ts.append((t, msg.numerator, msg.denominator))
            elif msg.type == "note_on" and msg.velocity:
                end = max(end, t)
                if msg.channel == 9:
                    drums.append(t)
    if not tempos:
        tempos = {0: 500000}
    tempos.setdefault(0, tempos[min(tempos)])
    marks = sorted(tempos.items())
    if not ts or min(x[0] for x in ts) > 0:
        ts.append((0, 4, 4))
    ts = sorted(ts)

    def sec(T):
        s = 0.0
        for j, (mt, tp) in enumerate(marks):
            if mt >= T:
                break
            nxt = marks[j + 1][0] if j + 1 < len(marks) else T
            s += (min(T, nxt) - mt) / ppq * tp / 1e6
        return s

    bars, tick, k = [], 0, 0
    while tick <= end:
        while k + 1 < len(ts) and ts[k + 1][0] <= tick:
            k += 1
        num, den = ts[k][1], ts[k][2]
        L = max(1, int(round(num * ppq * 4 / den)))
        bars.append(dict(t=round(sec(tick), 5), num=num, den=den, eighths=round(num * 8 / den, 3)))
        tick += L
    beats = [round(sec(q * ppq), 5) for q in range(int(end // ppq) + 2)]
    return dict(bars=bars, beats=beats, drums=[round(sec(t), 5) for t in sorted(set(drums))],
                tempos=[round(60e6 / tp, 3) for _, tp in marks])
