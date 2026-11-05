"""Train the harmonic-fit model on the library:  python -m mashlab.train_harmfit

Positives: each vocal over its own backing. The model scores all 12 relative transpositions per beat and
is trained (softmax over the 12) to put the real one on top, weighted by how clearly a note is sung.
Then it is calibrated: "fit" per beat = exp(-regret / tau); the harmony score's 0 % and 100 % anchors are
the median fit of random pairings (random song, random key) and of each vocal over its own song, both
measured on PLANNED arrangements (the planner picks the best-fitting phrases, which raw fits ignore).
Retraining changes the model version, so every stored mashup rating is recomputed on the next run.
Needs torch (only for training; the app runs the saved weights with numpy).
"""
import os
import sys
import time

import numpy as np

from . import automash as A, harmfit, library

REPS = ["mel", "cqt"]
HIDDEN = 48


def song_arrays(S):
    """-> (interval features (n, B, 12), weights (n,)) for the beats where S's vocal sings over harmony."""
    if not S.has_real_vocals or S.mash is None:
        return None
    w = S.vocal_weight() * S.harm_on()
    sel = w > 0.05
    if sel.sum() < 40:
        return None
    bass, oth, ctx = S.backing()
    X = harmfit.features([S.melody_rep(r)[sel] for r in REPS], bass[sel], oth[sel], ctx[sel])
    return X, w[sel].astype(np.float32), sel


def train(Xs, Ws, epochs=40, seed=0, dev=None):
    """Note-relative MLP (see harmfit.Model), softmax over the 12 transpositions."""
    import torch
    dev = dev or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)

    class Net(torch.nn.Module):
        def __init__(s):
            super().__init__()
            s.l1, s.l2, s.l3 = torch.nn.Linear(60, HIDDEN), torch.nn.Linear(HIDDEN, HIDDEN), torch.nn.Linear(HIDDEN, 1)

        def forward(s, X):
            g = torch.nn.functional.gelu
            v = X[:, 0]
            w = v / (v.sum(1, keepdim=True) + 1e-6)
            outs = []
            for k in range(12):
                Xk = torch.cat([X[:, :2], torch.roll(X[:, 2:], k, dims=2)], 1)
                rots = torch.stack([torch.roll(Xk, -p, dims=2) for p in range(12)], 1).flatten(2) * 5
                f = s.l3(g(s.l2(g(s.l1(rots))))).squeeze(-1)
                outs.append((f * w).sum(1))
            return torch.stack(outs, 1)

    X = torch.tensor(np.concatenate(Xs), dtype=torch.float32, device=dev)
    W = torch.tensor(np.concatenate(Ws), dtype=torch.float32, device=dev)
    net = Net().to(dev)
    opt = torch.optim.AdamW(net.parameters(), 3e-3, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    zero = torch.zeros(1024, dtype=torch.long, device=dev)
    for _ in range(epochs):
        perm = torch.randperm(len(X), device=dev)
        for i in range(0, len(X), 1024):
            idx = perm[i:i + 1024]
            loss = (torch.nn.functional.cross_entropy(net(X[idx]), zero[:len(idx)], reduction="none") * W[idx]).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
    return {k: v.detach().cpu().numpy() for k, v in
            dict(W1=net.l1.weight, b1=net.l1.bias, W2=net.l2.weight, b2=net.l2.bias, W3=net.l3.weight, b3=net.l3.bias).items()}


def main():
    t0 = time.time()
    ids = [s["id"] for s in library.list_songs() if s.get("status") == "ready"]
    songs, data = [], []
    for sid in ids:
        S = A.get_song(sid)
        d = song_arrays(S)
        if d is not None:
            songs.append(S)
            data.append(d)
    print(f"{len(songs)} songs with sung vocals, {sum(len(d[1]) for d in data)} beats", flush=True)
    params = train([d[0] for d in data], [d[1] for d in data])
    meta = dict(version=int(time.time()), arch="noterel", reps=REPS, scale=5.0, topk=3, tau=1.0, fit_random=0.5, fit_own=0.8,
                songs=len(songs))
    model = harmfit.Model(params, meta)

    # ---- calibration
    rng = np.random.default_rng(1)
    own_reg, rand_reg = [], []
    for S, (X, w, sel) in zip(songs, data):
        Sc = model.shift_scores(X)
        own_reg.append((model.regret(Sc, 0), w))
    # random pairings: vocal beats of one song over backing beats of another, random transposition
    for S, (X, w, sel) in zip(songs, data):
        other = songs[rng.integers(len(songs))]
        if other is S:
            continue
        bass, oth, ctx = other.backing()
        on = np.where(other.harm_on() > 0)[0]
        if len(on) < 40:
            continue
        pick = rng.choice(on, len(w))
        Xr = harmfit.features([S.melody_rep(r)[sel] for r in REPS], bass[pick], oth[pick], ctx[pick])
        rand_reg.append((model.regret(model.shift_scores(Xr), int(rng.integers(12))), w))
    allr = np.concatenate([r for r, _ in rand_reg])
    tau = float(np.median(allr))
    fit = lambda r: np.exp(-r / tau)
    wmean = lambda pairs: [float((fit(r) * w).sum() / w.sum()) for r, w in pairs]
    f_own, f_rand = wmean(own_reg), wmean(rand_reg)
    meta.update(tau=round(tau, 4), fit_own=round(float(np.median(f_own)), 4), fit_random=round(float(np.median(f_rand)), 4),
                fit_own_q=[round(float(q), 3) for q in np.quantile(f_own, [0.1, 0.25, 0.5, 0.75, 0.9])],
                fit_random_q=[round(float(q), 3) for q in np.quantile(f_rand, [0.1, 0.25, 0.5, 0.75, 0.9])])
    # how much key matters on a beat (score spread over transpositions), typical for a sung vocal
    u = [(float((model.shift_scores(X).std(1) * w).sum() / w.sum()), S.stability) for S, (X, w, sel) in zip(songs, data)]
    meta["u_ref"] = round(float(np.median([x for x, st in u if st >= 0.6] or [x for x, _ in u])), 4)
    harmfit.save(params, meta)

    # ---- the 0 % / 100 % anchors of the harmony score, measured on PLANNED arrangements (the planner
    # picks the best-fitting phrases, so raw per-beat fits would be too easy to beat)
    print("calibrating against planned arrangements...", flush=True)
    from concurrent.futures import ProcessPoolExecutor
    ids_v = [S.id for S in songs]
    tasks = [(s, s, 0) for s in ids_v]
    for _ in range(300):
        v, i = ids_v[rng.integers(len(ids_v))], ids[rng.integers(len(ids))]
        if v != i:
            tasks.append((v, i, int(rng.integers(-6, 6))))
    from .pairs import _single_thread_init
    with ProcessPoolExecutor(max_workers=max(2, min(16, (os.cpu_count() or 4) - 4)), initializer=_single_thread_init) as ex:
        res = [r for r in ex.map(_plan_fit, tasks, chunksize=8) if r is not None]
    own = [f for v, i, f in res if v == i]
    rnd = [f for v, i, f in res if v != i]
    meta.update(plan_fit_own=round(float(np.median(own)), 4), plan_fit_random=round(float(np.median(rnd)), 4))
    harmfit.save(params, meta)
    print("saved", harmfit.MODEL_PATH, meta, f"{time.time() - t0:.0f}s", flush=True)


def _plan_fit(task):
    from . import planner
    v, i, r = task
    try:
        g = planner.Grid(A.get_song(v), A.get_song(i), 1.0)
        return v, i, planner.plan_vocals(g, r, 0.5)["fit_info"]
    except Exception:
        return None


if __name__ == "__main__":
    sys.exit(main())
