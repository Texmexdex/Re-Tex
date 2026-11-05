"""Autonomous "2+ songs -> finished, editable project" orchestration (design D).

This is the top of the Phase-2 stack: it ingests/separates the input songs, analyzes them, transcribes +
profiles their lyrics, pairs/role-assigns, scores with the lyrics-aware rating, plans the arrangement, builds
the transitions + multi-stem blend, and assembles a *normal, fully-editable* Mashup Lab project (D.3). Every
decision lands as a visible clip/fade/fx on the timeline, so "fully automatic" and "adjust anything after"
are not in tension.

Three pieces are pure and deterministic (unit-tested on CPU with stubbed scores, no GPU):
  * `resolve_style(prompt)`      — the D.2 style/mode prompt -> concrete parameter bundle mapper.
  * `select_bed(ids, scorer)`    — the N>2 bed selection (max aggregate lyrics-aware fit, OD-4).
  * `choose_direction(a, b, ...)`— the 2-song deterministic near-tie rule (D.1 step 4).

The heavy end-to-end `run(...)` glues the real backends together under the D.1a VRAM/single-worker-GPU/unload
discipline. It is submitted with `jobs.submit(..., gpu=True)` so it holds the single `_gpu` worker and nothing
else can load a heavy model underneath it; its stages run strictly in-line and sequentially, each freeing its
model before the next loads. Full GPU end-to-end verification (incl. the >=2 three-song runs and VRAM
measurement) is DEFERRED to the install-time acceptance pass — the pure pieces above and the server routes are
the CPU-testable surface.
"""
import copy
import re

# ================================================================== D.2 style/mode prompt -> parameters
#
# Every preset carries the FULL key set (no implicit defaults to guess), with the exact values pinned in
# design D.2. These are untuned musical defaults confirmed by ear at install time.

STYLE_BUNDLES = {
    "balanced": {
        "vocal_density": 0.50, "chop": False, "tempo_bias": 0.50, "global_rate": 1.00,
        "pitch_down": 0, "swing": 0.00, "energy_floor": 0.00, "swap_vocals": False,
        "add_sweeps": False, "add_fills": False, "lyrics_mode": "resonance",
    },
    "remix": {
        "vocal_density": 0.55, "chop": False, "tempo_bias": 0.50, "global_rate": 1.00,
        "pitch_down": 0, "swing": 0.00, "energy_floor": 0.00, "swap_vocals": True,
        "add_sweeps": True, "add_fills": True, "lyrics_mode": "resonance",
    },
    "chopped and screwed": {
        "vocal_density": 0.50, "chop": True, "tempo_bias": 0.00, "global_rate": 0.80,
        "pitch_down": -3, "swing": 0.15, "energy_floor": 0.00, "swap_vocals": False,
        "add_sweeps": False, "add_fills": False, "lyrics_mode": "off",
    },
    "workout/hype": {
        "vocal_density": 0.55, "chop": False, "tempo_bias": 0.65, "global_rate": 1.00,
        "pitch_down": 0, "swing": 0.00, "energy_floor": 0.50, "swap_vocals": True,
        "add_sweeps": True, "add_fills": True, "lyrics_mode": "resonance",
    },
    "contrast/mashup-of-opposites": {
        "vocal_density": 0.50, "chop": False, "tempo_bias": 0.50, "global_rate": 1.00,
        "pitch_down": 0, "swing": 0.00, "energy_floor": 0.00, "swap_vocals": True,
        "add_sweeps": False, "add_fills": False, "lyrics_mode": "contrast",
    },
}

# The five presets keyed under the aliases the UI/user is likely to type. The canonical bundles above are the
# source of truth; these aliases all resolve to one of the five without touching the pinned values.
STYLE_ALIASES = {
    "balanced": "balanced",
    "50/50": "balanced",
    "even": "balanced",
    "remix": "remix",
    "remixed": "remix",
    "chopped and screwed": "chopped and screwed",
    "chopped & screwed": "chopped and screwed",
    "chopped": "chopped and screwed",
    "screwed": "chopped and screwed",
    "chop and screw": "chopped and screwed",
    "workout": "workout/hype",
    "workout/hype": "workout/hype",
    "workout remix": "workout/hype",
    "hype": "workout/hype",
    "gym": "workout/hype",
    "contrast": "contrast/mashup-of-opposites",
    "contrast/mashup-of-opposites": "contrast/mashup-of-opposites",
    "mashup of opposites": "contrast/mashup-of-opposites",
    "opposites": "contrast/mashup-of-opposites",
}

# keyword -> preset for the deterministic keyword table (first match wins, order = specificity).
_KEYWORD_TABLE = [
    (("chopped and screwed", "chopped & screwed", "chop and screw", "chopped", "screwed"), "chopped and screwed"),
    (("workout", "hype", "gym", "pump", "energetic"), "workout/hype"),
    (("contrast", "opposite", "clash", "versus"), "contrast/mashup-of-opposites"),
    (("remix", "remixed"), "remix"),
    (("balanced", "even", "50/50", "half and half"), "balanced"),
]

# the full knob key set, so an override can never leave a preset half-formed.
STYLE_KEYS = tuple(STYLE_BUNDLES["balanced"].keys())


def _preset(name):
    """A fresh copy of a canonical preset bundle (never a shared reference)."""
    return copy.deepcopy(STYLE_BUNDLES[name])


def _keyword_match(text):
    """Deterministic keyword-table lookup. Returns a preset name or None (unknown free text)."""
    t = text.lower()
    direct = STYLE_ALIASES.get(t.strip())
    if direct:
        return direct
    for words, name in _KEYWORD_TABLE:
        for w in words:
            if w in t:
                return name
    return None


def _zeroshot_preset(text, classifier=None):
    """Local zero-shot fallback for unknown free text (design D.2): embed the prompt and pick the nearest
    preset label. `classifier(text, labels) -> label` lets a caller inject the real embedding classifier (or a
    stub in tests); without one, returns None so the caller keeps `balanced`."""
    if classifier is None:
        return None
    try:
        label = classifier(text, list(STYLE_BUNDLES.keys()))
    except Exception:
        return None
    return label if label in STYLE_BUNDLES else None


# numeric/word cue overrides parsed from otherwise-unknown free text (applied ON TOP of `balanced`, D.2).
_BPM_RE = re.compile(r"(\d{2,3})\s*bpm", re.I)


def _apply_cue_overrides(bundle, text):
    """Override individual knobs from explicitly parsed cues in free text, VERBATIM per design D.2:
        "slower"   -> lower tempo_bias AND global_rate
        "faster"   -> raise tempo_bias
        "half-time"-> tempo_bias 0.0
        "double-time" -> tempo_bias 1.0
        "<n> bpm"  -> recorded as target_bpm (a cue the planner can read; tempo_bias nudged toward it)
    Mutates and returns `bundle`."""
    t = text.lower()
    if "half-time" in t or "half time" in t or "halftime" in t:
        bundle["tempo_bias"] = 0.0
    elif "double-time" in t or "double time" in t or "doubletime" in t:
        bundle["tempo_bias"] = 1.0
    if "slower" in t or "slow down" in t or "slowed" in t:
        bundle["tempo_bias"] = min(bundle["tempo_bias"], 0.25)
        bundle["global_rate"] = min(bundle["global_rate"], 0.90)
    if "faster" in t or "speed up" in t or "sped up" in t:
        bundle["tempo_bias"] = max(bundle["tempo_bias"], 0.75)
    m = _BPM_RE.search(t)
    if m:
        bundle["target_bpm"] = int(m.group(1))
    if "chop" in t or "chopped" in t:
        bundle["chop"] = True
    if "swing" in t or "swung" in t:
        bundle["swing"] = max(bundle["swing"], 0.15)
    return bundle


def resolve_style(prompt=None, classifier=None):
    """Resolve a free-text or preset prompt to the full D.2 knob bundle.

    * No prompt (None / empty / whitespace) -> `balanced` 50/50 verbatim.
    * A preset name/alias or a keyword hit -> that preset's exact pinned bundle.
    * Unknown free text -> the local zero-shot nearest preset if a `classifier` is supplied, otherwise
      `balanced`; then individual knobs are overridden from any explicitly parsed numeric/word cues.

    The returned dict always carries the full STYLE_KEYS set plus a "source" field recording how it was
    resolved (so the project desc shows it) and, when prompt text was given, "prompt". It is stored in the
    project desc verbatim.
    """
    if prompt is None or not str(prompt).strip():
        bundle = _preset("balanced")
        bundle["source"] = "default"
        return bundle

    text = str(prompt).strip()
    name = _keyword_match(text)
    if name is not None:
        bundle = _preset(name)
        bundle["source"] = "preset:" + name
        bundle["prompt"] = text
        return bundle

    # unknown free text: zero-shot nearest preset (if available) else balanced, then cue overrides.
    zs = _zeroshot_preset(text, classifier)
    base = zs if zs is not None else "balanced"
    bundle = _preset(base)
    _apply_cue_overrides(bundle, text)
    bundle["source"] = ("zeroshot:" + zs) if zs is not None else "balanced+cues"
    bundle["prompt"] = text
    return bundle


# ================================================================== D.1 step 4: direction / bed selection

TIE_WINDOW = 1.0          # |overall_VoverI - overall_IoverV| <= 1.0 on the 0..100 scale -> near-tie (MEDIUM-4)
JUDGE_WINDOW = 0.02       # tie-break (a) judgeable considered equal within 0.02


def _score_overall(res):
    """Pull the F1xlyrical overall (0..100) out of a pairs.evaluate result dict. Unworkable -> -inf so it can
    never win a direction choice."""
    if not res or not res.get("workable"):
        return float("-inf")
    return float(res.get("rating", 0.0))


def choose_direction(res_v_over_i, res_i_over_v, judge_v, judge_i, bars_v, bars_i,
                     id_v, id_i, swap_vocals=False):
    """The 2-song deterministic near-tie rule (design D.1 step 4, VERBATIM).

    Inputs are the two directed `pairs.evaluate` results (V's vocal over I, and I's vocal over V), the two
    songs' lyrical `judgeable` (A.2), their usable vocal-bar counts (planner `vocal_beats`/`bars`, F5), and
    their ids. Returns `"v_over_i"` or `"i_over_v"` naming which song leads the vocal.

    When the two directions' `overall` scores are within 1.0 point the ordering is a coin-toss on noise, so it
    is broken deterministically, in order:
        (a) higher lyrical `judgeable` on the leading vocal;
        (b) if those are within 0.02, more usable vocal bars;
        (c) if still tied, the lower song id (stable -> reproducible).
    `swap_vocals` picks only the *primary* lead (both vocals may still trade per the planner), so it does not
    silence anyone — it is accepted for signature symmetry and recorded by the caller; the direction choice is
    unchanged by it.
    """
    sv = _score_overall(res_v_over_i)     # score when V leads
    si = _score_overall(res_i_over_v)     # score when I leads

    if abs(sv - si) > TIE_WINDOW:
        return "v_over_i" if sv >= si else "i_over_v"

    # (a) higher judgeable of the LEADING vocal: v_over_i leads with V, i_over_v leads with I.
    if abs(float(judge_v) - float(judge_i)) > JUDGE_WINDOW:
        return "v_over_i" if float(judge_v) > float(judge_i) else "i_over_v"

    # (b) more usable vocal bars for the leading vocal.
    if int(bars_v) != int(bars_i):
        return "v_over_i" if int(bars_v) > int(bars_i) else "i_over_v"

    # (c) lower song id (stable).
    return "v_over_i" if str(id_v) <= str(id_i) else "i_over_v"


def select_bed(ids, scorer, judge=None, bars=None):
    """N>2 FULL-autonomy bed selection (OD-4, design D.1 step 4).

    The bed is the song maximizing its AGGREGATE lyrics-aware fit as bed against every other input: for each
    candidate bed `b`, sum `scorer(src, b)` (the directed `overall_v2` of each other song `src` as the
    vocal/layer source over `b`) across all `src != b`. Highest aggregate wins. Ties are broken by the same
    deterministic rule as the 2-song case: higher aggregate leading `judgeable`, then more usable bars, then
    lower song id.

    `scorer(src, bed) -> float` is the directed overall (unworkable -> treat as -inf via `_score_overall` if a
    dict is returned, else the float as given). `judge`/`bars` are optional {sid: value} maps used only for the
    tie-break; absent values default to 0. Returns the chosen bed id.
    """
    ids = list(ids)
    if not ids:
        raise ValueError("select_bed needs at least one song id")
    if len(ids) == 1:
        return ids[0]
    judge = judge or {}
    bars = bars or {}

    def agg(bed):
        total = 0.0
        for src in ids:
            if src == bed:
                continue
            s = scorer(src, bed)
            if isinstance(s, dict):
                s = _score_overall(s)
            s = float(s)
            if s == float("-inf"):
                s = 0.0                      # an unworkable direction contributes nothing (not a penalty)
            total += s
        return total

    scored = [(agg(b), b) for b in ids]
    best = max(s for s, _ in scored)
    tied = [b for s, b in scored if abs(s - best) <= TIE_WINDOW]
    if len(tied) == 1:
        return tied[0]

    # tie-break (a) aggregate leading judgeable (the bed is the layer-source-receiver; the "lead" here is the
    # bed's own judgeable as a stable, reproducible proxy), (b) usable bars, (c) lower id.
    def tie_key(b):
        return (-float(judge.get(b, 0.0)), -int(bars.get(b, 0)), str(b))

    return sorted(tied, key=tie_key)[0]


# ================================================================== D.1a VRAM budget / unload discipline

# Per-stage VRAM floors (bytes). ESTIMATES per design D.1a — remeasured at install. The pre-stage guard reads
# free, _ = torch.cuda.mem_get_info() and refuses to load blind when free < required.
GiB = 1024 ** 3
SEP_REQUIRED = 10 * GiB            # separation worst-case (one heavy model at stock batch) — the dominant stage
ASR_REQUIRED = 3 * GiB            # faster-whisper large-v3 fp16
LLM_REQUIRED = 6 * GiB            # optional llama.cpp 7B Q4 tag stage (absent in the v1 default)

# batch-reduction ladders handed through to separate._chunked (D.1a MEDIUM-1): halve on low free VRAM,
# floor at batch=1 (still correct, just slower — acceptable per constraint 3).
SEP_BATCH_LADDER = {
    "ultra": [4, 2, 1], "best": [4, 2, 1], "fast": [4, 2, 1], "demucs": [4, 2, 1],
    "sw": [2, 1], "blend": [2, 1],
}


def _free_vram():
    """Free CUDA VRAM in bytes, or None when torch/CUDA is unavailable (CPU-only test runs)."""
    from . import backends
    return backends.vram_free()


def plan_batch(quality, free_bytes, required=SEP_REQUIRED):
    """Pre-separation guard + batch-reduction fallback (D.1a, VERBATIM).

    Returns the largest batch from the quality's ladder whose working set is expected to fit `free_bytes`
    (halving ~halves the chunk working set, so required scales with batch/stock_batch), or `None` when even
    `batch=1` would exceed free VRAM. `free_bytes=None` (no CUDA / untestable) -> the stock batch (the guard is
    a no-op on a box without a VRAM reading; the real OOM guard is at load time)."""
    ladder = SEP_BATCH_LADDER.get(quality, SEP_BATCH_LADDER["ultra"])
    stock = ladder[0]
    if free_bytes is None:
        return stock
    for batch in ladder:
        need = required * (batch / stock)
        if free_bytes >= need:
            return batch
    return None


class StageOOM(RuntimeError):
    """A MANDATORY heavy stage (separation / ASR) cannot fit even at batch=1 / the smallest model. Names the
    free/required figures so the orchestrator job fails loud rather than OOM-ing mid-render (D.1a MEDIUM-3)."""


def guard_mandatory(stage, free_bytes, required):
    """Assert a mandatory stage can proceed; raise StageOOM naming free/required when it cannot. A `None` free
    reading (no CUDA) passes (the real guard is torch's own OOM on a GPU-less box this never runs on)."""
    if free_bytes is None:
        return
    if free_bytes < required:
        raise StageOOM(f"{stage}: not enough VRAM (free {free_bytes / GiB:.1f} GiB < "
                       f"required {required / GiB:.1f} GiB)")


def guard_optional(free_bytes, required=LLM_REQUIRED):
    """Pre-load guard for the OPTIONAL llama.cpp tag stage: returns True to run it, False to SKIP to the
    zero-shot/TF-IDF tags (A.2.2) — it NEVER raises. A skip is recorded by the caller as
    tags_backend='zeroshot-minilm', has_tags=False so the downgrade is visible (D.1a MEDIUM-3)."""
    if free_bytes is None:
        return True
    return free_bytes >= required


def _empty_cache():
    """Drop any freed CUDA blocks between stages (safe no-op without torch/CUDA)."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _unload_beats():
    """Set beats._model = None + empty_cache before ASR loads: beats._model is a persistent module global that
    would otherwise stay GPU-resident through ASR (D.1a required-unload step)."""
    try:
        from . import beats
        beats._model = None
    except Exception:
        pass
    _empty_cache()


# ================================================================== D.1 end-to-end pipeline

def _emit(progress, f, msg):
    try:
        progress(f, msg)
    except Exception:
        pass


def run(paths, prompt=None, density=None, job=None, progress=lambda f, m="": None,
        asr_factory=None, concept_backend=None, classifier=None, song_ids=None):
    """The autonomous "2+ songs -> finished editable project" pipeline (design D.1).

    `paths`    : list of input audio file paths (>=2). Order only affects deterministic tie-breaks.
    `song_ids` : OPTIONAL list of already-imported, already-'ready' library song ids (the normal UI path:
                 the Auto-Masher only lets the user pick songs already in the library). When given, these are
                 REUSED directly — no re-ingest, no re-separation (never re-stem the library). `paths` is then
                 only the defensive fallback for a hypothetical input that isn't a library song yet, which the
                 current UI can't reach.
    `prompt`   : the D.2 style/mode prompt (None -> balanced 50/50). Resolved via `resolve_style`.
    Returns a normal, fully-editable Mashup Lab project JSON (the same shape the manual timeline opens), with
    the resolved style bundle + transition/blend/lyrics summary stored in `project["mashup"]`.

    This runs IN-LINE on the single-worker `_gpu` slot (the caller submits it `gpu=True`); its stages are
    strictly sequential and each heavy model is unloaded before the next loads (D.1a). It is the install-time
    GPU acceptance target — the pure helpers above are the CPU-testable surface.
    """
    from . import analyze, automash, blend as _blend, library, mashfeat, pairs
    from . import backends, concept as _concept, lyrics as _lyrics, transitions as _trans

    style = resolve_style(prompt, classifier=classifier)
    if density is None:
        density = float(style.get("vocal_density", 0.5))
    lyrics_mode = style.get("lyrics_mode", "resonance")

    inputs = list(paths or [])
    song_ids = list(song_ids or [])
    if not inputs and not song_ids:
        raise ValueError("auto-mashup needs at least one input song")

    # ---- stage 1: resolve each input to a ready library song, REUSING existing stems/analysis ----------
    # The Auto-Masher UI only ever supplies already-'ready' library songs, so the common (and effectively
    # only) path is pure REUSE: skip ingest/separation entirely and go straight to pairing. A brand-new file
    # (not reachable from the current UI) is separated INLINE in the _gpu slot the orchestrator already holds
    # — never by resubmitting another gpu=True job and polling for it, which would self-deadlock the single
    # _gpu worker.
    _emit(progress, 0.02, "preparing songs")
    sids = []
    if song_ids:
        # primary path: explicit already-ready library songs. Reuse directly; do NOT separate.
        for i, sid in enumerate(song_ids):
            info = library.read(sid)
            if not info:
                raise ValueError(f"no such song {sid}")
            if not _song_ready_with_stems(library, sid, info):
                raise ValueError(f"song {sid} is not ready (status={info.get('status')!r}) - "
                                 "finish importing it before an auto-mashup")
            sids.append(sid)
    else:
        # fallback: raw paths. Reuse a matching ready library song when one exists; otherwise separate inline.
        quality = library.default_quality()          # SW + Ultra when both models are here (plan_batch: the Ultra ladder)
        need_sep = False
        for p in inputs:
            if _find_ready_song_by_source(library, p) is None:
                need_sep = True
        if need_sep:
            free = _free_vram()
            batch = plan_batch(quality, free, SEP_REQUIRED)
            if batch is None:
                guard_mandatory("separation", free, SEP_REQUIRED)   # raises StageOOM naming free/required
        for i, p in enumerate(inputs):
            _emit(progress, 0.02 + 0.18 * i / max(1, len(inputs)), f"preparing {i + 1}/{len(inputs)}")
            existing = _find_ready_song_by_source(library, p)
            if existing is not None:
                sids.append(existing)                       # reuse: no separation
                continue
            # genuinely new file: separate INLINE in the held _gpu slot (no new gpu job -> no deadlock)
            sids.append(library.ingest_inline(p, job, quality))
    _empty_cache()

    # ---- stage 2: analyze (grid/key/sections/loudness) + mashfeat (melody/groove) --------------------
    _emit(progress, 0.25, "analyzing")
    for sid in sids:
        if (library.read(sid) or {}).get("status") == "ready":
            try:
                mashfeat.compute(sid)
            except Exception:
                pass
    _unload_beats()

    # ---- stage 3: lyrics (ASR) + concept, batched + cached -------------------------------------------
    _emit(progress, 0.40, "transcribing lyrics")
    guard_mandatory("asr", _free_vram(), ASR_REQUIRED)
    for sid in sids:
        try:
            _lyrics.ensure(sid, backend_factory=asr_factory)
        except Exception:
            pass
    _empty_cache()
    _emit(progress, 0.52, "reading the lyrics")
    run_llm = guard_optional(_free_vram(), LLM_REQUIRED)
    cb = concept_backend if (concept_backend is not None and run_llm) else backends.local_concept()
    if concept_backend is None and not run_llm:
        cb = backends.MiniLmZeroShot()           # optional LLM skipped -> zero-shot/TF-IDF tags (A.2.2)
    for sid in sids:
        ly = _lyrics.load(sid)
        if ly is None:
            continue
        try:
            prof = _concept.profile(ly, backend=cb)
            _concept.write(sid, prof)
        except Exception:
            pass
    try:
        cb.unload()
    except Exception:
        pass
    _empty_cache()

    # ---- stage 4-5: pair / role-assign + lyrics-aware scoring ----------------------------------------
    _emit(progress, 0.62, "pairing + scoring")
    songs = {sid: automash.get_song(sid) for sid in sids}

    def scorer(src, bed):
        try:
            return pairs.evaluate(songs[src], songs[bed], density, lyrics_mode=lyrics_mode)
        except Exception:
            return float("-inf")

    if len(sids) == 2:
        a, b = sids
        r_ab = scorer(a, b)      # a's vocal over b
        r_ba = scorer(b, a)      # b's vocal over a
        ja = _judgeable_of(songs[a])
        jb = _judgeable_of(songs[b])
        bars_a = (r_ab or {}).get("vocal_bars", 0) if isinstance(r_ab, dict) else 0
        bars_b = (r_ba or {}).get("vocal_bars", 0) if isinstance(r_ba, dict) else 0
        choice = choose_direction(r_ab, r_ba, ja, jb, bars_a, bars_b, a, b,
                                  swap_vocals=style.get("swap_vocals", False))
        if choice == "v_over_i":
            vocal_id, bed_id, res = a, b, r_ab
        else:
            vocal_id, bed_id, res = b, a, r_ba
        if not (isinstance(res, dict) and res.get("workable")):
            raise ValueError("neither direction is workable for this pair")
        proj, desc = automash.build_pair(vocal_id, bed_id, res.get("shift_rel"), density)
    else:
        judge = {sid: _judgeable_of(songs[sid]) for sid in sids}
        bed_id = select_bed(sids, scorer, judge=judge)
        others = [s for s in sids if s != bed_id]
        roles = {"music": [bed_id], "vocals": others, "drums": others, "bass": others}
        variants = automash.generate_roles(roles, max_variants=500, density=density,
                                            progress=lambda f, m="": _emit(progress, 0.65 + 0.15 * f, m))
        if not variants:
            raise ValueError("no workable combination among the inputs")
        best = variants[0]
        proj, desc = best["project"], best["desc"]

    # ---- stage 7-9: transitions + blend attached by build_combo (already run inside build_pair/combo) -
    # build_pair/build_combo (D.1 steps 6-9) attach the TransitionPlan automation + blend fx + the lyrics
    # desc line. The style bundle is stored on the project desc so it is visible/editable (D.2).
    _emit(progress, 0.95, "assembling project")
    desc = dict(desc)
    desc["style"] = style
    desc["lyrics_mode"] = lyrics_mode
    proj["mashup"] = desc
    proj.setdefault("desc", {})
    _emit(progress, 1.0, "done")
    return proj


def _judgeable_of(song):
    """A song's lyrical judgeable (A.2) from its on-disk concept profile, 0 when absent."""
    co = getattr(song, "concept", None)
    if not co:
        return 0.0
    return float(co.get("judgeable", 0.0) or 0.0)


def _song_ready_with_stems(library, sid, info=None):
    """True iff `sid` is a 'ready' library song whose four stem files all exist on disk — i.e. it can be
    reused directly for a mashup with no re-separation."""
    from .config import STEMS
    info = info if info is not None else (library.read(sid) or {})
    if info.get("status") != "ready":
        return False
    d = library.song_dir(sid)
    return all((d / "stems" / f"{s}.flac").exists() for s in STEMS)


def _find_ready_song_by_source(library, path):
    """Return the id of an already-'ready' library song whose stored source resolves to the same file as
    `path` (so an auto-mashup input that is really a library song reuses its stems instead of re-separating),
    or None if no ready song matches. Resolution goes through config.resolve_source so it stays portable
    (project-relative sources, never a frozen Desktop/foreign absolute path)."""
    from pathlib import Path
    from .config import resolve_source
    try:
        want = Path(path).resolve()
    except OSError:
        want = Path(path)
    for s in library.list_songs():
        if s.get("status") != "ready":
            continue
        sid = s.get("id")
        info = library.read(sid) or {}
        try:
            got = Path(resolve_source(info.get("source"))).resolve()
        except OSError:
            continue
        if got == want and _song_ready_with_stems(library, sid, info):
            return sid
    return None
