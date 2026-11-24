"""Link to ACE Studio on this PC: drives `acestudio-cli.exe --json` as a subprocess (no Python SDK dependency),
checks unsaved changes before staging a Re-Tex bounce, polls jobs cleanly, exports rendered takes, reads cloud
generation history for 0-credit reuse, and writes through to the user's ACE Studio Library vault.

Configured by data/config.json "ace": {"cli": "C:/Program Files/ACE Studio/acestudio-cli.exe",
"vault": "C:/Users/<user>/Music/ACE Studio Library"}. Override with MASHLAB_ACE_CLI / MASHLAB_ACE_VAULT.
"""
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
import uuid
from pathlib import Path

from . import appcfg, components

DEFAULT_CLI = "C:/Program Files/ACE Studio/acestudio-cli.exe"
DEFAULT_VAULT = str(Path.home() / "Music" / "ACE Studio Library")

ACE_V2M_LANGUAGES = ("english", "chinese", "japanese", "spanish", "korean", "french", "italian", "portuguese")


class AceError(RuntimeError):
    def __init__(self, message, code=""):
        super().__init__(message)
        self.code = code or ""


def _cfg():
    """(cli_path, vault_dir)."""
    c = (appcfg.load_config() or {}).get("ace") or {}
    cli = os.environ.get("MASHLAB_ACE_CLI") or c.get("cli") or DEFAULT_CLI
    vault = os.environ.get("MASHLAB_ACE_VAULT") or c.get("vault") or DEFAULT_VAULT
    return str(cli), str(vault)


def cli_exe():
    cli, _ = _cfg()
    p = Path(cli)
    if p.exists():
        return str(p)
    found = shutil.which("acestudio-cli") or shutil.which("acestudio-cli.exe")
    return found or ""


def _run(args, timeout=120):
    """Run `acestudio-cli.exe <args> --json` and return parsed JSON. Raises AceError on CLI failure."""
    exe = cli_exe()
    if not exe:
        raise components.Missing("ace_studio", f"acestudio-cli.exe not found (checked {_cfg()[0]})")
    cmd = [exe, *list(args), "--json"]
    env = dict(os.environ, ACESTUDIO_CLI_NO_AGENT_HINT="1", PYTHONUTF8="1")
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
                              timeout=timeout, env=env, creationflags=flags)
    except subprocess.TimeoutExpired as e:
        raise AceError(f"ACE Studio CLI timed out after {timeout}s ({' '.join(args[:3])})", code="TIMEOUT") from e
    except OSError as e:
        raise AceError(f"Couldn't run ACE Studio CLI: {e}", code="EXEC_FAILED") from e

    out = (proc.stdout or "").strip()
    err = (proc.stderr or "").strip()
    if proc.returncode != 0:
        code, msg = "", err or out or f"exit {proc.returncode}"
        for blob in (err, out):
            if not blob:
                continue
            try:
                j = json.loads(blob)
                eobj = j.get("error") if isinstance(j, dict) else None
                if isinstance(eobj, dict):
                    code = str(eobj.get("code") or "")
                    msg = str(eobj.get("message") or msg)
                    hint = str(eobj.get("hint") or "").strip()
                    if hint and hint not in msg:
                        msg = f"{msg} — {hint}"
                    break
            except ValueError:
                pass
        if code == "CREDIT_INSUFFICIENT":
            msg = f"Not enough ACE Studio credits for this operation ({msg})"
        elif code == "UNSAVED_CHANGES":
            msg = f"ACE Studio has unsaved changes in the open project. Save it in ACE Studio first ({msg})"
        raise AceError(f"ACE Studio: {msg}", code=code)

    if not out:
        return {}
    try:
        return json.loads(out)
    except ValueError as e:
        raise AceError(f"ACE Studio returned non-JSON output: {out[:200]}", code="BAD_JSON") from e


def status():
    """Live ACE Studio status dict (never raises)."""
    exe = cli_exe()
    if not exe:
        return dict(installed=False, cli=_cfg()[0], running=False, bridge=False, awaiting_user="",
                    project_name="", dirty=False, is_temp=True, error="acestudio-cli.exe not found")
    try:
        st = _run(["app", "status"], timeout=6)
    except Exception as e:
        return dict(installed=True, cli=exe, running=False, bridge=False, awaiting_user="",
                    project_name="", dirty=False, is_temp=True, error=str(e))

    state_str = str(st.get("state") or "").lower()
    running = bool(
        state_str in ("bridge-ready", "running", "starting", "ready")
        or st.get("running") or st.get("bridgeServing") or st.get("serving")
    )
    bridge = bool(
        state_str in ("bridge-ready", "ready")
        or st.get("bridgeServing") or st.get("bridge") or st.get("serving") or st.get("ready")
    )
    # Verify actual bridge responsiveness by reading project info + awaiting-user when running
    awaiting, pname, dirty, is_temp, err = "", "", False, True, ""
    if running or bridge:
        try:
            aw = _run(["app", "awaiting-user"], timeout=6)
            au = aw.get("awaitingUser")
            if isinstance(au, dict):
                awaiting = str(au.get("message") or au.get("screen") or "")
            elif au:
                awaiting = str(au)
        except Exception:
            pass
        try:
            pi = _run(["project", "info"], timeout=6)
            pname = str(pi.get("projectName") or "")
            is_temp = bool(pi.get("isTempProject", True))
            bridge = True
            running = True
        except Exception as e:
            err = str(e)
            bridge = False
        if bridge:
            try:
                pd = _run(["project", "dirty"], timeout=6)
                dirty = bool(pd.get("dirty") or pd.get("isDirty") or pd.get("hasUnsavedChanges"))
            except Exception:
                pass
    return dict(installed=True, cli=exe, running=running, bridge=bridge, awaiting_user=awaiting,
                project_name=pname, dirty=dirty, is_temp=is_temp, error=err)


def ensure_running(wait_s=90):
    """Ensure ACE Studio is installed, running, and its bridge is serving (launches it if closed)."""
    components.require("ace_studio")
    st = status()
    if st["bridge"]:
        if st["awaiting_user"]:
            raise AceError(f"ACE Studio needs your attention first: {st['awaiting_user']}", code="AWAITING_USER")
        return st
    _run(["app", "launch", "--wait", "--timeout", f"{int(wait_s)}s"], timeout=int(wait_s) + 15)
    st = status()
    if not st["bridge"]:
        raise AceError(st.get("error") or "ACE Studio launched, but its CLI bridge isn't responding yet", code="NOT_READY")
    if st["awaiting_user"]:
        raise AceError(f"ACE Studio needs your attention first: {st['awaiting_user']}", code="AWAITING_USER")
    return st


def ensure_clean_stage(bpm=120.0, allow_discard_temp=True):
    """Open a fresh staging project in ACE Studio at `bpm` without clobbering an unsaved user project."""
    st = ensure_running()
    if st["dirty"] and not (allow_discard_temp and st["is_temp"] and not st["project_name"]):
        who = f"'{st['project_name']}'" if st["project_name"] else "an unsaved project"
        raise AceError(
            f"ACE Studio currently has unsaved changes in {who}. Please save or close it in ACE Studio first "
            "so Re-Tex does not overwrite your session.",
            code="UNSAVED_CHANGES",
        )
    new_args = ["project", "new", "--wait-busy", "10s"]
    if st["dirty"] and allow_discard_temp and st["is_temp"] and not st["project_name"]:
        new_args.append("--discard-changes")
    _run(new_args, timeout=30)
    bpm_clamped = max(20.0, min(300.0, float(bpm or 120.0)))
    _run(["tempo", "set-point", "--pos", "0t", "--bpm", f"{bpm_clamped:.3f}", "--wait-busy", "5s"], timeout=20)
    return bpm_clamped


def create_audio_track(name="Re-Tex Layer", index=None):
    args = ["track", "create", "--type", "audio", "--name", str(name), "--wait-busy", "5s"]
    if index is not None:
        args += ["--index", str(int(index))]
    return _run(args, timeout=20)


def import_audio(wav_path, track_index=0, pos="0t", wait_loaded_s=35.0):
    """Import WAV onto `track_index` and wait until ACE Studio finishes decoding it."""
    p = str(Path(wav_path).resolve())
    res = _run(["import", "file", "--path", p, "--track-index", str(int(track_index)), "--pos", str(pos),
                "--wait-busy", "5s"], timeout=60)
    tidx = res.get("trackIndex", track_index)
    t0 = time.time()
    while time.time() - t0 < wait_loaded_s:
        try:
            c = _run(["clip", "audio-content", "--track-index", str(int(tidx)), "--clip-index", "0"], timeout=10)
            ls = c.get("loadingState")
            if ls == "loaded_success":
                break
            if ls == "loaded_failed":
                raise AceError(f"ACE Studio failed to decode imported audio ({Path(p).name})", code="LOAD_FAILED")
        except AceError as e:
            if e.code == "LOAD_FAILED":
                raise
        time.sleep(0.6)
    return res


def set_arrangement_selection(begin_tick, end_tick, begin_index=0, end_index=0):
    horiz = json.dumps({"begin": int(begin_tick), "end": int(end_tick)})
    vert = json.dumps({
        "beginRegion": "arrangement", "beginIndex": int(begin_index),
        "endRegion": "arrangement", "endIndex": int(end_index),
    })
    return _run(["selection", "set", "--scope", "arrangement",
                 "--horizontal-selection", horiz, "--vertical-selection", vert, "--wait-busy", "5s"], timeout=20)


def music_enhancer(prompt=None, lyrics=None, influence=0.5, title=None):
    """Launch ACE Studio's Music Enhancer over the current arrangement selection (spends credits)."""
    args = ["generative", "music-enhancer", "--influence", f"{max(0.0, min(1.0, float(influence))):.2f}",
            "--wait-busy", "10s"]
    if prompt is not None and str(prompt).strip():
        args += ["--prompt", str(prompt).strip()]
    if lyrics is not None and str(lyrics).strip():
        args += ["--lyrics", str(lyrics)]
    if title is not None and str(title).strip():
        args += ["--title", str(title).strip()[:120]]
    return _run(args, timeout=60)


def add_a_layer(track_uuid, from_t, to_t, sound_type="instrumental", instrument=None, prompt=None, lyrics=None):
    """Launch ACE Studio's Add A Layer onto `track_uuid` over `[from_t, to_t)` (spends credits)."""
    st = str(sound_type or "instrumental").strip().lower()
    args = ["generative", "add-a-layer", "--track-uuid", str(track_uuid),
            "--from", str(from_t), "--to", str(to_t), "--sound-type", st, "--wait-busy", "10s"]
    if st == "custom" and instrument and str(instrument).strip():
        args += ["--instrument", str(instrument).strip()]
    if prompt and str(prompt).strip():
        args += ["--prompt", str(prompt).strip()]
    if st in ("song-track", "vocals", "backing-vocals") and lyrics and str(lyrics).strip():
        args += ["--lyrics", str(lyrics)]
    return _run(args, timeout=60)


def voice_changer_models(force_live=False):
    """Available free Voice Changer models [{id, name, kind, tags}]. Reads capabilities.json cache first for speed."""
    _, vault = _cfg()
    cap_path = Path(vault) / "capabilities.json"
    if not force_live and cap_path.exists():
        try:
            caps = json.loads(cap_path.read_text(encoding="utf-8-sig"))
            models = caps.get("voiceChangerModels") or []
            if models:
                return models
        except (OSError, ValueError):
            pass
    ensure_running()
    res = _run(["generative", "voice-changer", "models"], timeout=30)
    out = []
    for m in res.get("models") or []:
        tags = [t for t in (m.get("tags") or []) if str(t).strip()]
        out.append(dict(id=int(m["id"]), name=str(m.get("name") or f"Model {m['id']}"),
                        kind=str(m.get("kind") or "voice"), tags=tags))
    return out


def voice_changer_convert(track_uuids, from_t, to_t, model_ids, semitones=0, pitch_correction=20,
                          random_offset=0, correct_to_key=None, correct_to_scale="Major",
                          remove_instrument=False, remove_reverb=False):
    """Launch ACE Studio's FREE Voice Changer conversion across 1..N models in a single source render."""
    t_list = list(dict.fromkeys(str(u) for u in track_uuids if u))
    m_list = list(dict.fromkeys(int(m) for m in model_ids))
    if not t_list:
        raise ValueError("at least one track_uuid is required")
    if not m_list:
        raise ValueError("pick at least one Voice Changer model")
    args = ["generative", "voice-changer", "convert", "--from", str(from_t), "--to", str(to_t), "--wait-busy", "10s"]
    for u in t_list:
        args += ["--track-uuid", u]
    for mid in m_list:
        args += ["--model-id", str(mid)]
    args += ["--semitones", str(max(-24, min(24, int(semitones))))]
    args += ["--pitch-correction", str(max(0, min(100, int(pitch_correction))))]
    args += ["--random-offset", str(max(0, min(100, int(random_offset))))]
    if correct_to_key and str(correct_to_key).strip():
        args += ["--correct-to-key", str(correct_to_key).strip()]
        if correct_to_scale and str(correct_to_scale).strip():
            args += ["--correct-to-scale", str(correct_to_scale).strip()]
    if remove_instrument:
        args += ["--remove-instrument", "true"]
    if remove_reverb:
        args += ["--remove-reverb", "true"]
    return _run(args, timeout=60)


_LRC_ROW = re.compile(r"^\[(\d{1,3}):(\d{2}(?:\.\d+)?)\](.*)$")
_KEEP_DUP_WORDS = {"no", "oh", "ah", "la", "na", "ha", "uh", "ooh", "yeah", "hey", "go", "do", "so", "my", "me", "ya"}


def parse_lrc(lrc_text, total_dur=None):
    """Parse ACE Studio's exported LRC text into Re-Tex timed lyric lines [{t0, t1, text, words}].
    Cleans ACE's multi-note syllable continuation tags (word#1 / word#2) and accidental melisma duplicates."""
    raw_rows = []
    for line in str(lrc_text or "").splitlines():
        m = _LRC_ROW.match(line.strip())
        if not m:
            continue
        t = int(m.group(1)) * 60.0 + float(m.group(2))
        raw_rows.append((t, m.group(3).strip()))

    lines = []
    prev_tail_root = ""
    for i, (t0, txt) in enumerate(raw_rows):
        if not txt:
            prev_tail_root = ""
            continue
        # End time: next LRC timestamp (ACE emits empty [mm:ss.xx] end markers after phrases) or fallback +3s
        if i + 1 < len(raw_rows):
            t1 = max(t0 + 0.15, raw_rows[i + 1][0])
        elif total_dur and total_dur > t0:
            t1 = float(total_dur)
        else:
            t1 = t0 + 3.0

        # Clean ACE syllable markers (#1 kept without tag; #2/#3... continuations dropped) and accidental duplicates
        cleaned_tokens = []
        for tok in txt.split():
            m_tag = re.match(r"^(.+?)#(\d+)([.,!?;:'\"]*)$", tok)
            if m_tag:
                root, idx_num, punct = m_tag.group(1), int(m_tag.group(2)), m_tag.group(3)
                if idx_num > 1:
                    continue
                tok = root + punct
            norm = re.sub(r"[^a-z0-9']", "", tok.lower())
            if not norm:
                continue
            # Drop melismatic duplicate carried across adjacent line boundary or repeated back-to-back (>2 chars)
            if not cleaned_tokens and prev_tail_root and norm == prev_tail_root and len(norm) > 2 and norm not in _KEEP_DUP_WORDS:
                continue
            if cleaned_tokens:
                last_norm = re.sub(r"[^a-z0-9']", "", cleaned_tokens[-1].lower())
                if norm == last_norm and len(norm) > 2 and norm not in _KEEP_DUP_WORDS:
                    continue
            cleaned_tokens.append(tok)

        clean_txt = " ".join(cleaned_tokens).strip()
        if not clean_txt:
            continue
        prev_tail_root = re.sub(r"[^a-z0-9']", "", cleaned_tokens[-1].lower())
        lines.append(dict(t0=round(t0, 3), t1=round(t1, 3), text=clean_txt, words=[]))
    return lines


def vocal_to_midi_lyrics(wav_path, bpm=120.0, language="english",
                         progress=lambda f, m="": None, cancelled=lambda: False):
    """Run ACE Studio's FREE (0-credit) Vocal-to-MIDI V2 on an isolated vocal stem WAV and extract timestamped
    lyric lines via `export lrc`. Protects any unsaved user project via ensure_clean_stage()."""
    lang = str(language or "english").strip().lower()
    if lang not in ACE_V2M_LANGUAGES:
        raise ValueError(f"ACE Vocal-to-MIDI language must be one of {', '.join(ACE_V2M_LANGUAGES)}")

    progress(0.05, "connecting to ACE Studio (free Vocal-to-MIDI)")
    ensure_clean_stage(bpm=bpm)
    progress(0.14, "importing vocal stem into ACE Studio")
    imp = import_audio(wav_path, track_index=0, pos="0t")
    clip_uuid = imp.get("clipUuid")
    if not clip_uuid:
        cl = _run(["clip", "list", "--track-index", "0"], timeout=15)
        clips = cl.get("clips") or []
        clip_uuid = clips[0].get("uuid") if clips else None
    if not clip_uuid:
        raise AceError("Couldn't resolve imported vocal clip UUID in ACE Studio", code="NO_CLIP")

    dur_sec = float((imp.get("geometry") or {}).get("durSec") or imp.get("naturalDurSec") or 0.0)
    progress(0.22, f"ACE Studio (Free): transcribing vocal ({lang})")
    launch = _run([
        "generative", "vocal-to-midi",
        "--clip-uuid", str(clip_uuid),
        "--language", lang,
        "--apply-pitch", "false",
        "--wait-busy", "10s",
    ], timeout=60)
    wait_job(launch["jobId"], progress=progress, lo=0.25, hi=0.88, cancelled=cancelled,
             label="ACE Vocal-to-MIDI (Free)", poll_s=2.0, timeout_s=360)

    sing_tuuid = launch.get("trackUuid")
    if not sing_tuuid:
        raise AceError("ACE Studio Vocal-to-MIDI did not return a Sing track UUID", code="NO_TRACK")

    progress(0.92, "ACE Studio: exporting timestamped lyrics (LRC)")
    stage = Path(tempfile.mkdtemp(prefix="retex_ace_lrc_"))
    try:
        lrc_path = stage / "lyrics.lrc"
        exp = _run(["export", "lrc", "--path", str(lrc_path), "--track-uuid", str(sing_tuuid)], timeout=30)
        raw_lrc = lrc_path.read_text(encoding="utf-8", errors="replace") if lrc_path.exists() else ""
        lines = parse_lrc(raw_lrc, total_dur=(dur_sec if dur_sec > 0 else None))
        progress(1.0, f"done ({len(lines)} lines)")
        return dict(lines=lines, raw_lrc=raw_lrc, language=lang, track_name=str(exp.get("trackName") or ""))
    finally:
        shutil.rmtree(stage, ignore_errors=True)


def wait_job(job_id, progress=lambda f, m="": None, lo=0.0, hi=1.0, cancelled=lambda: False,
             label="ACE Studio", poll_s=2.5, timeout_s=900):
    """Poll `job get <job_id>` until lifecycle is terminal (`succeeded | failed | cancelled`)."""
    t0 = time.time()
    while True:
        if cancelled():
            try:
                j = _run(["job", "get", str(job_id)], timeout=10)
                if j.get("cancelable") or j.get("cancellable"):
                    _run(["job", "cancel", str(job_id)], timeout=15)
            except Exception:
                pass
            raise AceError("cancelled", code="CANCELLED")
        if time.time() - t0 > timeout_s:
            raise AceError(f"{label} timed out after {int(timeout_s)}s", code="TIMEOUT")
        j = _run(["job", "get", str(job_id)], timeout=20)
        lc = str(j.get("lifecycle") or "").lower()
        if lc == "succeeded":
            progress(hi, f"{label}: finished")
            return j
        if lc in ("failed", "cancelled", "canceled"):
            err = j.get("error") or j.get("message") or f"job {lc}"
            raise AceError(f"{label} {lc}: {err}", code=lc.upper())

        # Compute progress from explicit progress field, result settling ratio, or smooth elapsed curve
        raw_p = j.get("progress")
        if isinstance(raw_p, (int, float)):
            frac = float(raw_p) / 100.0 if float(raw_p) > 1.0 else float(raw_p)
        else:
            res_list = j.get("results") or []
            if res_list:
                settled = sum(1 for r in res_list if str(r.get("state")).lower() in ("settled", "local-ready"))
                streaming = sum(1 for r in res_list if str(r.get("state")).lower() == "streaming")
                frac = (settled + 0.5 * streaming) / max(1, len(res_list))
                frac = max(frac, min(0.85, (time.time() - t0) / 150.0))
            else:
                frac = min(0.88, (time.time() - t0) / 120.0)
        elapsed = int(time.time() - t0)
        progress(lo + (hi - lo) * max(0.0, min(0.98, frac)), f"{label}: {lc or 'working'} ({elapsed}s)")
        time.sleep(poll_s)


def job_results(job_id):
    return _run(["job", "results", str(job_id)], timeout=20)


def wait_settled_result_ids(job_id, kit="music-enhancer", title=None, timeout_s=35.0):
    """Wait until a succeeded staged job's results settle from ephemeral IDs (e.g. 'res-5') into durable
    'task_id:audio_id' handles, falling back to a refreshed cloud_history lookup by title if needed."""
    t0 = time.time()
    last_items = []
    while time.time() - t0 < timeout_s:
        try:
            info = job_results(job_id)
            items = info.get("results") or []
        except Exception:
            items = []
        if items:
            last_items = items
            all_settled = all(str(r.get("state") or "").lower() in ("settled", "local-ready") for r in items)
            all_durable = all(":" in str(r.get("id") or r.get("resultId") or "") for r in items)
            if all_settled and all_durable:
                return [str(r.get("id") or r.get("resultId")) for r in items]
        time.sleep(1.5)

    # Return any durable IDs already present in the job ledger
    durable = [str(r.get("id") or r.get("resultId")) for r in last_items
               if ":" in str(r.get("id") or r.get("resultId") or "")]
    if durable:
        return durable

    # Fallback: resolve durable IDs from account cloud history (matched by title when given)
    try:
        hist = cloud_history(kit=kit, max_pages=1, refresh=True)
        if title:
            base = str(title).strip().lower()
            matched = [h["result_id"] for h in hist if str(h.get("title") or "").lower().startswith(base)]
            if matched:
                return matched
        if hist:
            # Group newest task_id's takes
            top_task = hist[0].get("task_id")
            if top_task:
                return [h["result_id"] for h in hist if h.get("task_id") == top_task]
            return [h["result_id"] for h in hist[:2]]
    except Exception:
        pass

    return [str(r.get("id") or r.get("resultId")) for r in last_items if (r.get("id") or r.get("resultId"))]


def _paged_query(base_args, page=1, page_size=10, refresh=False, max_warm=8):
    args = list(base_args) + ["--page", str(int(page)), "--page-size", str(int(page_size))]
    if refresh:
        args += ["--refresh", "true"]
    for _ in range(max_warm):
        data = _run(args, timeout=30)
        if isinstance(data, dict) and data.get("status") == "warming":
            if refresh:
                args = list(base_args) + ["--page", str(int(page)), "--page-size", str(int(page_size))]
                refresh = False
            time.sleep(1.0)
            continue
        return data
    raise AceError(f"ACE Studio history stayed warming ({' '.join(base_args)})", code="WARMING_TIMEOUT")


def cloud_history(kit="all", max_pages=3, page_size=10, refresh=False):
    """Fetch ACE Studio account cloud history (0 credits, read-only) and register resultIds into the job ledger.
    kit: 'music-enhancer' | 'inspire-me' | 'all' -> list of flat audio entries."""
    ensure_running()
    items = []

    if kit in ("music-enhancer", "all"):
        for page in range(1, max_pages + 1):
            sd = _paged_query(["generative", "music-enhancer", "sessions", "list"],
                              page=page, page_size=page_size, refresh=(refresh and page == 1))
            sessions = sd.get("sessions") or []
            if not sessions:
                break
            for s in sessions:
                if s.get("status") != "ready" or not s.get("taskId"):
                    continue
                bpm_val = None
                try:
                    bpm_val = float(s.get("bpm")) if s.get("bpm") else None
                except (TypeError, ValueError):
                    pass
                for rp in range(1, max_pages + 1):
                    rd = _paged_query(["generative", "music-enhancer", "results", "list", "--session", str(s["taskId"])],
                                      page=rp, page_size=page_size, refresh=(refresh and rp == 1))
                    gens = rd.get("generations") or []
                    if not gens:
                        break
                    for g in gens:
                        if g.get("status") == "failed":
                            continue
                        for k, a in enumerate(g.get("audios") or [], 1):
                            rid = a.get("resultId")
                            if not rid:
                                continue
                            items.append(dict(
                                result_id=str(rid), kit="music-enhancer", task_id=str(g.get("taskId") or ""),
                                session_id=str(s["taskId"]),
                                title=str(g.get("title") or s.get("title") or "Enhanced Track") + (
                                    f" (Take {k})" if len(g.get("audios") or []) > 1 else ""),
                                tags=str(g.get("tags") or s.get("tags") or ""),
                                lyrics=str(g.get("lyrics") or s.get("lyrics") or ""),
                                bpm=bpm_val, duration=a.get("duration"),
                                state=str(a.get("state") or "need-download"),
                                local_path=a.get("localPath") or "",
                            ))
                    if rp * page_size >= int(rd.get("total") or 0):
                        break
            if page * page_size >= int(sd.get("total") or 0):
                break

    if kit in ("inspire-me", "all"):
        for page in range(1, max_pages + 1):
            d = _paged_query(["generative", "inspire-me", "history", "list"],
                             page=page, page_size=page_size, refresh=(refresh and page == 1))
            gens = d.get("generations") or []
            if not gens:
                break
            for g in gens:
                if g.get("status") == "failed":
                    continue
                for k, a in enumerate(g.get("audios") or [], 1):
                    rid = a.get("resultId")
                    if not rid:
                        continue
                    items.append(dict(
                        result_id=str(rid), kit="inspire-me", task_id=str(g.get("taskId") or ""),
                        session_id="",
                        title=str(g.get("title") or "Inspire Me Song") + (
                            f" (Take {k})" if len(g.get("audios") or []) > 1 else ""),
                        tags=str(g.get("tags") or ""),
                        lyrics=str(g.get("lyrics") or ""),
                        bpm=None, duration=a.get("duration"),
                        state=str(a.get("state") or "need-download"),
                        local_path=a.get("localPath") or "",
                    ))
            if page * page_size >= int(d.get("total") or 0):
                break

    return items


def fetch_staged_result(result_id, dst_path, kit="music-enhancer", progress=lambda f, m="": None,
                        lo=0.0, hi=1.0, cancelled=lambda: False):
    """Download (if needed) a staged ACE resultId and copy/export it to `dst_path` (local file path)."""
    dst = Path(dst_path)
    dst.parent.mkdir(parents=True, exist_ok=True)

    # First check history to see if it already has a localPath on disk
    rid = str(result_id)

    def _lookup_history(refresh_flag=False):
        try:
            for item in cloud_history(kit=kit, max_pages=3, refresh=refresh_flag):
                if item["result_id"] == rid:
                    return item
        except Exception:
            pass
        return None

    def _save_local(local_file):
        local_p = Path(local_file)
        if dst.suffix.lower() != local_p.suffix.lower() and dst.suffix.lower() in (".flac", ".wav"):
            from . import audio_io
            from .config import SR
            audio_io.save(dst, audio_io.load(local_p, SR), SR)
        else:
            shutil.copy2(local_p, dst)
        return dst

    # 1. Check if ACE Studio already downloaded the file locally (or is actively auto-downloading it)
    item = _lookup_history(refresh_flag=True)
    if item:
        if item.get("local_path") and Path(item["local_path"]).exists():
            return _save_local(item["local_path"])
        if item.get("state") == "downloading":
            for _ in range(25):
                if cancelled():
                    raise AceError("cancelled", code="CANCELLED")
                time.sleep(1.5)
                item = _lookup_history(refresh_flag=True)
                if item and item.get("local_path") and Path(item["local_path"]).exists():
                    return _save_local(item["local_path"])
                if not item or item.get("state") != "downloading":
                    break

    # 2. Request explicit download if still need-download
    progress(lo, "ACE Studio: downloading take")
    try:
        dl = _run(["job", "download", rid], timeout=30)
        if dl.get("jobId"):
            wait_job(dl["jobId"], progress=progress, lo=lo, hi=lo + 0.8 * (hi - lo), cancelled=cancelled,
                     label="ACE Studio download", poll_s=1.5, timeout_s=300)
    except AceError as e:
        # INVALID_ARG = already local; NOT_FOUND = check refreshed cloud history before falling back
        if e.code not in ("INVALID_ARG", "NOT_FOUND"):
            raise

    item = _lookup_history(refresh_flag=True)
    if item and item.get("local_path") and Path(item["local_path"]).exists():
        return _save_local(item["local_path"])

    # 3. Fallback: place onto a fresh audio track and export audio cleanly (no silent data loss)
    progress(lo + 0.7 * (hi - lo), "ACE Studio: bouncing staged take")
    tr = create_audio_track(name="Re-Tex Export")
    tuuid = tr.get("trackUuid") or tr.get("uuid")
    if not tuuid:
        raise AceError(f"Couldn't resolve downloaded audio path for result {rid}", code="NO_LOCAL_PATH")
    _run(["job", "place", rid, "--track-uuid", str(tuuid), "--at", "0t", "--wait-busy", "10s"], timeout=60)
    return export_track(tuuid, dst)


def export_track(track_uuid, dst_path, from_t=None, to_t=None):
    """Export one ACE Studio track over optional [from_t, to_t) to `dst_path` (handles track-name suffixing)."""
    dst = Path(dst_path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix="retex_ace_exp_"))
    try:
        base = stage / "bounce.wav"
        args = ["export", "audio", "--scope", "tracks", "--track-uuid", str(track_uuid),
                "--path", str(base), "--wait", "--wait-busy", "10s"]
        if from_t is not None:
            args += ["--from", str(from_t)]
        if to_t is not None:
            args += ["--to", str(to_t)]
        _run(args, timeout=300)
        files = sorted(stage.glob("*.*"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not files:
            raise AceError("ACE Studio audio export produced no file", code="EMPTY_EXPORT")
        src = files[0]
        if dst.suffix.lower() == ".flac":
            from . import audio_io
            from .config import SR
            audio_io.save(dst, audio_io.load(src, SR), SR)
        else:
            shutil.copy2(src, dst)
        return dst
    finally:
        shutil.rmtree(stage, ignore_errors=True)


# ------------------------------------------------------------------ Personal Agent Vault sync (0-credit reuse)

def vault_register(kind, source, title, prompt="", tags=None, instrument=None, bpm=None, key=None,
                   duration_sec=None, result_id=None, local_path=None, credit_cost=None, notes=""):
    """Write-through registration into `C:/Users/<user>/Music/ACE Studio Library/catalog.json` (schema v2)."""
    _, vault = _cfg()
    vroot = Path(vault)
    cat_path = vroot / "catalog.json"
    try:
        vroot.mkdir(parents=True, exist_ok=True)
        if cat_path.exists():
            cat = json.loads(cat_path.read_text(encoding="utf-8-sig"))
        else:
            cat = {
                "version": 2, "libraryRoot": str(vroot), "lastCloudSync": None,
                "stats": {"totalEntries": 0, "creditsSavedReuses": 0, "estimatedCreditsSaved": 0.0, "totalCreditsLogged": 0.0},
                "entries": [],
            }
        entries = list(cat.get("entries") or [])
        eid = str(result_id) if result_id else f"{kind}-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        tag_list = list(dict.fromkeys([t.strip().lower() for t in (tags or []) if str(t).strip()] + [kind, source]))
        norm_lp = str(Path(local_path).resolve()).lower() if local_path else ""

        idx = -1
        for i, cur in enumerate(entries):
            cur_lp = str(Path(cur["localPath"]).resolve()).lower() if cur.get("localPath") else ""
            if cur.get("id") == eid or (result_id and cur.get("resultId") == result_id) or (norm_lp and cur_lp == norm_lp):
                idx = i
                break

        prev = entries[idx] if idx >= 0 else {}
        entry = {
            "id": prev.get("id") or eid,
            "kind": kind,
            "source": source,
            "title": str(title).strip(),
            "prompt": str(prompt or ""),
            "lyrics": "",
            "tags": tag_list,
            "instrument": instrument,
            "bpm": float(bpm) if bpm is not None else prev.get("bpm"),
            "key": key or prev.get("key"),
            "durationSec": round(float(duration_sec), 2) if duration_sec is not None else prev.get("durationSec"),
            "resultId": result_id or prev.get("resultId"),
            "localPath": str(local_path) if local_path else prev.get("localPath"),
            "cloudState": "local-ready" if (result_id and local_path) else ("need-download" if result_id else "local-only"),
            "creditCost": credit_cost if credit_cost is not None else prev.get("creditCost"),
            "rating": prev.get("rating"),
            "reuseCount": int(prev.get("reuseCount") or 0),
            "createdAt": prev.get("createdAt") or time.strftime("%Y-%m-%dT%H:%M:%S"),
            "lastReusedAt": prev.get("lastReusedAt"),
            "notes": notes or prev.get("notes") or "Generated via Re-Tex ACE Reimagine.",
        }
        if idx >= 0:
            entries[idx] = entry
        else:
            entries.append(entry)
        cat["entries"] = entries
        stats = cat.setdefault("stats", {})
        stats["totalEntries"] = len(entries)
        stats["totalCreditsLogged"] = round(sum(float(x.get("creditCost") or 0) for x in entries), 2)
        tmp = cat_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(cat, indent=2), encoding="utf-8")
        tmp.replace(cat_path)
        return entry
    except Exception:
        return None
