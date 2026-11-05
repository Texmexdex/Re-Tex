# Re-Tex (code name Mashup Lab) - notes for AI coding assistants

This file loads into Claude Code sessions in this folder (other assistants: read it first too). It records what the
code does and what was learned the hard way. Keep it current: when a decision is made or something is learned, add
it here, in place.

## What it is, and who it's for
- A local AI DJ assistant and mashup mixer. Import songs; they're separated into stems and analysed. Mix them on a
  timeline with tempo, key and lyrics handled, and let the DJ brain plan mixes. A separate YuE2 Studio install can
  re-imagine a mix as a new production.
- Built for an audio engineer: the ear is the judge. Value audible, verified progress over scaffolding.
- UI: dark glass, Spotify-like, green accent. No emoji, no purple/gradient "AI" look.

## Where things live
| What | Where |
|---|---|
| App Python | `.venv-phase2` (made by `install.bat`) |
| Run the app | `Re-Tex.bat` (port 8765; `--port`, `--no-browser`) |
| The library (songs, stems, projects) | `data/` by default; the Settings dialog or `%APPDATA%\MashupLab\settings.json` "data" can move it; env `MASHLAB_DATA` overrides |
| Library: per-song folders | `songs/<id>/` (stems FLAC, song.json, features.npz, peaks) |
| Library: other folders | `cache/` (warped renders), `samples/smp_*` (sample-editor bakes), `uploads/` (source audio), `projects/`, `yue2/` (takes), `meta.json`, `config.json` |
| Separation checkpoints | `models/` (`config.SEP_MODELS`) |
| GGUF models | `data/models/` (`config.MODELS`) |
| Music Flamingo | its own environment `.venv-listen`, configured in `data/config.json` "listen" |

## Rules
- **Never delete, purge or compress the user's stems, renders or library files without an explicit yes.** Removing
  true duplicates is fine once shown what is duplicated and where.
- Share audio test results as files (MP3s with notes on what each clip is), not through the in-app Listening window.
- Never edit across lanes or ripple the arrangement. Clip edits belong in the pop-out sample editor.
- mixer.py stays as it is unless the change is the point; the project schema is additive only.
- Don't play audio through the user's speakers when testing (set the monitor volume to 0, or render to a file).
- Never commit a Hugging Face token, API keys, the library, or model weights.
- Scratch scripts that import mashlab and can reach import or analysis need `if __name__ == "__main__":`. Spawned
  workers re-run an unguarded top level; that once imported a take 15 times.

## The tools, what they're for, and what was learned
- **Separation** (separate.py, library.default_quality):
  - Modes: Demucs htdemucs_ft "best"; BS-Roformer vocals + Demucs "ultra"; BS-Roformer SW 6-stem "sw"; and
    **"blend", SW + Ultra averaged stem by stem** - it won a blind listening test (best vocals, as good as the best on
    the rest) and is the default for imports. Separation shows as badges and a Stems group in the library.
  - **Import stems** (web/js/stemimport.js, library.import_stems, /api/stems/*): already-split files become ONE song
    and are never separated. Each file gets a stem guessed from its name; files on one stem are mixed; a missing stem
    is silent (an acapella or a drum loop works alone). Badge "STEMS"; song.json keeps separation "imported".
- **Beat grid** (beats.py, timing.py, beatfix.py; the Beat grid window is web/grid.html):
  - Beat This! is refined into a grid. Songs play AS RECORDED at one fixed speed.
  - Taps in Re-Tex are already latency-compensated: never subtract a monitoring delay. The ear decides bar phase;
    drum onsets decide fine timing.
  - Detector failure mode: half/double-time switches between sections. beatfix.repair re-counts them, gating 3:2
    recounts on drum hits. A kick/snare "backbeat" bar-phase rule FAILED on known songs. Don't use it.
  - Check web tempo claims against the audio before using them (a pasted "steady 128 BPM" map had no 128 pulse and
    didn't even add up). The test that discriminates is pulse strength: the Fourier magnitude of the drum + bass
    onsets at a rate. "Clicks landing near hits" can't discriminate: dense drums put any steady click near a hit.
  - **Beat grid window editing** (gridedit.js, tested in node): a bar's beats stay EVEN between its two bar lines
    unless placed by hand; nothing before the edited bar moves. Drag a bar line: the bar before squeezes, the rest
    moves along (Ctrl: only the bars either side). Drag a beat: that bar's tempo changes (Ctrl: only that beat,
    marked "placed"). Shift+drag: everything. Del deletes and the bar re-spreads; deleting a bar line joins two bars.
    Double-click toggles bar's 1 / adds a beat. Uneven bars show red in the ruler.
- **Steady takes: don't lock to marks that follow the drummer's groupings.** A song can keep one tempo while the
  drummer moves from 16ths to triplets, and the detector marks pairs of triplet notes as beats. Every lock of those
  marks broke it. Playing as recorded keeps coming back around to the click (polyrhythm). The lock dialog offers
  "Play it as recorded (don't lock)" first; a lock is for takes that really drift (live recordings). Measuring
  "steadiness" from drum-hit phase failed both ways tried - don't trust such numbers without the ear.
- **Main mix metronome** (player.py `_add_clicks`, the Click button / K): clicks every project beat, louder on each
  bar's 1 and loudest every 4th bar, added in the player callback after the master (sample-locked). Never in exports.
- **How a song sits on the mix's grid:** the project grid is fixed (one bpm, bar 1 = beat 0). Unlocked, a song plays
  as recorded at ONE fixed speed (project bpm / its own bpm); its marks show white on the grid, red off it. Lock
  re-renders its stems so EVERY mark lands on a steady grid.
- **Timing versions: a song's grid change is for projects from then on.** The song's grid = song.json user_marks; a
  project's own timing = project.songs[sid].marks (per song, not per stem). Saving a song's grid freezes every saved
  project that followed it (it gets a copy of the old grid). Locks are per grid; a lock still played by a frozen
  project moves to stems_locked/<rev>/ + info.locks[sig]. Never rmtree stems_locked/.
- **Song tools menu** (songtools.js): the same list on every right-click (clip, lane header, arrangement bar, library,
  song view) and the arrangement bar's "Tools" button.
- **Lock** (lock.py): beat-by-beat Rubber Band is clean on right marks; warble came from wrong marks. When marks run
  at another speed the lock button ASKS ("Repair, then lock", "Line it up by ear", "Lock anyway").
- **DJ Mix, Remix style** (djremix.py, the default): modelled on a reference mix by the author - a bed played as
  recorded, a guest rapping on top, a third song's drums. In it: the bed's music holds for long stretches with one
  jump to its chorus at the drop; the bed's vocal is off under the guest's verses; a short guest phrase loops every 4
  bars; stutters lead into sections; a one-bar break before the drop; the guest gets long stretches that end where he
  stops. The engine picks the bed by fit; only the guest's key moves. The planner plans the bed's path and places
  whole lines by part and bar; the engine keeps each line's pickup and refuses lines over clashing sections.
  - Lines come from the vocal's LEVEL, not pitch tracking: a stop = 24 dB down for at least 150 ms.
  - Songs without words get Whisper'd on first use (lyrics.json).
- **DJ Mix, Blend style** (djbrain.py, llm.py; kept as an option): a per-stem part script. It lost to the ear twice
  ("a weird fade out mid word"; a bass under the other song's chords "sounds pretty bad, as they are stems").
  - Planning goes through Claude Code headless (`claude -p --json-schema`, tools off) on the user's own Claude login;
    no API key. Never use `--bare` (it ignores the login). Gemma (local GGUF) is the offline option and plans badly.
- **Listen and describe** (listen.py, listen_worker.py): Music Flamingo, in its own environment, writes a genre,
  sub-genre and a sound line per song from 30 s of audio; it beat a title-based text model in a blind test. Audio LLMs
  CAN'T plan mixes or judge timing; they take ONE audio per request.
- **Reimagine / YuE2** (reimagine.py, yue2link.py, takealign.py, align.py, userlyrics.py): bounces the live mix and
  sends it to a YuE2 Studio server (127.0.0.1:8791) for a cover, aligned bar-for-bar, then locked. YuE2 keeps its own
  clock for words: score-side steering FAILED twice. What works: typed lyrics + forced alignment (wav2vec2) +
  per-line re-timing after generation.
- **Sample editor** (sampler.py, web/sample.html): one clip at a time on a grid from 1/2 to 1/64 (triplets too):
  chop, stutter, layer, tape delay, echo. Cut vocals on word starts; place effects at section boundaries.
- **Render cache** (warp.py, draft.py): warped stems per tempo/key as raw float32 .npy, plus drafts/ and clips/.
- **ReadHead** (mashlab/readhead.py + readhead_js/, web/js/readhead.js; right-click a lane header): automatic
  beat-aligned chops written as ordinary clips on ONE lane, one undo step. "Arrange vocal" re-arranges a vocal against
  the rest of the project; its "Flow" style was learned from a hand-made edit (loops of 1-2 bars, alternating
  neighbour bars, a restarted opening, a 16th ratchet before changes; every move a whole number of the SINGER's bars;
  cuts only between syllables). "Glitch chops" chops the lane in place. Reverse is off (clips can't play backwards).
  The engine is ReadHead's JS (src/engine of the standalone tool), run with Node on mono inputs built from the warped
  renders; inputs and caches go to %TEMP%\retex_readhead.
- **Perform mode** (mashlab/perform_live.py, web/js/perform.js; the Perform button in the top bar): chops and samples
  played from the keyboard while the song plays, then written as clips. A MODE: off (default) every Re-Tex key is
  as before; on, A-L / Z-M / 1-9 are pads. Space still plays.
  - A-L on the chosen lane: loop 2 / 1 / 1/2 / 1/4 beat, bar back, bar ahead, roll, chop (1/16 on/off), drop out.
    Slip only (the lane comes back where the song is).
  - Z-M: Shift+drag a clip (Re-Tex's highlight), press the key = that span is on it (project.perform_samples). Hold
    the key = it loops on top on a layer lane (`perform_layer: {key, src}`).
  - While held, the PLAYER plays it: Player._mix -> perform_live.LIVE.render, a copy of Mixer.render reading the lane
    from a shifted position (mixer.py untouched; no takes = exactly mixer.render; errors fall back to the plain mix).
  - Key up: /api/perform/release fixes the end on the grid at once; /api/perform/take turns it into clips against the
    project the UI sends, ONE take at a time, each applied before the next. Computing against the server's copy raced
    during fast chords and corrupted a lane - don't go back to that.
  - project.perform_seen = newest applied take id; set_project drops live takes the project now holds. Live fades =
    the clips' fades (in: MICRO_FADE, out: 0.03 beat; butt-joined where the take carries straight on), so live equals
    the saved clips sample for sample inside a take.

## Working practices (each learned by breaking something)
- After ANY web/js change: copy each file to `.mjs` and run `node --check` (on `.js` with imports, node silently
  passes broken code), then load the app in a FRESH browser tab and check the console. A broken string once blanked
  the whole UI.
- Edit code with editor tools. Heredoc + Python replace mangles `\n` and `\"`.
- Try UI changes on a sandbox: env `MASHLAB_DATA` = a scratch library of copied songs, another `--port`.
- Tests: `.venv-phase2\Scripts\python.exe -m pytest tests -q`.
- Verify by rendering and measuring, then let the ear decide (blind A/B when it matters).
- The #1 complaint about vocals is a line that stops, or fades, while the singer is still going. Measure it: vocal
  level 100 ms after a clip's end, against the song's loud level. A "quiet spot" inside continuous singing is still
  heard as a cut. Pitch-tracking gaps are not breaths. Whisper's word times are rough on sung vocals.
- When the user points at a mix they made, study it read-only first (clip by clip, against the songs' lines). It says
  more than any rule list.
