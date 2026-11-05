# Re-Tex

Make mashups and remixes of any songs that are **in time, in key, and mixed**, in about a minute.

## Start

First time on a PC: double-click **`install.bat`** (see *Install* below). After that, double-click **`Re-Tex.bat`** (or the Re-Tex shortcut the installer makes). The app opens in your browser at http://127.0.0.1:8765. Close the black window to quit.

(Or run `python -m mashlab` from this folder. Add `--port 9000` to use another port, or `--no-browser` to skip opening the browser.)

## Install

Windows 10 or 11 (64-bit) with an **NVIDIA GPU** (a recent driver; anything that runs CUDA 12.8). It runs without one,
but separating songs gets very slow.

1. Download this repository (green **Code** button → *Download ZIP*, then unzip it), or `git clone` it.
2. Double-click **`install.bat`**. It asks one question: **Full** (everything, about 50 GB) or **Core** (about 15 GB).
3. When it's done, double-click **`Re-Tex.bat`** or the **Re-Tex** shortcut on your desktop.

The installer can be run again at any time: it skips what's already there and finishes what isn't.

What it sets up, all inside this folder unless noted:

| Part | What for | Size |
|---|---|---|
| Python 3.13 (from python.org via winget, for your user) | runs Re-Tex | 0.1 GB |
| A private Python environment (`.venv-phase2`) with pinned versions of everything Re-Tex uses, PyTorch built for CUDA 12.8 | the app | ~6 GB |
| FFmpeg and Node.js (via winget) | reading mp3/m4a/mp4 and exporting mp3; ReadHead's engine | 0.3 GB |
| Separation models: Demucs htdemucs_ft, BS-RoFormer vocals (viperx), BS-RoFormer SW 6-stem | splitting songs into stems | 1.5 GB |
| Beat This!, faster-whisper large-v3, wav2vec2 | beat grids, lyrics, word timing | 3.5 GB |

**Full** adds:

| Part | What for | Size | Needs |
|---|---|---|---|
| MuScriptor large | a second opinion on beat grids (notes transcribed from the song) | ~5 GB | a free Hugging Face login |
| Gemma 3 12B (GGUF) + llama.cpp | the offline DJ planner and mood tags | 8 GB | a free Hugging Face login |
| Music Flamingo (its own environment, `.venv-listen`) | writes a genre and a one-line description of each song from its audio | ~21 GB | — |

For the two marked "Hugging Face login", the installer asks you to log in (it opens the page for a free account and a
token) and to accept each model's terms on its page. Your token is stored by Hugging Face's own tools in your user
folder, never in this folder. Skip it and those two features just stay off.

**Optional, not installed by `install.bat`:**
- **DJ Mix planning with Claude** uses Claude Code on your own Claude login (no API key). Install Claude Code
  (https://docs.claude.com/en/docs/claude-code) and log in once; Re-Tex finds it. Without it, DJ Mix uses Gemma.
- **Reimagine** sends a mix to a separate **YuE2 Studio** install. Point Re-Tex at it in `data/config.json`:
  `"yue2": {"studio": "<your YuE2 folder>", "url": "http://127.0.0.1:8791"}`.

Your songs, stems, projects and renders live in `data/` (created on first run). Nothing in `data/` is ever uploaded
anywhere; it's ignored by git.

**No NVIDIA GPU?** Everything still runs on the CPU, just much slower for separation and lyrics. Music Flamingo
needs an NVIDIA GPU.

## The fast route

1. **Add songs.** Drop audio files anywhere in the window, or click **+ Add**. Any format ffmpeg can read works (wav, mp3, flac, m4a, even mp4). You can also paste a path to a folder of songs, or to a folder that already has `drums/bass/other/vocals` stems (UVR / Demucs output), which skips separation.
   Each song is split into 4 stems on the GPU (Demucs htdemucs_ft), then analysed for its beat grid, key, tuning, sections and loudness.
2. Click **✦ Auto-Mashup** and press **Rate every pairing**. Every song's vocal is arranged over every other song's music, and each finished arrangement is rated. You get the full list, best first, sortable by any column and filterable by song name or minimum rating. Click a row for a plain-language breakdown:
   - **Harmony:** how well the vocal's actual sung notes sit on the new chords, note by note. 100% means as well as a vocal typically fits its own song; 0% means what a random song in a random key gives.
   - **Tempo** and **Key:** how far the vocal and music have to be stretched and transposed. Vocals tolerate less than instrumentals.
   - **Structure:** how much of the music carries vocal, whether the hooks land on the big sections, and whether lines are only cut where the singer pauses.
   - **Groove:** straight versus swung feel.
   - **Acapella:** how clean the vocal stem is.
   - **Music** and **Length** only appear when the backing is mostly drums, or when only a short part of it is at a steady tempo.

   The overall rating multiplies these, so one bad aspect can't hide behind good ones. Other keys that also work are offered as buttons.
   Songs that are **the same song** (duplicates, alternate versions, demos, instrumentals of each other) are recognised from the audio and never paired.
   The first run on a big library takes a while: every vocal melody is tracked (about a minute of CPU per song, spread over your cores), then roughly 20 pairings are rated per second. Results are saved, so later runs only rate what's new or changed.
3. **Open & play.** Stems are warped to the new tempo and key (usually under a minute, in parallel), then playback starts automatically at the first vocal entry.
4. Tweak it on the timeline, then **Export** to WAV or MP3. Exports are normalised to −12 LUFS with a true −1 dBFS ceiling.

### Auto-Mashup: By roles

The **By roles** tab lets you choose candidates for each part:

- 🎤 **Vocals from:** any songs. You can also keep the bed song's own vocals, or have no vocals.
- 🎹 **Music (bed) from:** the song whose melody, chords and structure carry the arrangement.
- 🥁 **Drums from:** any songs, or keep the bed's own drums.
- 🎸 **Bass from:** any songs, or keep the bed's own bass.

Every combination is built and rated, and all of them are listed (best first, 40 at a time). Vocals are planned and rated exactly as in **Every pairing**. Drums are energy-matched and only play where the bed originally had drums, so breakdowns stay breakdowns. Bass is matched against the bed's chords using the older chroma measure. Both tabs let you tick whole **folders**, filter by name, or use all/none.

## How mashups are rated (and why you can trust the numbers)

The old Auto-Mashup gave out 8 results, and many said 100% for the wrong reasons. Duplicates of the same song "fit" perfectly. Songs with similar-sounding chord colour scored the same as songs that actually work together. Each piece of the new engine was measured on your own library:

- **Same-song detection** uses a cover-song method: harmony sequences are compared in every key and at double and half tempo, against a time-reversed control, and each song is judged against how it compares with everything else. It found 12 groups, including "Ligeia" = "Tomb of Liegia", "Blvd. Nights" = "Boulevard Nights" (instrumental) and "Your Skull Is Red" (two recordings in different keys). "Let's Go" and "On Letting Go" are correctly *not* the same song.
- **Harmony** comes from a small network trained on your library. Every vocal over its own music is a known-good combination, and the same vocal a few semitones off is a known-bad one. It looks at each sung note (tracked with pYIN, not the smeared spectral chroma) against the bass, the chords and the key of the surrounding bars. Tested on songs it never saw, it ranks a sung vocal's real key above the clashing ones (a half step, a whole step or a tritone off) for **90%** of songs and **82%** of 8-bar phrases. The old measure managed 60% and 52%. Where key can't be judged (screams, rap, drum-only passages), harmony counts as neutral rather than a free 100%.
- **Phrase placement** was tuned by making the planner rebuild songs from their own vocal and instrumental. On songs not used for tuning, it puts **63%** of phrases back exactly where they were (hand-set weights: 38%), and gets vocal-versus-silence right on 77% of the bars. Vocal lines stay whole: they're only cut where the singer pauses, and short pickups and held words are kept.
- **Keys:** the vocal is moved rather than the whole instrumental, and the music only moves by one or two semitones when the vocal alone would have to go far.
- **Tempo:** sections at a different tempo (half-time breakdowns, rubato intros) are left out. Clips are laid out bar by bar from the detected beats, but songs play as recorded (no beat-by-beat warping), so where a song drifts, its layers drift against the bed - check those by ear.

After adding lots of new songs, you can retrain the harmony model on the bigger library with `python -m mashlab.train_harmfit` (about 2 minutes, needs the GPU build of torch). All ratings then refresh on the next run.

## Finding songs that match

Right-click a song and choose **🔎 Find matches** (also available in the song panel, or on a song chip above the timeline). The library then lists every other song, best match first. Each card shows:

- **Score:** the full mashup rating (see above) for the better direction: their vocal over your music, or yours over theirs. A **≈** means it's only a quick estimate, because that pairing hasn't been rated yet. Press **Rate properly** at the top to rate every pairing with this song.
- ⏱ **Tempo:** how much stretching it takes. Half and double time count, e.g. 62 and 124 BPM line up.
- ♪ **Harmony and use:** which way round works, and how well.
- 🔑 **Keys** described in plain words ("same key", "relative major/minor", "neighbouring key"), so there's no Camelot wheel to learn.
- Versions of the same song are listed at the bottom as **=** (never a mashup).

Hover a card for the full breakdown. Press ✕ to go back to the normal library.

## Better stems (Ultra)

If the **BS-Roformer** vocal model is installed (`install.bat` puts `model_bs_roformer_ep_317_sdr_12.9755.ckpt` in `models/`; an Ultimate Vocal Remover install's copy is found too), new songs use **Ultra** separation by default. BS-Roformer takes the vocals out of the mix, then Demucs splits what's left into drums, bass and other. Drum leakage into the vocal stem dropped about 6× in testing, and the instrument stems no longer carry vocal remnants. It takes about 2–3 minutes per song on the GPU.

To upgrade songs you already have, right-click them and choose **✨ Re-separate stems (Ultra quality)** (Ctrl/Shift+click selects several). Each song's beat grid is kept, so your arrangements stay lined up. Songs with Ultra stems show an **ultra** tag.

**BS-Roformer SW (right-click only).** Right-click a song (or several) → **🎧 Re-separate with BS-Roformer SW**. One newer model makes all four stems. It splits six ways, and its guitar and piano go back into "other". Its drums, bass and other differ far more from Demucs' than Ultra's do, and it's fast: about 40–60 s per song, plus the analysis. Only the songs you pick change; the rest of the library keeps its stems. **+ Add** offers both too (Separation quality: SW + Ultra blend, BS-Roformer SW, Ultra, Best, Fast); Ultra stays preselected, and files dropped onto the window use Ultra. The grid is kept as with Ultra, and those songs show an **sw** tag.
- Model files: `models/BS-Rofo-SW-Fixed.ckpt` + `.yaml`, from huggingface.co/enerjazzer/BS-ROFO-SW-Fixed (sha256 `24e7d35e…916e`). The license is unknown: fine for personal use, not for commercial work.
- The file is loaded as plain weights (`weights_only`), so nothing in it can run code. Chunks overlap 4× (its config says 2), so every moment is the average of four runs.

**SW + Ultra blend (right-click only).** **🎧 Re-separate: SW + Ultra blend** separates the song with both SW and Ultra and averages the stems. The two leave different artifacts, so they partly cancel. In a blind listening test on two songs, the blend gave the best vocals and drums / bass as good as the best single version. It takes about 3–5 minutes per song. Those songs show an **sw+ultra** tag. Smoothing the separators' masks afterwards (to reduce the static) was tried in the same test and sounded clearly worse, so it is not used. Heavily distorted guitar still separates badly with every method.

**Nothing gets lost (since Oct 2026).** The four stems always add back up to the original song:
- Ultra used to throw away whatever Demucs called "vocals" in the vocal-free remainder (synth leads, pads, chops). That could be up to about 70 % of the mids in some sections; One test song lost a lot this way. That part now goes to "other".
- Any small leftover the separators don't place is shared between the stems by which one owns each frequency at each moment, so no stem gets an inverted copy of another.
- Plain Demucs separation gets the same leftover sharing.

**No more clipped stems.** Loud masters often separate into stems above full scale, which 24-bit FLAC can't store. They used to be clipped on save, flattening drum hits; 125 of the first 178 songs had some. Now all four stems of a song are saved a little lower when needed (`stem_scale` in `song.json`) and played back at their true level. Songs separated before this keep their old stems: re-separate them to get the clean, complete version.

`install.bat` downloads both models' `.ckpt` and `.yaml` files into `models/` and checks them against known hashes.

## Library folders

Click **+ Folder** to make a named folder. To file songs, drag them onto it (Ctrl/Shift+click to drag several at once), or right-click a song and choose **Move to folder**. Click a folder to collapse it; right-click it to rename, reorder or delete it (deleting a folder keeps its songs).

## Editing by hand

- Click a song in the library to open it in the bottom panel. It shows the four stems and detected sections (intro, verse, chorus, drop, break, and so on).
  - **Click** a section to hear it. Click a stem row to hear only that stem. **Ctrl+click** picks just the 4-bar phrase under the mouse.
  - **Drag** a section onto the timeline. Dragging the section strip brings all stems; dragging a stem row brings only that stem. It snaps to bars; hold **Alt** to snap to beats. Clips from a section are named after it ("verse E · other · …").
  - **Whole song:** the **all / drums / bass / other / vocals** buttons in the song panel's header lay in that stem (or every stem) for the whole song, one clip per section. Each clip is named after its section and sits exactly where it follows in the song, so you can line it up against another song's structure and delete the sections you don't want. Drag a button onto the timeline, or click it to add at the playhead. A stem is left out of sections where it doesn't play, which leaves a gap. One Ctrl+Z removes the lot.
  - Section clips play seamlessly: where the next clip on a track simply carries on the same audio (same song, stem, speed and gain, butt-joined), the mixer skips the 3 ms anti-click fade and both read one continuous stretch of the warp. Laid in by sections, a stem plays bit-identically to one long clip. A clip that has been split in two still gets the short fade-out that Split adds.
- On the timeline:
  - Drag clips to move them (snap to beat; **Shift** snaps to bar, **Alt** to 1/4 beat). Drag across tracks to move a clip to another track. **Ctrl+drag** copies.
  - Drag clip edges to trim. Drag the small squares at a clip's top corners to set fades.
  - Right-click for split, duplicate, gain, fades and loop. Double-click a clip to open its settings: gain, fades, "slip" (which part of the song plays) and stem swap.
  - Drag on the ruler to set a loop; click the ruler to jump there.
- The chips above the timeline control each song in the arrangement:
  - **transpose** in semitones (drums are never pitched)
  - **½× / 1× / 2×** to play it at half, normal or double time
  - the tempo change being applied to it
- Track **fx**: high/low-pass filters, a "vocal pocket" (cuts 2.8 kHz so a vocal sits on top), and ducking under the vocal tracks.
- If a song's bar lines are one beat off, use **◀ bar / bar ▶** in the song panel.

### Stretch, pitch, snap and layers

- **Stretch a clip smoothly:** drag the small ◢ triangles in a clip's **bottom corners** (they only grab inside the triangle; anywhere else on an edge trims). The source stays the same and only the speed changes, with no snapping (hold **Shift** to land the end on the grid). Dragging the upper part of an edge still **trims**, so you keep both. With several clips selected, every selected clip is stretched by the same amount, each around its own start (right triangle) or end (left triangle).
- **Keep pitch or not:** double-click a clip (or right-click it) and choose **Keep pitch** (time-stretch) or **Pitch follows speed** (varispeed, like a turntable). The clip editor also has a smooth speed slider, an exact ×factor box, and **Fit to N bars**.
- **Pitch** is its own control per clip: a smooth slider in semitones and cents, ±1 / ±0.1 buttons, or **Alt+mouse wheel** over a clip (0.1 st steps; Shift+Alt = 1 st). It stacks on top of the song's transpose.
- While you drag, you hear an instant varispeed preview (pitch moves with the speed). About a second after you let go, a quick draft takes over at the right pitch, then the high-quality Rubber Band render replaces it (5 to 30 s, depending on the clip's length). Clips show **HQ…** until it lands, and export always waits for the HQ versions.
- **Song panel zoom:** **Ctrl+wheel** zooms the stems view at the bottom, the wheel scrolls it, and there's a scrollbar and −/+/Fit buttons. **Drag along the bar ruler** (the numbered strip under the sections), or **Shift+drag** over the stems, to select any range of bars. Click inside the selection to hear it, or drag it onto the timeline. Drag it from a stem row to take just that stem.
- **Lining things up:** every clip shows its own song's beats where they really are in the audio: a short tick per beat, a tick and a faint line per downbeat. White ticks sit on the project grid, red ticks are off it. Since songs play as recorded, a song that drifts shows it as the ticks turn red, and that's where to slip, split or stretch by hand. The ticks come from the beat detector, so they can be wrong too (fix them with right-click → **Fix beat marks here**). When you move a clip, the downbeat nearest to where you grabbed it is what snaps to the grid, not its edge. Right-click → **Align downbeat to nearest bar** (or press **A**) fixes a selection in one go.
- **Tempo changes keep the pitch.** The high-quality warp takes about a minute per stem. For the first second or so you hear the original stems varispeeded onto the grid, with the pitch off by the tempo change. Then quick drafts take over: Rubber Band R2 renders of 16-beat chunks, the part under the playhead first. The drafts are at the right pitch and level, and within a few ms of the final timing, because they use the same click-pilot re-timing as the HQ warp. The HQ warp replaces them stem by stem without changing the pitch. A song played at its own tempo (and untransposed, no tuning match) isn't processed at all.
- **Changing the tempo never changes the ½× / 1× / 2× setting** of a song that has clips in the arrangement, since that would change which part of the song each clip plays. Use the ½× 1× 2× buttons for that.

### Stretch or contract part of a clip (no slicing)

1. **Shift+drag** across a clip to highlight a span (it snaps to the grid; add Alt for smooth).
2. **Ctrl+click** clips on other lanes to highlight the same span there too (Ctrl+click an empty spot in a lane for every clip of that lane under the span; Ctrl+click again to take one out).
3. **Drag the yellow right edge** of the highlight to stretch or contract that part of every highlighted clip. Everything after it in each clip moves along, and each clip's end moves with it. Nothing else on the timeline changes. It snaps like other drags (Alt = smooth, Shift = bars).
4. The toolbar next to the highlight has **Keep pitch** (time-stretch) or **Pitch follows** (turntable), which also changes parts you already stretched. It also has **Reset** (back to normal speed in the highlight) and **✕** / Esc to clear the highlight.

Stretched parts show as a violet band on the clip, with the factor and pitch mode. Every step can be undone with Ctrl+Z. The **⌨** button in the timeline toolbar lists every shortcut and mouse move.

### Songs play as recorded

Nothing is warped automatically. Each song plays at one fixed speed, the project tempo divided by the song's tempo (use ½× 1× 2× for half or double time), and keeps its own timing: drifts, odd bars, fills and tempo changes stay exactly as recorded. A clip starts where you put it and plays straight on from there. The grid lines are the project's grid; the ticks on a clip are the song's own beats, so you can see where it wanders off the grid and fix it yourself (split, slip, Shift+drag stretch).

Parts the beat detector found at another tempo show a thin blue strip along the bottom of the clip with the detected BPM. It's information only; they play as recorded like the rest.

**How sure the marks are.** The beat detector isn't always right, and it says where it isn't sure. Marks it's sure of are drawn white (on your grid) or red (off it). Marks it's unsure of are amber, under a dashed amber strip ("detector unsure - check by ear"), and so are marks that are only a guess (for example, extended over a silent intro). Snapping never uses an unsure mark. The song panel says how much of a song the detector is unsure of.

**🔒 Lock to marks.** For songs whose own timing drifts: a live take, or a download whose speed was altered (some re-uploads wander a few % to get past copyright matching). Click **🔒** next to the song in the arrangement bar.
- **First time:** each of the song's stems is warped once (Rubber Band R3, pitch kept), so every mark sits on a steady grid at the song's average tempo. This takes about a minute. Your own marks are used as they are. The detector's marks are first snapped onto the drum hit right next to them (within 40 ms). The locked stems are saved next to the originals (`stems_locked/`), which are never touched.
- **After that:** 🔒 toggles instantly between the locked and the as-recorded version, per song, per project. Ctrl+Z works.
- **Going stale:** a lock belongs to the marks, stems and beat grid it was made from. If you edit the marks, re-separate or re-analyze the song, the button says **🔒 re-lock**, and the song plays as recorded until you click it again.
- **Marks can't be edited while a song is locked.** Unlock, fix the marks, lock again.

Measured on It'll Be OK (a re-upload whose tempo swings 96.9–101 BPM): locked, its tempo holds 97.87–97.98 BPM. In a 98 BPM project, 23 of 24 four-bar blocks sit within ±2 ms of the grid.

**MuScriptor second opinion.** Right-click songs in the library → **🎼 MuScriptor second opinion**. MuScriptor (large) transcribes the song into notes in the background (about 2 minutes per song, one song at a time, on the GPU). Notes it re-struck at its 5-second chunk seams are joined back. For every mark, the kick, snare, bass and chord onsets around it are checked:
- **On the marks:** they agree.
- **Clearly on a shifted grid** (half a beat off, say): they disagree, and those marks turn amber even if Beat This was sure.
- **No clear pattern** (ambient, rubato, too few notes): no opinion, and nothing changes.

It also compares the drum pattern at 3, 4 and 6 beats. If the drums clearly repeat at another bar length than the marks have (Formant: 6/8, where Beat This counted 3), the song panel offers **Use 6/8**, with bars starting on the kick. That's one click, a normal marks edit, and Ctrl+Z undoes it; nothing changes until you click. Only MuScriptor's notes are used: its own tempo / beat grid comes from Beat This, so it would be no second opinion. The notes are kept with the song (`muscriptor_notes.json`), so a re-analysis re-checks the new marks without transcribing again.

**Fixing marks yourself.** Your marks always win. They're saved with the song, so every project shows them (the clip says **✎ your marks**). They never move the audio, and Ctrl+Z undoes every change.
- **✎ Marks** button (or **M**) turns on marks mode:
  - drag any mark to where the beat really is;
  - double-click a clip to add a mark;
  - **Del** deletes the mark under the mouse;
  - right-click for the marks menu;
  - **M** or **Esc** when done.
- **Tap them in:** in marks mode, select the clip, play, and press **T** on every beat. Your taps replace the marks they cover, and the bar count carries on from the downbeat before them. Stop tapping for 2 s and they're in.
- **Right-click a clip → Fix beat marks here ▸** (or the marks-mode menu) has:
  - shift the marks from here on, or before here, by ½ beat (or any amount);
  - bars start at this mark;
  - beats per bar;
  - "the marks in this clip are right" (no longer unsure);
  - back to the detector's marks.

**Moving a clip** snaps the song downbeat nearest to where you grabbed it onto the grid (only marks the detector is sure of, or yours; with none nearby, the clip's edge snaps), so you can line up any part of a song by grabbing it there (for example the main part after an intro that's half a beat off).

The song panel says **varies ±x %** only when the song's own tempo really moves by more than ±1 % over 16-beat stretches (the detector's beat-to-beat jitter averages out), and counts the parts it found at another tempo.

**Tuning is left alone** unless you ask: the tuning dropdown next to each song in the arrangement bar offers *Tune to A440* or *Match* another song of the arrangement, and shows how many cents that shifts. Drums are never pitched.

- **Only the settings you stop on get rendered.** A full-quality warp takes about a minute per stem, so renders start 0.5 s after your last tempo, transpose or pitch change. Anything the arrangement no longer needs is stopped right away, even halfway through. Clicking the tempo up 20 times renders just the final tempo, and a run of tempo clicks is a single undo step.
- **Track height:** **Shift+mouse wheel** over the timeline, or the ▁ █ buttons above the track names, makes every track and waveform taller or shorter. The size is remembered.
- **Follow playhead:** the **⇥ Follow** button (or **P**) toggles whether the view scrolls along while playing. Even when it's on, it waits 3 s after you touch the timeline, so it won't yank the view away while you edit.
- **Scrolling:** use the scrollbar under the timeline, or the mouse wheel over the ruler or scrollbar, to move sideways.
- **Snap:** the dropdown above the track names sets bar / beat / ½ / ¼ / ⅛ / off. Hold **Alt** while dragging to ignore snap for that move.
- **Layered clips:** clips that overlap on a track are shown in stacked rows (the track grows taller), so every layer can be seen, grabbed and moved on its own.
- **Undo / Redo:** the ↶ ↷ buttons in the top bar (or Ctrl+Z / Ctrl+Y). Hover them to see what they'll undo. A whole drag or slider sweep counts as one step, and about 300 steps are kept.

**Shortcuts:**

| Key | Action |
| --- | --- |
| Space | Play / stop |
| S | Split at the playhead |
| Del | Delete |
| Ctrl+D | Duplicate |
| Ctrl+Z / Ctrl+Y | Undo / redo (also the ↶ ↷ buttons) |
| Alt+wheel | Fine pitch on the clip under the mouse |
| Shift+wheel | Taller / shorter tracks |
| P | Follow playhead on/off |
| A | Align selected clips' downbeats to the nearest bar |
| Alt (while dragging) | No snap |
| Ctrl+S | Save |
| L | Loop |
| ← / → | Nudge the selection (hold Shift for a whole bar) |
| F | Fit the arrangement to the window |
| + / − | Zoom |
| Home | Back to start |
| Esc | Stop audition |

Projects autosave a couple of seconds after each edit (**Projects** reopens them).

## How the sound quality is kept high

- **One fixed speed per song.** The warp is a single constant time-stretch (project tempo / song tempo), so a song keeps its own timing. Its detected beats are only used for the marks, snapping and the song's sections.
- **Beat grid, tempo and meter.** Beats come from Beat This!, then get cleaned up:
  - The main tempo is the one covering the most of the song, not just its longest unbroken stretch.
  - Sections the tracker followed at 2×, 3× or 4× (eighths or sixteenths) or at half speed are folded back to the song's beat.
  - Tracks detected above 200 BPM are counted in quarter notes, unless they're in compound time.
  - Beats per bar (4, 3, or 6 for compound time such as 6/8, shown as "6/8") and where each bar starts both come from Beat This!'s own downbeat probabilities, measured at every beat.
  - Songs only pair with songs in the same meter.
  - **Stretches the tracker loses** (busy fills it hears as eighth notes, pauses) are filled by searching the snare/clap hits and Beat This!'s beat confidence for the steadiest beat path. The old grid extrapolated across them, which could drift up to half a beat off: One test song ran off-beat for about 40 s, then jumped back with a short beat.
  - Where two same-tempo sections meet off the beat, the side the tracker slipped on is moved onto the music's beat. A genuine odd-length bar (a 7/8 bar, an extra half-beat fill) is kept, and the warp stretches that beat to a full beat.
  - **Triplet-feel passages.** A passage the tracker followed at 4:3 or 3:2 of the beat (drums switching to triplets) is put back on the song's main beat when the beat carries straight through it: it must be a whole number of beats long, with evenly spaced beats sitting on the music. One song's bridge used to be tracked as 43 long beats instead of 57. Auto-align then squeezed it 1.33× faster and moved everything after it by 3.5 bars. A real tempo change at such a ratio doesn't come out a whole number of beats, and stays as it is.
  - **Re-analyze** keeps saved arrangements lined up: clips of that song are moved so they start at the same moment of the audio and still cover the same audio; their stretched parts stay on the same moments. Nothing else in the project moves.
- **Rubber Band R3** (via pedalboard) handles the stretch and pitch shift. R3 drifts internally by tens of milliseconds, so a click "pilot" track goes through the identical stretch; where each beat actually landed is measured and re-timed with a windowed-sinc resampler. Beats land within about 1 ms (a cent-level correction). Tiny tempo changes skip the phase vocoder and use clean varispeed.
- **Tuning** is measured per song in semitones (1-cent resolution) and only applied when you pick *Tune to A440* or *Match* in the arrangement bar. (Until 2026-10-02 every song was shifted automatically, by a value that was really in thirds of a semitone and wrapped around: songs ended up a median 11 cents apart instead of 5.) Drums are never pitch-shifted.
- **Key choice and phrase mapping** are described under *How mashups are rated* above: learned note-against-chord fit, whole vocal lines placed on the music's sections, cut only at pauses, with natural pickups and tails.
- **Mix.** The new vocal is levelled against the vocal it replaces. The instrumental gets a small pocket EQ and ducking. The master has glue compression and a limiter. Playback and export use the same mixer, so what you hear is what you get.

## Files

- `data/songs/<id>/`: stems (FLAC), `song.json` (grid/key/sections), `features.npz`, `mash.npz` (tracked vocal melody + groove), waveform peaks
- `data/ratings.sqlite`: every mashup rating; `data/same_song.json`: the same-song comparisons. Both are safe to delete (they're rebuilt).
- `mashlab/models/harmfit.npz`: the harmony model trained on your library
- `data/cache/`: warped stems, pruned automatically past 45 GB and safe to delete
- `data/projects/`: saved arrangements (JSON)
- `data/exports/`: rendered WAV/MP3

## Needs

Everything is set up by `install.bat` (see *Install* above). To set it up by hand instead: Python 3.13, then
`installer/requirements-core.txt` (PyTorch from the CUDA 12.8 index first, see `installer/install.ps1`), FFmpeg and
Node.js on PATH, and the model files listed in `models/README.md`.

## Tests

`.venv-phase2\Scripts\python.exe -m pytest tests -q`

## License

Re-Tex is free software under the **GNU General Public License v3.0** (see `LICENSE`): you can use it, change it and
share it, as long as what you share stays under the same license with its source. It's GPL because it's built on
Spotify's Pedalboard (GPLv3), which includes the Rubber Band Library (GPLv2 or later).

The models and data it downloads are **not** part of this repository and keep their own terms. Some are for
non-commercial use only (MuScriptor, Music Flamingo, the NRC VAD Lexicon), and two separation models have no stated
license. Everything Re-Tex uses, with its license and where it comes from: **`THIRD_PARTY_NOTICES.md`**.
