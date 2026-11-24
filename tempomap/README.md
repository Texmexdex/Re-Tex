# TempoMap

A tempo map for a song in your Re-Tex library, the way the song was made on a grid: sections of steady tempo,
the beats, their subdivisions, and bars of whatever length the music actually plays (4/4, 7/8, 11/8, a triplet
feel at the same tempo...). It lives inside Re-Tex's folder (`tempomap/`) but is its own program,
run in its own process, so if it fails Re-Tex carries on.

**It only reads your library.** It writes only to its own `maps\` and `cache\` folders here. Saving a song's grid
is done in Re-Tex's Beat grid window, never here.

## Use it

- **In Re-Tex:** right-click a song - "Tempo map (find its grid structure)...". Its page opens in a pop-out;
  "Open in the Beat grid window" loads its beats and bar lines there to listen, fix and save. The Beat grid window
  also has a "Tempo map" button.
- **By hand:** double-click `run.bat`, type a few words of a song's title (or `run.bat fear inoculum`). It opens
  the song's page when it's done. `maps\index.html` lists every song mapped so far.
- `test.bat` makes four test songs with a known map and checks the tool against them; `tests\key_run.py` scores
  it on the MIDI answer key in `key\` (your own .mid files as Re-Tex renders them - exact bar lines).

It uses Re-Tex's own Python (`..\.venv-phase2`), which also has Beat This!, whose downbeat probabilities help find
bar 1.

## The page

- **Player**: the song with a click on **the tempo map** (bar 1 high, other beats lower, subdivisions a soft tick),
  on **the current grid** (Re-Tex's detector beats), on **one speed at the stored BPM** (a straight click at the BPM
  Re-Tex has for the song - what you hear against an unlocked song), or on **your saved marks** if the song has
  them. Switching keeps the position, so you can A/B the same spot.
- **Top strip**: the whole song - tempo sections, bars, and the drift lane: dots for where the drum hits sit against
  each grid, in ms (green = the map, grey = the current grid, amber = one speed at the stored BPM, blue = your marks). A grid at the right tempo gives a
  flat line; the wrong tempo slides and wraps. Click it to jump.
- **Lower view**: scroll to zoom, drag to move, click to jump. Green ticks = the map (tall = bar lines), grey = the
  current grid, blue = your marks, dark bars = drum hits.
- **Tables**: the tempo sections (how each was decided, how far the hits sit from it) and the bars (how sure).
  Click a row to jump there. Amber = not sure: listen there first.

## How it works

1. **Tempo sections.** The detector's beats are cut where their spacing changes (where the detector thinks the
   tempo changes). The longest stretch is taken to be counted at the right level; its exact tempo is the one that
   holds the whole stretch together best - the onsets summed at that rate over the entire stretch at once (a
   steady studio take keeps every bar in phase at its true tempo; even 0.3 % off loses the phase within a minute).
   Each other stretch first asks: is that same pulse still the one underneath here? Studio songs change the
   grouping far more often than the tempo, so a triplet feel or a 4-against-3 part keeps the tempo. Only a clearly
   different pulse gets a tempo of its own - and not even then if the detector's count there is a simple ratio of
   the main pulse (3:2, 4:3, 5:3, half time...): that is the same grid counted another way, shown as "regrouped". A stretch with no pulse at all (an ambient intro) carries the next
   grid on.
2. **Beats.** Each section's grid is put where the kick and snare land. Then each section is checked: if its
   straight grid slides against the drums over the section (a take recorded without a click, or one that speeds
   up), and a grid that follows the drums sits clearly better, the section "follows the drift" instead: a slow
   tracker, so fills and regroupings don't pull it, only a tempo that wanders over bars.
3. **Subdivisions.** Halves or thirds of the beat, judged a few beats at a time (a song can go triplet at the same
   tempo).
4. **Bars.** How often the rhythm (drums, bass, the rest) repeats - chords don't vote on the length, since a
   progression repeats every few bars. Bar 1 is where kick, crash, bass and chord changes land hardest. A change of
   meter must hold for about two bars, and is placed on the bar line where the bars on each side best match their
   own pattern.

Measured on the test songs (known maps): every tempo within 0.01 BPM, 97-100 % of subdivisions, 92-100 % of bar
lines, 84-100 % of bars the right length - including 7/8, 5/4, 11/8, a triplet feel at the same tempo, a tempo
change, and a take creeping from 88 to 94 BPM with no click (followed within 2 ms).

## What it can't do (yet)

- **A riff that alternates 5 + 7** comes out as one 12-subdivision bar: the bar lines every 12 are right, the line
  between the 5 and the 7 isn't drawn (rhythm alone can't tell it from a 5/4 that alternates kick and snare).
- **Following a drift** needs drums to follow; it can slip by a 16th where the drums go loose for a long stretch.
  The drift lane shows where.
- **Bar lengths on real music are the shakiest part.** Use the Bars table's confidence and your ears.
- No editing on the page. If it goes into Re-Tex, the Beat grid window is where it would be edited.

## Files

`tempomap/` the engine (features, pulse = tempo sections, meter = bars, check = drift, render = click tracks,
report = the page, cli). `tests/` the test songs and their scoring. `maps/<song id>/` per song: `report.html`,
`map.json`, `mix_map.mp3`, `mix_grid.mp3`, `mix_steady.mp3`, `mix_marks.mp3`.
