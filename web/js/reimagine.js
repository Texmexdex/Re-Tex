// Reimagine: the open mix (a DJ mix or one you made by hand), re-performed by YuE2 Studio as one new production.
// 1 Prepare (render the mix, hear its lyrics, YuE2 reads its melody + chords, the DJ writes the style) ->
// 2 Review: edit the style and the lyrics, pick how YuE2 covers it and how hard it follows the style -> 3 Perform ->
// 4 Keep the takes you like: each is saved as a song with its stems next to the originals, lined up with the mix
// bar for bar, and (optionally) dropped into the open mix on muted tracks. Past runs stay listed to reopen.
import { api } from "./api.js";
import { S, on, edit, toast, uid } from "./state.js";
import { modal } from "./ui.js";
import { mountAce, unmountAce } from "./acereimagine.js";

let m = null, job = null, jobKind = null, prep = null, takes = null;
const pendingAdd = new Map();             // song id -> project id it should be dropped into once lined up
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const saved = (() => { try { return JSON.parse(localStorage.getItem("reimagine") || "{}"); } catch { return {}; } })();
const st = Object.assign({ engine: "yue2", prompt: "", takes: 2, addToMix: true, mode: "cover_full", cfg: 1.0, flow: 32,
                           voices: "both", seed: "", cont: 30 }, saved);
const persist = () => { try { localStorage.setItem("reimagine", JSON.stringify(st)); } catch {} };
const mmss = (s) => { const t = Math.round(s); return `${Math.floor(t / 60)}:${String(t % 60).padStart(2, "0")}`; };

// how YuE2 covers the mix - YuE2 Studio's cover modes
const MODES = [
  ["cover_full", "Melody + harmony", "Keeps both songs' tunes and the blend's chords: the most recognisable reimagining.", "score"],
  ["cover_melody", "Melody cover", "Keeps the tunes; YuE2 writes new chords and arrangement - more freedom for the style. (YuE2 Studio's recommended cover.)", "score"],
  ["cover_instrumental", "Instrumental cover", "The melodies played on instruments, no vocals - an instrumental version of the mix. Lyrics aren't used.", "score"],
  ["rerender", "Real-audio re-render", "Re-synthesises the mix itself through YuE2's decoder, the style steering its sound - closest to the original. (Non-commercial weights.)", "audio"],
  ["continue", "Real-audio continuation", "Experimental: plays the mix and keeps going for the seconds you set - an extension of it.", "audio"],
];
const modeLabel = (k) => (MODES.find((x) => x[0] === k) || [k, k])[1];
const strengthWord = (v) => (v <= 1.0 ? "YuE2's default" : v < 1.15 ? "a little stronger" : v < 1.35 ? "stronger" : "strongest");

export function openReimagine() {
  if (m) return;
  m = modal("Reimagine", `
    <div class="ri-engines" role="tablist" aria-label="Reimagine engine">
      <button type="button" class="ri-eng${st.engine === "yue2" ? " on" : ""}" data-eng="yue2">
        <b>YuE2 Studio</b> <span class="dim">local GPU · score &amp; real-audio covers</span>
      </button>
      <button type="button" class="ri-eng${st.engine === "ace" ? " on" : ""}" data-eng="ace">
        <b>ACE Studio</b> <span class="dim">adherence/drift · genre &amp; stem swap · free voice swap</span>
      </button>
    </div>
    <div id="riBody"></div><span id="riProg" style="color:var(--dim)"></span>`,
    { wide: true, onClose: () => { unmountAce(); m = null; } });
  m.body.querySelectorAll(".ri-eng").forEach((b) => {
    b.onclick = () => {
      if (st.engine === b.dataset.eng) return;
      st.engine = b.dataset.eng;
      persist();
      m.body.querySelectorAll(".ri-eng").forEach((x) => x.classList.toggle("on", x.dataset.eng === st.engine));
      prog("");
      render();
    };
  });
  render();
}

function body() { return m?.body.querySelector("#riBody"); }
function prog(html) { const p = m?.body.querySelector("#riProg"); if (p) p.innerHTML = html; }

function render() {
  if (!m) return;
  if (st.engine === "ace") {
    mountAce({
      body,
      prog,
      close: () => m?.close(),
      isVisible: () => !!m && st.engine === "ace",
      trackPending: (sid, pid) => pendingAdd.set(sid, pid),
    });
    return;
  }
  unmountAce();
  if (job) prog(`<span class="spinner"></span> ${jobKind === "prep" ? "preparing…" : "YuE2 is performing…"}`);
  else prog("");
  if (takes) return showTakes();
  if (prep) return showReview();
  showStart();
}

async function showStart() {
  body().innerHTML = `
    <p class="lead">YuE2 re-performs the open mix - a DJ mix or one you made by hand - as one new production: its
      melodies, chords and lyrics, re-sung and re-instrumented in a new style, at the mix's tempo and key.
      First it prepares everything it will get and shows what it found in the songs, so you can shape the style (the style builder) and edit the lyrics before it performs.</p>
    <textarea id="riPrompt" rows="2" spellcheck="false" placeholder="Any direction for the sound? e.g. heavier drums, darker, live band, slower and dreamier (or leave empty)"></textarea>
    <div class="row-gap"><button id="riPrep" class="primary">1 · Prepare</button>
      <span class="dim">renders the mix, hears its lyrics, reads its melody and chords (a minute or two)</span>
      <button id="riStop" class="small" ${job ? "" : "hidden"}>■ Cancel the run in progress</button></div>
    <div id="riRecent"></div>`;
  const B = body();
  const pr = B.querySelector("#riPrompt");
  pr.value = st.prompt;
  pr.oninput = () => { st.prompt = pr.value; persist(); };
  B.querySelector("#riPrep").onclick = startPrep;
  B.querySelector("#riStop").onclick = cancelRun;
  let list = [];
  try { list = await api.get("/api/yue2/renders"); } catch { list = []; }
  const rec = body()?.querySelector("#riRecent");
  if (!rec || !list.length) return;
  rec.innerHTML = `<div class="ri-lab"><b>Recent reimaginings</b> <span class="dim">- reopen to listen and save their takes</span></div>
    <table class="dj-plan"><tbody>${list.map((r, i) => {
      const when = new Date((r.created || 0) * 1000);
      const savedN = Object.keys(r.saved || {}).length;
      return `<tr><td>${esc(r.title)}</td><td class="dim">${esc(r.mode_label)}${r.cfg_scale ? ` · strength ${r.cfg_scale}` : ""}</td>
        <td class="dim">${when.toLocaleDateString()} ${when.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}</td>
        <td>${r.takes.length} take${r.takes.length > 1 ? "s" : ""}${savedN ? ` <span class="tag">${savedN} saved</span>` : ""}</td>
        <td><button class="small" data-i="${i}">Open</button></td></tr>`;
    }).join("")}</tbody></table>`;
  rec.querySelectorAll("button[data-i]").forEach((b) => b.onclick = () => { takes = list[+b.dataset.i]; render(); });
}

async function startPrep() {
  if (!S.project?.clips?.length) { toast("Open a mix first (one you made, or one from DJ Mix)", true); return; }
  if (job && !confirm("A YuE2 run is still in progress. Start a new preparation anyway? (the run keeps going)")) return;
  prog('<span class="spinner"></span> starting…');
  try { job = (await api.post("/api/yue2/prepare", { project: S.project, prompt: st.prompt.trim() })).id; jobKind = "prep"; }
  catch (e) { prog(esc(e.message)); toast(e.message, true); }
}

async function cancelRun() {
  if (!job) return;
  if (!confirm("Cancel the YuE2 run in progress?")) return;
  try { await api.post(`/api/jobs/${job}/cancel`, {}); prog("cancelling…"); }
  catch (e) { toast(e.message, true); }
}

function fitRows(sections, fit) {
  const f = fit?.sections || [];
  return sections.map((s, i) => {
    const x = f[i] || {};
    const flag = x.flag && x.flag !== "ok" ? `<span class="warn">${esc(x.flag)}</span>` : (x.flag ? "ok" : "");
    return `<tr><td>${esc(s.label)}</td><td>${mmss(s.t0)}–${mmss(s.t1)}</td><td>${s.bars}</td>
      <td>${x.melody_notes ?? "-"}</td><td>${x.syllables ?? "-"}</td><td>${flag}</td></tr>`;
  }).join("");
}

// When YuE2 Studio's transcriber finds no sung notes (rap, mostly), every section shows up "note-less": YuE2 still
// raps the lyrics, placing them itself (roughly in place, loosely). Measured on a rap mix (Oct 13): lines 1-2 s late,
// the first one at 0:00 - re-cutting the sections around the sung lines didn't change that, and writing the vocal line
// into the score as notes made it worse (the whole vocal moved ~25 s). So saving re-times the take line by line
// (reimagine.retime_plan); a real-audio re-render keeps the mix's own timing (96% of words within 0.15 s).
function noteNote(p) {
  const f = p.fit?.sections || [];
  if (!f.length || f.some((x) => x.melody_notes > 0)) return "";
  return `<div class="ri-guide warn">YuE2 Studio's transcriber found no sung melody in this mix (usual for rap), so every
    section reads "note-less": YuE2 still performs the lyrics but on its own clock (a second or two off). Saving puts them
    back: <b>move the new vocals line by line</b> (on by default for covers) lines the lyrics up with the take and moves
    each line to where the mix sings it. Or use <b>Real-audio re-render</b>, which keeps the mix's own timing.</div>`;
}

// Where your lyrics were placed (reimagine.place): the sections YuE2 will get, with when each starts and its lines.
function placedTable(pl) {
  if (!pl?.sections?.length) return "";
  const lines = pl.lines || [];
  const weak = lines.filter((l) => l.conf < 0.08).length;
  const rows = pl.sections.map((s, i) => {
    const end = pl.sections[i + 1]?.t0 ?? 1e9;
    const mine = lines.filter((l) => l.t0 >= s.t0 - 0.5 && l.t0 < end - 0.5);
    return `<tr><td>${esc(s.label)}</td><td>${mmss(s.t0)}</td><td>bar ${s.bar + 1}</td>
      <td>${mine.length ? `${mine.length} line${mine.length > 1 ? "s" : ""} <span class="dim">· ${esc(mine[0].text.slice(0, 48))}${mine[0].text.length > 48 ? "…" : ""}</span>` : '<span class="dim">instrumental</span>'}</td></tr>`;
  }).join("");
  return `<div class="ri-lab"><b>Placed</b> <span class="dim">- the sections YuE2 gets (each starts singing at its first bar)${weak ? ` · ${weak} line${weak > 1 ? "s" : ""} hard to hear in the vocals - check them` : ""}</span></div>
    <table class="dj-plan"><thead><tr><th>section</th><th>starts</th><th></th><th>lyrics</th></tr></thead><tbody>${rows}</tbody></table>`;
}

// The style builder (Nov 17): the user reads what Re-Tex found, then shapes the style himself in plain descriptors,
// in the order YuE2 Studio's own examples use ("Instrumental, expressive chamber string ensemble, lyrical violin
// melody, warm viola and cello accompaniment, ..., natural concert hall acoustics, no vocals, no singing, no choir,
// no spoken words"). Chips are only suggestions; any words typed in the fields go in as they are.
const BUILD = [
  ["genre", "Genre / style", "e.g. indie folk, folktronica"],
  ["lead", "Lead instruments", "what carries the melody"],
  ["rhythm", "Drums and bass", "the rhythm section"],
  ["sound", "Sound and space", "production, room, texture"],
  ["mood", "Mood and dynamics", "how it feels and moves"],
];
const CHIPS = {
  genre: ["indie folk", "folktronica", "Icelandic post-rock", "ambient", "neoclassical", "chamber pop", "synthwave",
    "trip-hop", "lo-fi hip hop", "jazz", "progressive metal", "post-rock", "dream pop", "electronica", "cinematic"],
  lead: ["fingerpicked acoustic guitar", "acoustic guitar", "felt piano", "piano", "electric piano", "music box",
    "glockenspiel", "string ensemble", "cello", "violin", "brass section", "flute", "synth lead", "clean electric guitar",
    "distorted electric guitar", "banjo", "harp"],
  rhythm: ["glitchy electronic beats", "soft brushed drums", "live drums", "punchy drums", "tribal percussion",
    "polyrhythmic percussion", "hand claps", "no drums", "deep round bass", "upright bass", "warm synth bass", "heavy bass"],
  sound: ["warm analog synth pads", "ambient pads", "field recordings", "lo-fi tape texture", "reverb-drenched",
    "spacious", "intimate close-miked", "natural room acoustics", "concert hall acoustics", "crisp modern production"],
  mood: ["melancholic", "dreamy", "hypnotic", "dark", "brooding", "hopeful", "uplifting", "gentle", "tense",
    "aggressive", "slow-building", "dynamic swells", "minor key", "major key", "high energy"],
};
const NO_VOCALS = "no vocals, no singing, no choir, no spoken words";
function heardBox(p) {
  const h = p.heard || [];
  if (!h.length) return "";
  return `<div class="ri-guide">${h.map((x) => `<div><b>Found in the song${h.length > 1 ? ` (${esc(x.song)})` : ""}:</b>
      ${x.preset ? `artist style <b>${esc(x.preset)}</b> - ${esc(x.preset_style)}` : "no artist style for it"}
      ${x.measured?.length ? ` · measured: ${esc(x.measured.join(", "))}` : ""}
      ${x.genre ? ` · heard as <b>${esc(x.genre)}</b>${x.sound ? ` - ${esc(x.sound)}` : ""}` : ""}</div>`).join("")}</div>`;
}
function initBuilder(B) {
  const q = (s) => B.querySelector(s);
  const list = (k) => q(`#rb_${k}`).value.split(",").map((x) => x.trim()).filter(Boolean);
  const set = (k, items) => { q(`#rb_${k}`).value = items.join(", "); };
  const markChips = () => B.querySelectorAll(".ri-chip").forEach((c) => {
    c.classList.toggle("on", list(c.dataset.k).map((x) => x.toLowerCase()).includes(c.dataset.v.toLowerCase()));
  });
  const compose = () => {
    const parts = [];
    if (q("#rbIns").checked) parts.push("Instrumental");
    for (const [k] of BUILD) parts.push(...list(k));
    if (q("#rbIns").checked) parts.push(NO_VOCALS);
    return parts.join(", ");
  };
  const preview = () => { markChips(); q("#rbPreview").textContent = compose() ? "Will read: " + compose() : ""; };
  // start: your words as the genre, and what the song has that the chips name (its rhythm, mood) - all editable
  const b = prep.builder || {};
  if (!prep.builder) {
    const found = ((prep.heard || []).map((x) => `${x.preset_style}, ${(x.measured || []).join(", ")}`).join(", ")).toLowerCase();
    b.genre = (prep.prompt || "").split(",").map((x) => x.trim()).filter(Boolean);
    for (const k of ["rhythm", "mood"]) b[k] = CHIPS[k].filter((c) => found.includes(c.toLowerCase()));
    b.ins = st.mode === "cover_instrumental" || prep.sung === false;
  }
  for (const [k] of BUILD) set(k, b[k] || []);
  q("#rbIns").checked = !!b.ins;
  const keep = () => { prep.builder = { ins: q("#rbIns").checked }; for (const [k] of BUILD) prep.builder[k] = list(k); preview(); };
  B.querySelectorAll(".ri-brow input, #rbIns").forEach((e) => { e.oninput = keep; e.onchange = keep; });
  B.querySelectorAll(".ri-chip").forEach((c) => c.onclick = () => {
    const k = c.dataset.k, cur = list(k), i = cur.findIndex((x) => x.toLowerCase() === c.dataset.v.toLowerCase());
    if (i >= 0) cur.splice(i, 1); else cur.push(c.dataset.v);
    set(k, cur); keep();
  });
  q("#rbUse").onclick = () => {
    const s = compose();
    if (!s) { toast("Pick or type something first", true); return; }
    q("#riStyle").value = s; prep.styleEdit = s;
    toast("Style prompt set from the builder - edit it there if you like");
  };
  q("#rbFound").onclick = () => { q("#riStyle").value = prep.style; prep.styleEdit = prep.style; };
  preview();
}

function showReview() {
  body().innerHTML = `
    <div class="dj-head"><b>${esc(prep.title)}</b> · ${prep.bpm} BPM · ${mmss(prep.mix_seconds)} · ${prep.sections.length} sections <span class="dim">· edit anything below, then perform</span></div>
    <div class="ri-lab"><b>How YuE2 covers it</b></div>
    <div class="ri-modes">${MODES.map(([k, t, d, kind]) => `
      <label class="ri-mode${st.mode === k ? " on" : ""}"><input type="radio" name="riMode" value="${k}"${st.mode === k ? " checked" : ""}>
        <b>${t}</b>${kind === "audio" ? ' <span class="tag">real audio</span>' : ""}<div class="dim">${d}</div></label>`).join("")}</div>
    <div class="ri-row">
      <label class="ri-strength">Style strength <span id="riCfgV" class="dim"></span>
        <input type="range" id="riCfg" min="1" max="1.5" step="0.05">
        <span class="dim">How hard YuE2 follows the style prompt (its CFG). Higher = more of the new style, further from the mix's own sound.</span></label>
    </div>
    <label class="ri-lab">Style prompt <span class="dim">(what YuE2 is told about the sound; the tempo is added for you). It starts as what Re-Tex found in the songs - shape it below, or type over it</span></label>
    <textarea id="riStyle" rows="3" spellcheck="false"></textarea>
    ${heardBox(prep)}
    <details class="ri-adv ri-build" open><summary><b>Style builder</b> <span class="dim">- pick what you want; it writes the style the way YuE2 Studio's own examples do: genre, lead instruments, rhythm section, sound and space, mood${""}</span></summary>
      <label class="ri-chk"><input type="checkbox" id="rbIns"> Instrumental <span class="dim">(adds "no vocals, no singing, no choir, no spoken words" - on for an instrumental cover)</span></label>
      ${BUILD.map(([k, label, ph]) => `<div class="ri-brow"><label>${label}</label><input id="rb_${k}" spellcheck="false" placeholder="${esc(ph)}">
        <div class="ri-chips">${CHIPS[k].map((c) => `<button type="button" class="ri-chip" data-k="${k}" data-v="${esc(c)}">${esc(c)}</button>`).join("")}</div></div>`).join("")}
      <div class="dim" id="rbPreview"></div>
      <div class="row-gap"><button id="rbUse" class="small primary">Put it in the style prompt</button>
        <button id="rbFound" class="small" title="Back to what Re-Tex found in the songs">Back to what Re-Tex found</button></div>
    </details>
    <div id="riLyricsWrap">
      <label class="ri-lab">Lyrics <span class="dim">(${prep.lyrics_from === "yours" ? "the songs' own lyrics you typed (song view → Lyrics), already lined up" : prep.lyrics_from === "mixed" ? "yours where you typed a song's lyrics (song view → Lyrics), Whisper's guess for the rest" : "Whisper's guess - type a song's lyrics once in the song view (Lyrics) and every mix uses them"}. Edit or paste your own, no tags needed, then <b>Place my lyrics</b>: each line is lined up with the vocals and filed under the section it's sung in)</span></label>
      <textarea id="riLyrics" rows="12" spellcheck="false"></textarea>
      <table class="dj-plan ri-fit"><thead><tr><th>section</th><th>time in the mix</th><th>bars</th><th>sung notes</th><th>syllables</th><th>fit (an estimate)</th></tr></thead>
        <tbody>${fitRows(prep.placed?.sections || prep.sections, prep.fit)}</tbody></table>
      <div id="riPlaced">${placedTable(prep.placed)}</div>
      ${noteNote(prep)}
      <div class="row-gap"><button id="riPlace" class="small primary" title="Line your lyrics up with the mix's vocals (like YouTube does with an uploaded transcript) and cut the sections where the singing starts">Place my lyrics</button>
        <button id="riFit" class="small">Check fit</button><span id="riPlaceMsg" class="dim"></span></div>
    </div>
    <details class="ri-adv"><summary class="dim">Advanced</summary>
      <div class="ri-row">
        <label>Flow steps <select id="riFlow"><option value="32">32 (default)</option><option value="48">48</option><option value="64">64 (slower, maybe cleaner)</option></select></label>
        <label id="riVoicesL">Melody lines to keep <select id="riVoices"><option value="both">both (vocal + instrument)</option><option value="Vocal">vocal only</option><option value="Ins">instrument only</option></select></label>
        <label>Seed <input id="riSeed" type="number" placeholder="random" style="width:90px"></label>
        <label id="riContL">Continue for <input id="riCont" type="number" min="10" max="120" step="5" style="width:60px"> s</label>
      </div>
    </details>
    <div class="row-gap">
      <label>Takes <select id="riTakes">${[1, 2, 3, 4].map((n) => `<option value="${n}">${n}</option>`).join("")}</select></label>
      <button id="riPerform" class="primary">2 · Perform</button>
      <span class="dim" id="riEta"></span>
      <button id="riRestart" class="small">↺ Start over</button>
    </div>`;
  const B = body();
  const q = (s) => B.querySelector(s);
  q("#riStyle").value = prep.styleEdit ?? prep.style;
  q("#riLyrics").value = prep.lyricsEdit ?? prep.lyrics;
  q("#riStyle").oninput = (e) => { prep.styleEdit = e.target.value; };
  initBuilder(B);
  q("#riLyrics").oninput = (e) => { prep.lyricsEdit = e.target.value; };
  const cfg = q("#riCfg"), cfgV = q("#riCfgV");
  const showCfg = () => { cfgV.textContent = `${(+cfg.value).toFixed(2)} · ${strengthWord(+cfg.value)}`; };
  cfg.value = String(st.cfg); showCfg();
  cfg.oninput = () => { st.cfg = +cfg.value; persist(); showCfg(); };
  const sel = (id, k, num) => { const e = q(id); e.value = String(st[k]); e.onchange = () => { st[k] = num ? +e.value : e.value; persist(); applyMode(); }; };
  sel("#riFlow", "flow", true); sel("#riVoices", "voices"); sel("#riTakes", "takes", true);
  const seed = q("#riSeed"); seed.value = st.seed; seed.oninput = () => { st.seed = seed.value; persist(); };
  const cont = q("#riCont"); cont.value = st.cont; cont.oninput = () => { st.cont = +cont.value || 30; persist(); applyMode(); };
  B.querySelectorAll('input[name="riMode"]').forEach((r) => r.onchange = () => {
    st.mode = r.value; persist();
    if (st.mode === "cover_instrumental") { const c = q("#rbIns"); if (c && !c.checked) { c.checked = true; c.onchange(); } }
    B.querySelectorAll(".ri-mode").forEach((c) => c.classList.toggle("on", c.querySelector("input").checked));
    applyMode();
  });
  function applyMode() {
    const audio = st.mode === "rerender" || st.mode === "continue";
    q("#riLyricsWrap").style.display = st.mode === "cover_instrumental" ? "none" : "";
    q("#riVoicesL").style.display = audio || st.mode === "cover_instrumental" ? "none" : "";
    q("#riContL").style.display = st.mode === "continue" ? "" : "none";
    const per = audio ? (st.mode === "continue" ? (prep.mix_seconds + st.cont) * 0.8 : prep.mix_seconds * 0.5)
                      : prep.mix_seconds * 2.1 * (st.flow / 32 * 0.55 + 0.45);
    q("#riEta").textContent = `about ${Math.max(1, Math.round(per * st.takes / 60))} min for ${st.takes} take${st.takes > 1 ? "s" : ""}`;
  }
  applyMode();
  q("#riPlace").onclick = async () => {
    const btn = q("#riPlace"), msg = q("#riPlaceMsg");
    btn.disabled = true; msg.innerHTML = '<span class="spinner"></span> lining your words up with the vocals…';
    try {
      const r = await api.post("/api/yue2/place", { prep_id: prep.prep_id, lyrics: q("#riLyrics").value });
      prep.placed = { lyrics: r.lyrics, lines: r.lines, sections: r.sections };
      q("#riLyrics").value = prep.lyricsEdit = r.lyrics;
      prep.fit = r.fit;
      q(".ri-fit tbody").innerHTML = fitRows(prep.placed?.sections || prep.sections, prep.fit);
      q("#riPlaced").innerHTML = placedTable(prep.placed);
      msg.textContent = `${r.lines.length} lines placed in ${r.sections.length} sections`;
    } catch (e) { msg.textContent = ""; toast(e.message, true); }
    btn.disabled = false;
  };
  q("#riFit").onclick = async () => {
    try {
      prep.fit = await api.post("/api/yue2/fit", { prep_id: prep.prep_id, lyrics: q("#riLyrics").value });
      q(".ri-fit tbody").innerHTML = fitRows(prep.placed?.sections || prep.sections, prep.fit);
      const left = prep.fit.leftover || [];
      const noteless = (prep.fit.sections || []).length && (prep.fit.sections || []).every((x) => !x.melody_notes);
      toast(noteless ? "No sung notes in the score (rap) - YuE2 places the words itself; re-render keeps them exact"
        : left.length ? `Lyric sections with no matching score section: ${left.join(", ")}` : "Fit checked", !!left.length && !noteless);
    } catch (e) { toast(e.message, true); }
  };
  q("#riPerform").onclick = startPerform;
  q("#riRestart").onclick = async () => {
    if (job) {
      if (!confirm("A YuE2 run is still in progress. Cancel it and start over?")) return;
      try { await api.post(`/api/jobs/${job}/cancel`, {}); } catch {}
    }
    prep = null; takes = null; render();
  };
}

async function startPerform() {
  const B = body();
  const style = B.querySelector("#riStyle").value.trim(), lyrics = B.querySelector("#riLyrics").value;
  if (!style) { toast("The style prompt is empty", true); return; }
  if (job && !confirm("A YuE2 run is still in progress. Start another one after it?")) return;
  prep.styleEdit = style; prep.lyricsEdit = lyrics;
  prog('<span class="spinner"></span> starting…');
  const seed = String(st.seed).trim();
  try {
    job = (await api.post("/api/yue2/perform", {
      prep_id: prep.prep_id, style, lyrics, takes: st.takes, mode: st.mode, cfg_scale: st.cfg > 1.0 ? st.cfg : null,
      flow_steps: st.flow, voices: st.voices, seed: seed === "" ? null : +seed, continuation_seconds: st.cont,
    })).id;
    jobKind = "perform";
  } catch (e) { prog(esc(e.message)); toast(e.message, true); }
}

function metricLine(t) {
  const pct = (x) => (x == null ? null : `${Math.round(100 * x)}%`);
  if (t.melody != null || t.chroma != null)
    return `follows the mix's melody ${pct(t.melody) ?? "-"} · chords ${pct(t.chroma) ?? "-"}`;
  const mt = t.metrics || {};
  const bits = Object.entries(mt).filter(([k, v]) => typeof v === "number" && !["rank"].includes(k)).slice(0, 3)
    .map(([k, v]) => `${k} ${Math.abs(v) < 10 ? v.toFixed(2) : Math.round(v)}`);
  return bits.join(" · ");
}

async function showTakes() {
  const r = takes;
  body().innerHTML = `
    <div class="dj-head"><b>${esc(r.title)}</b> · ${esc(r.mode_label || modeLabel(r.mode || "cover_full"))}${r.cfg_scale ? ` · style strength ${r.cfg_scale}` : ""} · ${r.bpm} BPM
      <span class="dim">· style: ${esc(r.style)}</span></div>
    <div class="ri-takes">${r.takes.map((t) => {
      const sv = (r.saved || {})[String(t.n)];
      return `<label class="ri-take"><input type="checkbox" value="${t.n}"${t.top && !sv ? " checked" : ""}${sv ? " disabled" : ""}>
        <span><b>Take ${t.n}</b>${t.top ? ' <span class="tag">top pick</span>' : ""}${sv ? ' <span class="tag">saved</span>' : ""}
        <span class="dim">· ${mmss(t.seconds)} · ${metricLine(t) || (/rerender|continue/.test(r.mode) ? "real-audio take (YuE2 Studio doesn't score these)" : "no scores")}</span></span>
        <audio controls preload="none" src="/api/yue2/audio/${encodeURIComponent(r.render_id)}/${encodeURIComponent(t.file)}"></audio></label>`;
    }).join("")}</div>
    <div class="ri-row"><label><input type="checkbox" id="riAdd"${st.addToMix ? " checked" : ""}> also drop it into the open mix, lined up bar for bar:</label>
      <select id="riHow"><option value="vocals">only the new vocals, playing, over your music (the mix's own vocals muted)</option>
        <option value="all">all four stems, on muted tracks</option></select></div>
    <div class="ri-row"><label title="YuE2 sings the words roughly in place but on its own clock (a second or two off, worst in rap). This lines the same lyrics up with the take's vocals and moves each line to where the mix sings it - one clip per line, so you can still drag or trim any of them."><input type="checkbox" id="riRetime"> move the new vocals line by line to where the lyrics are sung in the mix</label></div>
    <div class="row-gap"><button id="riSave" class="primary">Save selected · split stems · line up</button>
      <button id="riAgain">↻ New takes</button><button id="riBack">← Edit lyrics / style</button><button id="riHome" class="small">All reimaginings</button></div>`;
  const B = body();
  B.querySelector("#riAdd").onchange = (e) => { st.addToMix = e.target.checked; persist(); };
  const how = B.querySelector("#riHow"), retime = B.querySelector("#riRetime");
  // a real-audio take keeps the mix's own timing; a cover keeps its own clock for the words
  retime.checked = !/rerender|continue/.test(r.mode || "");
  how.value = st.addHow || "vocals";
  how.onchange = () => { st.addHow = how.value; persist(); };
  B.querySelector("#riSave").onclick = async () => {
    const chosen = [...B.querySelectorAll(".ri-take input:checked:not(:disabled)")].map((c) => +c.value);
    if (!chosen.length) { toast("Tick the takes to keep", true); return; }
    const addTo = st.addToMix && S.project ? S.project.id : null;
    try {
      const res = await api.post("/api/yue2/save", { render_id: r.render_id, takes: chosen, add_to: addTo, vocals_only: how.value === "vocals", retime: retime.checked });
      for (const x of res) if (addTo) pendingAdd.set(x.id, addTo);
      toast(`Saving ${res.length} take${res.length > 1 ? "s" : ""}: splitting stems, then lining up with the mix (a few minutes). They go in the originals' folder${addTo ? (how.value === "vocals" ? ", and the new vocals into the open mix over your music" : " and onto muted tracks in the open mix") : ""}.`, false, 9000);
      m?.close();
    } catch (e) { toast(e.message, true); }
  };
  const needPrep = async () => {
    if (prep && prep.prep_id === r.prep_id) return true;
    try { prep = await api.get(`/api/yue2/prep/${encodeURIComponent(r.prep_id)}`); prep.styleEdit = r.style; prep.lyricsEdit = r.lyrics; return true; }
    catch (e) { toast("That run's preparation is gone - prepare the mix again", true); return false; }
  };
  B.querySelector("#riAgain").onclick = async () => { if (await needPrep()) { takes = null; render(); startPerform(); } };
  B.querySelector("#riBack").onclick = async () => { if (await needPrep()) { takes = null; render(); } };
  B.querySelector("#riHome").onclick = () => { takes = null; prep = null; render(); };
}

on("event:job", (msg) => {
  if (!job || msg.job.id !== job) return;
  const visible = !!m && st.engine === "yue2";
  if (msg.job.status === "running" && visible) prog(`<span class="spinner"></span> ${esc(msg.job.message || "")} <span class="pct">${Math.round(100 * (msg.job.progress || 0))}%</span>`);
  if (msg.job.status === "error" || msg.job.status === "cancelled") {
    const txt = msg.job.status === "cancelled" || /cancelled/.test(msg.job.error || "") ? "cancelled" : msg.job.error;
    if (visible) prog(esc(txt)); else toast(txt, txt !== "cancelled");
    job = null;
    if (visible && !prep && !takes) showStart();
  }
  if (msg.job.status === "done" && msg.result) {
    if (jobKind === "prep") { prep = msg.result; takes = null; } else { takes = msg.result; }
    job = null; if (visible) prog("");
    if (visible) render(); else toast(jobKind === "prep" ? "Reimagining prepared - open Reimagine to review the lyrics and style" : "Your YuE2 takes are ready - open Reimagine to listen", false, 8000);
  }
});

// a saved take has been lined up with the mix: drop its stems into the project it came from
on("event:reimagine", (msg) => {
  const target = msg.project || pendingAdd.get(msg.sid);     // the server carries it, so a reload in between is fine
  pendingAdd.delete(msg.sid);
  if (!msg.aligned || !target) return;
  if (S.project?.id !== target) { toast(`${msg.title} is lined up - its stems are in the library (the mix it came from isn't open)`, false, 7000); return; }
  if (S.project.clips.some((c) => c.song === msg.sid)) return;   // already added (e.g. a second open page)

  const engine = msg.engine === "ace" ? "ACE" : "YuE2";
  const dropMode = msg.drop_mode || (msg.vocals_only ? "vocals" : "all");
  const startBeat = Number(msg.start_beat || 0);
  const singleStem = msg.target_stem || "other";
  const stemsToAdd = dropMode === "single" ? [singleStem]
    : dropMode === "vocals" ? ["vocals"]
    : dropMode === "instrumental" ? ["drums", "bass", "other"]
    : ["drums", "bass", "other", "vocals"];
  const playLive = (dropMode === "vocals" || dropMode === "instrumental" || dropMode === "single");
  const shortTitle = engine === "ACE"
    ? msg.title.replace(/^.*\(ACE /, "ACE ").replace(/\)$/, "")
    : msg.title.replace(/^.*\(YuE2 /, "YuE2 ").replace(/\)$/, "");

  let muted = 0;
  edit((p) => {
    p.songs[msg.sid] = { transpose: 0, mult: 1, lock: !!msg.lock };   // its stems locked to the mix's grid
    if (dropMode === "vocals") {                                       // new vocals replace the mix's own
      for (const t of p.tracks) {
        const cl = p.clips.filter((c) => c.track === t.id);
        if (!t.mute && cl.length && cl.every((c) => c.stem === "vocals")) { t.mute = true; muted++; }
      }
    } else if (dropMode === "instrumental") {                          // new backing band under mix's vocals
      for (const t of p.tracks) {
        const cl = p.clips.filter((c) => c.track === t.id);
        if (!t.mute && cl.length && cl.every((c) => c.stem !== "vocals")) { t.mute = true; muted++; }
      }
    }
    for (const stem of stemsToAdd) {
      const lines = stem === "vocals" ? (msg.retime || []) : [];
      const t = { id: uid("t"), name: `${shortTitle} · ${stem}${lines.length ? " (line by line)" : ""}`, role: "any",
                  color: engine === "ACE" ? "#34c77b" : "#8e9198",
                  gain_db: 0, pan: 0, mute: !playLive, solo: false, hpf: 0, lpf: 0, duck: 0, pocket: 0 };
      p.tracks.push(t);
      if (lines.length) {                    // each sung line of the take, moved to where the mix sings it (reimagine.retime_plan)
        for (const l of lines) p.clips.push({ id: uid("c"), track: t.id, song: msg.sid, stem, start: startBeat + l.start,
                                              src: l.src, len: l.len, gain_db: 0, fade_in: 0.03, fade_out: 0.08,
                                              mute: false, rate: l.rate, pitch: 0, keep_pitch: true });
        continue;
      }
      p.clips.push({ id: uid("c"), track: t.id, song: msg.sid, stem, start: startBeat, src: msg.first_beat || 0,
                     len: msg.n_beats, gain_db: 0, fade_in: 0, fade_out: 0.05, mute: false, rate: 1, pitch: 0,
                     keep_pitch: true });
    }
  }, { label: `Add ${engine} take` });

  const moved = (msg.retime || []).length ? ` - ${msg.retime.length} lines moved to where the mix sings them` : "";
  const note = dropMode === "single"
    ? `${msg.title}: dropped as an isolated ${singleStem} lane on the mix grid`
    : dropMode === "instrumental"
    ? `${msg.title}: new instrumental (drums/bass/other) is playing under your vocals${muted ? ` - ${muted} original music lane${muted > 1 ? "s" : ""} muted` : ""}`
    : dropMode === "vocals"
    ? `${msg.title}: its vocals are lined up${moved} and playing over your music${muted ? ` - the mix's ${muted} vocal track${muted > 1 ? "s are" : " is"} muted (unmute to compare)` : ""}`
    : `${msg.title}: lined up and added on 4 muted tracks - unmute to hear it with the mix`;
  toast(note, false, 9000);
});
