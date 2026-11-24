// ACE Studio panel inside the unified Reimagine modal (mounted by reimagine.js when engine === "ace").
// Modes:
//   1. enhancer   : Full Reimagine / AI Cover (music-enhancer) with Adherence <-> Drift slider, multi-axis style builder
//   2. layer      : Instrument / Stem Swap & Layer Injection (add-a-layer) with stem masking + 15 sound types
//   3. voice_swap : Multi-model Voice & Timbre Swap (voice-changer convert) - FREE (0 credits), 33 voices + 14 instruments
//   4. cloud_reuse: Browse & pull prior ACE Studio cloud history takes into Re-Tex - FREE (0 credits)
import { api } from "./api.js";
import { S, on, toast } from "./state.js";

let host = null;         // { body(), prog(html), close(), isVisible(), trackPending(sid, pid) }
let job = null, jobKind = null, prep = null, takes = null;
let aceStatus = null, modelsCache = null, sttCache = null, cloudHistory = null;

const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const mmss = (s) => { const t = Math.max(0, Math.round(s || 0)); return `${Math.floor(t / 60)}:${String(t % 60).padStart(2, "0")}`; };

const saved = (() => { try { return JSON.parse(localStorage.getItem("ace_reimagine") || "{}"); } catch { return {}; } })();
const st = Object.assign({
  prompt: "", skipLyrics: false, sttEngine: "whisper_vad", sttLanguage: "english", sttScopeBars: true,
  mode: "enhancer", influence: 0.50,
  sourceScope: "full", muteStems: [], startBar: "", endBar: "",
  lyricsMode: "mix", soundType: "drums", customInstrument: "", layerPrompt: "",
  modelIds: [], vcKindFilter: "all", vcSearch: "",
  semitones: 0, pitchCorrection: 20, randomOffset: 0,
  correctKey: "", correctScale: "Major", removeInstrument: false, removeReverb: false,
  addToMix: true, dropMode: "all", retime: false,
}, saved);
const persist = () => { try { localStorage.setItem("ace_reimagine", JSON.stringify(st)); } catch {} };

const STT_DEFAULTS = [
  { id: "whisper_vad", label: "Whisper large-v3 (Default · VAD on)", note: "Re-Tex's proven default on GPU. Best general accuracy; Silero VAD can drop soft or breathy singing.", needs_language: false, ok: true },
  { id: "whisper_novad", label: "Whisper large-v3 (VAD off · breathy/soft vocals)", note: "Disables Silero VAD and normalizes each vocal phrase separately to catch quiet, whispered, or reverb-drenched lines.", needs_language: false, ok: true },
  { id: "ace_v2m", label: "ACE Vocal-to-MIDI V2 (Free · singing-trained)", note: "ACE Studio's 0-credit singing transcriber. Strong on melodic/reverberant vocals; requires choosing language.", needs_language: true, ok: true },
  { id: "parakeet_tdt", label: "NVIDIA Parakeet-TDT 0.6B v3 (Local ONNX)", note: "Fast local ONNX recognizer on CPU (zero CUDA conflict). Strong on crisp/dry pop, rock, and rap vocals.", needs_language: false, ok: true },
];
const V2M_LANGS = ["english", "chinese", "japanese", "spanish", "korean", "french", "italian", "portuguese"];

const sttEngines = () => (sttCache?.engines?.length ? sttCache.engines : STT_DEFAULTS);
const sttLanguages = () => (sttCache?.languages?.length ? sttCache.languages : V2M_LANGS);
const sttLabel = (id) => {
  if (id === "whisper") return "Whisper large-v3 (VAD on)";
  const e = sttEngines().find((x) => x.id === id);
  return e ? e.label : id;
};
async function ensureSttInfo() {
  if (sttCache) return sttCache;
  try { sttCache = await api.get("/api/ace/stt"); }
  catch { sttCache = { engines: STT_DEFAULTS, languages: V2M_LANGS }; }
  return sttCache;
}

const ACE_MODES = [
  ["enhancer", "Full Reimagine / Genre Swap", "Re-produces the mix as one unified track. Continuous Adherence to Drift slider controls how far it moves from the original.", "credits"],
  ["layer", "Instrument / Stem Swap", "Mute any stem(s) from the reference mix and generate a locked replacement layer — or inject a new instrument or backing vocal.", "credits"],
  ["voice_swap", "Voice & Timbre Swap", "Re-sing vocals across up to 6 voice models — or turn a monophonic line into an instrument (synth bass, sax, violin, talkbox).", "free"],
];

const DRIFT_LABEL = (v) => (
  v <= 0.20 ? "Strict adherence · mix glue & polish"
  : v <= 0.45 ? "Remix / restyle · keeps arrangement & phrasing"
  : v <= 0.75 ? "Genre & instrument swap · bold re-instrumentation"
  : "Full reimagining · maximum creative drift"
);

const ACE_BUILD = [
  ["genre", "Primary genre", "e.g. progressive metal, synthwave, chamber folk"],
  ["fusion", "Mashup fusion", "optional second genre to blend in"],
  ["vocals", "Vocals & delivery", "singer timbre, register, style, vocal FX"],
  ["lead", "Lead instruments", "melodic & harmonic focal instruments"],
  ["rhythm", "Drums & bass", "rhythm section, groove & low end"],
  ["mood", "Mood & dynamics", "emotional colour, energy & arc"],
  ["production", "Era & production", "mix aesthetic, room & sonic texture"],
  ["avoid", "Exclude / avoid", "optional elements to keep out"],
];

const ACE_CHIPS = {
  genre: ["synthwave", "progressive metal", "industrial", "trip-hop", "drum and bass", "post-rock", "Icelandic post-rock",
    "shoegaze", "darkwave", "indie folk", "folktronica", "neoclassical", "lo-fi hip hop", "jazz fusion", "doom jazz",
    "funk", "chiptune", "hyperpop", "cinematic orchestral", "grunge", "psychedelic rock"],
  fusion: ["orchestral hybrid", "electronic rock", "metal x synthwave", "jazz x hip hop", "folk x IDM",
    "ambient x breakbeat", "8-bit x metal", "acoustic unplugged", "cyberpunk industrial"],
  vocals: ["male baritone", "male tenor", "raspy rock vocal", "female soprano", "breathy intimate vocal", "soulful alto",
    "duet vocals", "layered harmonies", "choir / gang vocals", "rhythmic rap delivery", "spoken word",
    "harsh screamed vocals", "operatic", "vocoder / talkbox", "reverb-drenched vocals", "dry close-miked vocals"],
  lead: ["distorted electric guitar", "clean electric guitar", "fingerpicked acoustic guitar", "felt piano", "grand piano",
    "Rhodes electric piano", "Hammond organ", "analog synth lead", "8-bit chiptune lead", "string orchestra",
    "solo cello", "lyrical violin", "brass section", "tenor saxophone", "banjo", "harp", "sitar"],
  rhythm: ["punchy live rock kit", "double-kick metal drums", "brushed jazz drums", "breakbeat / jungle drums",
    "glitchy IDM percussion", "808 trap drums", "polyrhythmic percussion", "half-time groove", "no drums",
    "growling distorted bass", "warm synth bass", "Reese sub bass", "upright acoustic bass", "slap funk bass", "acid 303 bass"],
  mood: ["dark", "brooding", "melancholic", "aggressive", "high energy", "hypnotic", "dreamy", "euphoric",
    "cinematic swells", "slow-building", "tense", "intimate", "stripped-back", "chaotic", "anthemic"],
  production: ["modern crisp hi-fi", "70s analog tape warmth", "80s gated reverb", "90s raw live room",
    "lo-fi cassette texture", "vinyl crackle", "concert hall acoustics", "intimate bedroom recording",
    "dry punchy club mix", "wall-of-sound", "wide stereo cinematic"],
  avoid: ["no vocals", "no drums", "no guitars", "no synths", "no heavy distortion", "no rap", "no choir"],
};

const LAYER_CHIPS = [
  "tight acoustic kit", "heavy double-kick drums", "brushed jazz groove", "glitchy breakbeat",
  "warm analog synth bass", "growling distorted bass", "upright acoustic bass", "lush string section",
  "distorted rhythm guitars", "fingerpicked acoustic guitar", "warm Rhodes chords", "analog arp synth",
  "brass stabs", "cinematic risers & impacts", "soulful backing harmonies", "whispered double vocal",
];

export function mountAce(h) {
  host = h;
  renderAce();
}

export function unmountAce() {
  host = null;
}

export function aceJobActive() {
  return !!job;
}

function body() { return host?.body(); }
function prog(html) { host?.prog(html); }

export function renderAce() {
  if (!host || !host.isVisible()) return;
  if (job) {
    const lbl = jobKind === "prep" ? "preparing mix for ACE Studio…"
      : jobKind === "stt" ? "transcribing mix vocals…"
      : jobKind === "reuse" ? "fetching cloud take from ACE Studio…"
      : "ACE Studio is rendering…";
    prog(`<span class="spinner"></span> ${lbl}`);
  } else {
    prog("");
  }
  if (takes) return showTakes();
  if (prep) return showReview();
  showStart();
}

async function refreshStatus() {
  try { aceStatus = await api.get("/api/ace/status"); }
  catch (e) { aceStatus = { installed: false, running: false, bridge: false, error: e.message }; }
  return aceStatus;
}

function statusStripHtml(s) {
  if (!s) return `<div class="ace-stat dim">Checking ACE Studio…</div>`;
  if (!s.installed) {
    return `<div class="ace-stat bad"><b>ACE Studio CLI not found</b> · ${esc(s.error || s.cli)} <button class="small" id="aceRefStat">Retry</button></div>`;
  }
  if (!s.bridge) {
    return `<div class="ace-stat warn"><b>ACE Studio closed</b> · Re-Tex will launch it automatically when you run (or open it now). <button class="small" id="aceRefStat">Refresh</button></div>`;
  }
  if (s.awaiting_user) {
    return `<div class="ace-stat warn"><b>ACE Studio needs attention:</b> ${esc(s.awaiting_user)} <button class="small" id="aceRefStat">Refresh</button></div>`;
  }
  if (s.dirty && !(s.is_temp && !s.project_name)) {
    return `<div class="ace-stat warn"><b>Unsaved project open in ACE Studio</b> (${esc(s.project_name || "untitled")}) · save it in ACE Studio before running so Re-Tex won't overwrite your session. <button class="small" id="aceRefStat">Refresh</button></div>`;
  }
  return `<div class="ace-stat ok"><b>ACE Studio connected</b> · bridge ready ${s.project_name ? `(${esc(s.project_name)})` : "(clean stage)"} <button class="small" id="aceRefStat">Refresh</button></div>`;
}

function bindStatusRefresh(B) {
  const b = B.querySelector("#aceRefStat");
  if (!b) return;
  b.onclick = async () => {
    b.disabled = true;
    await refreshStatus();
    if (prep) prep.ace_status = aceStatus;
    renderAce();
  };
}

// ------------------------------------------------------------------ Stage 1: Start + Recent + Cloud History (0 credits)

async function showStart() {
  const B = body();
  if (!B) return;
  B.innerHTML = `
    <div id="aceStatSlot">${statusStripHtml(aceStatus)}</div>
    <p class="lead">Reimagine the open Re-Tex mix with <b>ACE Studio</b>: full genre & style covers with continuous
      <b>Adherence ↔ Drift</b> control, stem-masked instrument swapping, or <b>0-credit</b> multi-model voice and
      instrument timbre swaps — then split, bar-align, and drop straight onto your timeline.</p>
    <textarea id="acePrompt" rows="2" spellcheck="false" placeholder="Initial direction? e.g. turn this into 80s dark synthwave with heavy drums, or swap lead vocal to soulful female (optional)"></textarea>
    <div class="ri-row">
      <label class="ri-chk" style="margin:0"><input type="checkbox" id="aceSkipLyr"${st.skipLyrics ? " checked" : ""}> Skip lyric transcription <span class="dim">(instant prepare — you can still transcribe anytime in Stage 2)</span></label>
    </div>
    <div class="ri-row ace-stt-bar" id="acePrepSttRow">
      <label>Lyrical STT engine
        <select id="acePrepStt">
          ${sttEngines().map((e) => `<option value="${esc(e.id)}">${esc(e.label)}${e.ok === false ? " [download needed]" : ""}</option>`).join("")}
        </select>
      </label>
      <label id="acePrepLangWrap">Language
        <select id="acePrepLang">
          ${sttLanguages().map((l) => `<option value="${esc(l)}">${esc(l[0].toUpperCase() + l.slice(1))}</option>`).join("")}
        </select>
      </label>
      <span class="dim" id="acePrepSttNote"></span>
    </div>
    <div class="row-gap">
      <button id="acePrep" class="primary">1 · Prepare open mix</button>
      <span class="dim">snapshots the open mix, detects tempo/key/traits, and unlocks all ACE modes</span>
      <button id="aceStop" class="small" ${job ? "" : "hidden"}>■ Cancel run</button>
    </div>
    <div id="aceRecent"></div>
    <div id="aceCloudBox" class="ace-cloud-box">
      <div class="row-gap">
        <b>ACE Studio Cloud History</b> <span class="tag ace-free">0 credits</span>
        <span class="dim">Reuse prior Music Enhancer or Inspire Me generations from your account</span>
        <button id="aceLoadCloud" class="small">${cloudHistory ? "↻ Refresh cloud history" : "Load cloud history"}</button>
      </div>
      <div id="aceCloudList"></div>
    </div>`;

  const pr = B.querySelector("#acePrompt");
  pr.value = st.prompt;
  pr.oninput = () => { st.prompt = pr.value; persist(); };
  const skipEl = B.querySelector("#aceSkipLyr");
  const sttRow = B.querySelector("#acePrepSttRow");
  const sttSel = B.querySelector("#acePrepStt");
  const langWrap = B.querySelector("#acePrepLangWrap");
  const langSel = B.querySelector("#acePrepLang");
  const sttNote = B.querySelector("#acePrepSttNote");
  sttSel.value = st.sttEngine || "whisper_vad";
  langSel.value = st.sttLanguage || "english";
  const syncPrepStt = () => {
    const eng = sttEngines().find((x) => x.id === sttSel.value) || sttEngines()[0];
    sttRow.style.display = skipEl.checked ? "none" : "";
    langWrap.style.display = eng?.needs_language ? "" : "none";
    if (sttNote) sttNote.textContent = eng?.ok === false && eng?.problem ? `Requires download: ${eng.problem}` : (eng?.note || "");
  };
  skipEl.onchange = (e) => { st.skipLyrics = e.target.checked; persist(); syncPrepStt(); };
  sttSel.onchange = () => { st.sttEngine = sttSel.value; persist(); syncPrepStt(); };
  langSel.onchange = () => { st.sttLanguage = langSel.value; persist(); };
  syncPrepStt();
  ensureSttInfo().then(() => {
    const sel = body()?.querySelector("#acePrepStt");
    if (!sel) return;
    sel.innerHTML = sttEngines().map((e) => `<option value="${esc(e.id)}">${esc(e.label)}${e.ok === false ? " [download needed]" : ""}</option>`).join("");
    sel.value = st.sttEngine || "whisper_vad";
    syncPrepStt();
  });
  B.querySelector("#acePrep").onclick = startPrep;
  B.querySelector("#aceStop").onclick = cancelRun;
  B.querySelector("#aceLoadCloud").onclick = () => loadCloudHistory(true);
  bindStatusRefresh(B);

  // Live status check in background
  refreshStatus().then((s) => {
    const slot = body()?.querySelector("#aceStatSlot");
    if (slot) { slot.innerHTML = statusStripHtml(s); bindStatusRefresh(body()); }
  });

  // Render cached cloud history if already loaded
  if (cloudHistory) renderCloudList(B);

  // Load recent local ACE renders
  let list = [];
  try { list = await api.get("/api/ace/renders"); } catch { list = []; }
  const rec = body()?.querySelector("#aceRecent");
  if (!rec || !list.length) return;
  rec.innerHTML = `<div class="ri-lab"><b>Recent ACE reimaginings</b> <span class="dim">- reopen to audition and save their takes</span></div>
    <table class="dj-plan"><tbody>${list.map((r, i) => {
      const when = new Date((r.created || 0) * 1000);
      const savedN = Object.keys(r.saved || {}).length;
      const drift = r.influence != null ? ` · drift ${Math.round(r.influence * 100)}%` : "";
      const costTag = r.credit_billed ? "" : ' <span class="tag ace-free">free</span>';
      return `<tr><td>${esc(r.title)}</td>
        <td class="dim">${esc(r.mode_label)}${drift}${costTag}</td>
        <td class="dim">${when.toLocaleDateString()} ${when.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}</td>
        <td>${r.takes.length} take${r.takes.length > 1 ? "s" : ""}${savedN ? ` <span class="tag">${savedN} saved</span>` : ""}</td>
        <td><button class="small" data-i="${i}">Open</button></td></tr>`;
    }).join("")}</tbody></table>`;
  rec.querySelectorAll("button[data-i]").forEach((b) => {
    b.onclick = () => { takes = list[+b.dataset.i]; renderAce(); };
  });
}

async function loadCloudHistory(refresh = false) {
  const B = body();
  const listEl = B?.querySelector("#aceCloudList");
  if (!listEl) return;
  listEl.innerHTML = `<div class="dim"><span class="spinner"></span> reading ACE Studio cloud history (0 credits)…</div>`;
  try {
    const r = await api.get(`/api/ace/history?kit=all&refresh=${refresh ? "true" : "false"}`);
    cloudHistory = r.items || [];
    renderCloudList(body());
  } catch (e) {
    listEl.innerHTML = `<div class="ri-guide warn">${esc(e.message)}</div>`;
  }
}

function renderCloudList(B) {
  const listEl = B?.querySelector("#aceCloudList");
  if (!listEl || !cloudHistory) return;
  if (!cloudHistory.length) {
    listEl.innerHTML = `<div class="dim" style="margin-top:6px">No past Inspire Me or Music Enhancer results found on your ACE Studio account yet.</div>`;
    return;
  }
  listEl.innerHTML = `<table class="dj-plan" style="margin-top:6px"><tbody>${cloudHistory.slice(0, 20).map((it, i) => `
    <tr>
      <td><b>${esc(it.title)}</b> <span class="tag">${esc(it.kit)}</span></td>
      <td class="dim">${esc((it.tags || "").slice(0, 56))}</td>
      <td class="dim">${it.duration ? mmss(it.duration) : "-"}</td>
      <td><button class="small primary" data-ci="${i}">Pull (0 cr)</button></td>
    </tr>`).join("")}</tbody></table>`;
  listEl.querySelectorAll("button[data-ci]").forEach((b) => {
    b.onclick = async () => {
      const it = cloudHistory[+b.dataset.ci];
      if (!it) return;
      prog('<span class="spinner"></span> pulling cloud take…');
      try {
        job = (await api.post("/api/ace/reuse", {
          result_id: it.result_id, title: it.title, kit: it.kit,
          prep_id: prep?.prep_id || null, bpm: S.project?.bpm || it.bpm || 120,
        })).id;
        jobKind = "reuse";
      } catch (e) { prog(esc(e.message)); toast(e.message, true); }
    };
  });
}

async function startPrep() {
  if (!S.project?.clips?.length) { toast("Open a mix first (one you made, or one from DJ Mix)", true); return; }
  if (job && !confirm("An ACE Studio run is still in progress. Prepare a new mix anyway?")) return;
  prog('<span class="spinner"></span> preparing mix…');
  try {
    job = (await api.post("/api/ace/prepare", {
      project: S.project, prompt: st.prompt.trim(), skip_lyrics: !!st.skipLyrics,
      stt_engine: st.sttEngine || "whisper_vad", stt_language: st.sttLanguage || "english",
    })).id;
    jobKind = "prep";
  } catch (e) { prog(esc(e.message)); toast(e.message, true); }
}

async function cancelRun() {
  if (!job) return;
  if (!confirm("Cancel the ACE Studio run in progress?")) return;
  try { await api.post(`/api/jobs/${job}/cancel`, {}); prog("cancelling…"); }
  catch (e) { toast(e.message, true); }
}

// ------------------------------------------------------------------ Stage 2: Review & Configure All Variables

function heardBox(p) {
  const h = p.heard || [];
  if (!h.length) return "";
  return `<div class="ri-guide">${h.map((x) => `<div><b>Detected${h.length > 1 ? ` (${esc(x.song)})` : ""}:</b>
    ${x.preset ? `style <b>${esc(x.preset)}</b> - ${esc(x.preset_style)}` : ""}
    ${x.measured?.length ? ` · measured: ${esc(x.measured.join(", "))}` : ""}
    ${x.genre ? ` · heard as <b>${esc(x.genre)}</b>${x.sound ? ` - ${esc(x.sound)}` : ""}` : ""}</div>`).join("")}</div>`;
}

async function ensureModels() {
  if (modelsCache) return modelsCache;
  try { modelsCache = await api.get("/api/ace/models"); }
  catch (e) { toast(`Couldn't load ACE models: ${e.message}`, true); modelsCache = { models: [], sound_types: [], keys: [], scales: ["Major", "Minor"] }; }
  return modelsCache;
}

function showReview() {
  const B = body();
  if (!B) return;
  const p = prep;
  if (p.ace_status) aceStatus = p.ace_status;
  if (!st.correctKey && p.key) { st.correctKey = p.key; st.correctScale = p.scale || "Major"; }

  const secOpts = (p.sections || []).map((s, i) =>
    `<option value="${i}">${esc(s.label)} (bars ${s.start_bar}–${s.end_bar})</option>`
  ).join("");
  const stemList = ["drums", "bass", "other", "vocals"];

  B.innerHTML = `
    <div id="aceStatSlot">${statusStripHtml(aceStatus)}</div>
    <div class="dj-head">
      <b>${esc(p.title)}</b> · ${p.bpm} BPM${p.key ? ` · ${esc(p.key)} ${esc(p.scale || "")}` : ""} · ${p.total_bars} bars (${mmss(p.mix_seconds)})
      <span class="dim">· pick a mode, dial Adherence/Drift & variables, then run</span>
    </div>

    <div class="ri-lab"><b>ACE Studio Mode</b></div>
    <div class="ri-modes">${ACE_MODES.map(([k, title, desc, cost]) => `
      <label class="ri-mode${st.mode === k ? " on" : ""}">
        <input type="radio" name="aceMode" value="${k}"${st.mode === k ? " checked" : ""}>
        <b>${esc(title)}</b>
        <span class="tag ${cost === "free" ? "ace-free" : "ace-cost"}">${cost === "free" ? "FREE · 0 credits" : "credits"}</span>
        <div class="dim">${esc(desc)}</div>
      </label>`).join("")}</div>

    <div class="ace-box">
      <div class="ri-row">
        <label>Input source
          <select id="aceScope">
            <option value="full">Full open mix</option>
            <option value="instrumental">Instrumental only (vocals muted)</option>
            <option value="vocals">Vocals stem only</option>
            <option value="other">Other / melodic stem only</option>
            <option value="bass">Bass stem only</option>
            <option value="drums">Drums stem only</option>
          </select>
        </label>
        <label>Bars
          <input id="aceBarA" type="number" min="1" max="${p.total_bars}" placeholder="1" style="width:62px">
          –
          <input id="aceBarB" type="number" min="1" max="${p.total_bars}" placeholder="${p.total_bars}" style="width:62px">
        </label>
        ${secOpts ? `<label>Section <select id="aceSecJump"><option value="">Full mix (1–${p.total_bars})</option>${secOpts}</select></label>` : ""}
        <span class="dim" id="aceRangeInfo"></span>
      </div>
      <div id="aceRangeWarn"></div>
      <div class="ri-row" id="aceMaskRow" style="margin-top:8px">
        <span class="dim">Mute from reference bounce (for stem replacement):</span>
        ${stemList.map((s) => `<label class="ace-inline-chk"><input type="checkbox" class="ace-mask" value="${s}"${(st.muteStems || []).includes(s) ? " checked" : ""}> ${s}</label>`).join("")}
      </div>
    </div>

    <!-- MODE 1: FULL REIMAGINE / AI COVER (music-enhancer) -->
    <div id="acePaneEnhancer">
      <div class="ace-box">
        <label class="ri-strength">
          <span><b>Adherence ↔ Reimagine Drift</b> · <span id="aceInfVal" class="ace-inf-val"></span></span>
          <input type="range" id="aceInf" min="0" max="1" step="0.05">
          <div class="ri-row" style="margin-top:4px; gap:6px">
            <button type="button" class="ri-chip ace-inf-pre" data-v="0.15">15% Polish &amp; Glue</button>
            <button type="button" class="ri-chip ace-inf-pre" data-v="0.40">40% Remix / Restyle</button>
            <button type="button" class="ri-chip ace-inf-pre" data-v="0.70">70% Genre Swap</button>
            <button type="button" class="ri-chip ace-inf-pre" data-v="0.90">90% Full Reimagine</button>
          </div>
          <span class="dim">Maps to ACE Studio Music Enhancer <code>--influence</code> (0.00 follows the original mix closely; 1.00 lets your target genre, instruments, mood &amp; vocals override freely).</span>
        </label>
      </div>

      <label class="ri-lab">Style &amp; Reimagine Prompt <span class="dim">(edit directly or compose with the variable builder below)</span></label>
      <textarea id="aceStyle" rows="3" spellcheck="false"></textarea>
      ${heardBox(p)}

      <details class="ri-adv ri-build" open>
        <summary><b>Multi-Axis Style &amp; Swap Builder</b> <span class="dim">- genre, mashup fusion, vocals, lead instruments, drums &amp; bass, mood, era/production, excludes</span></summary>
        ${ACE_BUILD.map(([k, lbl, ph]) => `
          <div class="ri-brow">
            <label>${esc(lbl)}</label>
            <input id="ab_${k}" spellcheck="false" placeholder="${esc(ph)}">
            <div class="ri-chips">${ACE_CHIPS[k].map((c) => `<button type="button" class="ri-chip ab-chip" data-k="${k}" data-v="${esc(c)}">${esc(c)}</button>`).join("")}</div>
          </div>`).join("")}
        <div class="dim" id="abPreview"></div>
        <div class="row-gap">
          <button type="button" id="abUse" class="small primary">Apply to style prompt</button>
          <button type="button" id="abClear" class="small">Clear builder</button>
          <button type="button" id="abFound" class="small">Reset to detected mix traits</button>
        </div>
      </details>

      <div class="ace-box" style="margin-top:10px">
        <div class="ri-row">
          <label><b>Vocals &amp; Lyrics</b>
            <select id="aceLyrMode">
              <option value="mix">Re-sing mix lyrics (editable below)</option>
              <option value="custom">Custom / rewritten lyrics</option>
              <option value="auto">Let ACE Studio auto-transcribe from mix</option>
              <option value="instrumental">Instrumental only (strip vocals)</option>
            </select>
          </label>
          <span class="dim" id="aceLyrSrc">${p.lyrics_from === "yours" ? "Source: your typed song lyrics" : p.lyrics_from === "mixed" ? `Source: your lyrics + STT (${(p.lines || []).length} lines)` : p.lyrics_from === "skipped" ? "Skipped during prepare" : p.lyrics_from && p.lyrics_from !== "none" ? `Source: ${esc(sttLabel(p.lyrics_from))} (${(p.lines || []).length} lines)` : "No lyrics detected"}</span>
          ${(p.lines || []).length ? `<button type="button" id="aceSliceLyr" class="small">Slice to selected bars</button>` : ""}
        </div>
        ${(p.stems_present || []).includes("vocals") ? `
        <div class="ri-row ace-stt-bar" id="aceRetransBar" style="margin-top:6px">
          <label>STT Engine
            <select id="aceRetransEng">
              ${sttEngines().map((e) => `<option value="${esc(e.id)}">${esc(e.label)}${e.ok === false ? " [download needed]" : ""}</option>`).join("")}
            </select>
          </label>
          <label id="aceRetransLangWrap">Lang
            <select id="aceRetransLang">
              ${sttLanguages().map((l) => `<option value="${esc(l)}">${esc(l[0].toUpperCase() + l.slice(1))}</option>`).join("")}
            </select>
          </label>
          <label class="ace-inline-chk"><input type="checkbox" id="aceRetransScope"${st.sttScopeBars ? " checked" : ""}> Selected bars only</label>
          <button type="button" id="aceRetransBtn" class="small primary">Re-transcribe vocals (0 cr)</button>
        </div>
        <div class="dim" id="aceRetransNote" style="font-size:11.5px; margin-top:2px"></div>` : ""}
        <div id="aceLyrBox" style="margin-top:6px">
          <textarea id="aceLyrics" rows="7" spellcheck="false" placeholder="Paste or edit the exact lyrics to sing (leaving this blank lets ACE auto-transcribe, which often mishears effected/chopped vocals)"></textarea>
        </div>
        <div id="aceLyrNote" class="dim" style="margin-top:4px; font-size:12px"></div>
        <details id="aceSendPrev" class="ace-send-prev" open>
          <summary><b>Pre-send check: exact lyrics &amp; audio sent to ACE</b></summary>
          <pre id="aceSendPrevBody"></pre>
        </details>
      </div>
    </div>

    <!-- MODE 2: INSTRUMENT / STEM SWAP (add-a-layer) -->
    <div id="acePaneLayer">
      <div class="ace-box">
        <div class="ri-row">
          <label><b>Target Layer Type</b>
            <select id="aceSoundType">
              ${["drums", "bass", "guitar", "keyboard", "strings", "synth", "brass", "woodwinds", "percussion", "fx", "vocals", "backing-vocals", "instrumental", "song-track", "custom"].map((t) => `<option value="${t}">${t}</option>`).join("")}
            </select>
          </label>
          <label id="aceCustInstWrap"><b>Custom instrument</b>
            <input id="aceCustInst" spellcheck="false" placeholder="e.g. baritone sax, taiko drums, 8-bit arp" style="width:240px">
          </label>
        </div>
        <label class="ri-lab" style="margin-top:8px">Layer Timbre, Style &amp; Groove <span class="dim">(keep concise — do NOT describe chords or rhythm; ACE listens directly to your mix's harmony and groove)</span></label>
        <textarea id="aceLayerPrompt" rows="2" spellcheck="false" placeholder="e.g. punchy organic indie rock drum kit, dry close-miked room, dynamic ghost notes"></textarea>
        <div class="ri-chips" style="margin-top:6px">${LAYER_CHIPS.map((c) => `<button type="button" class="ri-chip al-chip" data-v="${esc(c)}">${esc(c)}</button>`).join("")}</div>
        <div id="aceLayerLyrWrap" style="margin-top:8px">
          <label class="ri-lab">Layer Lyrics <span class="dim">(for vocals / backing-vocals / song-track)</span></label>
          <textarea id="aceLayerLyrics" rows="4" spellcheck="false" placeholder="Words or harmony lines for ACE to sing over the mix"></textarea>
        </div>
      </div>
    </div>

    <!-- MODE 3: FREE VOICE & TIMBRE SWAP (voice-changer) -->
    <div id="acePaneVoice">
      <div class="ace-box">
        <div class="ri-row" style="justify-content:space-between">
          <div><b>Pick 1–6 Models to Batch-Audition</b> <span class="tag ace-free">FREE · 0 credits</span> <span class="dim" id="aceVcCount"></span></div>
          <div class="ri-row" style="margin:0; gap:6px">
            <select id="aceVcFilter">
              <option value="all">All models</option>
              <option value="voice">Voices only (33)</option>
              <option value="instrument">Instruments only (14)</option>
            </select>
            <input id="aceVcSearch" spellcheck="false" placeholder="Search name or tag (e.g. soul, metal, synth, sax)…" style="width:220px">
          </div>
        </div>
        <div id="aceVcGrid" class="ace-vc-grid"><div class="dim">Loading ACE Studio Voice Changer models…</div></div>
      </div>

      <div class="ace-box" style="margin-top:8px">
        <div class="ri-row">
          <label class="ace-slider">Transpose <span id="aceSemiV" class="dim"></span>
            <input type="range" id="aceSemi" min="-24" max="24" step="1">
          </label>
          <label class="ace-slider">Pitch correction <span id="acePitchV" class="dim"></span>
            <input type="range" id="acePitch" min="0" max="100" step="5">
          </label>
          <label class="ace-slider">Take drift (random offset) <span id="aceDriftV" class="dim"></span>
            <input type="range" id="aceDrift" min="0" max="100" step="5">
          </label>
        </div>
        <div class="ri-row" style="margin-top:8px">
          <label>Snap to key
            <select id="aceKey">
              <option value="">Off (preserve original pitch)</option>
              ${["C","C#","Db","D","D#","Eb","E","F","F#","Gb","G","G#","Ab","A","A#","Bb","B"].map((k) => `<option value="${k}">${k}</option>`).join("")}
            </select>
          </label>
          <label>Scale
            <select id="aceScale"><option value="Major">Major</option><option value="Minor">Minor</option></select>
          </label>
          <label class="ace-inline-chk"><input type="checkbox" id="aceRmInst"> Strip backing music</label>
          <label class="ace-inline-chk"><input type="checkbox" id="aceRmRev"> Strip reverb</label>
        </div>
      </div>
    </div>

    <div class="row-gap" style="margin-top:12px">
      <button id="aceRun" class="primary">2 · Run in ACE Studio</button>
      <span id="aceCostNote" class="dim"></span>
      <button id="aceRestart" class="small">↺ Start over</button>
    </div>`;

  const q = (s) => B.querySelector(s);
  bindStatusRefresh(B);

  // Source scope & range controls
  const scopeEl = q("#aceScope"), barA = q("#aceBarA"), barB = q("#aceBarB");
  scopeEl.value = st.sourceScope || "full";
  barA.value = st.startBar || "";
  barB.value = st.endBar || "";
  const updateRangeInfo = () => {
    const a = Math.max(1, +barA.value || 1);
    const b = Math.min(p.total_bars, +barB.value || p.total_bars);
    const bars = Math.max(1, b - a + 1);
    const barSec = (p.bpb || 4) * (60 / (p.bpm || 120));
    const secs = bars * barSec;
    const infoEl = q("#aceRangeInfo");
    const warnEl = q("#aceRangeWarn");
    if (infoEl) infoEl.textContent = `${bars} bar${bars > 1 ? "s" : ""} (~${mmss(secs)})`;
    if (!warnEl) return;
    if (st.mode === "enhancer" && (secs > 239.5 || secs < 3.0)) {
      const maxBars = Math.max(1, Math.floor(235 / barSec));
      const firstB = Math.min(p.total_bars, maxBars);
      const lastA = Math.max(1, p.total_bars - maxBars + 1);
      const midA = Math.max(1, Math.floor((p.total_bars - maxBars) / 2) + 1);
      const midB = Math.min(p.total_bars, midA + maxBars - 1);
      const over = secs > 239.5;
      warnEl.innerHTML = `<div class="ri-guide warn" style="flex-direction:column; gap:6px; margin-top:8px">
        <div><b>${over ? "Range too long" : "Range too short"} (${mmss(secs)} / ${Math.round(secs)}s):</b>
          ACE Studio Music Enhancer accepts <b>0:03 to 4:00 (240s max)</b>. Pick a bar window up to ${maxBars} bars:</div>
        ${over ? `<div class="ri-row" style="margin:0; gap:6px">
          <button type="button" class="small primary ace-fit-btn" data-a="1" data-b="${firstB}">First ~3:55 (bars 1–${firstB})</button>
          ${midA > 1 && midB < p.total_bars ? `<button type="button" class="small ace-fit-btn" data-a="${midA}" data-b="${midB}">Middle ~3:55 (bars ${midA}–${midB})</button>` : ""}
          <button type="button" class="small ace-fit-btn" data-a="${lastA}" data-b="${p.total_bars}">Last ~3:55 (bars ${lastA}–${p.total_bars})</button>
        </div>` : ""}
      </div>`;
      warnEl.querySelectorAll(".ace-fit-btn").forEach((btn) => {
        btn.onclick = () => {
          barA.value = btn.dataset.a;
          barB.value = btn.dataset.b;
          st.startBar = barA.value;
          st.endBar = barB.value;
          if (secSel) secSel.value = "";
          persist();
          updateRangeInfo();
        };
      });
    } else {
      warnEl.innerHTML = "";
    }
  };
  scopeEl.onchange = () => {
    st.sourceScope = scopeEl.value;
    if (st.mode === "voice_swap") {
      st.removeInstrument = (st.sourceScope === "full");
      q("#aceRmInst").checked = st.removeInstrument;
    }
    persist();
  };
  barA.oninput = () => { st.startBar = barA.value; persist(); updateRangeInfo(); };
  barB.oninput = () => { st.endBar = barB.value; persist(); updateRangeInfo(); };
  const secSel = q("#aceSecJump");
  if (secSel) {
    secSel.onchange = () => {
      if (secSel.value === "") { barA.value = ""; barB.value = ""; }
      else {
        const s = p.sections[+secSel.value];
        if (s) { barA.value = String(s.start_bar); barB.value = String(s.end_bar); }
      }
      st.startBar = barA.value; st.endBar = barB.value; persist(); updateRangeInfo();
    };
  }
  B.querySelectorAll(".ace-mask").forEach((c) => {
    c.onchange = () => {
      st.muteStems = [...B.querySelectorAll(".ace-mask:checked")].map((x) => x.value);
      persist();
    };
  });
  updateRangeInfo();

  // Enhancer: Adherence <-> Drift slider + presets
  const inf = q("#aceInf"), infVal = q("#aceInfVal");
  const syncInf = () => {
    const v = +inf.value;
    infVal.textContent = `${Math.round(v * 100)}% drift (${v.toFixed(2)}) · ${DRIFT_LABEL(v)}`;
    B.querySelectorAll(".ace-inf-pre").forEach((b) => b.classList.toggle("on", Math.abs(+b.dataset.v - v) < 0.03));
  };
  inf.value = String(st.influence);
  syncInf();
  inf.oninput = () => { st.influence = +inf.value; persist(); syncInf(); };
  B.querySelectorAll(".ace-inf-pre").forEach((b) => {
    b.onclick = () => { inf.value = b.dataset.v; st.influence = +b.dataset.v; persist(); syncInf(); };
  });

  // Enhancer style prompt + multi-axis builder
  q("#aceStyle").value = p.styleEdit ?? p.style ?? "";
  q("#aceStyle").oninput = (e) => { p.styleEdit = e.target.value; };
  initAceBuilder(B);

  // Enhancer lyrics
  const lyrMode = q("#aceLyrMode"), lyrArea = q("#aceLyrics"), lyrNote = q("#aceLyrNote"), prevBody = q("#aceSendPrevBody");
  lyrMode.value = st.lyricsMode || "mix";
  lyrArea.value = p.lyricsEdit ?? p.lyrics ?? "";
  const linesInRange = () => {
    const a = Math.max(1, +barA.value || 1);
    const b = Math.min(p.total_bars, +barB.value || p.total_bars);
    const barSec = (p.bpb || 4) * (60 / (p.bpm || 120));
    const t0 = (a - 1) * barSec, t1 = b * barSec;
    return (p.lines || []).filter((l) => {
      const mid = 0.5 * ((l.t0 || 0) + (l.t1 || 0));
      return mid >= t0 && mid <= t1 && String(l.text || "").trim();
    });
  };
  const applyLyrMode = () => {
    const lm = lyrMode.value;
    q("#aceLyrBox").style.display = (lm === "mix" || lm === "custom") ? "" : "none";
    const a = Math.max(1, +barA.value || 1);
    const b = Math.min(p.total_bars, +barB.value || p.total_bars);
    const rangeLines = linesInRange();
    if (lyrNote) {
      if (lm === "instrumental") {
        lyrNote.innerHTML = `<span style="color:var(--accent)">Mutes your mix's vocals from the ACE reference bounce, strips vocal tags from the prompt, and locks <code>[Instrumental]</code></span> — ideal for genre-swapping the backing band to drop under your original vocals.`;
      } else if (lm === "auto" || !lyrArea.value.trim()) {
        lyrNote.innerHTML = `<span style="color:var(--warn)">Heads up:</span> without explicit lyrics, ACE Studio auto-transcribes from the full mix and often mishears heavily effected or chopped vocals. Paste accurate lyrics here (or save them once in Song View → Lyrics) for faithful words.`;
      } else {
        lyrNote.textContent = "Sends these exact lyrics to ACE Studio (automatically trimmed to your selected Bars if unedited).";
      }
    }
    if (prevBody) {
      let audioLine = `Audio sent: ${scopeEl.value === "full" ? "Full mix" : scopeEl.value} · bars ${a}–${b}`;
      let lyrText = "";
      if (lm === "instrumental") {
        audioLine = `Audio sent: Instrumental only (vocals auto-muted) · bars ${a}–${b}`;
        lyrText = "[Instrumental]  (vocal descriptors stripped from style prompt)";
      } else if (lm === "auto") {
        lyrText = "(None sent — ACE Studio will auto-transcribe lyrics from the audio)";
      } else {
        const raw = lyrArea.value.trim();
        const unchanged = raw === (p.lyrics || "").trim();
        if (unchanged && rangeLines.length && (a > 1 || b < p.total_bars)) {
          lyrText = rangeLines.map((l) => `[${mmss(l.t0)}] ${l.text}`).join("\n");
        } else if (raw) {
          lyrText = raw;
        } else {
          lyrText = "(Empty — ACE Studio will fall back to auto-transcribing from the audio)";
        }
      }
      prevBody.textContent = `${audioLine}\n---\n${lyrText}`;
    }
  };
  const sliceBtn = q("#aceSliceLyr");
  if (sliceBtn) {
    sliceBtn.onclick = () => {
      const rl = linesInRange();
      if (!rl.length) { toast("No transcribed lines detected inside that bar range", true); return; }
      lyrArea.value = rl.map((l) => l.text).join("\n");
      p.lyricsEdit = lyrArea.value;
      applyLyrMode();
      toast(`Loaded ${rl.length} line${rl.length > 1 ? "s" : ""} for the selected bar range`);
    };
  }
  const rtEng = q("#aceRetransEng"), rtLangWrap = q("#aceRetransLangWrap"), rtLang = q("#aceRetransLang");
  const rtScope = q("#aceRetransScope"), rtBtn = q("#aceRetransBtn"), rtNote = q("#aceRetransNote");
  if (rtEng && rtBtn) {
    rtEng.value = st.sttEngine || p.stt_engine || "whisper_vad";
    if (rtLang) rtLang.value = st.sttLanguage || p.stt_language || "english";
    const syncRtStt = () => {
      const eng = sttEngines().find((x) => x.id === rtEng.value) || sttEngines()[0];
      if (rtLangWrap) rtLangWrap.style.display = eng?.needs_language ? "" : "none";
      if (rtNote) rtNote.textContent = eng?.ok === false && eng?.problem ? `Requires download: ${eng.problem}` : (eng?.note || "");
    };
    rtEng.onchange = () => { st.sttEngine = rtEng.value; persist(); syncRtStt(); };
    if (rtLang) rtLang.onchange = () => { st.sttLanguage = rtLang.value; persist(); };
    if (rtScope) rtScope.onchange = () => { st.sttScopeBars = rtScope.checked; persist(); };
    syncRtStt();
    ensureSttInfo().then(() => {
      const sel = body()?.querySelector("#aceRetransEng");
      if (!sel) return;
      sel.innerHTML = sttEngines().map((e) => `<option value="${esc(e.id)}">${esc(e.label)}${e.ok === false ? " [download needed]" : ""}</option>`).join("");
      sel.value = st.sttEngine || p.stt_engine || "whisper_vad";
      syncRtStt();
    });
    rtBtn.onclick = async () => {
      if (job) { toast("Wait for the current ACE job to finish first", true); return; }
      const engine = rtEng.value;
      const language = rtLang ? rtLang.value : "english";
      const scoped = rtScope ? rtScope.checked : false;
      const payload = {
        prep_id: p.prep_id, engine, language,
        start_bar: scoped && barA.value ? +barA.value : null,
        end_bar: scoped && barB.value ? +barB.value : null,
      };
      rtBtn.disabled = true;
      prog(`<span class="spinner"></span> transcribing vocals (${esc(sttLabel(engine))})…`);
      try {
        job = (await api.post("/api/ace/transcribe", payload)).id;
        jobKind = "stt";
      } catch (e) {
        rtBtn.disabled = false;
        prog(esc(e.message));
        toast(e.message, true);
      }
    };
  }
  lyrArea.oninput = (e) => { p.lyricsEdit = e.target.value; applyLyrMode(); };
  lyrMode.onchange = () => { st.lyricsMode = lyrMode.value; persist(); applyLyrMode(); };
  // Also refresh preview whenever bar range or scope changes
  const origUpdateRange = updateRangeInfo;
  const syncAllRange = () => { origUpdateRange(); applyLyrMode(); };
  barA.oninput = () => { st.startBar = barA.value; persist(); syncAllRange(); };
  barB.oninput = () => { st.endBar = barB.value; persist(); syncAllRange(); };
  scopeEl.addEventListener("change", applyLyrMode);
  if (secSel) {
    const origSec = secSel.onchange;
    secSel.onchange = () => { origSec(); applyLyrMode(); };
  }
  applyLyrMode();

  // Layer mode controls
  const stEl = q("#aceSoundType"), custWrap = q("#aceCustInstWrap"), custIn = q("#aceCustInst");
  const lpEl = q("#aceLayerPrompt"), llWrap = q("#aceLayerLyrWrap"), llEl = q("#aceLayerLyrics");
  stEl.value = st.soundType || "drums";
  custIn.value = st.customInstrument || "";
  lpEl.value = st.layerPrompt || "";
  llEl.value = p.layerLyricsEdit ?? p.lyrics ?? "";
  const applyLayerType = () => {
    custWrap.style.display = stEl.value === "custom" ? "" : "none";
    llWrap.style.display = ["vocals", "backing-vocals", "song-track"].includes(stEl.value) ? "" : "none";
  };
  stEl.onchange = () => { st.soundType = stEl.value; persist(); applyLayerType(); };
  custIn.oninput = () => { st.customInstrument = custIn.value; persist(); };
  lpEl.oninput = () => { st.layerPrompt = lpEl.value; persist(); };
  llEl.oninput = () => { p.layerLyricsEdit = llEl.value; };
  B.querySelectorAll(".al-chip").forEach((c) => {
    c.onclick = () => {
      const cur = lpEl.value.split(",").map((x) => x.trim()).filter(Boolean);
      const i = cur.findIndex((x) => x.toLowerCase() === c.dataset.v.toLowerCase());
      if (i >= 0) cur.splice(i, 1); else cur.push(c.dataset.v);
      lpEl.value = cur.join(", ");
      st.layerPrompt = lpEl.value; persist();
      c.classList.toggle("on", i < 0);
    };
  });
  applyLayerType();

  // Voice Swap controls (FREE)
  const semi = q("#aceSemi"), pitch = q("#acePitch"), drift = q("#aceDrift");
  const keySel = q("#aceKey"), scaleSel = q("#aceScale"), rmInst = q("#aceRmInst"), rmRev = q("#aceRmRev");
  semi.value = String(st.semitones || 0);
  pitch.value = String(st.pitchCorrection ?? 20);
  drift.value = String(st.randomOffset || 0);
  keySel.value = st.correctKey || "";
  scaleSel.value = st.correctScale || "Major";
  rmInst.checked = !!st.removeInstrument;
  rmRev.checked = !!st.removeReverb;
  const syncSliders = () => {
    const sv = +semi.value;
    q("#aceSemiV").textContent = `${sv > 0 ? "+" : ""}${sv} st`;
    q("#acePitchV").textContent = `${pitch.value}%`;
    q("#aceDriftV").textContent = `${drift.value}%`;
  };
  syncSliders();
  semi.oninput = () => { st.semitones = +semi.value; persist(); syncSliders(); };
  pitch.oninput = () => { st.pitchCorrection = +pitch.value; persist(); syncSliders(); };
  drift.oninput = () => { st.randomOffset = +drift.value; persist(); syncSliders(); };
  keySel.onchange = () => { st.correctKey = keySel.value; persist(); };
  scaleSel.onchange = () => { st.correctScale = scaleSel.value; persist(); };
  rmInst.onchange = () => { st.removeInstrument = rmInst.checked; persist(); };
  rmRev.onchange = () => { st.removeReverb = rmRev.checked; persist(); };

  const vcFilter = q("#aceVcFilter"), vcSearch = q("#aceVcSearch");
  vcFilter.value = st.vcKindFilter || "all";
  vcSearch.value = st.vcSearch || "";
  const drawVcGrid = () => renderVcGrid(B);
  vcFilter.onchange = () => { st.vcKindFilter = vcFilter.value; persist(); drawVcGrid(); };
  vcSearch.oninput = () => { st.vcSearch = vcSearch.value; persist(); drawVcGrid(); };
  ensureModels().then(() => { if (body()) drawVcGrid(); });

  // Mode switching
  const applyMode = () => {
    const md = st.mode;
    q("#acePaneEnhancer").style.display = md === "enhancer" ? "" : "none";
    q("#acePaneLayer").style.display = md === "layer" ? "" : "none";
    q("#acePaneVoice").style.display = md === "voice_swap" ? "" : "none";
    q("#aceMaskRow").style.display = md === "layer" ? "" : "none";
    if (md === "voice_swap") {
      q("#aceRun").textContent = "2 · Convert in ACE Studio (Free · 0 credits)";
      q("#aceCostNote").innerHTML = `<span class="tag ace-free">0 credits</span> renders once across all selected models`;
    } else if (md === "layer") {
      q("#aceRun").textContent = "2 · Generate Layer in ACE Studio";
      q("#aceCostNote").textContent = "Spends ACE Studio credits (add-a-layer) · archived automatically to your personal library";
    } else {
      q("#aceRun").textContent = "2 · Reimagine in ACE Studio";
      q("#aceCostNote").textContent = "Spends ACE Studio credits (music-enhancer, 3s–240s max, non-cancellable once started) · archived automatically";
    }
    updateRangeInfo();
  };
  B.querySelectorAll('input[name="aceMode"]').forEach((r) => {
    r.onchange = () => {
      st.mode = r.value;
      // Smart source default when switching to voice_swap if user has vocals
      if (st.mode === "voice_swap" && (p.stems_present || []).includes("vocals") && st.sourceScope === "full") {
        st.sourceScope = "vocals";
        scopeEl.value = "vocals";
        st.removeInstrument = false;
        rmInst.checked = false;
      } else if (st.mode === "enhancer" && st.sourceScope === "vocals") {
        st.sourceScope = "full";
        scopeEl.value = "full";
      }
      persist();
      B.querySelectorAll(".ri-mode").forEach((c) => c.classList.toggle("on", c.querySelector("input").checked));
      applyMode();
    };
  });
  applyMode();

  q("#aceRun").onclick = startPerform;
  q("#aceRestart").onclick = async () => {
    if (job) {
      if (!confirm("An ACE Studio run is still in progress. Cancel and start over?")) return;
      try { await api.post(`/api/jobs/${job}/cancel`, {}); } catch {}
    }
    prep = null; takes = null; renderAce();
  };
}

function initAceBuilder(B) {
  const q = (s) => B.querySelector(s);
  const list = (k) => q(`#ab_${k}`).value.split(",").map((x) => x.trim()).filter(Boolean);
  const set = (k, arr) => { q(`#ab_${k}`).value = arr.join(", "); };
  const mark = () => B.querySelectorAll(".ab-chip").forEach((c) => {
    c.classList.toggle("on", list(c.dataset.k).map((x) => x.toLowerCase()).includes(c.dataset.v.toLowerCase()));
  });
  const compose = () => {
    const parts = [];
    for (const [k] of ACE_BUILD) parts.push(...list(k));
    return parts.join(", ");
  };
  const preview = () => { mark(); q("#abPreview").textContent = compose() ? `Composed prompt: ${compose()}` : ""; };

  if (!prep.aceBuilder) {
    const b = {};
    const found = ((prep.heard || []).map((x) => `${x.preset_style}, ${(x.measured || []).join(", ")}, ${x.genre}`).join(", ")).toLowerCase();
    b.genre = (prep.prompt || "").split(",").map((x) => x.trim()).filter(Boolean);
    for (const k of ["mood", "rhythm"]) b[k] = ACE_CHIPS[k].filter((c) => found.includes(c.toLowerCase()));
    prep.aceBuilder = b;
  }
  for (const [k] of ACE_BUILD) set(k, prep.aceBuilder[k] || []);

  const keep = () => { for (const [k] of ACE_BUILD) prep.aceBuilder[k] = list(k); preview(); };
  B.querySelectorAll(".ri-brow input").forEach((e) => { e.oninput = keep; e.onchange = keep; });
  B.querySelectorAll(".ab-chip").forEach((c) => {
    c.onclick = () => {
      const k = c.dataset.k, cur = list(k), i = cur.findIndex((x) => x.toLowerCase() === c.dataset.v.toLowerCase());
      if (i >= 0) cur.splice(i, 1); else cur.push(c.dataset.v);
      set(k, cur); keep();
    };
  });
  q("#abUse").onclick = () => {
    const s = compose();
    if (!s) { toast("Pick or type at least one descriptor first", true); return; }
    q("#aceStyle").value = s; prep.styleEdit = s;
    toast("Style prompt updated from builder");
  };
  q("#abClear").onclick = () => {
    for (const [k] of ACE_BUILD) set(k, []);
    keep();
  };
  q("#abFound").onclick = () => { q("#aceStyle").value = prep.style || ""; prep.styleEdit = prep.style || ""; };
  preview();
}

function renderVcGrid(B) {
  const grid = B.querySelector("#aceVcGrid"), cnt = B.querySelector("#aceVcCount");
  if (!grid) return;
  const all = modelsCache?.models || [];
  if (!all.length) { grid.innerHTML = `<div class="dim">No Voice Changer models available. Check that ACE Studio is running.</div>`; return; }

  const kindF = st.vcKindFilter || "all";
  const qText = (st.vcSearch || "").trim().toLowerCase();
  const filtered = all.filter((m) => {
    if (kindF !== "all" && m.kind !== kindF) return false;
    if (!qText) return true;
    const hay = `${m.name} ${m.kind} ${(m.tags || []).join(" ")}`.toLowerCase();
    return hay.includes(qText);
  });

  const selSet = new Set((st.modelIds || []).map(Number));
  if (cnt) cnt.textContent = `· ${selSet.size} selected (max 6)`;

  grid.innerHTML = filtered.map((m) => {
    const on = selSet.has(Number(m.id));
    const tags = (m.tags || []).slice(0, 4).join(", ");
    return `<button type="button" class="ace-vc-card${on ? " on" : ""}" data-mid="${m.id}">
      <div class="ace-vc-top"><b>${esc(m.name)}</b> <span class="tag">${esc(m.kind)}</span></div>
      <div class="dim">${esc(tags || "preset model")}</div>
    </button>`;
  }).join("") || `<div class="dim">No models match "${esc(qText)}".</div>`;

  grid.querySelectorAll(".ace-vc-card").forEach((btn) => {
    btn.onclick = () => {
      const id = +btn.dataset.mid;
      const cur = new Set((st.modelIds || []).map(Number));
      if (cur.has(id)) cur.delete(id);
      else {
        if (cur.size >= 6) { toast("Select up to 6 models per batch", true); return; }
        cur.add(id);
      }
      st.modelIds = [...cur];
      persist();
      renderVcGrid(B);
    };
  });
}

async function startPerform() {
  const B = body();
  if (!B || !prep) return;
  if (job && !confirm("An ACE Studio run is already in progress. Start another one?")) return;

  const payload = {
    prep_id: prep.prep_id,
    mode: st.mode,
    source_scope: st.sourceScope || "full",
    mute_stems: st.mode === "layer" ? (st.muteStems || []) : [],
    start_bar: st.startBar ? +st.startBar : null,
    end_bar: st.endBar ? +st.endBar : null,
  };

  if (st.mode === "enhancer") {
    const a = Math.max(1, +st.startBar || 1);
    const b = Math.min(prep.total_bars, +st.endBar || prep.total_bars);
    const estSec = Math.max(1, b - a + 1) * (prep.bpb || 4) * (60 / (prep.bpm || 120));
    if (estSec > 240.0 || estSec < 3.0) {
      toast(`Selected range (~${mmss(estSec)}) is outside ACE Music Enhancer's 0:03–4:00 limit — click a ~3:55 fit button or adjust Bars first`, true, 7000);
      return;
    }
    const style = B.querySelector("#aceStyle").value.trim();
    const lyrics = B.querySelector("#aceLyrics").value;
    prep.styleEdit = style; prep.lyricsEdit = lyrics;
    if (!style && st.lyricsMode !== "auto" && st.influence > 0.25) {
      if (!confirm("The style prompt is empty — ACE Studio will rely on its own analysis of the mix. Continue?")) return;
    }
    payload.style = style;
    payload.lyrics = lyrics;
    payload.lyrics_mode = st.lyricsMode || "mix";
    payload.influence = +st.influence;
  } else if (st.mode === "layer") {
    const lp = B.querySelector("#aceLayerPrompt").value.trim();
    const ci = B.querySelector("#aceCustInst").value.trim();
    if (st.soundType === "custom" && !ci) { toast("Enter a custom instrument name", true); return; }
    payload.sound_type = st.soundType || "drums";
    payload.instrument = ci;
    payload.style = lp;
    payload.lyrics = B.querySelector("#aceLayerLyrics").value;
  } else if (st.mode === "voice_swap") {
    if (!(st.modelIds || []).length) { toast("Pick at least one Voice Changer model first", true); return; }
    payload.model_ids = st.modelIds;
    payload.semitones = +st.semitones || 0;
    payload.pitch_correction = +st.pitchCorrection;
    payload.random_offset = +st.randomOffset || 0;
    payload.correct_to_key = st.correctKey || "";
    payload.correct_to_scale = st.correctScale || "Major";
    payload.remove_instrument = !!st.removeInstrument;
    payload.remove_reverb = !!st.removeReverb;
  }

  prog('<span class="spinner"></span> sending to ACE Studio…');
  try {
    job = (await api.post("/api/ace/perform", payload)).id;
    jobKind = "perform";
  } catch (e) { prog(esc(e.message)); toast(e.message, true); }
}

// ------------------------------------------------------------------ Stage 3: Audition & Save Takes

function showTakes() {
  const B = body();
  if (!B || !takes) return;
  const r = takes;
  const isSingle = r.mode === "layer" || r.mode === "voice_swap";
  const driftTxt = r.influence != null ? ` · drift ${Math.round(r.influence * 100)}%` : "";
  const rangeTxt = r.start_bar > 1 || (r.end_bar && r.end_bar > 0) ? ` · bars ${r.start_bar || 1}–${r.end_bar || "end"}` : "";

  B.innerHTML = `
    <div class="dj-head">
      <b>${esc(r.title)}</b> · ${esc(r.mode_label || r.mode)}${driftTxt}${rangeTxt} · ${r.bpm} BPM
      ${r.credit_billed ? "" : ' <span class="tag ace-free">0 credits</span>'}
      ${r.style ? `<div class="dim" style="margin-top:3px">Prompt: ${esc(r.style)}</div>` : ""}
    </div>

    <div class="ri-takes">${r.takes.map((t) => {
      const sv = (r.saved || {})[String(t.n)];
      return `<label class="ri-take">
        <input type="checkbox" value="${t.n}"${t.top && !sv ? " checked" : ""}${sv ? " disabled" : ""}>
        <span><b>${esc(t.label || `Take ${t.n}`)}</b>${sv ? ' <span class="tag">saved</span>' : ""}
          <span class="dim">· ${mmss(t.seconds)}${t.target_stem ? ` · ${esc(t.target_stem)} stem` : ""}</span>
        </span>
        <audio controls preload="none" src="/api/ace/audio/${encodeURIComponent(r.render_id)}/${encodeURIComponent(t.file)}"></audio>
      </label>`;
    }).join("")}</div>

    <div class="ri-row">
      <label><input type="checkbox" id="aceAdd"${st.addToMix ? " checked" : ""}> drop into the open mix, aligned to the grid:</label>
      ${isSingle
        ? `<span class="dim">adds as an isolated stem lane at bar ${r.start_bar || 1} (skips unnecessary 4-stem separation)</span>`
        : `<select id="aceDropMode">
            <option value="instrumental">new instrumental only (drums + bass + other) under your mix's vocals</option>
            <option value="vocals">new vocals only, playing over your mix's music (mix vocals muted)</option>
            <option value="all">all four stems, on muted tracks (A/B &amp; cherry-pick)</option>
          </select>`}
    </div>
    ${isSingle ? "" : `
      <div class="ri-row">
        <label><input type="checkbox" id="aceRetime"${st.retime ? " checked" : ""}> move new vocals line by line to where lyrics are sung in the mix</label>
      </div>`}

    <div class="row-gap">
      <button id="aceSave" class="primary">${isSingle ? "Save selected · add stem to mix" : "Save selected · split stems · line up"}</button>
      ${r.prep_id ? `<button id="aceBack">← Edit settings &amp; try another mode</button>` : ""}
      <button id="aceHome" class="small">All ACE runs</button>
    </div>`;

  B.querySelector("#aceAdd").onchange = (e) => { st.addToMix = e.target.checked; persist(); };
  const dmEl = B.querySelector("#aceDropMode");
  if (dmEl) {
    dmEl.value = st.dropMode || "all";
    dmEl.onchange = () => { st.dropMode = dmEl.value; persist(); };
  }
  const rtEl = B.querySelector("#aceRetime");
  if (rtEl) { rtEl.onchange = () => { st.retime = rtEl.checked; persist(); }; }

  B.querySelector("#aceSave").onclick = async () => {
    const chosen = [...B.querySelectorAll(".ri-take input:checked:not(:disabled)")].map((c) => +c.value);
    if (!chosen.length) { toast("Tick at least one take to keep", true); return; }
    const addTo = st.addToMix && S.project ? S.project.id : null;
    const dropMode = isSingle ? "single" : (dmEl?.value || "all");
    try {
      const res = await api.post("/api/ace/save", {
        render_id: r.render_id, takes: chosen, add_to: addTo,
        drop_mode: dropMode, retime: !!rtEl?.checked,
      });
      for (const x of res) if (addTo) host?.trackPending(x.id, addTo);
      toast(`Saving ${res.length} ACE take${res.length > 1 ? "s" : ""}${isSingle ? " as isolated stem(s)" : ": splitting stems and aligning bar-for-bar"}.`, false, 8000);
      host?.close();
    } catch (e) { toast(e.message, true); }
  };

  const backBtn = B.querySelector("#aceBack");
  if (backBtn) {
    backBtn.onclick = async () => {
      if (!prep || prep.prep_id !== r.prep_id) {
        try { prep = await api.get(`/api/ace/prep/${encodeURIComponent(r.prep_id)}`); }
        catch (e) { toast("That preparation is gone — prepare the mix again", true); return; }
      }
      takes = null; renderAce();
    };
  }
  B.querySelector("#aceHome").onclick = () => { takes = null; prep = null; renderAce(); };
}

// ------------------------------------------------------------------ Job Event Listener

on("event:job", (msg) => {
  if (!job || msg.job.id !== job) return;
  if (msg.job.status === "running") {
    prog(`<span class="spinner"></span> ${esc(msg.job.message || "")} <span class="pct">${Math.round(100 * (msg.job.progress || 0))}%</span>`);
  }
  if (msg.job.status === "error" || msg.job.status === "cancelled") {
    const txt = msg.job.status === "cancelled" || /cancelled/.test(msg.job.error || "") ? "cancelled" : msg.job.error;
    prog(esc(txt));
    if (!host || !host.isVisible() || jobKind === "stt") toast(txt, txt !== "cancelled");
    const wasStt = jobKind === "stt";
    job = null;
    if (host?.isVisible()) {
      if (wasStt && prep) {
        const b = body()?.querySelector("#aceRetransBtn");
        if (b) b.disabled = false;
      } else if (!prep && !takes) {
        showStart();
      }
    }
  }
  if (msg.job.status === "done" && msg.result) {
    if (jobKind === "prep") {
      prep = msg.result; takes = null;
    } else if (jobKind === "stt" && prep) {
      const r = msg.result;
      prep.lines = r.lines || [];
      prep.lyrics = r.lyrics || "";
      prep.lyricsEdit = r.lyrics || "";
      prep.lyrics_from = r.lyrics_from || r.engine;
      prep.stt_engine = r.engine;
      prep.stt_language = r.language;
      const scopeTxt = r.scoped_bars ? ` · bars ${r.scoped_bars[0]}–${r.scoped_bars[1]}` : "";
      toast(`${r.engine_label}: ${r.line_count} line${r.line_count === 1 ? "" : "s"} (${r.elapsed_s}s${scopeTxt})`, r.line_count === 0, 6000);
    } else {
      takes = msg.result;
    }
    const doneKind = jobKind;
    job = null; prog("");
    if (host?.isVisible()) renderAce();
    else if (doneKind !== "stt") toast(doneKind === "prep" ? "ACE Reimagine prepared — open Reimagine to configure" : "Your ACE Studio takes are ready — open Reimagine to audition", false, 8000);
  }
});
