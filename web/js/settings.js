// Settings: the audio output device (switches instantly, playback carries on), and where the library (songs,
// stems, projects, uploads, YuE2 takes) and the exports live - a new empty folder becomes a new library; changing
// those restarts Re-Tex.
import { api } from "./api.js";
import { toast } from "./state.js";
import { modal, showMenu } from "./ui.js";

// ---------------------------------------------------------------- audio output
async function switchOutput(name) {
  const r = await api.post("/api/audio/output", { name: name || null });
  if (r.error) toast(r.error, true);
  else toast(`Playing through ${r.active || "the system default"}`);
  return r;
}

/** Quick switcher: click the speaker icon next to the volume slider. */
export async function outputMenu(e) {
  let r;
  try { r = await api.get("/api/audio/outputs"); } catch (err) { toast(err.message, true); return; }
  const mark = (on, label) => `${on ? "✓ " : "    "}${label}`;
  showMenu(e, [
    ["Play through:", null],
    [mark(!r.current, `System default${!r.current && r.active ? " (" + r.active + ")" : ""}`), () => switchOutput(null)],
    ...r.devices.map((n) => [mark(r.current === n, n), () => switchOutput(n)]),
    "-",
    ["Look for new devices", async () => { try { await api.get("/api/audio/outputs?refresh=1"); outputMenu(e); } catch (err) { toast(err.message, true); } }],
    ["Play a test sound", () => api.post("/api/audio/test").catch((err) => toast(err.message, true))],
  ]);
}

let m = null;
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

function describe(info, isLibrary) {
  if (!info) return "";
  if (!info.ok) return `<span class="set-bad">✕ ${esc(info.error)}</span>`;
  if (!info.exists) return isLibrary ? "new folder - becomes a new, empty library" : "new folder - will be created";
  if (!isLibrary) return "✓ folder exists";
  return info.songs || info.projects
    ? `✓ library: ${info.songs} song${info.songs === 1 ? "" : "s"}, ${info.projects} project${info.projects === 1 ? "" : "s"}`
    : "empty folder - becomes a new library";
}

export async function openSettings() {
  if (m) return;
  let cur;
  try { cur = await api.get("/api/settings"); } catch (e) { toast("Settings: " + e.message, true); return; }
  const usingDefaultExports = cur.exports.replace(/[\\/]+$/, "") === (cur.data.replace(/[\\/]+$/, "") + "\\exports");
  m = modal("Settings", `
    <div class="set-sec">Audio output</div>
    <div class="set-row">
      <div class="set-field"><select id="setOut"><option>loading…</option></select>
        <button id="setOutTest" class="small" title="A short, quiet chime through this output">Test</button>
        <button id="setOutScan" class="small" title="Look for devices connected or switched since Re-Tex started">Refresh</button></div>
      <div id="setOutInfo" class="set-info">Switches right away - playback carries on. Also on the speaker icon next to the volume.</div>
    </div>
    <div class="set-sec">Library</div>
    <p class="lead">Where your songs live. The library folder holds songs, stems, projects, uploads and YuE2 takes -
      it keeps growing as you add and delete songs. Point it at another folder for a separate library (a new empty
      folder starts one). Paste a path or browse.</p>
    <div class="set-row">
      <label for="setData"><b>Library folder</b></label>
      <div class="set-field"><input id="setData" spellcheck="false" value="${esc(cur.saved.data || cur.data)}">
        <button id="setDataBrowse" class="small">Browse…</button></div>
      <div id="setDataInfo" class="set-info">${describe(cur.library, true)}</div>
    </div>
    <div class="set-row">
      <label for="setExports"><b>Exports folder</b> <span class="set-dim">(finished mixes; empty = "exports" inside the library)</span></label>
      <div class="set-field"><input id="setExports" spellcheck="false" placeholder="inside the library"
        value="${esc(cur.saved.exports || (usingDefaultExports ? "" : cur.exports))}">
        <button id="setExpBrowse" class="small">Browse…</button></div>
      <div id="setExpInfo" class="set-info"></div>
    </div>
    ${cur.pinned.data || cur.pinned.exports ? `<p class="set-bad">This copy was started with MASHLAB_DATA / MASHLAB_EXPORTS set, which win over these settings.</p>` : ""}
    <div class="row-gap"><button id="setSave" class="primary">Save</button><span id="setMsg"></span></div>
    <details class="set-about"><summary>About this copy</summary>
      <div>${cur.release ? `Frozen release <b>${esc(cur.release)}</b>` : "Development copy"}<br>
      In use now: library <code>${esc(cur.data)}</code>, exports <code>${esc(cur.exports)}</code><br>
      Program: <code>${esc(cur.code)}</code><br>Models: <code>${esc(cur.models)}</code><br>
      Settings file: <code>${esc(cur.settings_file)}</code></div></details>`, { onClose: () => (m = null) });
  const B = m.body;
  const sel = B.querySelector("#setOut"), outInfo = B.querySelector("#setOutInfo");
  const fillOutputs = (r) => {
    sel.innerHTML = `<option value="">System default${!r.current && r.active ? " (" + esc(r.active) + ")" : ""}</option>` +
      r.devices.map((n) => `<option value="${esc(n)}"${r.current === n ? " selected" : ""}>${esc(n)}</option>`).join("");
    if (r.current && !r.devices.includes(r.current)) sel.insertAdjacentHTML("beforeend", `<option value="${esc(r.current)}" selected>${esc(r.current)} (not connected)</option>`);
    outInfo.innerHTML = r.error ? `<span class="set-bad">${esc(r.error)}</span>` : `Playing through <b>${esc(r.active || "the system default")}</b>. Switches right away - playback carries on.`;
  };
  const loadOutputs = (refresh) => api.get(`/api/audio/outputs${refresh ? "?refresh=1" : ""}`).then(fillOutputs)
    .catch((e) => { outInfo.innerHTML = `<span class="set-bad">✕ ${esc(e.message)}</span>`; });
  loadOutputs(false);
  sel.onchange = async () => { try { fillOutputs(await switchOutput(sel.value)); } catch (e) { toast(e.message, true); } };
  B.querySelector("#setOutTest").onclick = () => api.post("/api/audio/test").catch((e) => toast(e.message, true));
  B.querySelector("#setOutScan").onclick = () => loadOutputs(true);
  const data = B.querySelector("#setData"), exp = B.querySelector("#setExports");
  const check = async (input, out, isLibrary) => {
    const v = input.value.trim();
    if (!v) { out.textContent = isLibrary ? "" : `"exports" inside the library`; return; }
    try { out.innerHTML = describe(await api.post("/api/settings/check", { path: v }), isLibrary); }
    catch (e) { out.innerHTML = `<span class="set-bad">✕ ${esc(e.message)}</span>`; }
  };
  let t1 = 0, t2 = 0;
  data.oninput = () => { clearTimeout(t1); t1 = setTimeout(() => check(data, B.querySelector("#setDataInfo"), true), 300); };
  exp.oninput = () => { clearTimeout(t2); t2 = setTimeout(() => check(exp, B.querySelector("#setExpInfo"), false), 300); };
  check(exp, B.querySelector("#setExpInfo"), false);
  const browse = async (input, title) => {
    try {
      const r = await api.post("/api/settings/browse", { path: input.value, title });
      if (r.path) { input.value = r.path; input.oninput(); }
    } catch (e) { toast(e.message, true); }
  };
  B.querySelector("#setDataBrowse").onclick = () => browse(data, "Library folder");
  B.querySelector("#setExpBrowse").onclick = () => browse(exp, "Exports folder");
  B.querySelector("#setSave").onclick = async () => {
    const msg = B.querySelector("#setMsg");
    try {
      const r = await api.post("/api/settings", { data: data.value, exports: exp.value });
      if (!r.restart) { msg.textContent = "✓ saved (already in use)"; return; }
      msg.innerHTML = `✓ saved. Re-Tex restarts to use it (anything running stops). <button id="setRestart" class="small primary">Restart now</button>`;
      msg.querySelector("#setRestart").onclick = () => restart(msg);
    } catch (e) { msg.innerHTML = `<span class="set-bad">✕ ${esc(e.message)}</span>`; }
  };
}

async function restart(msg) {
  let before = null;
  try { before = (await api.get("/api/whoami")).pid; } catch {}
  try { await api.post("/api/restart"); } catch (e) { msg.innerHTML = `<span class="set-bad">✕ ${esc(e.message)}</span>`; return; }
  msg.innerHTML = '<span class="spinner"></span> restarting…';
  for (let i = 0; i < 120; i++) {           // up to ~60 s
    await new Promise((r) => setTimeout(r, 500));
    try { if ((await api.get("/api/whoami")).pid !== before) { location.reload(); return; } } catch {}
  }
  msg.innerHTML = '<span class="set-bad">✕ it didn\'t come back - close Re-Tex and start it again</span>';
}
