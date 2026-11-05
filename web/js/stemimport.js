// Import stems: already-split audio (a session's exports, a stem pack, an acapella, a drum loop) as songs that are
// never separated again. A song has four stems (drums, bass, other, vocals); each file goes on one - guessed from its
// name, yours to change - and files on the same stem are mixed together (kick + snare + hats = drums; guitars +
// piano/strings = other); a stem with no file is silent. Files from several songs (a folder holding two remix-contest
// stem sets) are split into one song each by their names (stemnames.js). POST /api/stems/upload (files you pick or
// drop) or /api/stems/import (a folder on this PC, read in place).
import { api } from "./api.js";
import { toast } from "./state.js";
import { modal } from "./ui.js";
import { ROLES, groupSongs, startRoles } from "./stemnames.js";

export { looksLikeStems } from "./stemnames.js";

const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

/** opts: {files: File[]} (picked or dropped) or {folder: {folder, name, files: [{path, name, size}]}}; fromDrop
 * adds a way out to the old behaviour (each file separated as its own song). */
export function openStemImport({ files = null, folder = null, fromDrop = false, onSeparate = null } = {}) {
  const items = files ? files.map((f) => ({ name: f.name, size: f.size, file: f }))
    : folder.files.map((f) => ({ name: f.name, size: f.size, path: f.path }));
  const names = items.map((i) => i.name);
  const groups = groupSongs(names, folder?.name || "");
  startRoles(names, groups).forEach((r, i) => (items[i].role = r));
  const m = modal("Import stems", `
    <p class="lead">Each song below is imported from its files as they are - nothing is separated again. A song has four
      stems; give each file one: files on the same stem are mixed together (guitars + piano = other, kick + snare =
      drums), a stem with no file stays silent, so an acapella or a drum loop works too. A song's files should all
      start at the same point, as exported from one session.</p>
    <div id="siSongs"></div>
    <div class="row-gap"><button class="primary" id="siGo"></button>
      ${fromDrop ? '<button id="siSep">No - separate each file as its own song</button>' : ""}</div>`, { wide: true });
  const B = m.body, box = B.querySelector("#siSongs");
  const go = B.querySelector("#siGo");
  const live = () => groups.filter((g) => g.items.some((i) => items[i].role));
  const refresh = () => {
    for (const g of groups) {
      const c = Object.fromEntries(ROLES.map((r) => [r, g.items.filter((i) => items[i].role === r).length]));
      const out = g.items.filter((i) => !items[i].role).length;
      g.sum.textContent = Object.values(c).some(Boolean)
        ? ROLES.map((r) => `${r}: ${c[r] ? `${c[r]} file${c[r] > 1 ? "s, mixed together" : ""}` : "silent"}`).join(" · ") + (out ? ` · ${out} left out` : "")
        : "left out - nothing in its file names says stem (a whole song? add it normally to separate it)";
      g.sec.classList.toggle("si-off", !Object.values(c).some(Boolean));
    }
    const n = live().length;
    go.textContent = n > 1 ? `Import as ${n} songs` : "Import as one song";
    go.disabled = !n;
  };
  for (const g of groups) {
    const sec = document.createElement("div");
    sec.className = "si-song";
    sec.innerHTML = `<div class="field"><label>Song title</label><input type="text" spellcheck="false" style="width:100%"></div>
      <table class="dj-plan si-tab"><thead><tr><th>file</th><th>stem</th><th>size</th></tr></thead><tbody></tbody></table>
      <p class="dim si-sum"></p>`;
    const title = sec.querySelector("input");
    title.value = g.title;
    title.oninput = () => (g.title = title.value);
    const tb = sec.querySelector("tbody");
    for (const i of g.items) {
      const it = items[i], tr = document.createElement("tr");
      tr.innerHTML = `<td>${esc(it.name)}</td><td><select>${ROLES.map((r) => `<option value="${r}">${r}</option>`).join("")}<option value="">leave out</option></select></td>
        <td class="dim">${(it.size / 1048576).toFixed(1)} MB</td>`;
      const sel = tr.querySelector("select");
      sel.value = it.role;
      sel.onchange = () => { it.role = sel.value; refresh(); };
      tb.appendChild(tr);
    }
    g.sec = sec;
    g.sum = sec.querySelector(".si-sum");
    box.appendChild(sec);
  }
  refresh();
  go.onclick = async () => {
    const todo = live();
    if (!todo.length) { toast("Give at least one file a stem", true); return; }
    m.close();
    for (const g of todo) {
      const use = g.items.map((i) => items[i]).filter((i) => i.role);
      const t = (g.title || "").trim() || "Stems";
      try {
        if (files) {
          const fd = new FormData();
          use.forEach((i) => fd.append("files", i.file));
          fd.append("roles", JSON.stringify(use.map((i) => i.role)));
          fd.append("title", t);
          toast(`Uploading "${t}" (${use.length} stem file${use.length > 1 ? "s" : ""})…`);
          await api.post("/api/stems/upload", fd);
        } else {
          await api.post("/api/stems/import", { title: t, files: use.map((i) => ({ path: i.path, role: i.role })) });
          toast(`Importing "${t}" from its stems…`);
        }
      } catch (e) { toast(`Stem import of "${t}" failed: ${e.message}`, true); }
    }
  };
  if (fromDrop) B.querySelector("#siSep").onclick = () => { m.close(); onSeparate?.(); };
}

/** Pick stem files, or a folder of them on this PC, then assign. */
export function stemPicker() {
  const m = modal("Import stems", `
    <p class="lead">Already have the stems? Bring them in without separating them again - one song's, or a folder
      holding several songs' stems (each becomes its own song).</p>
    <div class="row-gap" style="margin-bottom:14px"><button class="primary" id="spFiles">Choose stem files…</button>
      <span class="dim">or drop them anywhere in the window</span></div>
    <div class="field"><label>…or paste the path of a folder of stems on this PC (read in place, not copied)</label>
      <div class="row-gap"><input type="text" id="spPath" placeholder="C:\\Sessions\\MySong\\Stems" spellcheck="false" style="flex:1">
      <button id="spGo">Open</button></div></div>`);
  const inp = document.createElement("input");
  inp.type = "file"; inp.multiple = true; inp.accept = "audio/*,.wav,.flac,.mp3,.aif,.aiff,.ogg,.m4a";
  inp.onchange = () => { const fs = [...inp.files]; if (fs.length) { m.close(); openStemImport({ files: fs }); } };
  m.body.querySelector("#spFiles").onclick = () => inp.click();
  m.body.querySelector("#spGo").onclick = async () => {
    const path = m.body.querySelector("#spPath").value.trim();
    if (!path) return;
    try {
      const folder = await api.get(`/api/stems/folder?path=${encodeURIComponent(path)}`);
      m.close();
      openStemImport({ folder });
    } catch (e) { toast(e.message, true); }
  };
}
