// User-named folders for grouping library songs (shared by the library panel and the Auto-Mashup pickers).
import { api } from "./api.js";
import { S, emit, uid, toast } from "./state.js";

S.folders = { folders: [], assign: {} };

export async function loadFolders() {
  try { S.folders = await api.get("/api/folders"); } catch { S.folders = { folders: [], assign: {} }; }
  emit("folders");
}
async function save() {
  emit("folders");
  try { await api.put("/api/folders", S.folders); } catch (e) { toast("Couldn't save folders: " + e.message, true); }
}

export function folderOf(sid) { return S.folders.assign[sid] || null; }

/** [{folder: {id,name,collapsed} | null (= unsorted), songs: [...]}] in folder order, unsorted last. */
export function groups(songs) {
  const by = new Map(S.folders.folders.map((f) => [f.id, []]));
  const loose = [];
  for (const s of songs) {
    const f = S.folders.assign[s.id];
    if (f && by.has(f)) by.get(f).push(s); else loose.push(s);
  }
  const out = S.folders.folders.map((f) => ({ folder: f, songs: by.get(f.id) }));
  out.push({ folder: null, songs: loose });
  return out;
}

export async function createFolder(name, songIds = []) {
  name = (name ?? prompt("Folder name", "New folder"))?.trim();
  if (!name) return null;
  const f = { id: uid("f"), name, collapsed: false };
  S.folders.folders.push(f);
  for (const id of songIds) S.folders.assign[id] = f.id;
  await save();
  return f;
}
export async function renameFolder(f) {
  const name = prompt("Folder name", f.name)?.trim();
  if (!name || name === f.name) return;
  f.name = name;
  await save();
}
export async function deleteFolder(f) {
  if (!confirm(`Delete folder "${f.name}"? (the songs stay in your library, just unsorted)`)) return;
  S.folders.folders = S.folders.folders.filter((x) => x.id !== f.id);
  for (const [k, v] of Object.entries(S.folders.assign)) if (v === f.id) delete S.folders.assign[k];
  await save();
}
export async function moveToFolder(songIds, fid) {
  for (const id of songIds) {
    if (fid) S.folders.assign[id] = fid; else delete S.folders.assign[id];
  }
  await save();
}
export async function toggleCollapsed(f) {
  f.collapsed = !f.collapsed;
  await save();
}
export async function moveFolder(f, dir) {
  const a = S.folders.folders, i = a.indexOf(f), j = i + dir;
  if (j < 0 || j >= a.length) return;
  [a[i], a[j]] = [a[j], a[i]];
  await save();
}
