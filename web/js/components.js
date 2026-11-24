// When something Re-Tex needs from outside (a model, a library, a program) isn't there, or a fallback was used,
// this says so: what isn't working, the error, what it's needed for, how to fix it, a link - and a Download button
// where it can be fetched from here. Nothing downloads until that button is pressed (mashlab/components.py).
import { api } from "./api.js";
import { modal } from "./ui.js";

const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
const open = new Map();          // key -> dialog shown for it right now (one at a time per component)
const waiting = new Map();       // download job id -> {key, out}

function row(label, html) {
  return html ? `<div class="cmp-row"><span class="cmp-l">${label}</span><span>${html}</span></div>` : "";
}
function link(url) {
  return url ? `<a href="${esc(url)}" target="_blank" rel="noopener noreferrer">${esc(url)}</a>` : "";
}

// info: {key, name, problem, used_for, fix, url, where, download, instead?}
export function showComponent(info, { fallback = false } = {}) {
  if (!info || open.has(info.key)) return;
  const title = fallback ? `Fallback in use: ${info.name}` : `${info.name} isn't working`;
  const m = modal(title, `
    ${fallback
      ? `<p class="cmp-warn">This is a fallback. Re-Tex carried on, but not the way it should: ${esc(info.instead)}.</p>`
      : `<p class="cmp-warn">This stopped because ${esc(info.name)} isn't there or isn't working. Nothing was substituted.</p>`}
    ${row("The error", esc(info.problem || "(it reports no problem now - it may have been fixed since)"))}
    ${row("Needed for", esc(info.used_for))}
    ${row("On this PC", info.where ? `<code>${esc(info.where)}</code>` : "")}
    ${row("To fix it", esc(info.fix))}
    ${row("Link", link(info.url))}
    <div class="cmp-out"></div>`, { onClose: () => open.delete(info.key) });
  open.set(info.key, m);
  m.footer.hidden = false;
  m.footer.innerHTML = `${info.download ? `<button class="primary" id="cmpDl">Download now</button>` : ""}
    <button id="cmpAll">All components…</button><button id="cmpClose">Close</button>`;
  m.footer.querySelector("#cmpClose").onclick = m.close;
  m.footer.querySelector("#cmpAll").onclick = () => { m.close(); openComponents(); };
  const out = m.body.querySelector(".cmp-out");
  m.footer.querySelector("#cmpDl")?.addEventListener("click", (e) => download(info, out, e.target));
}

async function download(info, out, btn) {
  if (btn) btn.disabled = true;
  out.innerHTML = `<span class="spinner"></span> downloading ${esc(info.name)} - progress is in the status bar`;
  try {
    const j = await api.post(`/api/components/${info.key}/download`);
    waiting.set(j.id, { key: info.key, out, btn });
  } catch (e) {
    out.innerHTML = `<span class="set-bad">✕ ${esc(e.message)}</span>`;
    if (btn) btn.disabled = false;
  }
}

// main.js passes every job update here, so a dialog that started a download says how it ended
export function componentJob(j) {
  const w = waiting.get(j.id);
  if (!w || (j.status !== "done" && j.status !== "error" && j.status !== "cancelled")) return;
  waiting.delete(j.id);
  if (j.status === "done") w.out.innerHTML = `✓ downloaded and checked - try again now`;
  else { w.out.innerHTML = `<span class="set-bad">✕ ${esc(j.error || j.status)}</span>`; if (w.btn) w.btn.disabled = false; }
}

// The list (Settings > Components): every outside piece Re-Tex uses, where it is on this PC, missing ones first.
export async function openComponents() {
  const m = modal("Components", `<p class="lead">Everything Re-Tex uses that isn't its own code. Missing ones can be
    downloaded from here (only when you press Download), or the link says how to get them.</p>
    <div class="cmp-list"><span class="spinner"></span></div>`, { wide: true });
  const list = m.body.querySelector(".cmp-list");
  let r;
  try { r = await api.get("/api/components"); } catch (e) { list.innerHTML = `<span class="set-bad">✕ ${esc(e.message)}</span>`; return; }
  const items = [...r.items].sort((a, b) => a.ok - b.ok);
  const fb = new Map((r.fallbacks || []).map((f) => [f.key, f.instead]));
  list.innerHTML = items.map((c) => `<div class="cmp-item${c.ok ? "" : " bad"}" data-k="${esc(c.key)}">
      <div class="cmp-h"><b>${c.ok ? "✓" : "✕"} ${esc(c.name)}</b> <span class="cmp-dim">${esc(c.used_for)}</span></div>
      ${c.ok ? "" : row("The error", esc(c.problem)) + row("To fix it", esc(c.fix))}
      ${fb.has(c.key) ? row("Fallback used", `<span class="set-bad">${esc(fb.get(c.key))}</span>`) : ""}
      ${row("On this PC", c.where ? `<code>${esc(c.where)}</code>` : "")}
      ${row("Link", link(c.url))}
      ${!c.ok && c.download ? `<div class="row-gap"><button class="small primary cmp-dl">Download now</button><span class="cmp-out"></span></div>` : ""}
    </div>`).join("");
  list.querySelectorAll(".cmp-dl").forEach((b) => {
    const el = b.closest(".cmp-item");
    const c = items.find((x) => x.key === el.dataset.k);
    b.onclick = () => download(c, el.querySelector(".cmp-out"), b);
  });
}

// Fallbacks that happened before this page was open (once per Re-Tex run, per component).
export async function showEarlierFallbacks() {
  let r;
  try { r = await api.get("/api/components"); } catch { return; }
  let seen = {};
  try { seen = JSON.parse(localStorage.getItem("retexFallbacksSeen") || "{}"); } catch {}
  const run = String(r.since || "");
  if (seen.run !== run) seen = { run, keys: [] };
  for (const f of r.fallbacks || []) {
    if (seen.keys.includes(f.key)) continue;
    seen.keys.push(f.key);
    showComponent(f, { fallback: true });
  }
  try { localStorage.setItem("retexFallbacksSeen", JSON.stringify(seen)); } catch {}
}
export function markFallbackSeen(key, since) {
  try {
    const seen = JSON.parse(localStorage.getItem("retexFallbacksSeen") || "{}");
    const s = seen.run === String(since) ? seen : { run: String(since), keys: [] };
    if (!s.keys.includes(key)) s.keys.push(key);
    localStorage.setItem("retexFallbacksSeen", JSON.stringify(s));
  } catch {}
}

// api.js raises this for a 424 (or an error message carrying a [component:key] mark)
window.addEventListener("retex-missing", (e) => showComponent(e.detail));
