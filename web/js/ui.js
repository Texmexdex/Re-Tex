// Context menus + modal dialogs.
const menu = document.getElementById("ctxMenu");

export function showMenu(e, items) {
  menu.innerHTML = "";
  for (const it of items) {
    if (it === "-") { const s = document.createElement("div"); s.className = "sep"; menu.appendChild(s); continue; }
    const [label, fn, key] = it;
    const d = document.createElement("div");
    d.className = "it";
    d.innerHTML = `<span></span>${key ? `<span class="k">${key}</span>` : ""}`;
    d.firstChild.textContent = label;
    if (fn) d.onclick = () => { hideMenu(); fn(); };
    else { d.style.color = "var(--dim)"; d.style.cursor = "default"; d.style.fontSize = "11px"; }
    menu.appendChild(d);
  }
  menu.hidden = false;
  const r = menu.getBoundingClientRect();
  menu.style.left = Math.min(e.clientX, innerWidth - r.width - 6) + "px";
  menu.style.top = Math.min(e.clientY, innerHeight - r.height - 6) + "px";
  e.stopPropagation?.();
}
// A long list to pick from (folders...), in the same spot as a menu: a filter box, the list scrolling inside a
// short box instead of running down the screen, then a few fixed items (extra). Enter picks the first match.
export function showPicker(e, { title = "", items = [], extra = [], placeholder = "filter…" } = {}) {
  menu.innerHTML = "";
  if (title) { const h = document.createElement("div"); h.className = "pk-title"; h.textContent = title; menu.appendChild(h); }
  const q = document.createElement("input");
  q.className = "pk-q"; q.placeholder = placeholder; q.spellcheck = false;
  const list = document.createElement("div");
  list.className = "pk-list";
  const row = ([label, fn]) => {
    const d = document.createElement("div");
    d.className = "it"; d.innerHTML = "<span></span>"; d.firstChild.textContent = label;
    d.onclick = () => { hideMenu(); fn(); };
    return d;
  };
  const fill = () => {
    const f = q.value.trim().toLowerCase();
    const shown = items.filter(([label]) => !f || label.toLowerCase().includes(f));
    list.replaceChildren(...shown.map(row));
    if (!shown.length) { const d = document.createElement("div"); d.className = "pk-none"; d.textContent = f ? "no folder by that name" : "no folders yet"; list.appendChild(d); }
    return shown;
  };
  q.oninput = fill;
  q.onkeydown = (k) => {
    if (k.key === "Enter") { const s = fill(); if (s.length) { hideMenu(); s[0][1](); } }
    else if (k.key === "Escape") hideMenu();
    k.stopPropagation();
  };
  menu.append(q, list);
  if (extra.length) { const s = document.createElement("div"); s.className = "sep"; menu.appendChild(s); extra.forEach((x) => menu.appendChild(row(x))); }
  fill();
  menu.hidden = false;
  const r = menu.getBoundingClientRect();
  menu.style.left = Math.min(e.clientX, innerWidth - r.width - 6) + "px";
  menu.style.top = Math.min(e.clientY, innerHeight - r.height - 6) + "px";
  setTimeout(() => q.focus(), 0);
}
export function hideMenu() { menu.hidden = true; }
window.addEventListener("mousedown", (e) => { if (!menu.contains(e.target)) hideMenu(); });
window.addEventListener("blur", hideMenu);

export function modal(title, bodyHtml, { wide = false, onClose } = {}) {
  const root = document.getElementById("modalRoot");
  const back = document.createElement("div");
  back.className = "modal-back";
  back.innerHTML = `<div class="modal"${wide ? ' style="width:min(1100px,96vw)"' : ""}>
    <header><h3></h3><button class="small close" title="Close (Esc)">✕</button></header>
    <div class="body"></div><footer hidden></footer></div>`;
  back.querySelector("h3").textContent = title;
  back.querySelector(".body").innerHTML = bodyHtml;
  const close = () => { back.remove(); document.removeEventListener("keydown", esc, true); onClose?.(); };
  const esc = (e) => { if (e.key === "Escape") { e.stopPropagation(); close(); } };
  document.addEventListener("keydown", esc, true);
  back.querySelector(".close").onclick = close;
  back.addEventListener("mousedown", (e) => { if (e.target === back) close(); });
  root.appendChild(back);
  const el = back.querySelector(".modal");
  return { el, body: el.querySelector(".body"), footer: el.querySelector("footer"), close };
}
