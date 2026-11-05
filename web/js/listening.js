// Listening: test results to hear and weigh in on (listening.py). Each session says what was being tried, what each
// clip is, what was measured and decided - and keeps your verdict and notes, read back before the next round.
// A/B: starting one clip while another plays picks up at the same moment (versions of one mix, same timeline).
import { api } from "./api.js";
import { toast } from "./state.js";
import { modal } from "./ui.js";

const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const para = (s) => esc(s).replace(/\n/g, "<br>");
const VERDICT_LABEL = { right: "Sounds right", close: "Close - needs work", wrong: "Not there" };
const btn = document.getElementById("btnListen");

export async function refreshListenBadge() {
  let r;
  try { r = await api.get("/api/listening"); } catch { return; }
  btn.hidden = !r.enabled;
  const n = r.sessions.reduce((a, s) => a + s.waiting, 0);
  btn.querySelector(".badge").textContent = n ? String(n) : "";
  btn.title = n ? `Listening: ${n} test clip${n > 1 ? "s" : ""} waiting for your ears` : "Listening: test results to hear and weigh in on";
}

function item(s, it, fb) {
  const f = fb.items?.[it.id] || {};
  return `<div class="ls-item" data-item="${it.id}">
    <div class="ls-item-head"><b>${esc(it.label)}</b>${f.verdict ? ` <span class="tag ls-v-${f.verdict}">${esc(VERDICT_LABEL[f.verdict])}</span>` : ""}</div>
    ${it.what ? `<div class="dim">${para(it.what)}</div>` : ""}
    <audio controls preload="metadata" src="/api/listening/${encodeURIComponent(s.id)}/audio/${encodeURIComponent(it.file)}"></audio>
    ${it.measured ? `<div class="ls-note"><span class="ls-k">Measured</span>${para(it.measured)}</div>` : ""}
    ${it.mine ? `<div class="ls-note"><span class="ls-k">My read</span>${para(it.mine)}</div>` : ""}
    <div class="ls-yours"><span class="ls-k">Yours</span>
      <span class="seg">${Object.entries(VERDICT_LABEL).map(([k, t]) => `<button class="small${f.verdict === k ? " on" : ""}" data-v="${k}">${t}</button>`).join("")}</span>
      <textarea rows="2" spellcheck="true" placeholder="What you hear - timing, artifacts, joins, tone, what you'd do instead…">${esc(f.notes || "")}</textarea>
      <span class="dim ls-saved"></span></div>
  </div>`;
}

function session(s, open) {
  const fb = s.feedback || {};
  return `<details class="ls-session" data-sid="${esc(s.id)}"${open ? " open" : ""}>
    <summary><b>${esc(s.title)}</b> <span class="dim">· ${new Date(s.created * 1000).toLocaleString()}</span>
      ${s.waiting ? `<span class="tag">${s.waiting} waiting</span>` : '<span class="tag ls-v-right">heard</span>'}</summary>
    <div class="ls-body">
      <div class="ls-note"><span class="ls-k">What I was trying</span>${para(s.goal)}</div>
      <div class="row-gap"><label class="dim"><input type="checkbox" class="ls-ab" checked> A/B: switching clips keeps the position</label>
        <button class="small ls-reveal" title="Open the folder - drag the files into your DAW">Open folder</button>
        <button class="small ls-done" title="Done with these: the audio is deleted, your notes are kept (listening/_archive)">Done - remove</button></div>
      ${s.items.map((it) => item(s, it, fb)).join("")}
      ${s.decision ? `<div class="ls-note"><span class="ls-k">What I decided</span>${para(s.decision)}</div>` : ""}
      ${s.questions?.length ? `<div class="ls-note"><span class="ls-k">What I can't hear - your call</span><ul>${s.questions.map((q) => `<li>${esc(q)}</li>`).join("")}</ul></div>` : ""}
      <div class="ls-yours ls-overall" data-item="overall"><span class="ls-k">Overall</span>
        <textarea rows="3" spellcheck="true" placeholder="Anything about this round as a whole - what was really going on, what to try next…">${esc(fb.overall?.notes || "")}</textarea>
        <span class="dim ls-saved"></span></div>
    </div></details>`;
}

export async function openListening() {
  let r;
  try { r = await api.get("/api/listening"); } catch (e) { toast(e.message, true); return; }
  const m = modal("Listening", `
    <p class="dim" style="margin:0 0 10px">Test results from working on Re-Tex, to hear for yourself. Each one says what I was trying, what
      I measured and what I decided from it - the numbers can't hear joins, artifacts or feel, you can. Your verdicts and notes
      are saved with the files and read before the next round. Plays through the system's default output.</p>
    ${r.sessions.length ? r.sessions.map((s, i) => session(s, i === 0 || s.waiting > 0)).join("") : '<p class="dim">Nothing to hear yet.</p>'}`,
    { wide: true, onClose: () => { m.body.querySelectorAll("audio").forEach((a) => a.pause()); refreshListenBadge(); } });

  const send = async (sid, el, body) => {
    const note = el.querySelector(".ls-saved");
    try { await api.put(`/api/listening/${encodeURIComponent(sid)}/feedback`, body); if (note) note.textContent = "saved"; }
    catch (e) { toast(e.message, true); }
  };
  m.body.querySelectorAll(".ls-session").forEach((S) => {
    const sid = S.dataset.sid;
    S.querySelector(".ls-reveal").onclick = () => api.post(`/api/listening/${encodeURIComponent(sid)}/reveal`).catch((e) => toast(e.message, true));
    S.querySelector(".ls-done").onclick = async () => {
      if (!confirm("Remove this session? Its audio is deleted; your verdicts and notes are kept.")) return;
      S.querySelectorAll("audio").forEach((a) => { a.pause(); a.removeAttribute("src"); a.load(); });   // let go of the files
      try { await api.del(`/api/listening/${encodeURIComponent(sid)}`); S.remove(); refreshListenBadge(); }
      catch (e) { toast(e.message, true); }
    };
    S.querySelectorAll("[data-item]").forEach((el) => {
      const item = el.dataset.item;
      el.querySelectorAll("[data-v]").forEach((b) => {
        b.onclick = () => {
          const v = b.classList.contains("on") ? "" : b.dataset.v;      // click again to take it back
          el.querySelectorAll("[data-v]").forEach((x) => x.classList.toggle("on", x === b && v !== ""));
          send(sid, el, { item, verdict: v });
        };
      });
      const ta = el.querySelector("textarea");
      let t = null;
      ta.oninput = () => { el.querySelector(".ls-saved").textContent = ""; clearTimeout(t); t = setTimeout(() => send(sid, el, { item, notes: ta.value }), 700); };
      ta.onblur = () => { if (t) { clearTimeout(t); t = null; send(sid, el, { item, notes: ta.value }); } };
    });
    // A/B: one clip at a time; with the box ticked, the next one starts where the last one was
    S.querySelectorAll("audio").forEach((a) => {
      a.addEventListener("play", () => {
        m.body.querySelectorAll("audio").forEach((o) => {
          if (o === a || o.paused) return;
          if (S.querySelector(".ls-ab").checked && S.contains(o)) {
            const at = o.currentTime, go = () => { a.currentTime = Math.min(at, a.duration || at); };
            if (a.readyState >= 1) go(); else a.addEventListener("loadedmetadata", go, { once: true });
          }
          o.pause();
        });
      });
    });
  });
}

btn.onclick = openListening;
refreshListenBadge();
