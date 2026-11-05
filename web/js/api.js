// Thin fetch wrappers + the server-sent event stream.
async function req(method, url, body) {
  const opt = { method, headers: {} };
  if (body !== undefined) {
    if (body instanceof FormData) opt.body = body;
    else { opt.body = JSON.stringify(body); opt.headers["Content-Type"] = "application/json"; }
  }
  const r = await fetch(url, opt);
  if (!r.ok) {
    let msg = r.statusText;
    try { const j = await r.json(); msg = j.detail || msg; } catch {}
    throw new Error(msg);
  }
  const ct = r.headers.get("content-type") || "";
  return ct.includes("json") ? r.json() : r.arrayBuffer();
}

export const api = {
  get: (u) => req("GET", u),
  post: (u, b) => req("POST", u, b ?? {}),
  put: (u, b) => req("PUT", u, b),
  patch: (u, b) => req("PATCH", u, b),
  del: (u) => req("DELETE", u),
};

export function connectEvents(onMsg) {
  let es;
  const open = () => {
    es = new EventSource("/api/events");
    es.onmessage = (e) => { try { onMsg(JSON.parse(e.data)); } catch (err) { console.error(err); } };
    es.onerror = () => { es.close(); setTimeout(open, 1500); onMsg({ type: "disconnected" }); };
  };
  open();
}

const peakCache = new Map();
export async function peaks(songId, stem) {
  const k = songId + "/" + stem;
  if (!peakCache.has(k)) {
    peakCache.set(k, api.get(`/api/songs/${songId}/peaks/${stem}`).then((b) => new Int8Array(b)).catch(() => null));
  }
  return peakCache.get(k);
}
export function dropPeaks(songId) {
  for (const k of [...peakCache.keys()]) if (k.startsWith(songId + "/")) peakCache.delete(k);
  for (const k of [...hiCache.keys()]) if (k.startsWith(songId + "/")) hiCache.delete(k);
}

// close-up waveform (16-bit min/max, a few thousand per second) - only fetched once you zoom in on a song
const hiCache = new Map();
export function peaksHi(songId, stem) {
  const k = songId + "/" + stem;
  if (!hiCache.has(k)) {
    hiCache.set(k, fetch(`/api/songs/${songId}/peaks_hi/${stem}`).then(async (r) => {
      if (!r.ok) return null;
      return { pk: new Int16Array(await r.arrayBuffer()), pps: +r.headers.get("X-Peaks-Per-Sec") || 2205 };
    }).catch(() => null));
  }
  return hiCache.get(k);
}
