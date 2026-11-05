// Reading stem file names (no DOM, so node can test it): which stem a file is, and which song it belongs to - a
// folder can hold several songs' stems (Radiohead's remix releases: "01 Reckoner (Bass Stem).wav", "Nude (Voice
// Stem).wav" side by side), plus full songs that aren't stems at all.

export const ROLES = ["drums", "bass", "other", "vocals"];

const base = (n) => n.replace(/\.[a-z0-9]{2,4}$/i, "");
const words = (n) => " " + base(n).toLowerCase().replace(/[_\-.()[\]&+,]+/g, " ").replace(/\s+/g, " ") + " ";

/** {role, sure}: sure = the name says so (not just "anything else is other"). */
export function guessRole(name) {
  const w = words(name);
  if (/ (no|without|minus) ?(vocals?|vox|voice) /.test(w) || / (instrumental|inst|karaoke|backing track) /.test(w)) return { role: "other", sure: true };
  if (/(vocal|vox|voice|acap|a ?cappella|acapella| bgv |backing vo|ad ?libs? | choir| rap | lead v)/.test(w)) return { role: "vocals", sure: true };
  if (/(bass ?drum|kick| kik | bd |snare| snr | sd |hi ?hats?| hh | hats? |cymbal|crash| ride |perc|clap| toms? |overhead| oh |shaker|tamb| rim|break ?beat|drum)/.test(w)) return { role: "drums", sure: true };
  if (/(bass| 808| sub |sub ?bass|low ?end)/.test(w)) return { role: "bass", sure: true };
  if (/(guitar| gtr|keys|piano|synth| pads? |string|lead|pluck|organ|brass|horn| fx | sfx | arp|chord|melod| other | music | stem )/.test(w)) return { role: "other", sure: true };
  return { role: "other", sure: false };
}

/** Does a drop of these files look like stems rather than songs? */
export function looksLikeStems(names) {
  return names.some((n) => guessRole(n).sure && !/ (instrumental|inst|karaoke) /.test(words(n)));
}

const STEMWORD = /\b(stems?|vocals?|vox|voice|acapella|a ?cappella|backing|lead|bgv|drums?|percussion|perc|kick|snare|hats?|hi ?hat|cymbals?|toms?|overheads?|bass|808|sub|guitars?|gtrs?|piano|keys|strings?|synths?|pads?|fx|sfx|etc|other|music|instrumental|and|n)\b/gi;

/** The song a stem file belongs to, from its name: track number, bracketed parts ("(Bass Stem)", even cut off:
 * "(String FX Etc. S") and " - Kick and Snare"-style stem parts dropped. "" when the name is only a stem. */
export function songOf(name) {
  let s = base(name).replace(/^\s*\d{1,3}(?:\s*[.\-_)]\s*|\s+)/, "");    // "01 ", "07. ", "1_"
  s = s.replace(/[([].*$/, "");                                           // "(Bass Stem)", "[stems]", unclosed too
  const parts = s.split(/\s+-\s+|_+|\s-|-\s/).map((x) => x.trim()).filter(Boolean);
  const keep = parts.filter((x) => !new RegExp(STEMWORD.source, "i").test(x));
  s = (keep.length ? keep : parts).join(" - ").replace(STEMWORD, " ");
  return s.replace(/\s+/g, " ").replace(/^[\s\-–_.,]+|[\s\-–_.,]+$/g, "").trim();
}

/** Files -> songs: [{title, items: [index...]}]. Files with the same song name go together; a file the name says
 * nothing about joins the stems if there's only one song, else stays its own (probably a full song, not a stem). */
export function groupSongs(names, folder = "") {
  const groups = new Map();
  names.forEach((n, i) => {
    const t = songOf(n), k = t.toLowerCase();
    if (!groups.has(k)) groups.set(k, { title: t, items: [] });
    groups.get(k).items.push(i);
  });
  const named = [...groups.entries()].filter(([k]) => k);
  const loose = groups.get("")?.items || [];
  let out = named.map(([, g]) => g);
  if (loose.length) {
    if (out.length === 1) out[0].items.push(...loose);
    else out.push({ title: folder || "Stems", items: loose });
  }
  if (!out.length) out = [{ title: folder || "Stems", items: names.map((_, i) => i) }];
  out.forEach((g) => { if (!g.title) g.title = folder || "Stems"; g.items.sort((a, b) => a - b); });
  return out;
}

/** The stem each file starts on: its guess; "" (left out) for a file that looks like a whole song - alone in its
 * group with nothing in the name saying stem. */
export function startRoles(names, groups) {
  const role = names.map((n) => guessRole(n).role);
  for (const g of groups) {
    const sure = g.items.filter((i) => guessRole(names[i]).sure);
    if (!sure.length && (g.items.length === 1 || groups.length > 1)) g.items.forEach((i) => (role[i] = ""));
  }
  return role;
}
