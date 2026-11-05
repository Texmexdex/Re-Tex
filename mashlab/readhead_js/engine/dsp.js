// ReadHead engine — frame-level audio features (pure JS, runs in a Worker or Node).
// One pass per stem: RMS, spectral-flux onset strength, low-band (kick) flux,
// 12-bin chroma, and 16 log-spaced timbre bands.

export const FFT_N = 2048;
export const HOP = 512;
export const NBANDS = 16;

function makeFFT(n) {
  const levels = Math.log2(n) | 0;
  const rev = new Uint32Array(n);
  for (let i = 0; i < n; i++) {
    let x = i, r = 0;
    for (let j = 0; j < levels; j++) { r = (r << 1) | (x & 1); x >>= 1; }
    rev[i] = r;
  }
  const cos = new Float64Array(n / 2), sin = new Float64Array(n / 2);
  for (let i = 0; i < n / 2; i++) {
    cos[i] = Math.cos((2 * Math.PI * i) / n);
    sin[i] = Math.sin((2 * Math.PI * i) / n);
  }
  return function fft(re, im) {
    for (let i = 0; i < n; i++) {
      const j = rev[i];
      if (j > i) {
        let t = re[i]; re[i] = re[j]; re[j] = t;
        t = im[i]; im[i] = im[j]; im[j] = t;
      }
    }
    for (let size = 2; size <= n; size <<= 1) {
      const half = size >> 1, step = n / size;
      for (let i = 0; i < n; i += size) {
        for (let j = i, k = 0; j < i + half; j++, k += step) {
          const l = j + half;
          const tre = re[l] * cos[k] + im[l] * sin[k];
          const tim = -re[l] * sin[k] + im[l] * cos[k];
          re[l] = re[j] - tre; im[l] = im[j] - tim;
          re[j] += tre; im[j] += tim;
        }
      }
    }
  };
}

/**
 * @param {Float32Array} mono  mono samples
 * @param {number} sr          sample rate
 * @returns frame features; frame k is centred at time k*HOP/sr
 */
export function analyzeFrames(mono, sr, onProgress) {
  const N = FFT_N, H = HOP;
  const nFrames = Math.max(1, Math.ceil(mono.length / H));
  const fft = makeFFT(N);
  const win = new Float64Array(N);
  for (let i = 0; i < N; i++) win[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / N);

  const nb = N / 2;
  const binHz = sr / N;
  // bin -> pitch class (or -1), bin -> band
  const pc = new Int8Array(nb).fill(-1);
  const pcW = new Float32Array(nb);
  for (let k = 1; k < nb; k++) {
    const f = k * binHz;
    if (f >= 55 && f <= 4200) {
      const midi = 69 + 12 * Math.log2(f / 440);
      const m = Math.round(midi);
      pc[k] = ((m % 12) + 12) % 12;
      // de-emphasise bins far from the semitone centre
      pcW[k] = Math.cos(Math.min(1, Math.abs(midi - m) * 2) * Math.PI / 2) ** 2;
    }
  }
  const band = new Int8Array(nb).fill(-1);
  const fLo = 40, fHi = Math.min(16000, sr / 2 - 1);
  for (let k = 1; k < nb; k++) {
    const f = k * binHz;
    if (f >= fLo && f < fHi) {
      band[k] = Math.min(NBANDS - 1, Math.floor((NBANDS * Math.log(f / fLo)) / Math.log(fHi / fLo)));
    }
  }
  const lowBin = Math.max(2, Math.round(150 / binHz));

  const rms = new Float32Array(nFrames);
  const flux = new Float32Array(nFrames);
  const low = new Float32Array(nFrames);
  const chroma = new Float32Array(nFrames * 12);
  const bands = new Float32Array(nFrames * NBANDS);

  const re = new Float64Array(N), im = new Float64Array(N);
  const prev = new Float64Array(nb);
  const cur = new Float64Array(nb);
  const bandAcc = new Float64Array(NBANDS);
  const bandCnt = new Float64Array(NBANDS);
  for (let k = 1; k < nb; k++) if (band[k] >= 0) bandCnt[band[k]]++;

  for (let fIdx = 0; fIdx < nFrames; fIdx++) {
    const centre = fIdx * H;
    const start = centre - N / 2;
    let e = 0;
    for (let i = 0; i < N; i++) {
      const idx = start + i;
      const x = idx >= 0 && idx < mono.length ? mono[idx] : 0;
      re[i] = x * win[i];
      im[i] = 0;
      if (i >= N / 2 - H / 2 && i < N / 2 + H / 2) e += x * x;
    }
    rms[fIdx] = Math.sqrt(e / H);
    fft(re, im);
    let fl = 0, lw = 0;
    bandAcc.fill(0);
    const cOff = fIdx * 12;
    for (let k = 1; k < nb; k++) {
      const mag = Math.sqrt(re[k] * re[k] + im[k] * im[k]);
      const lm = Math.log1p(100 * mag);
      cur[k] = lm;
      const d = lm - prev[k];
      if (d > 0) { fl += d; if (k <= lowBin) lw += d; }
      if (pc[k] >= 0) chroma[cOff + pc[k]] += mag * pcW[k];
      if (band[k] >= 0) bandAcc[band[k]] += mag * mag;
    }
    flux[fIdx] = fl;
    low[fIdx] = lw;
    const bOff = fIdx * NBANDS;
    for (let b = 0; b < NBANDS; b++) bands[bOff + b] = Math.log10(1e-10 + bandAcc[b] / Math.max(1, bandCnt[b]));
    prev.set(cur);
    if (onProgress && (fIdx & 1023) === 0) onProgress(fIdx / nFrames);
  }
  return { sr, hop: H, hopSec: H / sr, nFrames, rms, flux, low, chroma, bands, nbands: NBANDS };
}

export function toMono(channels) {
  const n = channels[0].length;
  if (channels.length === 1) return channels[0];
  const out = new Float32Array(n);
  for (const ch of channels) for (let i = 0; i < n; i++) out[i] += ch[i];
  const g = 1 / channels.length;
  for (let i = 0; i < n; i++) out[i] *= g;
  return out;
}
