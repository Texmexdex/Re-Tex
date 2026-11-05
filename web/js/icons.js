// Line icons (16-unit grid, currentColor). The UI uses these instead of emoji.
const svg = (body) => `<svg class="ic" viewBox="0 0 16 16" aria-hidden="true">${body}</svg>`;
const S = 'fill="none" stroke="currentColor" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round"';

export const ICONS = {
  play: svg('<path d="M4.5 2.8v10.4L13 8z" fill="currentColor"/>'),
  stop: svg('<rect x="3.5" y="3.5" width="9" height="9" rx="0.5" fill="currentColor"/>'),
  lock: svg(`<rect x="3.5" y="7" width="9" height="6.5" rx="1" ${S}/><path d="M5.5 7V5.2a2.5 2.5 0 0 1 5 0V7" ${S}/>`),
  search: svg(`<circle cx="7" cy="7" r="4.2" ${S}/><path d="M10.2 10.2L13.5 13.5" ${S}/>`),
  keyboard: svg(`<rect x="1.5" y="4" width="13" height="8" rx="1" ${S}/><path d="M4 6.8h.01M6.7 6.8h.01M9.3 6.8h.01M12 6.8h.01M5 9.5h6" ${S}/>`),
};
