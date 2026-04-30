/**
 * VideoPlayerFrameSelector — vanilla JS widget for frame-accurate video scrubbing.
 *
 * Layout:
 *   - Marker zone (upper 7px): double-click to add marker, drag to move, alt+click to remove
 *   - Track bar  (lower 10px): click/drag to seek
 *
 * Visual:
 *   - Seeker: thin 1px red vertical line, no decorations
 *   - Markers: equilateral/right-triangle caps above track, gray dashes through track
 *   - Playback buttons: borderless; text buttons (Enlarge, Clear): outlined
 *
 * Reads sibling parameters (input_frame_numbers, frame_selection_mode, every_n)
 * via React fiber tree traversal. Writes back via handleParamChange.
 */

const SESSIONS = new Map();

/* ─── Icons ──────────────────────────────────────────────────────────────── */

const _svg = (d) =>
  `<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="${d}"/></svg>`;

const ICONS = {
  stepBack:  _svg('M15 5L3 12l12 7z M20 5v14'),
  playBack:  _svg('M16 5v14L5 12z'),
  pause:     _svg('M6 4h4v16H6z M14 4h4v16h-4z'),
  playFwd:   _svg('M8 5v14l11-7z'),
  stepFwd:   _svg('M4 5v14 M9 5l12 7-12 7z'),
  enlarge:   `<svg xmlns="http://www.w3.org/2000/svg" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="15 3 21 3 21 9"/><polyline points="9 21 3 21 3 15"/><line x1="21" y1="3" x2="14" y2="10"/><line x1="3" y1="21" x2="10" y2="14"/></svg>`,
};

/* ─── CSS ────────────────────────────────────────────────────────────────── */

const CSS_ID = 'vpw-global-styles';
const CSS = `
  /* ── Playback buttons (no border) ────────────────────────────────────── */
  .vpw-btn {
    background:transparent; border:none; border-radius:4px;
    color:#aaa; width:34px; height:34px;
    display:flex; align-items:center; justify-content:center;
    cursor:pointer; flex-shrink:0; transition:color .12s;
  }
  .vpw-btn:hover:not([disabled]) { color:#ddd; }
  .vpw-btn:active:not([disabled]) { color:#fff; }
  .vpw-btn[disabled] { opacity:.25; cursor:not-allowed; pointer-events:none; }

  /* ── Text buttons (outlined) ─────────────────────────────────────────── */
  .vpw-btn-text {
    background:transparent; border:1px solid #aaa; border-radius:4px;
    color:#aaa; padding:2px 8px; font-size:10px;
    display:flex; align-items:center; gap:4px;
    cursor:pointer; white-space:nowrap;
    transition:color .12s, border-color .12s;
  }
  .vpw-btn-text:hover { color:#ddd; border-color:#ccc; }

  /* ── Timecode ────────────────────────────────────────────────────────── */
  .vpw-timecode {
    color:#aaa; font-size:11px; padding:0 10px; white-space:nowrap;
    font-variant-numeric:tabular-nums;
    font-family:'SF Mono','Monaco','Menlo',monospace;
    text-align:center;
  }

  /* ── Numeric inputs ──────────────────────────────────────────────────── */
  .vpw-num {
    background:transparent; border:1px solid #aaa; border-radius:5px;
    color:#aaa; padding:4px 8px; font-size:11px; text-align:center;
    transition:background .15s, border-color .15s;
  }
  .vpw-num:focus { outline:none; border-color:#ccc; }
  .vpw-num[disabled] { opacity:.3; }
  .vpw-num.vpw-highlight { background:#1a3a5c; border-color:#4a7aaa; color:#ccc; }

  /* ── Total frames (selectable, not editable) ─────────────────────────── */
  .vpw-total {
    color:#aaa; font-size:11px; user-select:all; cursor:text;
    font-family:'SF Mono','Monaco','Menlo',monospace;
  }

  /* ── Timeline container (17px: 7px marker zone + 10px track) ──────── */
  .vpw-timeline {
    position:relative; height:17px;
    user-select:none; touch-action:none;
  }

  /* ── Marker interaction zone (top 7px, same height as triangles) ─────── */
  .vpw-tl-marker-zone {
    position:absolute; top:0; left:0; right:0; bottom:10px;
    cursor:crosshair; z-index:4;
  }

  /* ── Track bar (dark gray, 10px) ─────────────────────────────────────── */
  .vpw-tl-track {
    position:absolute; bottom:0; left:0; right:0; height:10px;
    background:#333; border-radius:3px;
    cursor:pointer; z-index:1; overflow:hidden;
  }

  /* ── Played fill (medium gray) ───────────────────────────────────────── */
  .vpw-tl-fill {
    position:absolute; bottom:0; left:0; height:10px;
    background:#666; border-radius:3px;
    pointer-events:none; z-index:2;
  }

  /* ── Seeker (thin red vertical line, no triangles, no transform) ──────── */
  .vpw-tl-seeker {
    position:absolute; bottom:0; width:1px; height:100%;
    background:#cc3333;
    pointer-events:none; z-index:5;
  }

  /* ── Frame selection markers ─────────────────────────────────────────── */
  .vpw-marker {
    position:absolute; top:0; bottom:-10px; z-index:3; /* bottom:-10px = track height, extends marker to timeline bottom */
    pointer-events:auto; cursor:grab;
  }
  .vpw-marker:active { cursor:grabbing; }

  /* Vertical dash: starts at triangle bottom, extends to timeline bottom */
  .vpw-marker-line {
    position:absolute; top:7px; bottom:0; left:0; width:1px;
    background:#aaa; pointer-events:none;
  }

  /* Triangle base (shared): sits in marker zone, receives hover for highlight */
  .vpw-marker-tri {
    position:absolute; top:0; pointer-events:auto; cursor:grab;
    width:0; height:0;
  }

  /* Single-frame: equilateral triangle pointing down, tip centered on dash */
  .vpw-marker-tri-full {
    left:0.5px; transform:translateX(-50%);
    border-left:4px solid transparent;
    border-right:4px solid transparent;
    border-top:7px solid #aaa;
  }

  /* Range start: right-triangle pointing left (outward), vertical leg aligned with dash */
  .vpw-marker-tri-start {
    right:0;
    border-right:4px solid #aaa;
    border-bottom:7px solid transparent;
  }

  /* Range end: right-triangle pointing right (outward), vertical leg aligned with dash */
  .vpw-marker-tri-end {
    left:0;
    border-left:4px solid #aaa;
    border-bottom:7px solid transparent;
  }

  /* Marker hover/active states */
  .vpw-marker:hover .vpw-marker-line { background:#d4a030; }
  .vpw-marker:hover .vpw-marker-tri-full { border-top-color:#d4a030; }
  .vpw-marker:hover .vpw-marker-tri-start { border-right-color:#d4a030; }
  .vpw-marker:hover .vpw-marker-tri-end { border-left-color:#d4a030; }
  .vpw-marker:active .vpw-marker-line { background:#e0a830; }
  .vpw-marker:active .vpw-marker-tri-full { border-top-color:#e0a830; }
  .vpw-marker:active .vpw-marker-tri-start { border-right-color:#e0a830; }
  .vpw-marker:active .vpw-marker-tri-end { border-left-color:#e0a830; }

  /* Range bar in marker zone (visual fill + hover target between start/end triangles) */
  .vpw-marker-range-bar {
    position:absolute; top:0; height:7px;
    background:rgba(170,170,170,0.08);
    pointer-events:auto; cursor:grab; z-index:2;
  }
  .vpw-marker-range-bar:hover { background:rgba(212,160,48,0.15); }

  /* JS-applied class when hovering the range bar — highlights both markers */
  .vpw-marker.vpw-range-hover .vpw-marker-line { background:#d4a030; }
  .vpw-marker.vpw-range-hover .vpw-marker-tri-start { border-right-color:#d4a030; }
  .vpw-marker.vpw-range-hover .vpw-marker-tri-end { border-left-color:#d4a030; }

  /* Range fill on track (purely visual, non-interactive — track handles seeking) */
  .vpw-range-fill {
    position:absolute; bottom:0; height:10px;
    background:rgba(170,170,170,0.2);
    pointer-events:none; z-index:2;
  }

  /* ── Every-Nth tick marks (non-interactive) ──────────────────────────── */
  .vpw-nth-tick {
    position:absolute; bottom:0; width:1px; height:10px;
    background:rgba(170,170,170,0.4);
    pointer-events:none; z-index:2;
  }

  /* ── Error state (invalid input_frame_numbers) ──────────────────────── */
  .vpw-root.vpw-error {
    outline:2px solid #cc3333;
    outline-offset:-2px;
    animation:vpw-error-pulse 1.5s ease-in-out infinite alternate;
  }
  @keyframes vpw-error-pulse {
    0%   { outline-color:rgba(204,51,51,0.8); }
    100% { outline-color:rgba(204,51,51,0.25); }
  }

  /* ── Enlarged dialog ─────────────────────────────────────────────────── */
  .vpw-dialog-backdrop {
    position:fixed; inset:0; background:rgba(0,0,0,.8);
    z-index:9999; backdrop-filter:blur(3px);
    display:flex; align-items:center; justify-content:center;
  }
  .vpw-dialog-box {
    background:#111; border:1px solid #2d2d2d; border-radius:14px;
    padding:20px; width:96vw; max-height:96vh; overflow-y:auto;
    position:relative; display:flex; flex-direction:column; gap:10px;
    font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
    font-size:12px; color:#ddd;
  }
  .vpw-dialog-title {
    margin:0; color:#888; font-size:13px;
    padding-right:40px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;
  }
  .vpw-dialog-close {
    position:absolute; top:14px; right:14px;
    background:transparent; border:1px solid #555; border-radius:6px;
    color:#999; width:28px; height:28px;
    display:flex; align-items:center; justify-content:center;
    cursor:pointer; font-size:15px;
  }
  .vpw-dialog-close:hover { border-color:#888; color:#ccc; }
  .vpw-dialog-canvas-wrap {
    background:#000; border-radius:8px; overflow:hidden;
    min-height:180px; display:flex; align-items:center; justify-content:center;
  }
  .vpw-dialog-canvas { display:block; width:100%; height:auto; max-height:78vh; }
  .vpw-dialog-placeholder { color:#3a3a3a; font-size:13px; }
`;

function injectGlobalStyles() {
  let el = document.getElementById(CSS_ID);
  if (!el) {
    el = document.createElement('style');
    el.id = CSS_ID;
    document.head.appendChild(el);
  }
  el.textContent = CSS;
}

/* ─── Frame string parsing / generation ─────────────────────────────────── */

/** Parse "1,4,5-9,11" → sorted array of {type:'single',frame} | {type:'range',start,end}. */
function parseFrameString(str) {
  if (!str || !str.trim()) return [];
  const markers = [];
  for (const token of str.split(',')) {
    const t = token.trim();
    if (!t) continue;
    if (t.includes('-')) {
      const [sStr, eStr] = t.split('-');
      const s = parseInt(sStr, 10), e = parseInt(eStr, 10);
      if (!Number.isFinite(s) || !Number.isFinite(e) || s < 1 || e < s) continue;
      markers.push(s === e ? { type: 'single', frame: s } : { type: 'range', start: s, end: e });
    } else {
      const n = parseInt(t, 10);
      if (Number.isFinite(n) && n >= 1) markers.push({ type: 'single', frame: n });
    }
  }
  markers.sort((a, b) => (a.type === 'single' ? a.frame : a.start) - (b.type === 'single' ? b.frame : b.start));
  return markers;
}

/** Return false if the string contains malformed tokens (used for error outline). */
function validateFrameString(str) {
  if (!str || !str.trim()) return true;
  for (const token of str.split(',')) {
    const t = token.trim();
    if (!t) continue;
    if (t.includes('-')) {
      const parts = t.split('-');
      if (parts.length !== 2) return false;
      const s = parseInt(parts[0], 10), e = parseInt(parts[1], 10);
      if (!Number.isFinite(s) || !Number.isFinite(e) || s < 1 || e < s) return false;
    } else {
      const n = parseInt(t, 10);
      if (!Number.isFinite(n) || n < 1 || String(n) !== t.trim()) return false;
    }
  }
  return true;
}

function markersToFrameString(markers) {
  const sorted = [...markers].sort((a, b) =>
    (a.type === 'single' ? a.frame : a.start) - (b.type === 'single' ? b.frame : b.start)
  );
  return sorted.map(m => m.type === 'single' ? String(m.frame) : `${m.start}-${m.end}`).join(',');
}

function isFrameSelected(frameNum, markers) {
  for (const m of markers) {
    if (m.type === 'single' && m.frame === frameNum) return true;
    if (m.type === 'range' && frameNum >= m.start && frameNum <= m.end) return true;
  }
  return false;
}

function markerStart(m) { return m.type === 'single' ? m.frame : m.start; }
function markerEnd(m)   { return m.type === 'single' ? m.frame : m.end; }

/** Merge adjacent/overlapping markers into consolidated ranges. */
function mergeOverlapping(arr) {
  if (arr.length <= 1) return arr;
  const sorted = [...arr].sort((a, b) => markerStart(a) - markerStart(b));
  const result = [sorted[0]];
  for (let i = 1; i < sorted.length; i++) {
    const prev = result[result.length - 1];
    const cur = sorted[i];
    const pEnd = markerEnd(prev);
    const cStart = markerStart(cur);
    const cEnd = markerEnd(cur);
    if (cStart <= pEnd + 1) {
      const newEnd = Math.max(pEnd, cEnd);
      const newStart = markerStart(prev);
      result[result.length - 1] = newStart === newEnd
        ? { type: 'single', frame: newStart }
        : { type: 'range', start: newStart, end: newEnd };
    } else {
      result.push(cur);
    }
  }
  return result;
}

/* ─── HTML helpers ───────────────────────────────────────────────────────── */

function controlsHtml(pfx) {
  return `
    <div style="display:flex;justify-content:center;">
      <div style="display:inline-flex;align-items:center;gap:4px;">
        <button class="${pfx}step-back vpw-btn" title="Back one frame">${ICONS.stepBack}</button>
        <button class="${pfx}play-back vpw-btn" title="Play backwards">${ICONS.playBack}</button>
        <button class="${pfx}pause-back vpw-btn" title="Pause" style="display:none;">${ICONS.pause}</button>
        <span class="${pfx}timecode vpw-timecode">00:00:00 / 00:00:00</span>
        <button class="${pfx}play-fwd vpw-btn" title="Play forwards">${ICONS.playFwd}</button>
        <button class="${pfx}pause-fwd vpw-btn" title="Pause" style="display:none;">${ICONS.pause}</button>
        <button class="${pfx}step-fwd vpw-btn" title="Forward one frame">${ICONS.stepFwd}</button>
      </div>
    </div>`;
}

function timelineHtml(id) {
  return `
    <div class="vpw-timeline nowheel" data-tl="${id}">
      <div class="vpw-tl-marker-zone" data-tl-mzone="${id}"></div>
      <div class="vpw-tl-track" data-tl-track="${id}"></div>
      <div class="vpw-tl-fill" data-tl-fill="${id}"></div>
      <div class="vpw-tl-seeker" data-tl-seeker="${id}"></div>
    </div>`;
}

function bottomHtml(pfx, initFps) {
  return `
    <div style="display:flex;justify-content:center;align-items:center;gap:12px;">
      <label style="display:flex;align-items:center;gap:6px;color:#777;">
        FPS <input class="${pfx}fps vpw-num" type="number" min="1" max="240" step="0.001" value="${initFps}" style="width:58px;">
      </label>
      <label style="display:flex;align-items:center;gap:6px;color:#777;">
        Frame <input class="${pfx}frame vpw-num" type="number" min="0" value="0" style="width:72px;">
      </label>
      <span class="${pfx}total vpw-total">/ 0</span>
      <button class="${pfx}clear-markers vpw-btn-text" title="Clear all markers">&#x2715; Markers</button>
    </div>`;
}

/* ─── Widget ─────────────────────────────────────────────────────────────── */

export default function VideoPlayerWidget(container, props) {
  injectGlobalStyles();

  const { value, onChange, disabled, height } = props;

  const MIN_FPS = 1, MAX_FPS = 240, DEFAULT_FPS = 24;
  const clamp = (n, lo, hi) => Math.max(lo, Math.min(hi, n));

  const sessionKey = String(props?.node_id || props?.nodeId || props?.id || 'vpw-default');
  const session = SESSIONS.get(sessionKey) || {
    fps: null, nativeFps: null, frameIndex: 0, currentTime: 0,
    markers: [], selectedFramesStr: '', selectionMode: 'list', everyN: 1,
    videoSrc: null,
  };
  SESSIONS.set(sessionKey, session);

  const initFps = (() => {
    const s = Number(session.fps), v = Number(value?.fps);
    if (Number.isFinite(s) && s > 0) return clamp(s, MIN_FPS, MAX_FPS);
    if (Number.isFinite(v) && v > 0) return clamp(v, MIN_FPS, MAX_FPS);
    return DEFAULT_FPS;
  })();

  const vpH = height && height > 0 ? Math.max(180, height - 380) : 280;
  container.style.height = 'auto';

  /* ── Main HTML ──────────────────────────────────────────────────────────── */

  container.innerHTML = `
    <div class="vpw-root nodrag" style="
      display:flex;flex-direction:column;gap:10px;
      background:#111;border-radius:10px;padding:10px;color:#ddd;
      font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;font-size:12px;">

      <div style="display:flex;justify-content:flex-end;">
        <button class="vpw-enlarge vpw-btn-text" style="gap:4px;">
          ${ICONS.enlarge} Enlarge
        </button>
      </div>

      <div style="position:relative;background:#000;border-radius:8px;overflow:hidden;height:${vpH}px;">
        <video class="vpw-video" style="width:100%;height:100%;object-fit:contain;" playsinline preload="metadata"></video>
        <div class="vpw-overlay" style="position:absolute;inset:0;display:${session.videoSrc ? 'none' : 'flex'};align-items:center;justify-content:center;color:#3a3a3a;font-size:13px;pointer-events:none;">
          No video loaded
        </div>
      </div>

      ${timelineHtml('main')}
      ${controlsHtml('vpw-m-')}
      ${bottomHtml('vpw-m-', initFps)}
    </div>`;

  /* ── Dialog HTML ────────────────────────────────────────────────────────── */

  const backdrop = document.createElement('div');
  backdrop.className = 'vpw-dialog-backdrop';
  backdrop.style.display = 'none';
  backdrop.innerHTML = `
    <div class="vpw-dialog-box">
      <p class="vpw-dialog-title">Video Player</p>
      <button class="vpw-dialog-close">✕</button>
      <div class="vpw-dialog-canvas-wrap">
        <canvas class="vpw-dcanvas vpw-dialog-canvas"></canvas>
        <span class="vpw-dno-video vpw-dialog-placeholder" style="display:none;">No video loaded</span>
      </div>
      ${timelineHtml('dialog')}
      ${controlsHtml('vpw-d-')}
      ${bottomHtml('vpw-d-', initFps)}
    </div>`;
  document.body.appendChild(backdrop);

  /* ── DOM refs ────────────────────────────────────────────────────────────── */

  const q  = (sel) => container.querySelector(sel);
  const qd = (sel) => backdrop.querySelector(sel);

  const widgetRoot = q('.vpw-root');
  const enlargeBtn = q('.vpw-enlarge');
  const video      = q('.vpw-video');
  const overlay    = q('.vpw-overlay');

  if (session.videoSrc) video.src = session.videoSrc;

  const tl         = q('[data-tl="main"]');
  const tlTrack    = q('[data-tl-track="main"]');
  const tlMzone    = q('[data-tl-mzone="main"]');
  const tlFill     = q('[data-tl-fill="main"]');
  const tlSeeker   = q('[data-tl-seeker="main"]');

  const mStepBack  = q('.vpw-m-step-back');
  const mPlayBack  = q('.vpw-m-play-back');
  const mPauseBack = q('.vpw-m-pause-back');
  const mPlayFwd   = q('.vpw-m-play-fwd');
  const mPauseFwd  = q('.vpw-m-pause-fwd');
  const mStepFwd   = q('.vpw-m-step-fwd');
  const mFpsInput  = q('.vpw-m-fps');
  const mFrameInput= q('.vpw-m-frame');
  const mTotalEl   = q('.vpw-m-total');
  const mClearBtn  = q('.vpw-m-clear-markers');

  const dialogTitleEl = qd('.vpw-dialog-title');
  const dialogClose   = qd('.vpw-dialog-close');
  const dialogCanvas  = qd('.vpw-dcanvas');
  const dNoVideo      = qd('.vpw-dno-video');

  const dtl        = qd('[data-tl="dialog"]');
  const dtlTrack   = qd('[data-tl-track="dialog"]');
  const dtlMzone   = qd('[data-tl-mzone="dialog"]');
  const dtlFill    = qd('[data-tl-fill="dialog"]');
  const dtlSeeker  = qd('[data-tl-seeker="dialog"]');

  const dStepBack  = qd('.vpw-d-step-back');
  const dPlayBack  = qd('.vpw-d-play-back');
  const dPauseBack = qd('.vpw-d-pause-back');
  const dPlayFwd   = qd('.vpw-d-play-fwd');
  const dPauseFwd  = qd('.vpw-d-pause-fwd');
  const dStepFwd   = qd('.vpw-d-step-fwd');
  const dFpsInput  = qd('.vpw-d-fps');
  const dFrameInput= qd('.vpw-d-frame');
  const dTotalEl   = qd('.vpw-d-total');
  const dClearBtn  = qd('.vpw-d-clear-markers');
  const mTimecode  = q('.vpw-m-timecode');
  const dTimecode  = qd('.vpw-d-timecode');

  /* ── State ──────────────────────────────────────────────────────────────── */

  let nativeFps    = Number.isFinite(Number(session.nativeFps)) && Number(session.nativeFps) > 0
                     ? Number(session.nativeFps) : initFps;
  let fps          = initFps;
  let duration     = 0;
  let totalFrames  = 0;
  let frameIndex   = 0;
  let markers      = Array.isArray(session.markers) && session.markers.length > 0
                     ? session.markers.map(m => ({...m})) : [];
  let selectedFramesStr = session.selectedFramesStr || '';
  let selectionMode     = session.selectionMode || 'list';
  let everyN            = session.everyN || 1;
  let videoName    = '';
  let isLoaded     = false;
  let playRafId    = null;
  let canvasRafId  = null;
  let currentSrc   = null;
  let pollId       = null;
  let hasError     = false;

  /* ── React fiber helpers ────────────────────────────────────────────────── */
  // These walk up the React component tree from this container's fiber node
  // to read/write sibling parameter values (input_frame_numbers, etc.).
  // This is necessary because the widget only receives its own parameter's
  // value via props — sibling values must be read from the DynamicNode's
  // data.elements array found higher in the fiber tree.

  /** Read the widget's own value from the nearest ancestor with {value, onChange}. */
  function readLatestValueFromFiber() {
    try {
      const key = Object.keys(container).find(k => k.startsWith('__reactFiber$'));
      if (!key) return undefined;
      let f = container[key];
      for (let i = 0; i < 6; i++) {
        f = f?.return;
        if (!f) break;
        const p = f.memoizedProps;
        if (p && 'onChange' in p && 'value' in p) return p.value;
      }
    } catch { /* ignore */ }
    return undefined;
  }

  /** Walk up to DynamicNode to get data.elements (all parameter values for this node). */
  function readNodeElements() {
    try {
      const key = Object.keys(container).find(k => k.startsWith('__reactFiber$'));
      if (!key) return undefined;
      let f = container[key];
      for (let i = 0; i < 50; i++) {
        f = f?.return;
        if (!f) break;
        const p = f.memoizedProps;
        if (p && p.data && Array.isArray(p.data.elements)) return p.data.elements;
      }
    } catch { /* ignore */ }
    return undefined;
  }

  /** Read the current value of a sibling parameter by name. */
  function readSiblingParamValue(paramName) {
    const elements = readNodeElements();
    if (!elements) return undefined;
    const flat = flattenElements(elements);
    const el = flat.find(e => e.name === paramName);
    return el?.value;
  }

  function flattenElements(elements) {
    const result = [];
    for (const el of elements) {
      if (el.name) result.push(el);
      if (Array.isArray(el.children)) result.push(...flattenElements(el.children));
    }
    return result;
  }

  /** Find the handleParamChange callback from NodeParamDisplay in the fiber tree. */
  function findHandleParamChange() {
    try {
      const key = Object.keys(container).find(k => k.startsWith('__reactFiber$'));
      if (!key) return undefined;
      let f = container[key];
      for (let i = 0; i < 20; i++) {
        f = f?.return;
        if (!f) break;
        const p = f.memoizedProps;
        if (p && typeof p.handleParamChange === 'function') return p.handleParamChange;
      }
    } catch { /* ignore */ }
    return undefined;
  }

  /** Write a value to a sibling parameter (triggers SetParameterValueRequest). */
  function setSiblingParamValue(paramName, val) {
    const fn = findHandleParamChange();
    if (fn) fn(paramName, val);
  }

  /* ── Helpers ─────────────────────────────────────────────────────────────── */

  const computeTotal = () =>
    duration > 0 && nativeFps > 0 ? Math.max(1, Math.round(duration * nativeFps)) : 0;

  function fmtTime(s) {
    if (!Number.isFinite(s) || s < 0) s = 0;
    const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), ss = Math.floor(s % 60);
    return `${String(h).padStart(2,'0')}:${String(m).padStart(2,'0')}:${String(ss).padStart(2,'0')}`;
  }

  function setTimecode(currentSecs, totalSecs) {
    const t = `${fmtTime(currentSecs)} / ${fmtTime(totalSecs)}`;
    mTimecode.textContent = t;
    dTimecode.textContent = t;
  }

  function saveSession() {
    session.fps             = fps;
    session.nativeFps       = nativeFps;
    session.frameIndex      = frameIndex;
    session.currentTime     = Number.isFinite(video.currentTime) ? video.currentTime : 0;
    session.markers         = markers.map(m => ({...m}));
    session.selectedFramesStr = selectedFramesStr;
    session.selectionMode   = selectionMode;
    session.everyN          = everyN;
  }

  /* ── Error state ───────────────────────────────────────────────────────── */

  /** Toggle the red pulsing outline when input_frame_numbers is malformed. */
  function updateErrorState(rawStr) {
    const isValid = validateFrameString(rawStr);
    if (!isValid && rawStr && rawStr.trim()) {
      if (!hasError) { widgetRoot.classList.add('vpw-error'); hasError = true; }
    } else {
      if (hasError) { widgetRoot.classList.remove('vpw-error'); hasError = false; }
    }
  }

  /* ── Marker rendering ──────────────────────────────────────────────────── */

  /** Create a marker DOM element with a triangle cap and vertical dash line. */
  function createMarkerEl(leftPx, idx, triClass) {
    const el = document.createElement('div');
    el.className = 'vpw-marker';
    el.style.left = leftPx;
    el.dataset.markerIndex = String(idx);

    const line = document.createElement('div');
    line.className = 'vpw-marker-line';
    el.appendChild(line);

    const tri = document.createElement('div');
    tri.className = `vpw-marker-tri ${triClass}`;
    el.appendChild(tri);

    return el;
  }

  /**
   * Render marker/tick DOM elements into the given timeline.
   * Positions are snapped to physical pixels via devicePixelRatio for uniform 1px rendering.
   */
  function renderMarkers(tlEl) {
    tlEl.querySelectorAll('.vpw-marker, .vpw-range-fill, .vpw-marker-range-bar, .vpw-nth-tick').forEach(m => m.remove());

    if (totalFrames <= 1) return;
    const max = totalFrames - 1;
    const mzone = tlEl.querySelector('[class*="vpw-tl-marker-zone"]');
    const w = tlEl.offsetWidth || 0;
    const dpr = window.devicePixelRatio || 1;
    const snap = (pct) => w > 0 ? `${Math.round(pct / 100 * w * dpr) / dpr}px` : `${pct}%`;
    const snapNum = (pct) => Math.round(pct / 100 * w * dpr) / dpr;

    if (selectionMode === 'every_Nth' && everyN > 0) {
      for (let frame = 0; frame < totalFrames; frame += everyN) {
        const tick = document.createElement('div');
        tick.className = 'vpw-nth-tick';
        tick.style.left = snap((frame / max) * 100);
        tlEl.appendChild(tick);
      }
      return;
    }

    if (selectionMode !== 'list') return;

    for (let i = 0; i < markers.length; i++) {
      const m = markers[i];
      if (m.type === 'single') {
        const pct = ((m.frame - 1) / max) * 100;
        const el = createMarkerEl(snap(pct), i, 'vpw-marker-tri-full');
        el.dataset.markerEdge = 'single';
        if (mzone) mzone.appendChild(el); else tlEl.appendChild(el);
      } else {
        const startPct = ((m.start - 1) / max) * 100;
        const endPct   = ((m.end - 1) / max) * 100;

        const fill = document.createElement('div');
        fill.className = 'vpw-range-fill';
        fill.style.left  = snap(startPct);
        fill.style.width = w > 0 ? `${snapNum(endPct) - snapNum(startPct)}px` : `${endPct - startPct}%`;
        fill.dataset.markerIndex = String(i);
        fill.dataset.markerEdge = 'fill';
        tlEl.appendChild(fill);

        const startEl = createMarkerEl(snap(startPct), i, 'vpw-marker-tri-start');
        startEl.dataset.markerEdge = 'start';
        if (mzone) mzone.appendChild(startEl); else tlEl.appendChild(startEl);

        const endEl = createMarkerEl(snap(endPct), i, 'vpw-marker-tri-end');
        endEl.dataset.markerEdge = 'end';
        if (mzone) mzone.appendChild(endEl); else tlEl.appendChild(endEl);

        const bar = document.createElement('div');
        bar.className = 'vpw-marker-range-bar';
        bar.style.left = snap(startPct);
        bar.style.width = w > 0 ? `${snapNum(endPct) - snapNum(startPct)}px` : `${endPct - startPct}%`;
        bar.addEventListener('mouseenter', () => { startEl.classList.add('vpw-range-hover'); endEl.classList.add('vpw-range-hover'); });
        bar.addEventListener('mouseleave', () => { startEl.classList.remove('vpw-range-hover'); endEl.classList.remove('vpw-range-hover'); });
        if (mzone) mzone.appendChild(bar); else tlEl.appendChild(bar);
      }
    }
  }

  /* ── Frame counter highlighting ────────────────────────────────────────── */

  function updateFrameHighlight() {
    const currentFrameNum = frameIndex + 1;
    let highlight = false;
    if (selectionMode === 'list') highlight = isFrameSelected(currentFrameNum, markers);
    else if (selectionMode === 'every_Nth') highlight = everyN > 0 && (frameIndex % everyN === 0);

    for (const inp of [mFrameInput, dFrameInput]) {
      if (highlight) inp.classList.add('vpw-highlight');
      else inp.classList.remove('vpw-highlight');
    }
  }

  /* ── Timeline sync & UI ────────────────────────────────────────────────── */

  /** Update only the seeker position and fill width — cheap, called on every frame. */
  function syncSeeker() {
    const pct = totalFrames > 1 ? (frameIndex / (totalFrames - 1)) * 100 : 0;
    tlFill.style.width   = `${pct}%`;
    tlSeeker.style.left  = `${pct}%`;
    dtlFill.style.width  = `${pct}%`;
    dtlSeeker.style.left = `${pct}%`;
    updateFrameHighlight();
  }

  /** Full timeline sync: seeker + markers. Call when markers change or layout shifts. */
  function syncTimeline() {
    syncSeeker();
    renderMarkers(tl);
    if (backdrop.style.display !== 'none') renderMarkers(dtl);
  }

  /** Refresh all UI state: button enabled/disabled, inputs, timecode, and full timeline. */
  function updateUi() {
    totalFrames = computeTotal();
    frameIndex  = clamp(frameIndex, 0, Math.max(0, totalFrames - 1));

    const maxF    = Math.max(0, totalFrames - 1);
    const canPlay = isLoaded && totalFrames > 0;

    for (const b of [mStepBack, mPlayBack, mPauseBack, mPlayFwd, mPauseFwd, mStepFwd,
                      dStepBack, dPlayBack, dPauseBack, dPlayFwd, dPauseFwd, dStepFwd])
      b.disabled = !canPlay;
    for (const inp of [mFrameInput, dFrameInput]) { inp.disabled = !canPlay; inp.max = String(maxF); }
    for (const inp of [mFpsInput, dFpsInput]) inp.disabled = !canPlay;

    mFpsInput.value    = dFpsInput.value    = String(fps);
    mFrameInput.value  = dFrameInput.value  = String(frameIndex);
    mTotalEl.textContent = dTotalEl.textContent = `/ ${maxF}`;
    dialogTitleEl.textContent = videoName || 'Video Player';
    setTimecode(Number.isFinite(video.currentTime) ? video.currentTime : 0, duration);

    syncTimeline();
    saveSession();
  }

  /* ── Seek / playback ────────────────────────────────────────────────────── */

  /** Seek to a specific frame index. Only moves the seeker — markers don't change. */
  function seekToFrame(idx) {
    const max = Math.max(0, totalFrames - 1);
    frameIndex = clamp(Math.round(idx), 0, max);
    if (isLoaded && nativeFps > 0) video.currentTime = frameIndex / nativeFps;
    mFrameInput.value = dFrameInput.value = String(frameIndex);
    setTimecode(Number.isFinite(video.currentTime) ? video.currentTime : 0, duration);
    syncSeeker();
    saveSession();
  }

  function stopPlayback() {
    if (playRafId) { cancelAnimationFrame(playRafId); playRafId = null; }
  }

  function showPlayState(direction) {
    for (const [play, p] of [[mPlayBack, mPauseBack], [dPlayBack, dPauseBack]]) {
      play.style.display  = direction === 'back' ? 'none' : '';
      p.style.display     = direction === 'back' ? '' : 'none';
    }
    for (const [play, p] of [[mPlayFwd, mPauseFwd], [dPlayFwd, dPauseFwd]]) {
      play.style.display  = direction === 'fwd' ? 'none' : '';
      p.style.display     = direction === 'fwd' ? '' : 'none';
    }
  }

  function playBackward() {
    if (!isLoaded) return;
    video.pause(); stopPlayback(); showPlayState('back');
    const frameMs = 1000 / fps;
    let last = null;
    function step(ts) {
      if (last === null) { last = ts; playRafId = requestAnimationFrame(step); return; }
      if (ts - last >= frameMs) {
        const next = video.currentTime - 1 / nativeFps;
        if (next <= 0) { video.currentTime = 0; frameIndex = 0; showPlayState(null); updateUi(); return; }
        video.currentTime = next; last = ts;
      }
      playRafId = requestAnimationFrame(step);
    }
    playRafId = requestAnimationFrame(step);
  }

  function pause() { stopPlayback(); video.pause(); showPlayState(null); }

  function playForward() {
    if (!isLoaded) return;
    video.pause(); stopPlayback(); showPlayState('fwd');
    const frameMs = 1000 / fps;
    let last = null;
    function step(ts) {
      if (last === null) { last = ts; playRafId = requestAnimationFrame(step); return; }
      if (ts - last >= frameMs) {
        const next = video.currentTime + 1 / nativeFps;
        if (next >= duration) { video.currentTime = duration; frameIndex = Math.max(0, totalFrames - 1); showPlayState(null); updateUi(); return; }
        video.currentTime = next; last = ts;
      }
      playRafId = requestAnimationFrame(step);
    }
    playRafId = requestAnimationFrame(step);
  }

  function stepFrame(d) { pause(); seekToFrame(frameIndex + d); }

  /* ── Dialog canvas mirror ───────────────────────────────────────────────── */

  function startMirror() {
    let hasDrawn = false;
    function draw() {
      if (video.readyState >= 2 && video.videoWidth > 0) {
        if (dialogCanvas.width  !== video.videoWidth)  dialogCanvas.width  = video.videoWidth;
        if (dialogCanvas.height !== video.videoHeight) dialogCanvas.height = video.videoHeight;
        dialogCanvas.getContext('2d').drawImage(video, 0, 0);
        if (!hasDrawn) { dialogCanvas.style.display = 'block'; dNoVideo.style.display = 'none'; hasDrawn = true; }
      } else if (!hasDrawn) { dialogCanvas.style.display = 'none'; dNoVideo.style.display = 'block'; }
      canvasRafId = requestAnimationFrame(draw);
    }
    canvasRafId = requestAnimationFrame(draw);
  }

  function stopMirror() { if (canvasRafId) { cancelAnimationFrame(canvasRafId); canvasRafId = null; } }

  function onEscKey(e) { if (e.key === 'Escape') closeDialog(); }
  function openDialog()  { backdrop.style.display = 'flex'; renderMarkers(dtl); startMirror(); document.addEventListener('keydown', onEscKey); }
  function closeDialog() { backdrop.style.display = 'none'; stopMirror(); document.removeEventListener('keydown', onEscKey); }

  /* ── Video loading ──────────────────────────────────────────────────────── */

  function urlBase(u) { return u ? u.split('?')[0] : ''; }

  /** Extract a URL string from a value that may be a string, VideoUrlArtifact, or dict. */
  function extractUrl(val) {
    if (typeof val === 'string' && val.trim()) return val.trim();
    if (val && typeof val === 'object') {
      const u = val.value || val.url || val.src;
      if (typeof u === 'string' && u.trim()) return u.trim();
    }
    return null;
  }

  /** Load a video URL into the <video> element. Skips if same base URL is already loaded. */
  function loadUrl(url, restoreTime = 0) {
    if (urlBase(url) === urlBase(currentSrc) && isLoaded) return;
    currentSrc = url; session.videoSrc = url;
    videoName = url.split('/').pop()?.split('?')[0] || 'video';
    isLoaded = false;
    overlay.textContent = 'Loading…'; overlay.style.display = 'flex';
    video.src = url; video.load();
    if (restoreTime > 0) video.addEventListener('loadedmetadata', () => { video.currentTime = restoreTime; }, { once: true });
    updateUi();
  }

  function initFromProps() {
    frameIndex        = session.frameIndex || 0;
    markers           = Array.isArray(session.markers) ? session.markers.map(m => ({...m})) : [];
    selectedFramesStr = session.selectedFramesStr || '';
    selectionMode     = session.selectionMode || 'list';
    everyN            = session.everyN || 1;

    if (session.videoSrc && urlBase(session.videoSrc) === urlBase(video.src)) {
      currentSrc = session.videoSrc;
      videoName  = session.videoSrc.split('/').pop()?.split('?')[0] || 'video';
      if (session.currentTime > 0) video.addEventListener('loadedmetadata', () => { video.currentTime = session.currentTime; }, { once: true });
      return;
    }
    const url = extractUrl(value);
    if (url) loadUrl(url, session.currentTime || 0);
  }

  function clearVideo() {
    pause(); video.removeAttribute('src'); video.load();
    currentSrc = null; session.videoSrc = null; isLoaded = false;
    videoName = ''; duration = 0; totalFrames = 0; frameIndex = 0;
    markers = []; selectedFramesStr = '';
    overlay.textContent = 'No video loaded'; overlay.style.display = 'flex';
    updateUi();
  }

  /* ── Sibling parameter polling ─────────────────────────────────────────── */

  /**
   * Poll sibling parameter values every 500ms for bidirectional sync.
   * Reads input_frame_numbers, frame_selection_mode, every_n from the fiber tree.
   */
  function startValuePoll() {
    pollId = setInterval(() => {
      const latestValue = readLatestValueFromFiber();
      if (latestValue !== undefined) {
        const latestUrl = extractUrl(latestValue);
        if (latestUrl === null && currentSrc !== null) { clearVideo(); return; }
        if (latestUrl && urlBase(latestUrl) !== urlBase(currentSrc)) { frameIndex = 0; loadUrl(latestUrl, 0); }
      } else {
        const siblingVal = readSiblingParamValue('input_video');
        if (siblingVal !== undefined) {
          const sibUrl = extractUrl(siblingVal);
          if (sibUrl === null && currentSrc !== null) clearVideo();
          else if (sibUrl && urlBase(sibUrl) !== urlBase(currentSrc)) { frameIndex = 0; loadUrl(sibUrl, 0); }
        }
      }

      let needsSync = false;

      const latestFrames = readSiblingParamValue('input_frame_numbers');
      if (latestFrames !== undefined) {
        const str = String(latestFrames || '');
        updateErrorState(str);
        if (str !== selectedFramesStr) {
          selectedFramesStr = str;
          markers = parseFrameString(str);
          needsSync = true;
        }
      }

      const latestMode = readSiblingParamValue('frame_selection_mode');
      if (latestMode !== undefined && latestMode !== selectionMode) { selectionMode = latestMode; needsSync = true; }

      const latestEveryN = readSiblingParamValue('every_n');
      if (latestEveryN !== undefined) {
        const n = Number(latestEveryN);
        if (Number.isFinite(n) && n !== everyN) { everyN = n; needsSync = true; }
      }

      if (needsSync) { syncTimeline(); saveSession(); }
    }, 500);
  }

  /* ── Marker commit ─────────────────────────────────────────────────────── */

  /** Serialize markers to string, write to the input_frame_numbers param, and re-render. */
  function commitMarkers() {
    selectedFramesStr = markersToFrameString(markers);
    setSiblingParamValue('input_frame_numbers', selectedFramesStr);
    syncTimeline();
    saveSession();
  }

  /* ── Seek zone interaction (track area) ────────────────────────────────── */

  /** Convert a pointer event's X position to a 0-based frame index. */
  function frameFromEvent(e, el) {
    const r = el.getBoundingClientRect();
    return Math.round(clamp((e.clientX - r.left) / r.width, 0, 1) * Math.max(0, totalFrames - 1));
  }

  /** Attach pointer handlers to the track bar for click/drag seeking. */
  function attachSeekZone(trackEl) {
    let dragging = false;
    trackEl.addEventListener('pointerdown', (e) => {
      trackEl.setPointerCapture(e.pointerId);
      dragging = true;
      seekToFrame(frameFromEvent(e, trackEl));
    });
    trackEl.addEventListener('pointermove', (e) => { if (dragging) seekToFrame(frameFromEvent(e, trackEl)); });
    trackEl.addEventListener('pointerup',     () => { dragging = false; });
    trackEl.addEventListener('pointercancel', () => { dragging = false; });
  }

  /* ── Marker zone interaction (above track) ─────────────────────────────── */

  const DBLCLICK_MS = 350;
  const HIT_TOLERANCE_PX = 8;

  /** Hit-test a pointer event against all markers. Returns {markerIndex, edge} or null. */
  function hitTestMarker(e, mzoneEl) {
    const r = mzoneEl.getBoundingClientRect();
    const clickX = e.clientX - r.left;
    const maxFrame = Math.max(0, totalFrames - 1);
    if (maxFrame === 0) return null;

    for (let i = 0; i < markers.length; i++) {
      const m = markers[i];
      if (m.type === 'single') {
        const markerX = ((m.frame - 1) / maxFrame) * r.width;
        if (Math.abs(clickX - markerX) <= HIT_TOLERANCE_PX) return { markerIndex: i, edge: 'single' };
      } else {
        const startX = ((m.start - 1) / maxFrame) * r.width;
        const endX   = ((m.end - 1)   / maxFrame) * r.width;
        if (Math.abs(clickX - startX) <= HIT_TOLERANCE_PX) return { markerIndex: i, edge: 'start' };
        if (Math.abs(clickX - endX) <= HIT_TOLERANCE_PX) return { markerIndex: i, edge: 'end' };
        if (clickX > startX + HIT_TOLERANCE_PX && clickX < endX - HIT_TOLERANCE_PX) return { markerIndex: i, edge: 'fill' };
      }
    }
    return null;
  }

  /**
   * Attach pointer handlers to the marker zone above the track.
   * Double-click: add marker. Drag existing: move. Alt+click: remove.
   * Drag from double-click: create range.
   */
  function attachMarkerZone(mzoneEl) {
    let isDragging = false;
    let dragInfo = null;
    let isCreatingRange = false;
    let rangeAnchorFrame = null;
    let rangeAnchorIndex = null;
    let lastClickTime = 0;
    let lastClickFrame = -1;

    function frameFromMzone(e) {
      const r = mzoneEl.getBoundingClientRect();
      return Math.round(clamp((e.clientX - r.left) / r.width, 0, 1) * Math.max(0, totalFrames - 1));
    }

    mzoneEl.addEventListener('pointerdown', (e) => {
      if (selectionMode !== 'list') return;
      const f = frameFromMzone(e);
      const now = Date.now();
      const hit = hitTestMarker(e, mzoneEl);

      // Alt+click: remove marker
      if (e.altKey) {
        if (hit) { markers.splice(hit.markerIndex, 1); commitMarkers(); }
        return;
      }

      // Click on existing marker: start drag
      if (hit) {
        mzoneEl.setPointerCapture(e.pointerId);
        isDragging = true;
        dragInfo = { ...hit, dragStartFrame: f + 1 };
        return;
      }

      // Double-click detection
      const isDblClick = (now - lastClickTime < DBLCLICK_MS) && (Math.abs(f - lastClickFrame) <= 2);

      if (isDblClick) {
        lastClickTime = 0; lastClickFrame = -1;
        mzoneEl.setPointerCapture(e.pointerId);
        isDragging = true;
        isCreatingRange = true;
        rangeAnchorFrame = f + 1;
        markers.push({ type: 'single', frame: rangeAnchorFrame });
        rangeAnchorIndex = markers.length - 1;
        syncTimeline();
        return;
      }

      lastClickTime = now;
      lastClickFrame = f;
    });

    mzoneEl.addEventListener('pointermove', (e) => {
      if (!isDragging) return;
      const f = frameFromMzone(e);

      if (dragInfo) {
        const m = markers[dragInfo.markerIndex];
        if (!m) return;
        const newFrame = clamp(f + 1, 1, totalFrames);

        if (dragInfo.edge === 'fill' && m.type === 'range') {
          const delta = newFrame - dragInfo.dragStartFrame;
          const rangeLen = m.end - m.start;
          let newStart = m.start + delta, newEnd = m.end + delta;
          if (newStart < 1) { newStart = 1; newEnd = 1 + rangeLen; }
          if (newEnd > totalFrames) { newEnd = totalFrames; newStart = totalFrames - rangeLen; }
          m.start = newStart; m.end = newEnd;
          dragInfo.dragStartFrame = newFrame;
        } else if (m.type === 'single') {
          m.frame = newFrame;
        } else if (dragInfo.edge === 'start') {
          m.start = Math.min(newFrame, m.end);
        } else if (dragInfo.edge === 'end') {
          m.end = Math.max(newFrame, m.start);
        }
        if (m.type === 'range' && m.start === m.end) markers[dragInfo.markerIndex] = { type: 'single', frame: m.start };
        syncTimeline();
        return;
      }

      if (isCreatingRange && rangeAnchorIndex !== null) {
        const currentFrame = clamp(f + 1, 1, totalFrames);
        const s = Math.min(rangeAnchorFrame, currentFrame);
        const en = Math.max(rangeAnchorFrame, currentFrame);
        markers[rangeAnchorIndex] = s === en ? { type: 'single', frame: s } : { type: 'range', start: s, end: en };
        syncTimeline();
      }
    });

    function endInteraction() {
      if (isDragging && (dragInfo || isCreatingRange)) {
        markers = mergeOverlapping(markers);
        commitMarkers();
      }
      isDragging = false; dragInfo = null;
      isCreatingRange = false; rangeAnchorFrame = null; rangeAnchorIndex = null;
    }

    mzoneEl.addEventListener('pointerup',     endInteraction);
    mzoneEl.addEventListener('pointercancel', endInteraction);
  }

  attachSeekZone(tlTrack);
  attachSeekZone(dtlTrack);
  attachMarkerZone(tlMzone);
  attachMarkerZone(dtlMzone);

  /* ── Video events ───────────────────────────────────────────────────────── */

  video.addEventListener('loadedmetadata', () => {
    const d = video.duration;
    if (!Number.isFinite(d) || d <= 0) return;
    duration = d; totalFrames = computeTotal(); isLoaded = true;
    overlay.style.display = 'none'; updateUi();
  });

  video.addEventListener('durationchange', () => {
    const d = video.duration;
    if (Number.isFinite(d) && d > 0) { duration = d; totalFrames = computeTotal(); updateUi(); }
  });

  video.addEventListener('canplay', () => { isLoaded = true; overlay.style.display = 'none'; updateUi(); });

  video.addEventListener('timeupdate', () => {
    if (!nativeFps || totalFrames === 0) return;
    const max = Math.max(0, totalFrames - 1);
    frameIndex = clamp(Math.round(video.currentTime * nativeFps), 0, max);
    mFrameInput.value = String(frameIndex); dFrameInput.value = String(frameIndex);
    setTimecode(video.currentTime, duration);
    syncSeeker(); saveSession();
  });

  video.addEventListener('seeked', () => updateUi());
  video.addEventListener('ended',  () => { showPlayState(null); updateUi(); });

  video.addEventListener('error', () => {
    isLoaded = false;
    const code = video.error?.code;
    const codeMsg = code ? ` (MediaError ${code})` : '';
    overlay.textContent = `Video could not be loaded${codeMsg}.`;
    overlay.style.display = 'flex'; updateUi();
  });

  /* ── FPS input ──────────────────────────────────────────────────────────── */

  function commitFps(inputEl, mirrorEl) {
    const v = Number(inputEl.value);
    if (Number.isFinite(v) && v >= MIN_FPS && v <= MAX_FPS) { fps = v; mirrorEl.value = String(fps); }
    else inputEl.value = String(fps);
  }
  mFpsInput.addEventListener('change', () => commitFps(mFpsInput, dFpsInput));
  mFpsInput.addEventListener('keydown', (e) => e.key === 'Enter' && commitFps(mFpsInput, dFpsInput));
  dFpsInput.addEventListener('change', () => commitFps(dFpsInput, mFpsInput));
  dFpsInput.addEventListener('keydown', (e) => e.key === 'Enter' && commitFps(dFpsInput, mFpsInput));

  /* ── Frame input ────────────────────────────────────────────────────────── */

  function commitFrame(inputEl) {
    const n = parseInt(inputEl.value, 10);
    if (Number.isFinite(n)) seekToFrame(n); else inputEl.value = String(frameIndex);
  }
  mFrameInput.addEventListener('input',   () => commitFrame(mFrameInput));
  mFrameInput.addEventListener('keydown', (e) => e.key === 'Enter' && commitFrame(mFrameInput));
  dFrameInput.addEventListener('input',   () => commitFrame(dFrameInput));
  dFrameInput.addEventListener('keydown', (e) => e.key === 'Enter' && commitFrame(dFrameInput));

  /* ── Controls ───────────────────────────────────────────────────────────── */

  mStepBack.addEventListener('click', () => stepFrame(-1));
  mPlayBack.addEventListener('click',  playBackward);
  mPauseBack.addEventListener('click', pause);
  mPlayFwd.addEventListener('click',   playForward);
  mPauseFwd.addEventListener('click',  pause);
  mStepFwd.addEventListener('click',   () => stepFrame(1));

  dStepBack.addEventListener('click',  () => stepFrame(-1));
  dPlayBack.addEventListener('click',  playBackward);
  dPauseBack.addEventListener('click', pause);
  dPlayFwd.addEventListener('click',   playForward);
  dPauseFwd.addEventListener('click',  pause);
  dStepFwd.addEventListener('click',  () => stepFrame(1));

  /* ── Clear markers ─────────────────────────────────────────────────────── */

  function clearAllMarkers() {
    if (markers.length === 0) return;
    markers = []; commitMarkers();
  }
  mClearBtn.addEventListener('click', clearAllMarkers);
  dClearBtn.addEventListener('click', clearAllMarkers);

  /* ── Dialog open/close ─────────────────────────────────────────────────── */

  enlargeBtn.addEventListener('click', openDialog);
  dialogClose.addEventListener('click', closeDialog);
  backdrop.addEventListener('click', (e) => { if (e.target === backdrop) closeDialog(); });

  /* ── Prevent node drag ─────────────────────────────────────────────────── */

  const root = container.firstElementChild;
  root.addEventListener('pointerdown', (e) => e.stopPropagation());
  root.addEventListener('mousedown',   (e) => e.stopPropagation());
  if (disabled) root.style.opacity = '0.8';

  /* ── Resize observer (re-snap px positions when width changes) ─────────── */

  const resizeObs = new ResizeObserver(() => syncTimeline());
  resizeObs.observe(tl);

  /* ── Init ───────────────────────────────────────────────────────────────── */

  initFromProps();
  updateUi();
  startValuePoll();

  /* ── Cleanup ────────────────────────────────────────────────────────────── */

  return () => {
    stopPlayback(); stopMirror(); closeDialog(); backdrop.remove();
    resizeObs.disconnect();
    if (pollId) clearInterval(pollId);
    saveSession();
  };
}
