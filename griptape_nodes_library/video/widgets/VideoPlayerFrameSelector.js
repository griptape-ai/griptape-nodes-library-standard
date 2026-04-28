/**
 * VideoPlayerWidget — vanilla JS
 * Controls: step back | play back | pause | play forward | step forward
 * Timeline: click to seek, Shift+click to add/remove a marker
 * Enlarge button opens a fullscreen dialog with a canvas mirror of the video.
 */

const SESSIONS = new Map();

/* ─── Icons ──────────────────────────────────────────────────────────────── */

const _svg = (d) =>
  `<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="currentColor"><path d="${d}"/></svg>`;

const ICONS = {
  stepBack:  _svg('M16 5L4 12l12 7zM18 5h2v14h-2z'),
  playBack:  _svg('M16 5v14L5 12z'),
  pause:     _svg('M6 19h4V5H6v14zm8-14v14h4V5h-4z'),
  playFwd:   _svg('M8 5v14l11-7z'),
  stepFwd:   _svg('M4 5h2v14H4zm4 0l12 7-12 7z'),
  enlarge:   `<svg xmlns="http://www.w3.org/2000/svg" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="15 3 21 3 21 9"/><polyline points="9 21 3 21 3 15"/><line x1="21" y1="3" x2="14" y2="10"/><line x1="3" y1="21" x2="10" y2="14"/></svg>`,
};

/* ─── CSS (injected once into document.head) ─────────────────────────────── */

const CSS_ID = 'vpw-global-styles';
const CSS = `
  .vpw-btn {
    background:none; border:none;
    color:#555; width:34px; height:34px;
    display:flex; align-items:center; justify-content:center;
    cursor:pointer; flex-shrink:0; border-radius:4px;
    transition:color .12s;
  }
  .vpw-btn:hover:not([disabled]) { color:#bbb; }
  .vpw-btn:active:not([disabled]) { color:#fff; }
  .vpw-btn[disabled] { opacity:.2; cursor:not-allowed; pointer-events:none; }
  .vpw-timecode {
    color:#666; font-size:11px; padding:0 10px; white-space:nowrap;
    font-variant-numeric:tabular-nums;
    font-family:'SF Mono','Monaco','Menlo',monospace;
    text-align:center;
  }
  .vpw-num {
    background:#1a1a1a; border:1px solid #333; border-radius:5px;
    color:#ccc; padding:4px 8px; font-size:11px; text-align:center;
  }
  .vpw-num:focus { outline:none; border-color:#4a7aaa; }
  .vpw-num[disabled] { opacity:.3; }
  .vpw-timeline {
    position:relative; height:36px;
    cursor:pointer; user-select:none; touch-action:none;
  }
  .vpw-tl-track {
    position:absolute; bottom:10px; left:0; right:0; height:3px;
    background:#252525; border-radius:2px; pointer-events:none;
  }
  .vpw-tl-fill {
    position:absolute; bottom:10px; left:0; height:3px;
    background:#2d5a8a; border-radius:2px; pointer-events:none;
  }
  .vpw-tl-seeker {
    position:absolute; bottom:8px; width:1px; height:calc(100% - 8px);
    background:#777; transform:translateX(-50%); pointer-events:none; z-index:3;
  }
  .vpw-tl-seeker::after {
    content:''; position:absolute; bottom:-1px; left:50%;
    width:9px; height:9px; background:#bbb; border:1px solid #444;
    transform:translateX(-50%) rotate(45deg); border-radius:1px;
  }
  .vpw-tl-marker {
    position:absolute; bottom:8px;
    width:9px; height:9px; background:#c07800;
    transform:translateX(-50%) rotate(45deg);
    pointer-events:none; z-index:2; border-radius:1px;
  }
  /* Dialog */
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
    background:#222; border:1px solid #444; border-radius:6px;
    color:#aaa; width:28px; height:28px;
    display:flex; align-items:center; justify-content:center;
    cursor:pointer; font-size:15px;
  }
  .vpw-dialog-close:hover { background:#333; color:#eee; }
  .vpw-dialog-canvas-wrap {
    background:#000; border-radius:8px; overflow:hidden;
    min-height:180px; display:flex; align-items:center; justify-content:center;
  }
  .vpw-dialog-canvas { display:block; width:100%; height:auto; max-height:78vh; }
  .vpw-dialog-placeholder { color:#3a3a3a; font-size:13px; }
`;

function injectGlobalStyles() {
  if (document.getElementById(CSS_ID)) return;
  const el = document.createElement('style');
  el.id = CSS_ID;
  el.textContent = CSS;
  document.head.appendChild(el);
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
    <div class="vpw-timeline nowheel" data-tl="${id}" title="Drag to seek · Shift+click to add/remove marker">
      <div class="vpw-tl-track"></div>
      <div class="vpw-tl-fill"   data-tl-fill="${id}"></div>
      <div class="vpw-tl-seeker" data-tl-seeker="${id}"></div>
    </div>`;
}

function bottomHtml(pfx, initFps) {
  return `
    <div style="display:flex;justify-content:center;align-items:center;gap:12px;">
      <label style="display:flex;align-items:center;gap:6px;color:#666;">
        FPS <input class="${pfx}fps vpw-num" type="number" min="1" max="240" step="0.001" value="${initFps}" style="width:58px;">
      </label>
      <label style="display:flex;align-items:center;gap:6px;color:#666;">
        Frame <input class="${pfx}frame vpw-num" type="number" min="0" value="0" style="width:72px;">
      </label>
      <span class="${pfx}total" style="color:#484848;">/ 0</span>
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
    fps: null, nativeFps: null, frameIndex: 0, currentTime: 0, markedFrames: [],
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

  /* ── Main HTML ──────────────────────────────────────────────────────────── */

  container.innerHTML = `
    <div class="nodrag" style="
      display:flex;flex-direction:column;gap:10px;
      background:#111;border-radius:10px;padding:10px;color:#ddd;
      font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;font-size:12px;">

      <div style="display:flex;justify-content:flex-end;">
        <button class="vpw-enlarge" style="background:none;border:1px solid #2a4560;border-radius:6px;color:#5090c0;padding:4px 8px;cursor:pointer;display:flex;align-items:center;gap:4px;font-size:11px;white-space:nowrap;">
          ${ICONS.enlarge} Enlarge
        </button>
      </div>

      <div style="position:relative;background:#000;border-radius:8px;overflow:hidden;height:${vpH}px;">
        <video class="vpw-video" style="width:100%;height:100%;object-fit:contain;" playsinline preload="metadata"
          ${session.videoSrc ? `src="${session.videoSrc}"` : ''}></video>
        <div class="vpw-overlay" style="position:absolute;inset:0;display:${session.videoSrc ? 'none' : 'flex'};align-items:center;justify-content:center;color:#3a3a3a;font-size:13px;pointer-events:none;">
          No video loaded
        </div>
      </div>

      ${timelineHtml('main')}
      ${controlsHtml('vpw-m-')}
      ${bottomHtml('vpw-m-', initFps)}
    </div>`;

  /* ── Dialog HTML (appended to body to escape stacking context) ──────────── */

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

  const enlargeBtn  = q('.vpw-enlarge');
  const video       = q('.vpw-video');
  const overlay     = q('.vpw-overlay');

  const tl          = q('[data-tl="main"]');
  const tlFill      = q('[data-tl-fill="main"]');
  const tlSeeker    = q('[data-tl-seeker="main"]');

  const mStepBack   = q('.vpw-m-step-back');
  const mPlayBack   = q('.vpw-m-play-back');
  const mPauseBack  = q('.vpw-m-pause-back');
  const mPlayFwd    = q('.vpw-m-play-fwd');
  const mPauseFwd   = q('.vpw-m-pause-fwd');
  const mStepFwd    = q('.vpw-m-step-fwd');
  const mFpsInput   = q('.vpw-m-fps');
  const mFrameInput = q('.vpw-m-frame');
  const mTotalEl    = q('.vpw-m-total');

  const dialogTitleEl = qd('.vpw-dialog-title');
  const dialogClose   = qd('.vpw-dialog-close');
  const dialogCanvas  = qd('.vpw-dcanvas');
  const dNoVideo      = qd('.vpw-dno-video');

  const dtl          = qd('[data-tl="dialog"]');
  const dtlFill      = qd('[data-tl-fill="dialog"]');
  const dtlSeeker    = qd('[data-tl-seeker="dialog"]');

  const dStepBack    = qd('.vpw-d-step-back');
  const dPlayBack    = qd('.vpw-d-play-back');
  const dPauseBack   = qd('.vpw-d-pause-back');
  const dPlayFwd     = qd('.vpw-d-play-fwd');
  const dPauseFwd    = qd('.vpw-d-pause-fwd');
  const dStepFwd     = qd('.vpw-d-step-fwd');
  const dFpsInput    = qd('.vpw-d-fps');
  const dFrameInput  = qd('.vpw-d-frame');
  const dTotalEl     = qd('.vpw-d-total');
  const mTimecode    = q('.vpw-m-timecode');
  const dTimecode    = qd('.vpw-d-timecode');

  /* ── State ──────────────────────────────────────────────────────────────── */

  let nativeFps    = Number.isFinite(Number(session.nativeFps)) && Number(session.nativeFps) > 0
                     ? Number(session.nativeFps) : initFps;
  let fps          = initFps;
  let duration     = 0;
  let totalFrames  = 0;
  let frameIndex   = 0;
  let markedFrames = Array.isArray(session.markedFrames) ? [...session.markedFrames] : [];
  let videoName    = '';
  let isLoaded     = false;
  let backRafId    = null;
  let canvasRafId  = null;
  let currentSrc   = null;
  let pollId       = null;

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
    session.fps          = fps;
    session.nativeFps    = nativeFps;
    session.frameIndex   = frameIndex;
    session.currentTime  = Number.isFinite(video.currentTime) ? video.currentTime : 0;
    session.markedFrames = [...markedFrames];
  }

  function renderMarkers(tlEl) {
    tlEl.querySelectorAll('.vpw-tl-marker').forEach(m => m.remove());
    if (totalFrames <= 1) return;
    const max = totalFrames - 1;
    for (const f of markedFrames) {
      const m = document.createElement('div');
      m.className = 'vpw-tl-marker';
      m.style.left = `${(f / max) * 100}%`;
      tlEl.appendChild(m);
    }
  }

  function syncTimeline() {
    const pct = totalFrames > 1 ? (frameIndex / (totalFrames - 1)) * 100 : 0;
    tlFill.style.width   = `${pct}%`;
    tlSeeker.style.left  = `${pct}%`;
    dtlFill.style.width  = `${pct}%`;
    dtlSeeker.style.left = `${pct}%`;
    renderMarkers(tl);
    renderMarkers(dtl);
  }

  function updateUi() {
    totalFrames = computeTotal();
    frameIndex  = clamp(frameIndex, 0, Math.max(0, totalFrames - 1));

    const maxF   = Math.max(0, totalFrames - 1);
    const canAct = isLoaded && totalFrames > 0 && !disabled;

    for (const b of [mStepBack, mPlayBack, mPauseBack, mPlayFwd, mPauseFwd, mStepFwd,
                      dStepBack, dPlayBack, dPauseBack, dPlayFwd, dPauseFwd, dStepFwd])
      b.disabled = !canAct;
    for (const inp of [mFrameInput, dFrameInput]) { inp.disabled = !canAct; inp.max = String(maxF); }
    for (const inp of [mFpsInput, dFpsInput]) inp.disabled = !!disabled;

    mFpsInput.value    = dFpsInput.value    = String(fps);
    mFrameInput.value  = dFrameInput.value  = String(frameIndex);
    mTotalEl.textContent = dTotalEl.textContent = `/ ${maxF}`;
    dialogTitleEl.textContent = videoName || 'Video Player';
    setTimecode(Number.isFinite(video.currentTime) ? video.currentTime : 0, duration);

    syncTimeline();
    saveSession();
  }

  /* ── Seek / playback ────────────────────────────────────────────────────── */

  function seekToFrame(idx) {
    const max = Math.max(0, totalFrames - 1);
    frameIndex = clamp(Math.round(idx), 0, max);
    if (isLoaded && nativeFps > 0) video.currentTime = frameIndex / nativeFps;
    updateUi();
  }

  function stopBackward() {
    if (backRafId) { cancelAnimationFrame(backRafId); backRafId = null; }
  }

  function showPlayState(direction) {
    // direction: 'back', 'fwd', or null (paused/stopped)
    for (const [play, pause] of [[mPlayBack, mPauseBack], [dPlayBack, dPauseBack]]) {
      play.style.display  = direction === 'back' ? 'none' : '';
      pause.style.display = direction === 'back' ? '' : 'none';
    }
    for (const [play, pause] of [[mPlayFwd, mPauseFwd], [dPlayFwd, dPauseFwd]]) {
      play.style.display  = direction === 'fwd' ? 'none' : '';
      pause.style.display = direction === 'fwd' ? '' : 'none';
    }
  }

  function playBackward() {
    if (!isLoaded) return;
    video.pause();
    stopBackward();
    showPlayState('back');
    const frameMs = 1000 / fps;
    let last = null;
    function step(ts) {
      if (last === null) { last = ts; backRafId = requestAnimationFrame(step); return; }
      if (ts - last >= frameMs) {
        const next = video.currentTime - 1 / nativeFps;
        if (next <= 0) { video.currentTime = 0; frameIndex = 0; showPlayState(null); updateUi(); return; }
        video.currentTime = next;
        last = ts;
      }
      backRafId = requestAnimationFrame(step);
    }
    backRafId = requestAnimationFrame(step);
  }

  function pause() { stopBackward(); video.pause(); showPlayState(null); }
  function playForward() {
    if (!isLoaded) return;
    video.pause();
    stopBackward();
    showPlayState('fwd');
    const frameMs = 1000 / fps;
    let last = null;
    function step(ts) {
      if (last === null) { last = ts; backRafId = requestAnimationFrame(step); return; }
      if (ts - last >= frameMs) {
        const next = video.currentTime + 1 / nativeFps;
        if (next >= duration) { video.currentTime = duration; frameIndex = Math.max(0, totalFrames - 1); showPlayState(null); updateUi(); return; }
        video.currentTime = next;
        last = ts;
      }
      backRafId = requestAnimationFrame(step);
    }
    backRafId = requestAnimationFrame(step);
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
        if (!hasDrawn) {
          dialogCanvas.style.display = 'block';
          dNoVideo.style.display     = 'none';
          hasDrawn = true;
        }
      } else if (!hasDrawn) {
        dialogCanvas.style.display = 'none';
        dNoVideo.style.display     = 'block';
      }
      canvasRafId = requestAnimationFrame(draw);
    }
    canvasRafId = requestAnimationFrame(draw);
  }

  function stopMirror() {
    if (canvasRafId) { cancelAnimationFrame(canvasRafId); canvasRafId = null; }
  }

  function openDialog()  { backdrop.style.display = 'flex'; startMirror(); }
  function closeDialog() { backdrop.style.display = 'none'; stopMirror(); }

  /* ── Video loading ──────────────────────────────────────────────────────── */

  // Read the latest `value` prop from GTN's React fiber tree without remounting.
  // GTN updates propsRef.current on every render (SetParameterValueResultSuccess),
  // but the widget mounts once. This traverses up to the WidgetLoader fiber to
  // read the current value after GTN processes a connection or file selection.
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
    } catch { /* ignore: different React version or internal change */ }
    return undefined;
  }

  function urlBase(u) { return u ? u.split('?')[0] : ''; }

  function loadUrl(url, restoreTime = 0) {
    if (urlBase(url) === urlBase(currentSrc) && isLoaded) return;
    currentSrc = url;
    session.videoSrc = url;
    videoName  = url.split('/').pop()?.split('?')[0] || 'video';
    isLoaded   = false;
    overlay.textContent = 'Loading…';
    overlay.style.display = 'flex';
    video.src = url;
    video.load();
    if (restoreTime > 0) {
      video.addEventListener('loadedmetadata', () => { video.currentTime = restoreTime; }, { once: true });
    }
    updateUi();
  }

  function initFromProps() {
    frameIndex   = session.frameIndex || 0;
    markedFrames = Array.isArray(session.markedFrames) ? [...session.markedFrames] : [];

    // Remount: session already has the video src baked into the HTML.
    // Just restore the time position — no reload needed.
    if (session.videoSrc && urlBase(session.videoSrc) === urlBase(video.src)) {
      currentSrc = session.videoSrc;
      videoName  = session.videoSrc.split('/').pop()?.split('?')[0] || 'video';
      if (session.currentTime > 0) {
        video.addEventListener('loadedmetadata', () => { video.currentTime = session.currentTime; }, { once: true });
      }
      return;
    }

    // First mount: load from props.value
    const url = typeof value === 'string' && value.trim() ? value.trim() : null;
    if (url) loadUrl(url, session.currentTime || 0);
  }

  function clearVideo() {
    pause();
    video.removeAttribute('src');
    video.load();
    currentSrc   = null;
    session.videoSrc = null;
    isLoaded     = false;
    videoName    = '';
    duration     = 0;
    totalFrames  = 0;
    frameIndex   = 0;
    overlay.textContent = 'No video loaded';
    overlay.style.display = 'flex';
    updateUi();
  }

  function startValuePoll() {
    pollId = setInterval(() => {
      const latestValue = readLatestValueFromFiber();
      if (latestValue === undefined) return;
      const latestUrl = typeof latestValue === 'string' && latestValue.trim() ? latestValue.trim() : null;
      if (latestUrl === null && currentSrc !== null) { clearVideo(); return; }
      if (!latestUrl || urlBase(latestUrl) === urlBase(currentSrc)) return;
      frameIndex   = 0;
      markedFrames = [];
      loadUrl(latestUrl, 0);
    }, 500);
  }

  /* ── Timeline interaction ───────────────────────────────────────────────── */

  function frameFromEvent(e, el) {
    const r = el.getBoundingClientRect();
    return Math.round(clamp((e.clientX - r.left) / r.width, 0, 1) * Math.max(0, totalFrames - 1));
  }

  function attachTimeline(tlEl) {
    let dragging = false;
    tlEl.addEventListener('pointerdown', (e) => {
      tlEl.setPointerCapture(e.pointerId);
      dragging = true;
      const f = frameFromEvent(e, tlEl);
      if (e.shiftKey) {
        const i = markedFrames.indexOf(f);
        if (i >= 0) markedFrames.splice(i, 1);
        else { markedFrames.push(f); markedFrames.sort((a, b) => a - b); }
        syncTimeline();
      } else {
        seekToFrame(f);
      }
    });
    tlEl.addEventListener('pointermove', (e) => {
      if (!dragging || e.shiftKey) return;
      seekToFrame(frameFromEvent(e, tlEl));
    });
    tlEl.addEventListener('pointerup',     () => { dragging = false; });
    tlEl.addEventListener('pointercancel', () => { dragging = false; });
  }

  attachTimeline(tl);
  attachTimeline(dtl);

  /* ── Video events ───────────────────────────────────────────────────────── */

  video.addEventListener('loadedmetadata', () => {
    const d = video.duration;
    if (!Number.isFinite(d) || d <= 0) return;
    duration    = d;
    totalFrames = computeTotal();
    isLoaded    = true;
    overlay.style.display = 'none';
    updateUi();
  });

  video.addEventListener('durationchange', () => {
    const d = video.duration;
    if (Number.isFinite(d) && d > 0) { duration = d; totalFrames = computeTotal(); updateUi(); }
  });

  video.addEventListener('canplay', () => {
    isLoaded = true;
    overlay.style.display = 'none';
    updateUi();
  });

  video.addEventListener('timeupdate', () => {
    if (!nativeFps || totalFrames === 0) return;
    const max = Math.max(0, totalFrames - 1);
    frameIndex = clamp(Math.round(video.currentTime * nativeFps), 0, max);
    // Fast path: skip full updateUi during playback
    const pct = totalFrames > 1 ? (frameIndex / max) * 100 : 0;
    tlFill.style.width    = `${pct}%`;
    tlSeeker.style.left   = `${pct}%`;
    dtlFill.style.width   = `${pct}%`;
    dtlSeeker.style.left  = `${pct}%`;
    mFrameInput.value     = String(frameIndex);
    dFrameInput.value     = String(frameIndex);
    setTimecode(video.currentTime, duration);
    saveSession();
  });

  video.addEventListener('seeked', () => updateUi());
  video.addEventListener('ended',  () => { showPlayState(null); updateUi(); });

  video.addEventListener('error', () => {
    isLoaded = false;
    const code = video.error?.code;
    // MediaError codes: 1=aborted 2=network 3=decode 4=src_not_supported
    const codeMsg = code ? ` (MediaError ${code})` : '';
    console.warn('[VideoPlayerWidget] video error', code, video.error?.message, 'src:', video.src);
    overlay.textContent = `Video could not be loaded${codeMsg}. Check console for URL details.`;
    overlay.style.display = 'flex';
    updateUi();
  });

  /* ── FPS input ──────────────────────────────────────────────────────────── */

  function commitFps(inputEl, mirrorEl) {
    const v = Number(inputEl.value);
    if (Number.isFinite(v) && v >= MIN_FPS && v <= MAX_FPS) {
      fps = v;
      mirrorEl.value = String(fps);
    } else {
      inputEl.value = String(fps);
    }
  }

  mFpsInput.addEventListener('change', () => commitFps(mFpsInput, dFpsInput));
  mFpsInput.addEventListener('keydown', (e) => e.key === 'Enter' && commitFps(mFpsInput, dFpsInput));
  dFpsInput.addEventListener('change', () => commitFps(dFpsInput, mFpsInput));
  dFpsInput.addEventListener('keydown', (e) => e.key === 'Enter' && commitFps(dFpsInput, mFpsInput));

  /* ── Frame input ────────────────────────────────────────────────────────── */

  function commitFrame(inputEl) {
    const n = parseInt(inputEl.value, 10);
    if (Number.isFinite(n)) seekToFrame(n);
    else inputEl.value = String(frameIndex);
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

  /* ── Dialog open/close ──────────────────────────────────────────────────── */

  enlargeBtn.addEventListener('click', openDialog);
  dialogClose.addEventListener('click', closeDialog);
  backdrop.addEventListener('click', (e) => { if (e.target === backdrop) closeDialog(); });

  /* ── Prevent node drag while using widget ───────────────────────────────── */

  const root = container.firstElementChild;
  root.addEventListener('pointerdown', (e) => e.stopPropagation());
  root.addEventListener('mousedown',   (e) => e.stopPropagation());

  if (disabled) root.style.opacity = '0.8';

  /* ── Init ───────────────────────────────────────────────────────────────── */

  initFromProps();
  updateUi();
  startValuePoll();

  /* ── Cleanup ────────────────────────────────────────────────────────────── */

  return () => {
    stopBackward();
    stopMirror();
    closeDialog();
    backdrop.remove();
    if (pollId) clearInterval(pollId);
    saveSession();
  };
}
