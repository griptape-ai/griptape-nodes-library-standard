// TrimAudioEditor: audio player with a peak waveform (the vsl-gui AudioWidget look) and draggable in/out
// handles. The trim points are the synced parameters `start_time` and `end_time`. An end time of 0 means the
// end of the clip.

const WIDGET_VERSION = "0.2.0";
const WAVEFORM_HEIGHT = 64;
const SAMPLES = 256;
const MIN_REGION = 0.1;
const HIT_TOLERANCE = 8; // px either side of a marker line, as in the Extract Frames timeline
const SCRUB_START_PX = 3; // a press that moves less than this is a click-to-seek, which stays silent
const FLAG_W = 4;
const FLAG_H = 7;
const MOVE_H = 10; // height of the strip along the bottom of the selection that drags it as a whole
const IDLE_COLOR = "#aaa";
const ACTIVE_COLOR = "#d4a030";
const MAX_DECODE_BYTES = 150 * 1024 * 1024; // decoding holds the whole clip in memory, so very long clips skip it
const END_EPSILON = 0.001;

const round3 = (t) => Math.round(t * 1000) / 1000;

function fmt(t) {
  if (!isFinite(t) || t < 0) t = 0;
  const cs = Math.round(t * 100);
  const m = Math.floor(cs / 6000);
  return `${m}:${((cs - m * 6000) / 100).toFixed(2).padStart(5, "0")}`;
}

// Reduce decoded audio to SAMPLES peaks, 0..1.
function computePeaks(buf) {
  const peaks = new Float32Array(SAMPLES);
  const bucket = Math.max(1, Math.floor(buf.length / SAMPLES));
  let loudest = 0;
  for (let c = 0; c < buf.numberOfChannels; c++) {
    const data = buf.getChannelData(c);
    for (let i = 0; i < SAMPLES; i++) {
      let p = peaks[i];
      for (let s = i * bucket, e = Math.min(data.length, s + bucket); s < e; s++) p = Math.max(p, Math.abs(data[s]));
      peaks[i] = p;
      loudest = Math.max(loudest, p);
    }
  }
  if (loudest > 0) for (let i = 0; i < SAMPLES; i++) peaks[i] /= loudest;
  return peaks;
}

export default function TrimAudioEditor(container, props) {
  if (container._instance?.wrapper?.isConnected) {
    container._instance.handleUpdate(props);
    return { cleanup: container._instance.cleanup, update: container._instance.handleUpdate };
  }

  let value = props.value || {};
  let onChange = props.onChange;
  let onLiveChange = props.onLiveChange;
  let locked = props.locked || [];
  let disabled = !!props.disabled;
  let url = "";
  let objectUrl = "";
  let peaks = null;
  let duration = 0;
  let start = 0;
  let end = 0;
  let drag = null; // "start" | "end" | "move" | "seek"
  let loadId = 0;
  let hovered = null;
  let raf = 0;
  let stopAtEnd = false; // playback pauses at the out point when it started before it
  let scrubWasPlaying = false;
  let scrubStarted = false;
  let downX = 0;
  let moveFrom = null;

  const audio = new Audio();
  audio.preload = "auto";

  // ── DOM ────────────────────────────────────────────────────────────────────
  const wrapper = document.createElement("div");
  wrapper.className = "nodrag nowheel";
  wrapper.style.cssText = "display:flex;flex-direction:column;gap:6px;width:100%;min-width:0;user-select:none;";

  const wave = document.createElement("div");
  wave.style.cssText = `position:relative;height:${WAVEFORM_HEIGHT}px;border-radius:6px;overflow:hidden;background:var(--accent);cursor:pointer;touch-action:none;`;
  const canvas = document.createElement("canvas");
  canvas.style.cssText = "position:absolute;inset:0;width:100%;height:100%;";
  wave.appendChild(canvas);

  // A 1px line with a small flag at the top, like the Extract Frames markers. The start flag points right
  // and the end flag points left, so both lean into the selection.
  const mkHandle = (side) => {
    const h = document.createElement("div");
    h.style.cssText = "position:absolute;top:0;bottom:0;width:0;pointer-events:none;";
    const line = document.createElement("div");
    line.style.cssText = "position:absolute;top:0;bottom:0;left:0;width:1px;";
    const flag = document.createElement("div");
    const edge = side === "start" ? "Left" : "Right";
    flag.style.cssText = `position:absolute;top:0;${side === "start" ? "left" : "right"}:0;width:0;height:0;border-${edge.toLowerCase()}:${FLAG_W}px solid;border-bottom:${FLAG_H}px solid transparent;`;
    h.append(line, flag);
    h._paint = (color) => { line.style.background = color; flag.style[`border${edge}Color`] = color; };
    h._paint(IDLE_COLOR);
    return h;
  };
  const startHandle = mkHandle("start");
  const endHandle = mkHandle("end");
  const playhead = document.createElement("div");
  playhead.style.cssText = "position:absolute;top:0;bottom:0;width:2px;background:var(--foreground);pointer-events:none;";
  const band = document.createElement("div");
  band.style.cssText = "position:absolute;top:0;bottom:0;pointer-events:none;background:rgba(170,170,170,0.2);";
  const moveStrip = document.createElement("div");
  moveStrip.style.cssText = `position:absolute;bottom:0;height:${MOVE_H}px;pointer-events:none;`;
  wave.append(band, moveStrip, startHandle, endHandle, playhead);

  const status = document.createElement("div");
  status.style.cssText = "position:absolute;inset:0;display:flex;align-items:center;justify-content:center;font-size:11px;color:var(--muted-foreground);pointer-events:none;";
  wave.appendChild(status);

  const controls = document.createElement("div");
  controls.style.cssText = "display:flex;align-items:center;gap:8px;font-size:12px;font-variant-numeric:tabular-nums;color:var(--foreground);";
  const playBtn = document.createElement("button");
  playBtn.style.cssText = "width:32px;height:32px;border-radius:9999px;border:1px solid var(--border);background:var(--muted);color:var(--foreground);cursor:pointer;flex-shrink:0;";
  const timeLabel = document.createElement("span");
  const rangeLabel = document.createElement("span");
  rangeLabel.style.cssText = "margin-left:auto;color:var(--muted-foreground);";
  const mkBtn = (label, title) => {
    const b = document.createElement("button");
    b.textContent = label;
    b.title = title;
    b.style.cssText = "height:24px;padding:0 8px;border-radius:6px;border:1px solid var(--border);background:var(--muted);color:var(--foreground);font-size:11px;cursor:pointer;";
    b.addEventListener("pointerdown", (e) => e.stopPropagation());
    return b;
  };
  const inBtn = mkBtn("Mark In", "Set the in point to the current time (I)");
  const outBtn = mkBtn("Mark Out", "Set the out point to the current time (O)");
  controls.append(playBtn, timeLabel, inBtn, outBtn, rangeLabel);

  wrapper.append(wave, controls);
  container.appendChild(wrapper);

  // ── Helpers ────────────────────────────────────────────────────────────────
  const isLocked = (name) => locked.includes(name);
  const t2x = (t) => (duration > 0 ? (t / duration) * wave.clientWidth : 0);
  const x2t = (clientX) => {
    const r = wave.getBoundingClientRect();
    return Math.min(duration, Math.max(0, ((clientX - r.left) / r.width) * duration));
  };

  // An out point at the end of the clip is sent as 0, so a clip swapped in later is not cut at this one's length.
  const trimValues = () => ({ start_time: start, end_time: end >= duration - END_EPSILON ? 0 : end });
  const commit = () => onChange({ ...value, ...trimValues() });

  function setStart(t) {
    start = round3(Math.max(0, Math.min(t, end - MIN_REGION)));
  }

  function setEnd(t) {
    end = round3(Math.min(duration, Math.max(t, start + MIN_REGION)));
  }

  function readParams() {
    start = Math.max(0, Number(value.start_time) || 0);
    const e = Number(value.end_time) || 0;
    end = e > 0 ? e : duration;
    if (duration > 0) {
      end = Math.min(end, duration);
      start = Math.min(start, Math.max(0, end - MIN_REGION));
    }
  }

  function draw() {
    const w = wave.clientWidth, h = WAVEFORM_HEIGHT;
    // The editor zooms nodes with a CSS transform; match the bitmap to the zoomed pixel density to stay sharp.
    const zoom = w > 0 ? wave.getBoundingClientRect().width / w : 1;
    const ratio = (window.devicePixelRatio || 1) * zoom;
    if (canvas.width !== Math.round(w * ratio)) canvas.width = Math.round(w * ratio);
    if (canvas.height !== Math.round(h * ratio)) canvas.height = Math.round(h * ratio);
    const ctx = canvas.getContext("2d");
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    ctx.clearRect(0, 0, w, h);
    const sx = t2x(start), ex = t2x(end);
    if (peaks && w > 0) {
      const cs = getComputedStyle(wrapper);
      const outside = cs.getPropertyValue("--muted-foreground").trim() || "#888";
      const inside = cs.getPropertyValue("--foreground").trim() || "#fff";
      const barW = 2, gap = 1, count = Math.max(1, Math.floor((w + gap) / (barW + gap)));
      for (let i = 0; i < count; i++) {
        const x = i * (barW + gap);
        const level = peaks[Math.min(SAMPLES - 1, Math.floor((i / count) * SAMPLES))];
        const bh = Math.max(2, level * h);
        const selected = x + barW > sx && x < ex;
        ctx.fillStyle = selected ? inside : outside;
        ctx.globalAlpha = selected ? 1 : 0.4;
        ctx.fillRect(x, (h - bh) / 2, barW, bh);
      }
      ctx.globalAlpha = 1;
    }
    startHandle.style.left = `${sx}px`;
    endHandle.style.left = `${ex}px`;
    band.style.left = `${sx}px`;
    band.style.width = `${Math.max(0, ex - sx)}px`;
    moveStrip.style.left = band.style.left;
    moveStrip.style.width = band.style.width;
    paintHandles();
    updatePlayhead();
    rangeLabel.textContent = peaks ? `In ${fmt(start)}  Out ${fmt(end)}  (${fmt(end - start)})` : "";
  }

  // The grabbed or hovered marker turns amber.
  function paintHandles() {
    const on = (name) => drag === name || (!drag && hovered === name);
    startHandle._paint(on("start") ? ACTIVE_COLOR : IDLE_COLOR);
    endHandle._paint(on("end") ? ACTIVE_COLOR : IDLE_COLOR);
    band.style.background = drag === "start" || drag === "end" || hovered ? "rgba(212,160,48,0.25)" : "rgba(170,170,170,0.2)";
    moveStrip.style.background = on("move") ? "rgba(212,160,48,0.6)" : "rgba(170,170,170,0.35)";
  }

  // The marker line within HIT_TOLERANCE of the pointer, nearest first, or the move strip: null | "start" | "end" | "move".
  function hitMarker(clientX, clientY) {
    // Pointer positions are screen px and t2x is layout px, so scale for the editor's zoom.
    const r = wave.getBoundingClientRect();
    const px = (clientX - r.left) * (wave.clientWidth / r.width);
    const ds = Math.abs(px - t2x(start)), de = Math.abs(px - t2x(end));
    const canStart = !disabled && !isLocked("start_time"), canEnd = !disabled && !isLocked("end_time");
    if (ds <= HIT_TOLERANCE && ds <= de && canStart) return "start";
    if (de <= HIT_TOLERANCE && canEnd) return "end";
    const inStrip = clientY - r.top >= r.height - MOVE_H * (r.height / wave.clientHeight);
    if (inStrip && px > t2x(start) && px < t2x(end) && canStart && canEnd) return "move";
    return null;
  }

  function updatePlayhead() {
    playhead.style.left = `${t2x(audio.currentTime) - 1}px`;
    timeLabel.textContent = `${fmt(audio.currentTime)} / ${fmt(duration)}`;
    playBtn.textContent = audio.paused ? "▶" : "❚❚";
  }

  // Playback that starts before the out point stops there and returns to the in point. Playback that starts
  // after it runs to the end of the clip, so a point past the selection can be heard and marked.
  function tick() {
    if (!audio.paused && stopAtEnd && drag !== "seek" && audio.currentTime >= end) {
      audio.pause();
      audio.currentTime = start;
    }
    updatePlayhead();
    if (!audio.paused) raf = requestAnimationFrame(tick);
  }

  function startPlayback() {
    stopAtEnd = audio.currentTime < end;
    audio.play().then(() => { cancelAnimationFrame(raf); tick(); }).catch(() => {});
  }

  playBtn.addEventListener("click", () => {
    if (!peaks) return;
    if (audio.paused) {
      if (audio.currentTime >= end) audio.currentTime = start;
      startPlayback();
    } else {
      audio.pause();
      updatePlayhead();
    }
  });
  playBtn.addEventListener("pointerdown", (e) => e.stopPropagation());

  // ── Mark in/out at the current time ────────────────────────────────────────
  function markPoint(which) {
    if (!peaks || disabled || isLocked(which === "in" ? "start_time" : "end_time")) return;
    if (which === "in") setStart(audio.currentTime);
    else setEnd(audio.currentTime);
    draw();
    commit();
  }
  inBtn.addEventListener("click", () => markPoint("in"));
  outBtn.addEventListener("click", () => markPoint("out"));
  wrapper.tabIndex = 0;
  wrapper.style.outline = "none";
  wrapper.addEventListener("keydown", (e) => {
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    const key = e.key.toLowerCase();
    if (key !== "i" && key !== "o") return;
    e.preventDefault();
    e.stopPropagation();
    markPoint(key === "i" ? "in" : "out");
  });

  // ── Dragging ───────────────────────────────────────────────────────────────
  function applyDrag(clientX) {
    const t = x2t(clientX);
    if (drag === "start") setStart(t);
    else if (drag === "end") setEnd(t);
    else if (drag === "move") {
      const len = moveFrom.end - moveFrom.start;
      start = round3(Math.min(duration - len, Math.max(0, moveFrom.start + t - moveFrom.t)));
      end = round3(start + len);
    } else {
      audio.currentTime = t;
      stopAtEnd = t < end;
      // Playing while scrubbing starts once the pointer has moved, so a plain click seeks without a blip.
      if (!scrubStarted && Math.abs(clientX - downX) >= SCRUB_START_PX) {
        scrubStarted = true;
        if (!scrubWasPlaying) startPlayback();
      }
    }
    if (drag !== "seek") onLiveChange?.(trimValues());
    draw();
  }

  wave.addEventListener("pointerdown", (e) => {
    e.stopPropagation();
    if (!peaks) return;
    drag = hitMarker(e.clientX, e.clientY) || "seek";
    wave.setPointerCapture(e.pointerId);
    wrapper.focus({ preventScroll: true });
    downX = e.clientX;
    if (drag === "move") moveFrom = { start, end, t: x2t(e.clientX) };
    if (drag === "seek") {
      // Dragging the waveform plays from the dragged position, as in the vsl-gui player. Letting go stops it
      // unless it was already playing.
      scrubWasPlaying = !audio.paused;
      scrubStarted = false;
    }
    applyDrag(e.clientX);
  });
  wave.addEventListener("pointermove", (e) => {
    if (drag) return applyDrag(e.clientX);
    const hit = peaks ? hitMarker(e.clientX, e.clientY) : null;
    if (hit !== hovered) { hovered = hit; wave.style.cursor = hit === "move" ? "grab" : hit ? "ew-resize" : "pointer"; paintHandles(); }
  });
  wave.addEventListener("pointerleave", () => { if (hovered) { hovered = null; paintHandles(); } });
  const endDrag = () => {
    if (!drag) return;
    const wasTrim = drag !== "seek";
    if (!wasTrim && !scrubWasPlaying) audio.pause();
    drag = null;
    paintHandles();
    if (wasTrim) commit();
  };
  wave.addEventListener("pointerup", endDrag);
  wave.addEventListener("pointercancel", endDrag);

  // ── Loading ────────────────────────────────────────────────────────────────
  function releaseObjectUrl() {
    if (objectUrl) URL.revokeObjectURL(objectUrl);
    objectUrl = "";
  }

  function metadataDuration() {
    return new Promise((resolve) => {
      if (isFinite(audio.duration) && audio.duration > 0) return resolve(audio.duration);
      audio.addEventListener("loadedmetadata", () => resolve(audio.duration), { once: true });
      audio.addEventListener("error", () => resolve(0), { once: true });
    });
  }

  // One download feeds both the waveform and the player.
  async function load(newUrl) {
    url = newUrl;
    const id = ++loadId;
    audio.pause();
    releaseObjectUrl();
    peaks = null;
    duration = 0;
    status.textContent = url ? "Loading…" : "Connect audio to trim";
    if (!url) { audio.removeAttribute("src"); audio.load(); readParams(); draw(); return; }
    try {
      const res = await fetch(url);
      if (!res.ok) throw new Error(`Could not load audio (${res.status})`);
      const bytes = await res.arrayBuffer();
      if (id !== loadId) return;
      objectUrl = URL.createObjectURL(new Blob([bytes], { type: res.headers.get("content-type") || "audio/mpeg" }));
      audio.src = objectUrl;
      if (bytes.byteLength > MAX_DECODE_BYTES) {
        duration = await metadataDuration();
        peaks = new Float32Array(SAMPLES).fill(0.15);
        status.textContent = "Clip too long to draw a waveform";
      } else {
        const buf = await new OfflineAudioContext(1, 1, 44100).decodeAudioData(bytes);
        peaks = computePeaks(buf);
        duration = buf.duration;
        status.textContent = "";
      }
      if (id !== loadId) return;
    } catch (err) {
      if (id !== loadId) return;
      status.textContent = String(err.message || err);
    }
    readParams();
    draw();
  }

  const ro = new ResizeObserver(() => draw());
  ro.observe(wave);

  function handleUpdate(newProps) {
    value = newProps.value || {};
    onChange = newProps.onChange;
    onLiveChange = newProps.onLiveChange;
    locked = newProps.locked || [];
    disabled = !!newProps.disabled;
    if ((value.audio_url || "") !== url) load(value.audio_url || "");
    else if (!drag) { readParams(); draw(); }
  }

  function cleanup() {
    cancelAnimationFrame(raf);
    ro.disconnect();
    loadId++;
    audio.pause();
    audio.removeAttribute("src");
    audio.load();
    releaseObjectUrl();
    wrapper.remove();
    delete container._instance;
  }

  container._instance = { handleUpdate, cleanup, wrapper, version: WIDGET_VERSION };
  load(value.audio_url || "");
  return { cleanup, update: handleUpdate };
}
