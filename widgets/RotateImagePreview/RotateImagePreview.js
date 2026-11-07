// Display-only preview of what Rotate Image (and Extract PDF Images) will produce. The rotation is drawn on a canvas in the
// browser; the node rotates the real image with PIL when it runs.

const WIDGET_VERSION = "1.1.0";
const CHECKER = 10;

const QUARTER_TURNS = {
  "none": 0,
  "90° clockwise": 90,
  "180°": 180,
  "90° counter-clockwise": -90,
};

// Degrees clockwise and output size for the current settings, mirroring RotateImage._process_image.
function resolveRotation(v, w, h) {
  const turn = QUARTER_TURNS[v.rotation];
  if (turn !== undefined) {
    const swap = Math.abs(turn) === 90;
    return { degrees: turn, outW: swap ? h : w, outH: swap ? w : h, custom: false };
  }
  const degrees = Number(v.angle) || 0;
  if (v.expand_canvas === false) return { degrees, outW: w, outH: h, custom: true };
  const rad = degrees * Math.PI / 180;
  const cos = Math.abs(Math.cos(rad)), sin = Math.abs(Math.sin(rad));
  return {
    degrees,
    outW: Math.max(1, Math.round(w * cos + h * sin)),
    outH: Math.max(1, Math.round(w * sin + h * cos)),
    custom: true,
  };
}

function fmtDegrees(d) {
  const r = Math.round(d * 100) / 100;
  return (r > 0 ? "+" : "") + r + "°";
}

export default function RotateImagePreview(container, props) {
  if (container._rotatePreview?.wrapper?.isConnected) {
    container._rotatePreview.handleUpdate(props);
    return { cleanup: container._rotatePreview.cleanup, update: container._rotatePreview.handleUpdate };
  }
  if (container._rotatePreview) delete container._rotatePreview;

  let value = props.value || {};
  let imageUrl = "";
  let imageLoaded = false;

  const wrapper = document.createElement("div");
  wrapper.className = "rotate-image-preview nodrag nowheel";
  wrapper.style.cssText = "display:flex;flex-direction:column;gap:6px;height:100%;user-select:none;-webkit-user-select:none;";

  const canvasWrap = document.createElement("div");
  canvasWrap.style.cssText = [
    "position:relative", "background:#111", "border-radius:6px", "overflow:hidden",
    "flex:1 1 0", "min-height:180px",
    "display:flex", "align-items:center", "justify-content:center",
  ].join(";");

  const canvas = document.createElement("canvas");
  canvas.style.cssText = "display:block;";
  const ctx = canvas.getContext("2d");
  canvasWrap.appendChild(canvas);

  const statusBar = document.createElement("div");
  statusBar.style.cssText = [
    "font-size:11px", "color:var(--muted-foreground)", "padding:4px 8px",
    "background:var(--muted)", "border-radius:4px",
    "display:flex", "justify-content:space-between", "align-items:center",
  ].join(";");
  const statusL = document.createElement("span");
  const statusR = document.createElement("span");
  statusR.style.cssText = "font-family:monospace;font-size:10px;opacity:0.6;";
  statusR.textContent = `v${WIDGET_VERSION}`;
  statusBar.appendChild(statusL);
  statusBar.appendChild(statusR);

  wrapper.appendChild(canvasWrap);
  wrapper.appendChild(statusBar);
  container.appendChild(wrapper);

  const img = new Image();
  img.crossOrigin = "anonymous";
  img.onload = () => { imageLoaded = true; render(); };
  img.onerror = () => { imageLoaded = false; render(); };

  function drawChecker(w, h) {
    ctx.fillStyle = "#3a3a3a";
    ctx.fillRect(0, 0, w, h);
    ctx.fillStyle = "#4a4a4a";
    for (let y = 0; y < h; y += CHECKER) {
      for (let x = (y / CHECKER) % 2 ? 0 : CHECKER; x < w; x += CHECKER * 2) {
        ctx.fillRect(x, y, CHECKER, CHECKER);
      }
    }
  }

  function render() {
    const areaW = canvasWrap.clientWidth || 480;
    const areaH = canvasWrap.clientHeight || 360;

    if (!imageLoaded) {
      canvas.width = areaW; canvas.height = areaH;
      ctx.fillStyle = "#1a1a1a";
      ctx.fillRect(0, 0, areaW, areaH);
      ctx.fillStyle = "#555";
      ctx.font = "13px sans-serif";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText("Connect an image to preview the rotation", areaW / 2, areaH / 2);
      statusL.textContent = "No image connected";
      return;
    }

    // naturalWidth/Height already reflect EXIF orientation, which the node also applies.
    const w = img.naturalWidth, h = img.naturalHeight;
    const { degrees, outW, outH, custom } = resolveRotation(value, w, h);

    const scale = Math.min(areaW / outW, areaH / outH);
    canvas.width = Math.max(1, Math.round(outW * scale));
    canvas.height = Math.max(1, Math.round(outH * scale));

    drawChecker(canvas.width, canvas.height);
    if (custom) {
      ctx.fillStyle = value.background_color || "#00000000";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
    }

    ctx.save();
    ctx.translate(canvas.width / 2, canvas.height / 2);
    ctx.rotate(degrees * Math.PI / 180);
    ctx.drawImage(img, -w * scale / 2, -h * scale / 2, w * scale, h * scale);
    ctx.restore();

    const label = custom ? fmtDegrees(degrees) : value.rotation;
    statusL.textContent = `${label}  →  ${outW} × ${outH} px`;
  }

  function handleUpdate(newProps) {
    value = newProps.value || {};
    const base = (u) => (u ? u.split("?")[0] : "");
    const newUrl = value.image_url || "";
    if (base(newUrl) !== base(imageUrl)) {
      imageUrl = newUrl;
      imageLoaded = false;
      if (newUrl) img.src = newUrl;
      else render();
      return;
    }
    render();
  }

  const ro = new ResizeObserver(() => render());
  ro.observe(canvasWrap);

  handleUpdate(props);

  function cleanup() {
    ro.disconnect();
    wrapper.remove();
    delete container._rotatePreview;
  }

  container._rotatePreview = { handleUpdate, cleanup, wrapper };
  return { cleanup, update: handleUpdate };
}
