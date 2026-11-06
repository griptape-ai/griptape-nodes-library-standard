// Status bar for BoundingBoxEditor — live coordinate readout and lock notice.

export function createFooter({ version }) {
  const statusBar = document.createElement("div");
  statusBar.style.cssText = [
    "font-size:11px", "color:var(--muted-foreground)", "padding:4px 8px",
    "background:var(--muted)", "border-radius:4px",
    "display:flex", "justify-content:space-between", "align-items:center",
  ].join(";");
  const statusL = document.createElement("span");
  const statusR = document.createElement("span");
  statusR.style.cssText = "font-family:monospace;font-size:10px;color:var(--muted-foreground);opacity:0.6;";
  statusR.textContent = `v${version}`;
  statusBar.appendChild(statusL);
  statusBar.appendChild(statusR);

  function updateStatus({ canvasReady, lockedParams, left, top, width, height }) {
    if (!canvasReady) {
      statusL.textContent = "Set a canvas width and height";
      statusL.style.color = "";
      return;
    }
    const coordLocked = ["left", "top", "width", "height"].filter(f => lockedParams.includes(f));
    if (coordLocked.length > 0) {
      statusL.style.color = "#f59e0b";
      statusL.textContent = "\u{1F512} Connected: " + coordLocked.join(", ");
      return;
    }
    statusL.style.color = "";
    statusL.textContent = `[${Math.round(left)}, ${Math.round(top)}, ${Math.round(width)}, ${Math.round(height)}]`;
  }

  return { statusBar, updateStatus };
}
