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

  function updateStatus({ canvasReady, lockedParams, x1, y1, x2, y2 }) {
    if (!canvasReady) {
      statusL.textContent = "Set a canvas width and height";
      statusL.style.color = "";
      return;
    }
    const coordLocked = ["x1", "y1", "x2", "y2"].filter(f => lockedParams.includes(f));
    if (coordLocked.length > 0) {
      statusL.style.color = "#f59e0b";
      statusL.textContent = "\u{1F512} Connected: " + coordLocked.join(", ");
      return;
    }
    statusL.style.color = "";
    statusL.textContent = `[${Math.round(x1)}, ${Math.round(y1)}, ${Math.round(x2)}, ${Math.round(y2)}]  ${Math.round(x2 - x1)} × ${Math.round(y2 - y1)}`;
  }

  return { statusBar, updateStatus };
}
