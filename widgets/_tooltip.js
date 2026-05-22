// Tooltip system — call createTooltip() once per widget instance.
// Returns { addTooltip(el, text), cleanup() }.

export function createTooltip() {
  const el = document.createElement("div");
  el.style.cssText =
    "position:fixed;z-index:999999;pointer-events:none;opacity:0;transition:opacity 0.1s;" +
    "background:var(--foreground);color:var(--background);font-size:11px;line-height:1.3;" +
    "padding:4px 8px;border-radius:5px;white-space:nowrap;box-shadow:0 2px 8px rgba(0,0,0,0.3);" +
    "transform:translateX(-50%) translateY(-4px);";

  const arrow = document.createElement("div");
  arrow.style.cssText =
    "position:absolute;left:50%;bottom:-4px;transform:translateX(-50%);" +
    "width:0;height:0;border-left:4px solid transparent;border-right:4px solid transparent;" +
    "border-top:4px solid var(--foreground);";
  el.appendChild(arrow);
  document.body.appendChild(el);

  let timer = null;

  function show(text, anchorEl) {
    clearTimeout(timer);
    timer = setTimeout(() => {
      el.textContent = text;
      el.appendChild(arrow);
      const rect = anchorEl.getBoundingClientRect();
      el.style.left = (rect.left + rect.width / 2) + "px";
      el.style.top = (rect.top - 8) + "px";
      el.style.transform = "translateX(-50%) translateY(-100%)";
      el.style.opacity = "1";
    }, 300);
  }

  function hide() {
    clearTimeout(timer);
    el.style.opacity = "0";
  }

  function addTooltip(targetEl, text) {
    targetEl.addEventListener("mouseenter", () => show(text, targetEl));
    targetEl.addEventListener("mouseleave", hide);
    targetEl.addEventListener("pointerdown", hide);
  }

  function cleanup() {
    clearTimeout(timer);
    el.remove();
  }

  return { addTooltip, cleanup };
}
