export const DEFAULT_CANVAS_WIDTH = 1920;
export const DEFAULT_CANVAS_HEIGHT = 1080;

export function defaultData() {
  return {
    image_url: "",
    raw_url: "",
    canvas_width: 0,
    canvas_height: 0,
    annotations: [],
    imported_annotations: [],
    overrides: {},
    active_tool: "select",
    tool_settings: {
      paint:   { color: "#ff0000", size: 8 },
      text:    { color: "#ff0000", font_size: 48 },
      arrow:   { color: "#ff0000", width: 8, has_start_arrow: false, has_end_arrow: true, is_bezier: false },
      rect:    { color: "#ff0000", width: 8, fill_color: "" },
      ellipse: { color: "#ff0000", width: 8, fill_color: "" },
    },
    selected_ids: [],
  };
}

export function injectStyles() {
  const id = "ais-styles";
  let el = document.getElementById(id);
  if (!el) { el = document.createElement("style"); el.id = id; document.head.appendChild(el); }
  el.textContent = `
    .ais-tool-btn { background:transparent; border:none; border-radius:4px; color:var(--muted-foreground); cursor:pointer; width:28px; height:28px; display:flex; align-items:center; justify-content:center; transition:background 0.15s,color 0.15s; flex-shrink:0; padding:0; }
    .ais-tool-btn:hover { background:var(--muted); color:var(--foreground); }
    .ais-tool-btn.active { background:transparent; color:var(--foreground); box-shadow:0 0 0 1.5px #7a9db8, 0 0 6px 1px rgba(122,157,184,0.4); }
    .ais-toggle-btn { background:transparent; border:none; border-radius:4px; color:var(--muted-foreground); cursor:pointer; display:flex; align-items:center; justify-content:center; transition:background 0.15s,color 0.15s; flex-shrink:0; padding:0; }
    .ais-toggle-btn:hover { background:var(--muted); color:var(--foreground); }
    .ais-toggle-btn.active { background:rgba(122,157,184,0.2); color:var(--foreground); border:1px solid rgba(122,157,184,0.5); }
    .ais-setting-label { font-size:11px; color:var(--muted-foreground); }
    .ais-range { accent-color:#7a9db8; cursor:pointer; width:80px; min-width:30px; flex-shrink:1; }
    .ais-color-btn { width:22px; height:22px; border-radius:4px; border:2px solid var(--border,#555); cursor:pointer; flex-shrink:0; }
    .ais-color-input { position:absolute; opacity:0; width:0; height:0; pointer-events:none; }
    .ais-val-label { font-size:11px; color:var(--foreground); min-width:20px; text-align:right; flex-shrink:0; }
  `;
}
