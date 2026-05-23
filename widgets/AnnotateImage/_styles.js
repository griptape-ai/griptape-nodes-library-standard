export const DEFAULT_CANVAS_WIDTH  = 1920;
export const DEFAULT_CANVAS_HEIGHT = 1080;
export const DEFAULT_COLOR         = "#ff0000";

export const DEFAULT_PAINT_SIZE    = 8;
export const MIN_PAINT_SIZE        = 1;
export const MAX_PAINT_SIZE        = 80;

export const DEFAULT_TEXT_SIZE     = 48;
export const MIN_TEXT_SIZE         = 8;
export const MAX_TEXT_SIZE         = 120;

export const DEFAULT_ARROW_WIDTH   = 8;
export const MIN_ARROW_WIDTH       = 1;
export const MAX_ARROW_WIDTH       = 40;

export const DEFAULT_SHAPE_WIDTH   = 8;
export const MIN_SHAPE_WIDTH       = 1;
export const MAX_SHAPE_WIDTH       = 40;

// ── Transform / selection chrome ──────────────────────────────────────────────

// Primary selection color (local annotations)
export const SEL_COLOR      = "#7a9db8";
export const SEL_COLOR_RGB  = "122,157,184";   // pre-split for rgba() use

// Imported-annotation selection color
export const IMP_COLOR      = "#c9830a";
export const IMP_COLOR_RGB  = "201,131,10";

// Frame / bounding-box opacities
export const FRAME_FILL_OPACITY    = 0.06;
export const FRAME_BORDER_OPACITY  = 0.75;
export const FRAME_CORNER_OPACITY  = 0.55;    // corner tick marks
export const FRAME_ROT_STEM_OPACITY = 0.35;   // dashed line to rotation handle

// Hover / hit-test overlays
export const HOVER_OPACITY         = 0.7;
export const LASSO_FILL_OPACITY    = 0.1;
export const LASSO_STROKE_OPACITY  = 0.8;
export const LAYER_HOVER_OPACITY   = 0.18;

// Handle fills
export const HANDLE_FILL           = "white";
export const ROT_HANDLE_INNER_OPACITY = 0.85; // white ring on rotation handle

// Handle stroke opacities
export const HANDLE_STROKE_OPACITY = 0.9;
export const CP_LINE_OPACITY       = 0.5;     // bezier control-point guide lines

// Stroke widths (canvas units, divided by displayScale at draw time)
export const LINE_WIDTH_PRIMARY    = 1.5;
export const LINE_WIDTH_SECONDARY  = 1.0;
export const LINE_WIDTH_TERTIARY   = 0.75;

// Handle radii (canvas units, divided by displayScale)
export const HANDLE_RADIUS         = 8;       // scale / corner handles
export const CP_HANDLE_RADIUS      = 4;       // bezier control-point handles
export const ROT_HANDLE_RADIUS     = 4.5;     // rotation handle
export const ARROW_HANDLE_RADIUS   = 10;      // arrow endpoint handles (min px)
export const CORNER_TICK_LEN       = 6;       // L-shaped corner tick arm length

// Dash patterns [dash, gap] (canvas units, divided by displayScale)
export const DASH_CP_LINE          = [3, 2];  // bezier control-point guide
export const DASH_ROT_STEM         = [2, 2];  // rotation handle stem
export const DASH_LASSO            = [4, 3];  // selection lasso

// Hover padding (extra pixels around annotation bounding box)
export const HOVER_PAD             = 4;

// ── Context HUD (floating action bar over canvas) ────────────────────────────
export const HUD_BG             = "rgba(18,18,20,0.88)";
export const HUD_BORDER         = "rgba(255,255,255,0.10)";
export const HUD_SHADOW         = "0 2px 16px rgba(0,0,0,0.55),0 0 0 1px rgba(255,255,255,0.06)";
export const HUD_BTN_HOVER_BG   = "rgba(255,255,255,0.10)";
export const HUD_SEP_COLOR      = "rgba(255,255,255,0.12)";

export function defaultData() {
  return {
    image_url: "",
    raw_url: "",
    canvas_width: DEFAULT_CANVAS_WIDTH,
    canvas_height: DEFAULT_CANVAS_HEIGHT,
    annotations: [],
    imported_annotations: [],
    overrides: {},
    active_tool: "select",
    tool_settings: {
      paint:   { color: DEFAULT_COLOR, size: DEFAULT_PAINT_SIZE },
      text:    { color: DEFAULT_COLOR, font_size: DEFAULT_TEXT_SIZE },
      arrow:   { color: DEFAULT_COLOR, width: DEFAULT_ARROW_WIDTH, has_start_arrow: false, has_end_arrow: true, is_bezier: false, taper: false },
      rect:    { color: DEFAULT_COLOR, width: DEFAULT_SHAPE_WIDTH, fill_color: "" },
      ellipse: { color: DEFAULT_COLOR, width: DEFAULT_SHAPE_WIDTH, fill_color: "" },
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
    .ais-hud { position:absolute; top:10px; left:50%; transform:translateX(-50%); display:flex; align-items:center; gap:2px; padding:3px; border-radius:10px; background:rgba(18,18,20,0.88); border:1px solid rgba(255,255,255,0.10); box-shadow:0 2px 16px rgba(0,0,0,0.55),0 0 0 1px rgba(255,255,255,0.06); pointer-events:auto; z-index:20; transition:opacity 0.12s; white-space:nowrap; }
    .ais-hud-btn { display:flex; align-items:center; justify-content:center; width:28px; height:28px; padding:0; border:none; border-radius:7px; background:transparent; color:#e0e0e0; cursor:pointer; line-height:1; transition:background 0.12s,color 0.12s; flex-shrink:0; }
    .ais-hud-btn:hover { background:rgba(255,255,255,0.10); color:#fff; }
    .ais-hud-btn.imp { color:#e8a040; }
    .ais-hud-btn.imp:hover { background:rgba(201,131,10,0.18); color:#f0b050; }
    .ais-hud-btn.danger:hover { background:rgba(220,60,60,0.22); color:#ff7070; }
    .ais-hud-sep { width:1px; height:18px; background:rgba(255,255,255,0.12); margin:0 2px; flex-shrink:0; }
  `;
}
