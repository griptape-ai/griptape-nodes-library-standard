export const WIDGET_VERSION = "0.1.0";

// ── Geometry ───────────────────────────────────────────────────────────────────
export const HANDLE_R = 6;

// ── Canvas chrome ──────────────────────────────────────────────────────────────
export const OVERLAY     = "rgba(0,0,0,0.55)";
export const CROP_BORDER = "rgba(255,255,255,0.9)";
export const GUIDE       = "rgba(255,255,255,0.25)";

// ── Resize handles ─────────────────────────────────────────────────────────────
// White fill + blue stroke matches the AnnotateImage widget convention.
// Works in both light and dark themes since white is always visible over the image canvas.
export const HANDLE_FILL         = "black";
export const HANDLE_STROKE       = "#ffffff";
export const HANDLE_STROKE_HOVER = "#60a5fa";
export const HANDLE_LOCKED       = "rgba(150,150,150,0.5)";
