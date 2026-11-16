// Preset sidebar for CropImageEditor — aspect ratio, resolution, and position shortcuts.

import { useState } from 'react';
import type { ReactNode, SyntheticEvent } from 'react';

import { type IconName, Icon } from './_icons';
import type { CropRect } from './_types';

interface RatioPreset { label: string; rw: number; rh: number }
interface ResolutionPreset { label: string; w: number; h: number }
interface PositionPreset { label: string; title: string; pos: PositionName }

type PositionName =
  | "top-left" | "top-center" | "top-right"
  | "left-center" | "center" | "right-center"
  | "bottom-left" | "bottom-center" | "bottom-right";

/** A crop in image pixels, as the sidebar hands it to `onApply`. Reset also resets the transform. */
export type AppliedCrop = CropRect | (CropRect & { zoom: number; rotate: number });

export interface CropSidebarProps {
  imgNatW: number;
  imgNatH: number;
  /** The crop the presets start from, in image pixels. */
  cropRect: { l: number; t: number; w: number; h: number };
  /** Every preset is off while a crop coordinate is connected or the widget is disabled. */
  off: boolean;
  onApply: (crop: AppliedCrop) => void;
}

const RATIO_PRESETS: readonly RatioPreset[] = [
  { label: "1:1",  rw: 1,  rh: 1  },
  { label: "16:9", rw: 16, rh: 9  },
  { label: "4:3",  rw: 4,  rh: 3  },
  { label: "3:2",  rw: 3,  rh: 2  },
  { label: "9:16", rw: 9,  rh: 16 },
  { label: "2:3",  rw: 2,  rh: 3  },
  { label: "3:4",  rw: 3,  rh: 4  },
];

const RESOLUTION_PRESETS: readonly ResolutionPreset[] = [
  { label: "480p",  w: 854,  h: 480  },
  { label: "720p",  w: 1280, h: 720  },
  { label: "1080p", w: 1920, h: 1080 },
  { label: "1440p", w: 2560, h: 1440 },
  { label: "4K",    w: 3840, h: 2160 },
];

const POSITION_PRESETS: readonly PositionPreset[] = [
  { label: "↖",    title: "Top Left",       pos: "top-left"      },
  { label: "↑",    title: "Top Center",     pos: "top-center"    },
  { label: "↗",    title: "Top Right",      pos: "top-right"     },
  { label: "←",    title: "Left Center",    pos: "left-center"   },
  { label: "⊕",    title: "Center",         pos: "center"        },
  { label: "→",    title: "Right Center",   pos: "right-center"  },
  { label: "↙",    title: "Bottom Left",    pos: "bottom-left"   },
  { label: "↓",    title: "Bottom Center",  pos: "bottom-center" },
  { label: "↘",    title: "Bottom Right",   pos: "bottom-right"  },
];

// Scale from the shortest side of the current crop to the new ratio.
function calcRatioRect(
  rw: number, rh: number, imgW: number, imgH: number,
  curL: number, curT: number, curW: number, curH: number,
): CropRect {
  const s = Math.min(curW, curH) / Math.min(rw, rh);
  let w = Math.round(rw * s);
  let h = Math.round(rh * s);
  if (w > imgW) { w = imgW; h = Math.round(w * rh / rw); }
  if (h > imgH) { h = imgH; w = Math.round(h * rw / rh); }
  const cx = curL + curW / 2, cy = curT + curH / 2;
  const l = Math.max(0, Math.min(Math.round(cx - w / 2), imgW - w));
  const t = Math.max(0, Math.min(Math.round(cy - h / 2), imgH - h));
  return { left: l, top: t, width: w, height: h };
}

// Center the new resolution on the current crop center, clamped to image bounds.
function calcResolutionRect(
  newW: number, newH: number, imgW: number, imgH: number,
  curL: number, curT: number, curW: number, curH: number,
): CropRect {
  const w = Math.min(newW, imgW);
  const h = Math.min(newH, imgH);
  const cx = curL + curW / 2, cy = curT + curH / 2;
  const l = Math.max(0, Math.min(Math.round(cx - w / 2), imgW - w));
  const t = Math.max(0, Math.min(Math.round(cy - h / 2), imgH - h));
  return { left: l, top: t, width: w, height: h };
}

// Compute left/top for a named position, keeping current crop size.
function calcPositionRect(
  posName: PositionName, imgW: number, imgH: number,
  curL: number, curT: number, curW: number, curH: number,
): CropRect {
  const w = curW, h = curH;
  const positions: Record<PositionName, { l: number; t: number }> = {
    "center":        { l: Math.round((imgW - w) / 2), t: Math.round((imgH - h) / 2) },
    "top-left":      { l: 0,           t: 0            },
    "top-center":    { l: Math.round((imgW - w) / 2), t: 0 },
    "top-right":     { l: imgW - w,    t: 0            },
    "left-center":   { l: 0,           t: Math.round((imgH - h) / 2) },
    "right-center":  { l: imgW - w,    t: Math.round((imgH - h) / 2) },
    "bottom-left":   { l: 0,           t: imgH - h     },
    "bottom-center": { l: Math.round((imgW - w) / 2), t: imgH - h },
    "bottom-right":  { l: imgW - w,    t: imgH - h     },
  };
  const { l, t } = positions[posName] || positions["center"];
  return { left: Math.max(0, l), top: Math.max(0, t), width: w, height: h };
}

const stopPropagation = (e: SyntheticEvent) => e.stopPropagation();

function Divider() {
  return <div style={{ height: "1px", background: "var(--border)", margin: "2px 0", flexShrink: 0 }} />;
}

function Label({ text }: { text: string }) {
  return <div style={{ fontSize: "9px", color: "var(--muted-foreground)", textAlign: "center", padding: "2px 0" }}>{text}</div>;
}

function Grid({ columns, children }: { columns: number; children: ReactNode }) {
  return <div style={{ display: "grid", gridTemplateColumns: `repeat(${columns},1fr)`, gap: "2px" }}>{children}</div>;
}

function PresetButton({ label, icon, title, off, onClick }: {
  label?: string; icon?: IconName; title: string; off: boolean; onClick: () => void;
}) {
  const [hovered, setHovered] = useState(false);
  return (
    <button
      title={title}
      disabled={off}
      style={{
        width: "100%", padding: "4px 2px", borderRadius: "4px",
        border: "1px solid var(--border)",
        background: hovered && !off ? "var(--accent)" : "var(--background)",
        color: "var(--foreground)", cursor: off ? "not-allowed" : "pointer",
        opacity: off ? 0.35 : 1,
        fontSize: "10px", fontWeight: "500", lineHeight: "1.3",
        whiteSpace: "nowrap", boxSizing: "border-box",
        display: "flex", alignItems: "center", justifyContent: "center", gap: "3px",
      }}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      onMouseDown={stopPropagation}
      onPointerDown={stopPropagation}
      onClick={(e) => { e.stopPropagation(); if (!off) onClick(); }}
    >
      {icon && <Icon name={icon} size={12} />}
      {label}
    </button>
  );
}

export function CropSidebar({ imgNatW, imgNatH, cropRect, off, onApply }: CropSidebarProps) {
  const { l, t, w, h } = cropRect;
  return (
    <div
      className="nodrag nowheel"
      style={{
        display: "flex", flexDirection: "column", gap: "3px",
        width: "108px", flexShrink: 0,
        background: "var(--muted)", borderRadius: "6px", padding: "4px",
        overflowY: "auto", boxSizing: "border-box",
      }}
    >
      {/* ── Reset ── */}
      <PresetButton
        icon="rotate-ccw" label="Reset" title="Reset crop to full image" off={off}
        onClick={() => onApply({ left: 0, top: 0, width: imgNatW, height: imgNatH, zoom: 100, rotate: 0 })}
      />

      {/* ── Aspect ratios ── */}
      <Divider />
      <Label text="Ratio" />
      <Grid columns={2}>
        {RATIO_PRESETS.map(({ label, rw, rh }) => (
          <PresetButton
            key={label} label={label} title={label} off={off}
            onClick={() => onApply(calcRatioRect(rw, rh, imgNatW, imgNatH, l, t, w, h))}
          />
        ))}
      </Grid>

      {/* ── Resolutions ── */}
      <Divider />
      <Label text="Res" />
      <Grid columns={2}>
        {RESOLUTION_PRESETS.map(({ label, w: pw, h: ph }) => (
          <PresetButton
            key={label} label={label} title={`${pw}×${ph}`} off={off}
            onClick={() => onApply(calcResolutionRect(pw, ph, imgNatW, imgNatH, l, t, w, h))}
          />
        ))}
      </Grid>

      {/* ── Positions ── */}
      <Divider />
      <Label text="Pos" />
      <Grid columns={3}>
        {POSITION_PRESETS.map(({ label, title, pos }) => (
          <PresetButton
            key={pos} label={label} title={title} off={off}
            onClick={() => onApply(calcPositionRect(pos, imgNatW, imgNatH, l, t, w, h))}
          />
        ))}
      </Grid>
    </div>
  );
}
