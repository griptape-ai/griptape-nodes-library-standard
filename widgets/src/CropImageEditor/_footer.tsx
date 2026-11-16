// Zoom + rotate slider controls and status bar for CropImageEditor.

import { useEffect, useRef, useState } from 'react';
import type { SyntheticEvent } from 'react';

import { Icon } from './_icons';
import { SLIDER_ACCENT } from './_styles';

export type SliderParam = "zoom" | "rotate";

export interface TransformControlsProps {
  zoom: number;
  rotate: number;
  /** A slider whose parameter is connected, or every slider while the widget is disabled, is off. */
  isOff: (param: SliderParam) => boolean;
  /** Shows a value while the slider moves. */
  onPreview: (param: SliderParam, value: number) => void;
  /** Commits the current values once the slider is let go or reset. */
  onCommit: () => void;
}

interface SliderRowProps {
  label: string;
  min: number;
  max: number;
  step: number;
  value: number;
  format: (v: number) => string;
  defaultVal: number;
  off: boolean;
  onPreview: (value: number) => void;
  onCommit: () => void;
}

const stopPropagation = (e: SyntheticEvent) => e.stopPropagation();

const formatZoom   = (v: number) => Math.round(v) + "%";
const formatRotate = (v: number) => (v >= 0 ? "+" : "") + Math.round(v) + "°";

function SliderRow({ label, min, max, step, value, format, defaultVal, off, onPreview, onCommit }: SliderRowProps) {
  const sliderRef = useRef<HTMLInputElement>(null);
  const [resetHovered, setResetHovered] = useState(false);

  // React's onChange on a range input fires on every input event, so the commit listens for the
  // native change event, which fires once the user lets go of the slider.
  useEffect(() => {
    const slider = sliderRef.current;
    if (!slider) return;
    const onNativeChange = (e: Event) => {
      if (off) return;
      e.stopPropagation();
      onCommit();
    };
    slider.addEventListener("change", onNativeChange);
    return () => slider.removeEventListener("change", onNativeChange);
  }, [off, onCommit]);

  return (
    <div style={{ display: "flex", alignItems: "center", gap: "8px", opacity: off ? 0.4 : 1 }}>
      <span style={{ fontSize: "11px", color: "var(--muted-foreground)", minWidth: "44px" }}>{label}</span>
      <input
        ref={sliderRef}
        type="range"
        min={min} max={max} step={step}
        value={value}
        disabled={off}
        style={{ flex: 1, accentColor: SLIDER_ACCENT }}
        onMouseDown={stopPropagation}
        onPointerDown={stopPropagation}
        onInput={(e) => {
          if (off) return;
          e.stopPropagation();
          onPreview(parseFloat(e.currentTarget.value));
        }}
      />
      <span style={{ fontSize: "11px", color: "var(--foreground)", minWidth: "38px", textAlign: "right", fontFamily: "monospace" }}>
        {format(value)}
      </span>
      <button
        title={`Reset to ${format(defaultVal)}`}
        disabled={off}
        style={{
          background: "none", border: "none", cursor: "pointer",
          color: resetHovered && !off ? "var(--foreground)" : "var(--muted-foreground)",
          padding: "0 2px", lineHeight: "1", flexShrink: 0,
          display: "flex", alignItems: "center",
        }}
        onMouseEnter={() => setResetHovered(true)}
        onMouseLeave={() => setResetHovered(false)}
        onMouseDown={stopPropagation}
        onPointerDown={stopPropagation}
        onClick={(e) => {
          if (off) return;
          e.stopPropagation();
          onPreview(defaultVal);
          onCommit();
        }}
      >
        <Icon name="rotate-ccw" size={13} />
      </button>
    </div>
  );
}

export function TransformControls({ zoom, rotate, isOff, onPreview, onCommit }: TransformControlsProps) {
  return (
    <div
      className="nodrag nowheel"
      style={{
        background: "var(--muted)", borderRadius: "4px", padding: "6px 8px",
        display: "flex", flexDirection: "column", gap: "5px",
      }}
    >
      <SliderRow
        label="Zoom" min={10} max={500} step={1}
        value={zoom} format={formatZoom} defaultVal={100} off={isOff("zoom")}
        onPreview={(v) => onPreview("zoom", v)} onCommit={onCommit}
      />
      <SliderRow
        label="Rotate" min={-180} max={180} step={1}
        value={rotate} format={formatRotate} defaultVal={0} off={isOff("rotate")}
        onPreview={(v) => onPreview("rotate", v)} onCommit={onCommit}
      />
    </div>
  );
}

// ── Status bar ─────────────────────────────────────────────────────────────────
export interface StatusBarProps {
  text: string;
  /** Highlights the text, or leaves the bar's colour when empty. */
  color: string;
  version: string;
}

export function StatusBar({ text, color, version }: StatusBarProps) {
  return (
    <div
      style={{
        fontSize: "11px", color: "var(--muted-foreground)", padding: "4px 8px",
        background: "var(--muted)", borderRadius: "4px",
        display: "flex", justifyContent: "space-between", alignItems: "center",
      }}
    >
      <span style={color ? { color } : undefined}>{text}</span>
      <span style={{ fontFamily: "monospace", fontSize: "10px", color: "var(--muted-foreground)", opacity: 0.6 }}>
        {`v${version}`}
      </span>
    </div>
  );
}
