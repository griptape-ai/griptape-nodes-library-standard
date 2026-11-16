// Frame scrubber and status bar for CropVideoEditor.

import type { ChangeEvent, SyntheticEvent } from 'react';
import { Icon } from './_icons';

export interface FrameScrubberProps {
  frame: number;
  totalFrames: number;
  disabled: boolean;
  onSeek: (frame: number) => void;
}

export interface StatusBarProps {
  videoLoaded: boolean;
  lockedParams: readonly string[];
  ecL: number;
  ecT: number;
  ecW: number;
  ecH: number;
  version: string;
}

const stopPropagation = (e: SyntheticEvent) => e.stopPropagation();

export function FrameScrubber({ frame, totalFrames, disabled, onSeek }: FrameScrubberProps) {
  const off = disabled || totalFrames === 0;

  function onInput(e: ChangeEvent<HTMLInputElement>) {
    if (disabled) return;
    e.stopPropagation();
    onSeek(parseInt(e.target.value, 10));
  }

  return (
    <div
      className="nodrag nowheel"
      style={{
        background: "var(--muted)", borderRadius: "4px", padding: "6px 8px",
        display: "flex", flexDirection: "column", gap: "5px",
      }}
    >
      {/* ── Frame scrubber row ─────────────────────────────────────────────── */}
      <div style={{ display: "flex", alignItems: "center", gap: "8px", opacity: off ? "0.4" : "1" }}>
        <span style={{ color: "var(--muted-foreground)", display: "flex", alignItems: "center", flexShrink: 0 }}>
          <Icon name="film" size={13} />
        </span>
        <input
          type="range" min="0" max={String(Math.max(0, totalFrames - 1))} step="1" value={String(frame)}
          disabled={off}
          style={{ flex: 1, accentColor: "#2563eb" }}
          onMouseDown={stopPropagation}
          onPointerDown={stopPropagation}
          onChange={onInput}
        />
        <span style={{ fontSize: "11px", color: "var(--foreground)", minWidth: "52px", textAlign: "right", fontFamily: "monospace" }}>
          {`${frame} / ${totalFrames}`}
        </span>
      </div>
    </div>
  );
}

function statusLine({ videoLoaded, lockedParams, ecL, ecT, ecW, ecH }: StatusBarProps): { text: string; color: string } {
  if (!videoLoaded) return { text: "No video connected", color: "" };
  const coordLocked = ["left", "top", "width", "height"].filter(f => lockedParams.includes(f));
  if (coordLocked.length > 0) return { text: "\u{1F512} Connected: " + coordLocked.join(", "), color: "#f59e0b" };
  return { text: `x: ${Math.round(ecL)}  y: ${Math.round(ecT)}  →  ${Math.round(ecW)} × ${Math.round(ecH)} px`, color: "" };
}

export function StatusBar(props: StatusBarProps) {
  const { text, color } = statusLine(props);
  return (
    <div
      style={{
        fontSize: "11px", color: "var(--muted-foreground)", padding: "4px 8px",
        background: "var(--muted)", borderRadius: "4px",
        display: "flex", justifyContent: "space-between", alignItems: "center",
      }}
    >
      <span style={{ color: color || undefined }}>{text}</span>
      <span style={{ fontFamily: "monospace", fontSize: "10px", color: "var(--muted-foreground)", opacity: 0.6 }}>
        {`v${props.version}`}
      </span>
    </div>
  );
}
