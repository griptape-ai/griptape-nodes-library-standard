// Frame scrubber and status bar for CropVideoEditor.

import { Icon } from './_icons';

export interface FooterOptions {
  isDisabled: () => boolean;
  onSeek: (frame: number) => void;
  version: string;
}

export interface FooterStatus {
  videoLoaded: boolean;
  lockedParams: readonly string[];
  ecL: number;
  ecT: number;
  ecW: number;
  ecH: number;
}

export interface Footer {
  controls: HTMLDivElement;
  statusBar: HTMLDivElement;
  sync: (totalFrames: number, disabled: boolean) => void;
  setFrame: (frame: number, totalFrames: number) => void;
  updateStatus: (status: FooterStatus) => void;
}

const stopPropagation = (e: Event) => e.stopPropagation();

export function createFooter({ isDisabled, onSeek, version }: FooterOptions): Footer {
  const scrubSlider = (
    <input
      type="range" min="0" max="0" step="1" value="0"
      style="flex:1;accent-color:#2563eb;"
      onMouseDown={stopPropagation}
      onPointerDown={stopPropagation}
      onInput={(e) => {
        if (isDisabled()) return;
        e.stopPropagation();
        const frame = parseInt((e.target as HTMLInputElement).value, 10);
        const total = parseInt(scrubSlider.max, 10);
        frameDisplay.textContent = `${frame} / ${total}`;
        onSeek(frame);
      }}
    />
  ) as HTMLInputElement;

  const frameDisplay = (
    <span style="font-size:11px;color:var(--foreground);min-width:52px;text-align:right;font-family:monospace;">
      0 / 0
    </span>
  ) as HTMLSpanElement;

  // ── Frame scrubber row ──────────────────────────────────────────────────────
  const scrubRow = (
    <div style="display:flex;align-items:center;gap:8px;">
      <span style="color:var(--muted-foreground);display:flex;align-items:center;flex-shrink:0;">
        <Icon name="film" size={13} />
      </span>
      {scrubSlider}
      {frameDisplay}
    </div>
  ) as HTMLDivElement;

  const controls = (
    <div
      class="nodrag nowheel"
      style={{
        background: "var(--muted)", borderRadius: "4px", padding: "6px 8px",
        display: "flex", flexDirection: "column", gap: "5px",
      }}
    >
      {scrubRow}
    </div>
  ) as HTMLDivElement;

  // ── Status bar ─────────────────────────────────────────────────────────────
  const statusL = (<span />) as HTMLSpanElement;
  const statusBar = (
    <div
      style={{
        fontSize: "11px", color: "var(--muted-foreground)", padding: "4px 8px",
        background: "var(--muted)", borderRadius: "4px",
        display: "flex", justifyContent: "space-between", alignItems: "center",
      }}
    >
      {statusL}
      <span style="font-family:monospace;font-size:10px;color:var(--muted-foreground);opacity:0.6;">
        {`v${version}`}
      </span>
    </div>
  ) as HTMLDivElement;

  function sync(totalFrames: number, disabled: boolean): void {
    scrubSlider.disabled = disabled || totalFrames === 0;
    scrubSlider.max = String(Math.max(0, totalFrames - 1));
    scrubRow.style.opacity = (disabled || totalFrames === 0) ? "0.4" : "1";
  }

  function setFrame(frame: number, totalFrames: number): void {
    scrubSlider.max = String(Math.max(0, totalFrames - 1));
    scrubSlider.value = String(frame);
    frameDisplay.textContent = `${frame} / ${totalFrames}`;
  }

  function updateStatus({ videoLoaded, lockedParams, ecL, ecT, ecW, ecH }: FooterStatus): void {
    if (!videoLoaded) {
      statusL.textContent = "No video connected";
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
    statusL.textContent = `x: ${Math.round(ecL)}  y: ${Math.round(ecT)}  →  ${Math.round(ecW)} × ${Math.round(ecH)} px`;
  }

  return { controls, statusBar, sync, setFrame, updateStatus };
}
