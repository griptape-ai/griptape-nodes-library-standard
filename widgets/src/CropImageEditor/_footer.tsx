// Zoom + rotate slider controls and status bar for CropImageEditor.

import { Icon } from './_icons';
import { SLIDER_ACCENT } from './_styles';

type SliderParam = "zoom" | "rotate";

interface SliderCallbacks {
  isLocked: (key: string) => boolean;
  isDisabled: () => boolean;
  onRender: () => void;
  onEmit: () => void;
}

interface SliderRowOptions extends SliderCallbacks {
  label: string;
  min: number;
  max: number;
  step: number;
  getVal: () => number;
  setVal: (v: number) => void;
  format: (v: number) => string;
  paramKey: SliderParam;
  defaultVal: number;
}

interface SliderRow {
  row: HTMLDivElement;
  sync: (lockedParams: readonly string[], disabled: boolean) => void;
}

export interface FooterOptions extends SliderCallbacks {
  getZoom: () => number;
  setZoom: (v: number) => void;
  getRotate: () => number;
  setRotate: (v: number) => void;
  version: string;
}

export interface FooterStatus {
  imageLoaded: boolean;
  lockedParams: readonly string[];
  ecL: number;
  ecT: number;
  ecW: number;
  ecH: number;
}

export interface Footer {
  controls: HTMLDivElement;
  statusBar: HTMLDivElement;
  sync: (lockedParams: readonly string[], disabled: boolean) => void;
  updateStatus: (status: FooterStatus) => void;
}

const stopPropagation = (e: Event) => e.stopPropagation();

function makeSliderRow({ label, min, max, step, getVal, setVal, format, paramKey, defaultVal, isLocked, isDisabled, onRender, onEmit }: SliderRowOptions): SliderRow {
  const slider = (
    <input
      type="range"
      min={String(min)} max={String(max)} step={String(step)}
      value={String(getVal())}
      style={`flex:1;accent-color:${SLIDER_ACCENT};`}
      onMouseDown={stopPropagation}
      onPointerDown={stopPropagation}
      onInput={(e) => {
        if (isLocked(paramKey) || isDisabled()) return;
        e.stopPropagation();
        setVal(parseFloat((e.target as HTMLInputElement).value));
        display.textContent = format(getVal());
        onRender();
      }}
      onChange={(e) => {
        if (isLocked(paramKey) || isDisabled()) return;
        e.stopPropagation();
        onEmit();
      }}
    />
  ) as HTMLInputElement;

  const display = (
    <span style="font-size:11px;color:var(--foreground);min-width:38px;text-align:right;font-family:monospace;">
      {format(getVal())}
    </span>
  ) as HTMLSpanElement;

  const resetBtn = (
    <button
      title={`Reset to ${format(defaultVal)}`}
      style={{
        background: "none", border: "none", color: "var(--muted-foreground)", cursor: "pointer",
        padding: "0 2px", lineHeight: "1", flexShrink: "0",
        display: "flex", alignItems: "center",
      }}
      onMouseEnter={() => { if (!resetBtn.disabled) resetBtn.style.color = "var(--foreground)"; }}
      onMouseLeave={() => { resetBtn.style.color = "var(--muted-foreground)"; }}
      onMouseDown={stopPropagation}
      onPointerDown={stopPropagation}
      onClick={(e) => {
        if (isLocked(paramKey) || isDisabled()) return;
        e.stopPropagation();
        setVal(defaultVal);
        slider.value = String(defaultVal);
        display.textContent = format(defaultVal);
        onRender();
        onEmit();
      }}
    >
      <Icon name="rotate-ccw" size={13} />
    </button>
  ) as HTMLButtonElement;

  const row = (
    <div style="display:flex;align-items:center;gap:8px;">
      <span style="font-size:11px;color:var(--muted-foreground);min-width:44px;">{label}</span>
      {slider}
      {display}
      {resetBtn}
    </div>
  ) as HTMLDivElement;

  function sync(lockedParams: readonly string[], disabled: boolean): void {
    const locked = lockedParams.includes(paramKey) || disabled;
    slider.disabled = locked;
    resetBtn.disabled = locked;
    slider.value = String(getVal());
    display.textContent = format(getVal());
    row.style.opacity = locked ? "0.4" : "1";
  }

  return { row, sync };
}

export function createFooter({ getZoom, setZoom, getRotate, setRotate, isLocked, isDisabled, onRender, onEmit, version }: FooterOptions): Footer {
  const zoomRow = makeSliderRow({
    label: "Zoom", min: 10, max: 500, step: 1,
    getVal: getZoom, setVal: setZoom,
    format: (v) => Math.round(v) + "%",
    paramKey: "zoom", defaultVal: 100,
    isLocked, isDisabled, onRender, onEmit,
  });

  const rotateRow = makeSliderRow({
    label: "Rotate", min: -180, max: 180, step: 1,
    getVal: getRotate, setVal: setRotate,
    format: (v) => (v >= 0 ? "+" : "") + Math.round(v) + "°",
    paramKey: "rotate", defaultVal: 0,
    isLocked, isDisabled, onRender, onEmit,
  });

  const controls = (
    <div
      class="nodrag nowheel"
      style={{
        background: "var(--muted)", borderRadius: "4px", padding: "6px 8px",
        display: "flex", flexDirection: "column", gap: "5px",
      }}
    >
      {zoomRow.row}
      {rotateRow.row}
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

  function sync(lockedParams: readonly string[], disabled: boolean): void {
    zoomRow.sync(lockedParams, disabled);
    rotateRow.sync(lockedParams, disabled);
  }

  function updateStatus({ imageLoaded, lockedParams, ecL, ecT, ecW, ecH }: FooterStatus): void {
    if (!imageLoaded) {
      statusL.textContent = "No image connected";
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

  return { controls, statusBar, sync, updateStatus };
}
