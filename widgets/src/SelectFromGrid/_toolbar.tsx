// _toolbar.tsx — Controls bar for the SelectFromGrid widget.
//
// Layout:
//   [Columns ──●── N]  [Grid|Masonry]  [☑ Labels] [Aa ──●── N]  · · ·  [N selected]  [Clear]  [⤢]
//
// The toolbar holds no state: it shows the values it's given and reports edits
// through the callbacks.

import { useEffect, useLayoutEffect, useRef, type SyntheticEvent } from 'react';

import { Icon } from './_icons';
import type { GridLayout } from './_types';

export interface ToolbarProps {
  layout: GridLayout;
  columns: number;
  disabled: boolean;
  showLabels: boolean;
  labelSize: number;
  selectedCount: number;
  onColumnsChange: (n: number) => void;
  onColumnsCommit: (n: number) => void;
  onLayoutChange: (layout: GridLayout) => void;
  onClear: () => void;
  onLabelsToggle: (show: boolean) => void;
  onLabelSizeChange: (n: number) => void;
  onLabelSizeCommit: (n: number) => void;
  onFullscreen: () => void;
}

const stopPropagation = (e: SyntheticEvent) => e.stopPropagation();

interface SliderProps {
  min: number;
  max: number;
  value: number;
  disabled: boolean;
  onInput: (n: number) => void;
  onCommit: (n: number) => void;
}

// A range input that reports every step through onInput and the release through onCommit.
// React's onChange fires per step, so the release is taken from the native change event.
function Slider({ min, max, value, disabled, onInput, onCommit }: SliderProps) {
  const ref = useRef<HTMLInputElement>(null);
  const commitRef = useRef(onCommit);
  useLayoutEffect(() => { commitRef.current = onCommit; });

  useEffect(() => {
    const input = ref.current;
    if (!input) return;
    const onChange = () => commitRef.current(parseInt(input.value, 10));
    input.addEventListener("change", onChange);
    return () => input.removeEventListener("change", onChange);
  }, []);

  return (
    <input
      ref={ref}
      type="range" className="sfg-slider" min={min} max={max}
      value={String(value)} disabled={disabled}
      onPointerDown={stopPropagation}
      onChange={(e) => onInput(parseInt(e.currentTarget.value, 10))}
    />
  );
}

export function Toolbar({
  layout, columns, disabled,
  showLabels, labelSize, selectedCount,
  onColumnsChange, onColumnsCommit,
  onLayoutChange,
  onClear,
  onLabelsToggle,
  onLabelSizeChange, onLabelSizeCommit,
  onFullscreen,
}: ToolbarProps) {
  const hasSelection = selectedCount > 0;

  const layoutButton = (target: GridLayout, text: string) => (
    <button
      type="button" className={"sfg-layout-btn" + (layout === target ? " active" : "")}
      disabled={disabled}
      onPointerDown={stopPropagation}
      onClick={(e) => {
        e.stopPropagation();
        onLayoutChange(target);
      }}
    >
      {text}
    </button>
  );

  return (
    <div className="sfg-controls">
      <label className="sfg-label">
        {"Columns "}
        <Slider
          min={1} max={8} value={columns} disabled={disabled}
          onInput={onColumnsChange} onCommit={onColumnsCommit}
        />
        <span>{String(columns)}</span>
      </label>

      {/* Layout toggle — segmented control */}
      <div className="sfg-layout-btns">
        {layoutButton("grid", "Grid")}
        {layoutButton("masonry", "Masonry")}
      </div>

      <button
        type="button" className={"sfg-labels-btn" + (showLabels ? " active" : "")}
        disabled={disabled}
        onPointerDown={stopPropagation}
        onClick={(e) => {
          e.stopPropagation();
          onLabelsToggle(!showLabels);
        }}
      >
        <span className="sfg-cb-box"><Icon name="check" size={9} /></span>
        Labels
      </button>

      <label className="sfg-label sfg-label-size-group">
        {"Aa "}
        <Slider
          min={8} max={32} value={labelSize} disabled={disabled}
          onInput={onLabelSizeChange} onCommit={onLabelSizeCommit}
        />
        <span>{String(labelSize)}</span>
      </label>

      <span className={"sfg-count" + (hasSelection ? " active" : "")}>
        {hasSelection ? `${selectedCount} selected` : ""}
      </span>

      <button
        type="button" className={"sfg-clear-btn" + (hasSelection ? " active" : "")} disabled={disabled}
        onPointerDown={stopPropagation}
        onClick={(e) => {
          e.stopPropagation();
          onClear();
        }}
      >
        Clear
      </button>

      {/* Stays enabled while the node is disabled: browsing doesn't change the value */}
      <button
        type="button" className="sfg-fullscreen-btn" title="Fullscreen"
        onPointerDown={stopPropagation}
        onClick={(e) => {
          e.stopPropagation();
          onFullscreen();
        }}
      >
        <Icon name="maximize-2" size={12} />
      </button>
    </div>
  );
}
