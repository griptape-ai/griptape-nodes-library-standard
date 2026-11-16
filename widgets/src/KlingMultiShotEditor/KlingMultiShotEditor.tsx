/**
 * MultiShotEditor - A list-based editor for managing video shot sequences.
 * Supports adding, deleting, reordering (drag-and-drop), duration selection,
 * and per-shot description text.
 */

import { useEffect, useRef, useState } from "react";
import type { PointerEvent as ReactPointerEvent, SyntheticEvent } from "react";
import { reactWidget } from "@griptape-ai/nodes-widget-api/react";

import type { KlingShot, KlingShotList } from './_types';
import type { WidgetProps } from '@griptape-ai/nodes-widget-api';

export type { KlingShot, KlingShotList } from './_types';

export type KlingMultiShotEditorProps = WidgetProps<KlingShotList>;

interface ShotItemRect {
  rect: DOMRect;
  index: number;
}

// Pointer-drag scratch state: the floating clone and the item rects taken when the drag began.
interface DragScratch {
  source: number;
  over: number | null;
  offsetY: number;
  clone: HTMLElement;
  rects: ShotItemRect[];
}

// What the drag renders: the dimmed source item and the drop marker.
interface DragView {
  source: number;
  over: number | null;
}

const WIDGET_VERSION = "0.6.1";

const MAX_SHOTS = 6;
const MAX_DESCRIPTION_LENGTH = 512;
const MAX_TOTAL_DURATION = 15;
const MIN_TOTAL_DURATION = 3;
const MIN_DURATION = 1;
const MAX_DURATION = 15;
const PLACEHOLDER =
  "Describe the shot information, such as: who is where and what is happening.";
const DROP_MARKER = "2px solid #4a9eff";
const ITEM_DIVIDER = "1px solid #2a2a2a";

const stopPropagation = (e: SyntheticEvent) => e.stopPropagation();

// -- helpers ------------------------------------------------------------

// Gives a shot an id if it has none, and keeps `nextId` past every id seen.
function assignId(shot: KlingShot, nextId: { current: number }): KlingShot {
  if (!shot.id) {
    shot.id = `shot-${nextId.current++}`;
  } else {
    const num = parseInt(shot.id.replace("shot-", ""), 10);
    if (!isNaN(num) && num >= nextId.current) {
      nextId.current = num + 1;
    }
  }
  return shot;
}

function renumberShots(shots: KlingShot[]): KlingShot[] {
  return shots.map((shot, i) => ({ ...shot, name: `Shot${i + 1}` }));
}

function totalDuration(shots: KlingShot[]) {
  return shots.reduce((sum, s) => sum + (s.duration || 0), 0);
}

function durationBudgetFor(shots: KlingShot[], index: number) {
  const othersTotal = shots.reduce(
    (sum, s, i) => sum + (i === index ? 0 : s.duration || 0),
    0,
  );
  return MAX_TOTAL_DURATION - othersTotal;
}

function withShot(shots: KlingShot[], index: number, patch: Partial<KlingShot>): KlingShot[] {
  return shots.map((s, i) => (i === index ? { ...s, ...patch } : s));
}

function TrashIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#888" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <polyline points="3 6 5 6 21 6" />
      <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
    </svg>
  );
}

// -- duration stepper button -------------------------------------------

function StepperButton({ label, enabled, onPress }: {
  label: string;
  enabled: boolean;
  onPress: (e: ReactPointerEvent) => void;
}) {
  const [hover, setHover] = useState(false);
  return (
    <span
      style={{
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        width: "22px",
        height: "22px",
        fontSize: "10px",
        color: enabled ? "#ccc" : "#555",
        cursor: enabled ? "pointer" : "default",
        userSelect: "none",
        opacity: enabled ? 1 : 0.4,
        background: enabled && hover ? "#444" : "transparent",
      }}
      // Hover is tracked while disabled too, so a button that re-enables under or away from the pointer shows the right colour.
      onPointerEnter={() => setHover(true)}
      onPointerLeave={() => setHover(false)}
      onPointerDown={enabled ? onPress : stopPropagation}
    >
      {label}
    </span>
  );
}

function TrashButton({ onPress }: { onPress: (e: ReactPointerEvent) => void }) {
  const [hover, setHover] = useState(false);
  return (
    <span
      style={{
        cursor: "pointer",
        flexShrink: "0",
        display: "flex",
        alignItems: "center",
        opacity: hover ? 1 : 0.5,
      }}
      onPointerEnter={() => setHover(true)}
      onPointerLeave={() => setHover(false)}
      onPointerDown={onPress}
    >
      <TrashIcon />
    </span>
  );
}

// -- description field --------------------------------------------------

function DescriptionField({ description, disabled, onInput, onCommit }: {
  description: string;
  disabled: boolean;
  onInput: (text: string) => void;
  onCommit: (text: string) => void;
}) {
  const [focused, setFocused] = useState(false);
  const atLimit = description.length >= MAX_DESCRIPTION_LENGTH;
  return (
    <>
      <textarea
        placeholder={PLACEHOLDER}
        value={description}
        disabled={disabled}
        maxLength={MAX_DESCRIPTION_LENGTH}
        style={{
          width: "100%",
          minHeight: "72px",
          marginTop: "4px",
          padding: "8px 10px",
          background: "#252525",
          border: `1px solid ${atLimit ? "#c44" : focused ? "#555" : "#333"}`,
          borderRadius: "6px",
          color: "#ccc",
          fontSize: "12px",
          fontFamily: "inherit",
          resize: "vertical",
          boxSizing: "border-box",
          outline: "none",
          userSelect: "text",
          WebkitUserSelect: "text",
        }}
        onFocus={() => setFocused(true)}
        onBlur={(e) => {
          setFocused(false);
          onCommit(e.currentTarget.value);
        }}
        onInput={(e) => onInput(e.currentTarget.value)}
        // Prevent node-level drag and keyboard shortcuts (e.g. Delete deleting the node)
        onPointerDown={stopPropagation}
        onMouseDown={stopPropagation}
        onKeyDown={stopPropagation}
      />
      <div
        style={{
          fontSize: "10px",
          color: atLimit ? "#c44" : "#555",
          textAlign: "right",
          marginTop: "2px",
        }}
      >
        {`${description.length} / ${MAX_DESCRIPTION_LENGTH}`}
      </div>
    </>
  );
}

// -- add-shot button ------------------------------------------------------

function AddButton({ disabled, onPress }: {
  disabled: boolean;
  onPress: (e: ReactPointerEvent) => void;
}) {
  const [hover, setHover] = useState(false);
  return (
    <div
      style={{
        display: "inline-flex",
        alignItems: "center",
        gap: "4px",
        padding: "5px 12px",
        marginTop: "4px",
        background: !disabled && hover ? "#333" : "#252525",
        border: "1px solid #333",
        borderRadius: "6px",
        color: "#ccc",
        fontSize: "12px",
        cursor: disabled ? "not-allowed" : "pointer",
        opacity: disabled ? 0.4 : 1,
        alignSelf: "flex-start",
        userSelect: "none",
      }}
      // Hover is tracked while disabled too, so a button that re-enables under or away from the pointer shows the right colour.
      onPointerEnter={() => setHover(true)}
      onPointerLeave={() => setHover(false)}
      onPointerDown={disabled ? undefined : onPress}
    >
      <span style={{ fontSize: "14px" }}>+</span>{" Shot"}
    </div>
  );
}

// -- editor -----------------------------------------------------------------

function MultiShotEditor({ value, onChange, disabled = false }: KlingMultiShotEditorProps) {
  const nextShotId = useRef(1);
  // Seeded from the value at mount; later values are not applied.
  const [shots, setShots] = useState<KlingShot[]>(() =>
    Array.isArray(value) && value.length > 0
      ? value.map((s) => assignId({ ...s }, nextShotId))
      : [assignId({ name: "Shot1", duration: 3, description: "" }, nextShotId)],
  );
  const [drag, setDrag] = useState<DragView | null>(null);
  const dragRef = useRef<DragScratch | null>(null);
  const listRef = useRef<HTMLDivElement>(null);

  function commit(next: KlingShot[]) {
    if (disabled || !onChange) return;
    onChange(next.map((s) => ({ ...s })));
  }

  function apply(next: KlingShot[]) {
    setShots(next);
    commit(next);
  }

  const total = totalDuration(shots);
  const remaining = MAX_TOTAL_DURATION - total;
  const addDisabled = disabled || shots.length >= MAX_SHOTS || remaining < 1;

  // -- shot actions -------------------------------------------------------

  function onAdd(e: ReactPointerEvent) {
    e.stopPropagation();
    const defaultDur = Math.min(2, remaining);
    apply([
      ...shots,
      assignId({ name: `Shot${shots.length + 1}`, duration: defaultDur, description: "" }, nextShotId),
    ]);
  }

  function onTrash(e: ReactPointerEvent, index: number) {
    e.stopPropagation();
    if (shots.length <= 1) return;
    const next = renumberShots(shots.filter((_, i) => i !== index));
    const nextTotal = totalDuration(next);
    if (nextTotal < MIN_TOTAL_DURATION) {
      const last = next.length - 1;
      next[last] = { ...next[last]!, duration: next[last]!.duration + MIN_TOTAL_DURATION - nextTotal };
    }
    apply(next);
  }

  // -- drag-and-drop reordering ---------------------------------------------

  function onDragStart(e: ReactPointerEvent<HTMLElement>, index: number) {
    if (disabled) return;
    e.preventDefault();
    e.stopPropagation();

    const itemEl = e.currentTarget.closest<HTMLElement>(".shot-item");
    if (!itemEl) return;

    const rect = itemEl.getBoundingClientRect();

    // Create a floating clone for visual feedback
    const clone = itemEl.cloneNode(true) as HTMLElement;
    Object.assign(clone.style, {
      position: "fixed",
      left: `${rect.left}px`,
      top: `${rect.top}px`,
      width: `${rect.width}px`,
      opacity: "0.85",
      pointerEvents: "none",
      zIndex: "9999",
      boxShadow: "0 4px 16px rgba(0,0,0,0.6)",
      background: "#1a1a1a",
      borderRadius: "6px",
    });
    document.body.appendChild(clone);

    // Collect all shot item rects for hit testing
    const rects = Array.from(listRef.current?.querySelectorAll<HTMLElement>(".shot-item") ?? []).map(
      (el) => ({ rect: el.getBoundingClientRect(), index: Number(el.dataset.index) }),
    );

    dragRef.current = { source: index, over: null, offsetY: e.clientY - rect.top, clone, rects };
    setDrag({ source: index, over: null });
  }

  const dragging = drag !== null;
  useEffect(() => {
    if (!dragging) return;

    const onDragMove = (e: PointerEvent) => {
      const d = dragRef.current;
      if (!d) return;
      e.preventDefault();

      d.clone.style.top = `${e.clientY - d.offsetY}px`;

      // Determine which item we're hovering over
      const hit = d.rects.find((item) => e.clientY < item.rect.top + item.rect.height / 2);
      const over = hit ? hit.index : shots.length;
      if (over !== d.over) {
        d.over = over;
        setDrag({ source: d.source, over });
      }
    };

    const onDragEnd = () => {
      const d = dragRef.current;
      if (!d) return;
      d.clone.remove();
      dragRef.current = null;
      setDrag(null);

      if (d.over !== null && d.over !== d.source) {
        const next = [...shots];
        const [moved] = next.splice(d.source, 1);
        const insertAt = d.over > d.source ? d.over - 1 : d.over;
        next.splice(insertAt, 0, moved!);
        apply(renumberShots(next));
      }
    };

    document.addEventListener("pointermove", onDragMove);
    document.addEventListener("pointerup", onDragEnd);
    return () => {
      document.removeEventListener("pointermove", onDragMove);
      document.removeEventListener("pointerup", onDragEnd);
    };
  });

  // Removes a drag clone left on the page when the widget goes away mid-drag.
  useEffect(() => () => {
    dragRef.current?.clone.remove();
    dragRef.current = null;
  }, []);

  // -- render ---------------------------------------------------------------

  const lastIndex = shots.length - 1;
  const durationColor = total > MAX_TOTAL_DURATION || total < MIN_TOTAL_DURATION ? "#c44" : "#666";

  return (
    <div
      ref={listRef}
      className="multi-shot-container nodrag nowheel"
      style={{
        display: "flex",
        flexDirection: "column",
        gap: "0",
        padding: "8px",
        backgroundColor: "#1a1a1a",
        borderRadius: "6px",
        userSelect: "none",
        boxSizing: "border-box",
        width: "100%",
        height: "100%",
        maxHeight: "100%",
        minHeight: "0",
        overflowY: "auto",
        overflowX: "hidden",
        overscrollBehavior: "contain",
      }}
    >
      {shots.map((shot, index) => {
        // duration stepper (+ value -)
        const budget = durationBudgetFor(shots, index);
        const canIncrease = !disabled && shot.duration < MAX_DURATION && shot.duration < budget;
        const canDecrease = !disabled && shot.duration > MIN_DURATION && total - 1 >= MIN_TOTAL_DURATION;
        const isTarget = drag !== null && index !== drag.source;

        return (
          <div
            key={shot.id}
            className="shot-item"
            data-index={index}
            style={{
              display: "flex",
              flexDirection: "column",
              gap: "0",
              borderBottom: isTarget && drag.over === shots.length && index === lastIndex ? DROP_MARKER : ITEM_DIVIDER,
              paddingBottom: "8px",
              marginBottom: "8px",
              borderTop: isTarget && drag.over === index ? DROP_MARKER : undefined,
              opacity: drag?.source === index ? 0.3 : undefined,
            }}
          >
            {/* header row */}
            <div style={{ display: "flex", alignItems: "center", gap: "8px", height: "32px" }}>
              {/* drag handle */}
              <span
                className="drag-handle"
                style={{
                  fontSize: "18px",
                  color: "#666",
                  cursor: disabled ? "default" : "grab",
                  flexShrink: "0",
                  width: "20px",
                  textAlign: "center",
                  lineHeight: "1",
                }}
                onPointerDown={disabled ? undefined : (e) => onDragStart(e, index)}
              >
                |||
              </span>
              {/* shot name badge */}
              <span
                style={{
                  background: "#333",
                  color: "#ddd",
                  fontSize: "12px",
                  fontWeight: "600",
                  padding: "3px 10px",
                  borderRadius: "4px",
                  whiteSpace: "nowrap",
                }}
              >
                {shot.name}
              </span>
              <div
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: "0",
                  background: "#333",
                  borderRadius: "4px",
                  overflow: "hidden",
                }}
              >
                <StepperButton
                  label="+"
                  enabled={canIncrease}
                  onPress={(e) => {
                    e.stopPropagation();
                    const maxAllowed = Math.min(MAX_DURATION, budget);
                    apply(withShot(shots, index, { duration: Math.min(maxAllowed, shot.duration + 1) }));
                  }}
                />
                <span
                  style={{
                    color: "#ddd",
                    fontSize: "12px",
                    padding: "2px 8px",
                    minWidth: "32px",
                    textAlign: "center",
                    whiteSpace: "nowrap",
                  }}
                >
                  {`${shot.duration} s`}
                </span>
                <StepperButton
                  label="-"
                  enabled={canDecrease}
                  onPress={(e) => {
                    e.stopPropagation();
                    apply(withShot(shots, index, { duration: Math.max(MIN_DURATION, shot.duration - 1) }));
                  }}
                />
              </div>
              {/* spacer */}
              <div style={{ flex: 1 }} />
              {disabled ? null : <TrashButton onPress={(e) => onTrash(e, index)} />}
            </div>
            <DescriptionField
              description={shot.description || ""}
              disabled={disabled}
              onInput={(text) => setShots(withShot(shots, index, { description: text }))}
              onCommit={(text) => commit(withShot(shots, index, { description: text }))}
            />
          </div>
        );
      })}

      {/* status bar */}
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          fontSize: "10px",
          color: "#666",
          padding: "4px 2px",
          marginBottom: "2px",
        }}
      >
        <span>{`${shots.length} / ${MAX_SHOTS} shots`}</span>
        <span style={{ color: durationColor }}>
          {`${total}s (${MIN_TOTAL_DURATION}-${MAX_TOTAL_DURATION}s)`}
        </span>
      </div>

      <AddButton disabled={addDisabled} onPress={onAdd} />

      {/* version label */}
      <div
        style={{
          fontSize: "9px",
          color: "#444",
          textAlign: "right",
          marginTop: "4px",
        }}
      >
        {`v${WIDGET_VERSION}`}
      </div>
    </div>
  );
}

export default reactWidget(MultiShotEditor);
