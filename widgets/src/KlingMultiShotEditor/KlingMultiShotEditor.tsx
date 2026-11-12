/**
 * MultiShotEditor - A list-based editor for managing video shot sequences.
 * Supports adding, deleting, reordering (drag-and-drop), duration selection,
 * and per-shot description text.
 */

import type { KlingShot, KlingShotList } from './_types';
import type { WidgetProps } from '@griptape-ai/nodes-widget-api';

export type { KlingShot, KlingShotList } from './_types';

export type KlingMultiShotEditorProps = WidgetProps<KlingShotList>;

interface ShotItemRect {
  el: HTMLElement;
  rect: DOMRect;
  index: number;
}

const WIDGET_VERSION = "0.6.1";

const stopPropagation = (e: Event) => e.stopPropagation();

function TrashIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#888" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
      <polyline points="3 6 5 6 21 6" />
      <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
    </svg>
  );
}

// -- duration stepper button -------------------------------------------

function StepperButton({ label, enabled, onPress }: {
  label: string;
  enabled: boolean;
  onPress: (e: PointerEvent) => void;
}) {
  const btn = (
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
      }}
      onPointerEnter={enabled ? () => { btn.style.background = "#444"; } : undefined}
      onPointerLeave={enabled ? () => { btn.style.background = "transparent"; } : undefined}
      onPointerDown={enabled ? onPress : stopPropagation}
    >
      {label}
    </span>
  ) as HTMLSpanElement;
  return btn;
}

export default function MultiShotEditor(
  container: HTMLElement,
  props: KlingMultiShotEditorProps,
): () => void {
  const { value, onChange, disabled } = props;

  let nextShotId = 1;

  function assignId(shot: KlingShot): KlingShot {
    if (!shot.id) {
      shot.id = `shot-${nextShotId++}`;
    } else {
      const num = parseInt(shot.id.replace("shot-", ""), 10);
      if (!isNaN(num) && num >= nextShotId) {
        nextShotId = num + 1;
      }
    }
    return shot;
  }

  let shots: KlingShot[] =
    Array.isArray(value) && value.length > 0
      ? value.map((s) => assignId({ ...s }))
      : [assignId({ name: "Shot1", duration: 3, description: "" })];

  let dragSourceIndex: number | null = null;
  let dragOverIndex: number | null = null;

  const MAX_SHOTS = 6;
  const MIN_SHOTS = 1;
  const MAX_DESCRIPTION_LENGTH = 512;
  const MAX_TOTAL_DURATION = 15;
  const MIN_TOTAL_DURATION = 3;
  const MIN_DURATION = 1;
  const MAX_DURATION = 15;
  const PLACEHOLDER =
    "Describe the shot information, such as: who is where and what is happening.";

  // -- helpers ----------------------------------------------------------

  function renumberShots() {
    shots.forEach((shot, i) => {
      shot.name = `Shot${i + 1}`;
    });
  }

  function totalDuration() {
    return shots.reduce((sum, s) => sum + (s.duration || 0), 0);
  }

  function durationBudgetFor(index: number) {
    const othersTotal = shots.reduce(
      (sum, s, i) => sum + (i === index ? 0 : s.duration || 0),
      0,
    );
    return MAX_TOTAL_DURATION - othersTotal;
  }

  function emitChange() {
    if (disabled || !onChange) return;
    onChange(shots.map((s) => ({ ...s })));
  }

  // -- render ----------------------------------------------------------

  function render() {
    container.innerHTML = "";

    container.append(
      <div
        class="multi-shot-container nodrag nowheel"
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
        {shots.map((shot, index) => buildShotItem(shot, index))}
        {buildStatusBar()}
        {buildAddButton()}
        {buildVersionLabel()}
      </div>,
    );
  }

  // -- shot item -------------------------------------------------------

  function buildShotItem(shot: KlingShot, index: number) {
    // duration stepper (+ value -)
    const budget = durationBudgetFor(index);
    const wouldGoBelow = totalDuration() - 1 < MIN_TOTAL_DURATION;
    const canIncrease = !disabled && shot.duration < MAX_DURATION && shot.duration < budget;
    const canDecrease = !disabled && shot.duration > MIN_DURATION && !wouldGoBelow;

    const onIncrease = (e: PointerEvent) => {
      e.stopPropagation();
      const maxAllowed = Math.min(MAX_DURATION, budget);
      shots[index]!.duration = Math.min(maxAllowed, shots[index]!.duration + 1);
      emitChange();
      render();
    };

    const onDecrease = (e: PointerEvent) => {
      e.stopPropagation();
      shots[index]!.duration = Math.max(MIN_DURATION, shots[index]!.duration - 1);
      emitChange();
      render();
    };

    const onTrash = (e: PointerEvent) => {
      e.stopPropagation();
      if (shots.length <= 1) return;
      shots.splice(index, 1);
      renumberShots();
      const total = totalDuration();
      if (total < MIN_TOTAL_DURATION) {
        const lastShot = shots[shots.length - 1]!;
        lastShot.duration += MIN_TOTAL_DURATION - total;
      }
      emitChange();
      render();
    };

    const trash = disabled ? null : (
      <span
        style={{
          cursor: "pointer",
          flexShrink: "0",
          display: "flex",
          alignItems: "center",
          opacity: 0.5,
        }}
        onPointerEnter={() => { trash!.style.opacity = "1"; }}
        onPointerLeave={() => { trash!.style.opacity = "0.5"; }}
        onPointerDown={onTrash}
      >
        <TrashIcon />
      </span>
    ) as HTMLSpanElement;

    // -- description textarea --
    const descLength = (shot.description || "").length;
    const textarea = (
      <textarea
        placeholder={PLACEHOLDER}
        value={shot.description || ""}
        disabled={disabled}
        maxLength={MAX_DESCRIPTION_LENGTH}
        style={{
          width: "100%",
          minHeight: "72px",
          marginTop: "4px",
          padding: "8px 10px",
          background: "#252525",
          border: `1px solid ${descLength >= MAX_DESCRIPTION_LENGTH ? "#c44" : "#333"}`,
          borderRadius: "6px",
          color: "#ccc",
          fontSize: "12px",
          fontFamily: "inherit",
          resize: "vertical",
          boxSizing: "border-box",
          outline: "none",
          userSelect: "text",
          webkitUserSelect: "text",
        }}
        onFocus={() => {
          const len = shots[index]!.description.length;
          textarea.style.borderColor =
            len >= MAX_DESCRIPTION_LENGTH ? "#c44" : "#555";
        }}
        onBlur={() => {
          shots[index]!.description = textarea.value;
          const len = textarea.value.length;
          textarea.style.borderColor =
            len >= MAX_DESCRIPTION_LENGTH ? "#c44" : "#333";
          counter.textContent = `${len} / ${MAX_DESCRIPTION_LENGTH}`;
          counter.style.color = len >= MAX_DESCRIPTION_LENGTH ? "#c44" : "#555";
          emitChange();
        }}
        onInput={(e) => {
          const target = e.target as HTMLTextAreaElement;
          shots[index]!.description = target.value;
          counter.textContent = `${target.value.length} / ${MAX_DESCRIPTION_LENGTH}`;
          counter.style.color =
            target.value.length >= MAX_DESCRIPTION_LENGTH ? "#c44" : "#555";
          textarea.style.borderColor =
            target.value.length >= MAX_DESCRIPTION_LENGTH ? "#c44" : "#555";
        }}
        // Prevent node-level drag and keyboard shortcuts (e.g. Delete deleting the node)
        onPointerDown={stopPropagation}
        onMouseDown={stopPropagation}
        onKeyDown={stopPropagation}
      />
    ) as HTMLTextAreaElement;

    const counter = (
      <div
        style={{
          fontSize: "10px",
          color: descLength >= MAX_DESCRIPTION_LENGTH ? "#c44" : "#555",
          textAlign: "right",
          marginTop: "2px",
        }}
      >
        {`${descLength} / ${MAX_DESCRIPTION_LENGTH}`}
      </div>
    ) as HTMLDivElement;

    return (
      <div
        class="shot-item"
        data-index={index}
        style={{
          display: "flex",
          flexDirection: "column",
          gap: "0",
          borderBottom: "1px solid #2a2a2a",
          paddingBottom: "8px",
          marginBottom: "8px",
          borderTop: dragOverIndex === index ? "2px solid #4a9eff" : null,
        }}
      >
        {/* header row */}
        <div style="display: flex; align-items: center; gap: 8px; height: 32px;">
          {/* drag handle */}
          <span
            class="drag-handle"
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
            <StepperButton label="+" enabled={canIncrease} onPress={onIncrease} />
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
            <StepperButton label="-" enabled={canDecrease} onPress={onDecrease} />
          </div>
          {/* spacer */}
          <div style="flex: 1;" />
          {trash}
        </div>
        {textarea}
        {counter}
      </div>
    ) as HTMLDivElement;
  }

  // -- status bar ------------------------------------------------------

  function buildStatusBar() {
    const total = totalDuration();
    const overBudget = total > MAX_TOTAL_DURATION;
    const underBudget = total < MIN_TOTAL_DURATION;
    const durationColor = overBudget || underBudget ? "#c44" : "#666";

    return (
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
        <span style={`color: ${durationColor};`}>
          {`${total}s (${MIN_TOTAL_DURATION}-${MAX_TOTAL_DURATION}s)`}
        </span>
      </div>
    );
  }

  // -- add-shot button -------------------------------------------------

  function buildAddButton() {
    const atMaxShots = shots.length >= MAX_SHOTS;
    const remaining = MAX_TOTAL_DURATION - totalDuration();
    const cannotFitAnother = remaining < 1;
    const addDisabled = disabled || atMaxShots || cannotFitAnother;

    const onAdd = (e: PointerEvent) => {
      e.stopPropagation();
      const defaultDur = Math.min(2, remaining);
      shots.push(
        assignId({
          name: `Shot${shots.length + 1}`,
          duration: defaultDur,
          description: "",
        }),
      );
      emitChange();
      render();
    };

    const btn = (
      <div
        style={{
          display: "inline-flex",
          alignItems: "center",
          gap: "4px",
          padding: "5px 12px",
          marginTop: "4px",
          background: "#252525",
          border: "1px solid #333",
          borderRadius: "6px",
          color: "#ccc",
          fontSize: "12px",
          cursor: addDisabled ? "not-allowed" : "pointer",
          opacity: addDisabled ? 0.4 : 1,
          alignSelf: "flex-start",
          userSelect: "none",
        }}
        onPointerEnter={addDisabled ? undefined : () => { btn.style.background = "#333"; }}
        onPointerLeave={addDisabled ? undefined : () => { btn.style.background = "#252525"; }}
        onPointerDown={addDisabled ? undefined : onAdd}
      >
        <span style="font-size:14px;">+</span>{" Shot"}
      </div>
    ) as HTMLDivElement;

    return btn;
  }

  // -- version label --------------------------------------------------

  function buildVersionLabel() {
    return (
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
    );
  }

  // -- drag-and-drop reordering ----------------------------------------

  let dragClone: HTMLElement | null = null;
  let dragOffsetY = 0;
  let shotItems: ShotItemRect[] = [];

  function onDragStart(e: PointerEvent, index: number) {
    if (disabled) return;
    e.preventDefault();
    e.stopPropagation();

    dragSourceIndex = index;
    dragOverIndex = null;

    const itemEl = (e.target as Element).closest<HTMLElement>(".shot-item");
    if (!itemEl) return;

    const rect = itemEl.getBoundingClientRect();
    dragOffsetY = e.clientY - rect.top;

    // Create a floating clone for visual feedback
    dragClone = itemEl.cloneNode(true) as HTMLElement;
    dragClone.style.position = "fixed";
    dragClone.style.left = `${rect.left}px`;
    dragClone.style.top = `${rect.top}px`;
    dragClone.style.width = `${rect.width}px`;
    dragClone.style.opacity = "0.85";
    dragClone.style.pointerEvents = "none";
    dragClone.style.zIndex = "9999";
    dragClone.style.boxShadow = "0 4px 16px rgba(0,0,0,0.6)";
    dragClone.style.background = "#1a1a1a";
    dragClone.style.borderRadius = "6px";
    document.body.appendChild(dragClone);

    // Dim the original
    itemEl.style.opacity = "0.3";

    // Collect all shot item rects for hit testing
    shotItems = Array.from(container.querySelectorAll<HTMLElement>(".shot-item")).map(
      (el) => ({
        el,
        rect: el.getBoundingClientRect(),
        index: Number(el.dataset.index),
      }),
    );

    document.addEventListener("pointermove", onDragMove);
    document.addEventListener("pointerup", onDragEnd);
  }

  function onDragMove(e: PointerEvent) {
    if (dragSourceIndex === null || !dragClone) return;
    e.preventDefault();

    dragClone.style.top = `${e.clientY - dragOffsetY}px`;

    // Determine which item we're hovering over
    let newOverIndex = null;
    for (const item of shotItems) {
      const midY = item.rect.top + item.rect.height / 2;
      if (e.clientY < midY) {
        newOverIndex = item.index;
        break;
      }
    }
    if (newOverIndex === null) {
      newOverIndex = shots.length;
    }

    if (newOverIndex !== dragOverIndex) {
      dragOverIndex = newOverIndex;
      const lastIndex = shots.length - 1;
      shotItems.forEach((item) => {
        item.el.style.borderTop = "none";
        item.el.style.borderBottom = "1px solid #2a2a2a";

        if (item.index === dragSourceIndex) return;

        if (dragOverIndex === shots.length && item.index === lastIndex) {
          item.el.style.borderBottom = "2px solid #4a9eff";
        } else if (item.index === dragOverIndex) {
          item.el.style.borderTop = "2px solid #4a9eff";
        }
      });
    }
  }

  function onDragEnd(_e: PointerEvent) {
    document.removeEventListener("pointermove", onDragMove);
    document.removeEventListener("pointerup", onDragEnd);

    if (dragClone) {
      dragClone.remove();
      dragClone = null;
    }

    if (
      dragSourceIndex !== null &&
      dragOverIndex !== null &&
      dragOverIndex !== dragSourceIndex
    ) {
      const [moved] = shots.splice(dragSourceIndex, 1);
      const insertAt =
        dragOverIndex > dragSourceIndex ? dragOverIndex - 1 : dragOverIndex;
      shots.splice(insertAt, 0, moved!);
      renumberShots();
      emitChange();
    }

    dragSourceIndex = null;
    dragOverIndex = null;
    shotItems = [];
    render();
  }

  // -- init -------------------------------------------------------------

  render();

  // -- cleanup ---------------------------------------------------------

  return () => {
    document.removeEventListener("pointermove", onDragMove);
    document.removeEventListener("pointerup", onDragEnd);
    if (dragClone) {
      dragClone.remove();
      dragClone = null;
    }
  };
}
