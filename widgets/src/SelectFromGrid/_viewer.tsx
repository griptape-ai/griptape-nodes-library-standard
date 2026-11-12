// _viewer.tsx — Fullscreen viewer for the SelectFromGrid widget.
//
// A modal over the whole page that shows one item at a time at full size.
//
// Layout:
//   [label]                                    [N / M]  [Select]  [×]
//   [‹]                  current item                              [›]
//   [selected strip: thumbnails of picked items, each with an ×        ]
//
// Keys: ←/→ step, Space toggles selection of the current item, Esc closes.
// Swiping the stage left/right also steps.
//
// The viewer holds no selection state of its own: it reads the widget's state
// through the getters and asks the widget to change it through the callbacks,
// then the widget calls refresh() so both views stay in step.
//
// createViewer(opts) → { open(idx), close(), refresh(), destroy() }

import { CARD_HUES } from './_styles';
import { Icon } from './_icons';
import type { GridItem } from './_types';

const SWIPE_THRESHOLD = 50;

export interface ViewerOptions {
  getItems: () => GridItem[];
  getSelected: () => number[];
  isMultiSelect: () => boolean;
  isDisabled: () => boolean;
  onToggle: (idx: number) => void;
  onRemove: (idx: number) => void;
}

export interface Viewer {
  open: (idx: number) => void;
  close: () => void;
  refresh: () => void;
  destroy: () => void;
}

export function createViewer({
  getItems, getSelected, isMultiSelect, isDisabled,
  onToggle, onRemove,
}: ViewerOptions): Viewer {
  let current  = 0;
  let open     = false;
  let shownKey: string | null = null;   // what the stage is showing, so refresh() skips reloading unchanged media

  // ── DOM ────────────────────────────────────────────────────────────────────

  const titleEl   = (<div class="sfg-viewer-title" />) as HTMLDivElement;
  const counterEl = (<span class="sfg-viewer-counter" />) as HTMLSpanElement;

  const selectBtn = (
    <button type="button" class="sfg-viewer-select" onClick={() => toggleCurrent()} />
  ) as HTMLButtonElement;

  const prevBtn = (
    <button type="button" class="sfg-viewer-nav prev" title="Previous (←)" onClick={() => step(-1)}>
      <Icon name="chevron-left" size={28} />
    </button>
  ) as HTMLButtonElement;

  const nextBtn = (
    <button type="button" class="sfg-viewer-nav next" title="Next (→)" onClick={() => step(1)}>
      <Icon name="chevron-right" size={28} />
    </button>
  ) as HTMLButtonElement;

  let swipe: { x: number; y: number } | null = null;
  const stage = (
    <div
      class="sfg-viewer-stage"
      onPointerDown={(e) => {
        // Leave media controls and text selection / scrolling to the element
        if ((e.target as Element).closest("video, audio, .sfg-viewer-text, .sfg-viewer-dict")) return;
        swipe = { x: e.clientX, y: e.clientY };
      }}
      onPointerUp={(e) => {
        if (!swipe) return;
        const dx = e.clientX - swipe.x;
        const dy = e.clientY - swipe.y;
        swipe = null;
        if (Math.abs(dx) > SWIPE_THRESHOLD && Math.abs(dx) > Math.abs(dy)) step(dx < 0 ? 1 : -1);
      }}
      onPointerCancel={() => { swipe = null; }}
    />
  ) as HTMLDivElement;

  const stripHint  = (<div class="sfg-viewer-strip-hint" />) as HTMLDivElement;
  const stripItems = (<div class="sfg-viewer-strip-items" />) as HTMLDivElement;

  const overlay = (
    <div class="sfg-viewer" tabIndex={-1} role="dialog" aria-modal="true">
      <div class="sfg-viewer-header">
        {titleEl}
        {counterEl}
        {selectBtn}
        <button type="button" class="sfg-viewer-icon-btn" title="Close (Esc)" onClick={() => close()}>
          <Icon name="x" size={18} />
        </button>
      </div>
      <div class="sfg-viewer-body">
        {prevBtn}
        {stage}
        {nextBtn}
      </div>
      <div class="sfg-viewer-strip">
        {stripHint}
        {stripItems}
      </div>
    </div>
  ) as HTMLDivElement;

  // ── Rendering ──────────────────────────────────────────────────────────────

  function TextBlock({ className, text }: { className: string; text: string }) {
    return <div class={className} textContent={text} />;
  }

  function ErrorCard() {
    return <div class="sfg-error-card"><Icon name="alert-circle" size={40} /></div>;
  }

  // Media identity only: a thumbnail or label arriving later shouldn't restart playback
  function stageKey(item: GridItem | undefined) {
    return item ? JSON.stringify([item.type, item.url, item.full_url, item.value]) : null;
  }

  function renderStage() {
    stage.querySelectorAll<HTMLMediaElement>("video, audio").forEach((m) => m.pause());
    stage.innerHTML = "";

    const items = getItems();
    const item  = items[current];
    shownKey = stageKey(item);
    if (!item) return;

    switch (item.type) {
      case "image": {
        let triedPreview = !item.full_url || !item.url;
        const img = (
          <img
            src={item.full_url || item.url || ""}
            alt={item.label || ""}
            onError={() => {
              // Fall back to the grid's preview before giving up
              if (triedPreview) { img.replaceWith(<ErrorCard />); return; }
              triedPreview = true;
              img.src = item.url as string;
            }}
          />
        ) as HTMLImageElement;
        stage.appendChild(img);
        break;
      }
      case "video": {
        const vid = (
          <video
            src={item.url || ""}
            poster={item.thumbnail || null}
            controls loop playsInline
            onError={() => vid.replaceWith(<ErrorCard />)}
          />
        ) as HTMLVideoElement;
        stage.appendChild(vid);
        void vid.play().catch(() => {});
        break;
      }
      case "audio": {
        const card = (
          <div class="sfg-viewer-audio">
            <Icon name="music" size={72} />
            {item.url ? (
              <audio src={item.url} controls onError={() => card.replaceWith(<ErrorCard />)} />
            ) : null}
          </div>
        ) as HTMLDivElement;
        stage.appendChild(card);
        break;
      }
      case "dict":
        stage.appendChild(<TextBlock className="sfg-viewer-dict" text={(item.value as string) || "{}"} />);
        break;
      default:
        stage.appendChild(
          <TextBlock
            className="sfg-viewer-text"
            text={item.value !== undefined ? String(item.value) : (item.label || "")}
          />,
        );
        break;
    }
  }

  function mkThumb(item: GridItem, idx: number) {
    const thumb = (
      <div
        class={"sfg-viewer-thumb" + (idx === current ? " current" : "")}
        title={item.label || `Item ${idx + 1}`}
        data-idx={String(idx)}
        onClick={() => show(idx)}
      />
    ) as HTMLDivElement;

    switch (item.type) {
      case "image":
        thumb.appendChild(<img src={item.url || ""} alt="" decoding="async" />);
        break;
      case "video": {
        const Frame = () => (
          <video src={item.url ? `${item.url}#t=0.1` : ""} muted preload="metadata" />
        );
        if (item.thumbnail) {
          const img = (<img src={item.thumbnail} alt="" />) as HTMLImageElement;
          img.addEventListener("error", () => img.replaceWith(<Frame />), { once: true });
          thumb.appendChild(img);
        } else {
          thumb.appendChild(<Frame />);
        }
        break;
      }
      case "audio":
        thumb.classList.add("sfg-viewer-thumb-icon");
        thumb.appendChild(<Icon name="music" size={22} />);
        break;
      default: {
        const text = item.type === "dict"
          ? ((item.value as string) || "{}")
          : (item.value !== undefined ? String(item.value) : (item.label || ""));
        thumb.style.background = `hsla(${CARD_HUES[idx % CARD_HUES.length]}, 22%, 50%, 0.18)`;
        thumb.appendChild(<TextBlock className="sfg-viewer-thumb-text" text={text} />);
        break;
      }
    }

    if (!isDisabled()) {
      thumb.appendChild(
        <button
          type="button" class="sfg-viewer-thumb-remove" title="Remove from selection"
          onClick={(e) => {
            e.stopPropagation();
            onRemove(idx);
          }}
        >
          <Icon name="x" size={12} />
        </button>,
      );
    }

    return thumb;
  }

  function renderChrome() {
    const items    = getItems();
    const selected = getSelected();
    const item     = items[current];
    const isSel    = selected.includes(current);
    const multi    = isMultiSelect();

    titleEl.textContent   = item?.label || "";
    counterEl.textContent = items.length ? `${current + 1} / ${items.length}` : "";
    prevBtn.disabled = current <= 0;
    nextBtn.disabled = current >= items.length - 1;

    selectBtn.disabled = isDisabled();
    selectBtn.classList.toggle("active", isSel);
    selectBtn.replaceChildren();
    if (isSel) selectBtn.appendChild(<Icon name="check" size={14} />);
    let label: string;
    if (isSel)                            label = "Selected";
    else if (!multi && selected.length)   label = "Replace selection";
    else                                  label = "Select";
    selectBtn.appendChild(document.createTextNode(label));
    selectBtn.title = "Toggle selection (Space)";

    if (!selected.length) {
      stripHint.textContent = multi ? "Nothing selected · press Space to select" : "Nothing selected";
    } else if (multi) {
      stripHint.textContent = `${selected.length} selected`;
    } else {
      stripHint.textContent = "Selected · selecting another item replaces it";
    }
  }

  // Rebuilds every selected thumbnail. Only needed when the selection or items
  // change, not when stepping.
  function renderStrip() {
    const items = getItems();
    stripItems.replaceChildren();
    getSelected().forEach((idx) => {
      if (items[idx]) stripItems.appendChild(mkThumb(items[idx], idx));
    });
  }

  function updateStripCurrent() {
    for (const thumb of stripItems.children) {
      thumb.classList.toggle("current", Number((thumb as HTMLElement).dataset.idx) === current);
    }
  }

  // ── Navigation ─────────────────────────────────────────────────────────────

  function show(idx: number) {
    const items = getItems();
    if (!items.length) return;
    const next = Math.max(0, Math.min(items.length - 1, idx));
    const changed = next !== current;
    current = next;
    if (changed) renderStage();
    renderChrome();
    updateStripCurrent();
  }

  function step(delta: number) { show(current + delta); }

  function toggleCurrent() {
    if (isDisabled() || !getItems()[current]) return;
    onToggle(current);
  }

  // ── Events ─────────────────────────────────────────────────────────────────

  // Keys are taken in the capture phase on window so the editor's own shortcuts
  // (delete node, undo, …) don't fire underneath the modal.
  function onKeyDown(e: KeyboardEvent) {
    if (!open) return;
    e.stopPropagation();
    switch (e.key) {
      case "ArrowLeft":  e.preventDefault(); step(-1);        break;
      case "ArrowRight": e.preventDefault(); step(1);         break;
      case " ":          e.preventDefault(); if (!e.repeat) toggleCurrent(); break;
      case "Escape":     e.preventDefault(); close();         break;
    }
  }

  // ── API ────────────────────────────────────────────────────────────────────

  function openAt(idx: number) {
    const items = getItems();
    if (!items.length) return;
    current = Math.max(0, Math.min(items.length - 1, idx));
    if (!open) {
      open = true;
      document.body.appendChild(overlay);
      window.addEventListener("keydown", onKeyDown, true);
    }
    renderStage();
    renderChrome();
    renderStrip();
    overlay.focus();
  }

  function close() {
    if (!open) return;
    open = false;
    stage.querySelectorAll<HTMLMediaElement>("video, audio").forEach((m) => m.pause());
    stage.innerHTML = "";
    shownKey = null;
    window.removeEventListener("keydown", onKeyDown, true);
    overlay.remove();
  }

  // Re-read widget state. Called after any selection or item change.
  function refresh() {
    if (!open) return;
    const items = getItems();
    if (!items.length) { close(); return; }
    current = Math.min(current, items.length - 1);
    if (stageKey(items[current]) !== shownKey) renderStage();
    else {
      const vid = stage.querySelector("video");
      const thumbnail = items[current]?.thumbnail;
      if (vid && thumbnail) vid.poster = thumbnail;
    }
    renderChrome();
    renderStrip();
  }

  return {
    open: openAt,
    close,
    refresh,
    destroy: close,
  };
}
