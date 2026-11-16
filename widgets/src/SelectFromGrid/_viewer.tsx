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
// The viewer holds no selection state of its own: it shows the widget's items
// and selection, and asks the widget to change them through the callbacks. The
// widget renders it only while it's open, at the index it keeps for it.

import { useEffect, useLayoutEffect, useRef, useState, type RefObject } from 'react';
import { createPortal } from 'react-dom';

import { CARD_HUES } from './_styles';
import { Icon } from './_icons';
import type { GridItem } from './_types';

const SWIPE_THRESHOLD = 50;

export interface ViewerProps {
  items: GridItem[];
  selected: number[];
  multiSelect: boolean;
  disabled: boolean;
  current: number;
  onShow: (idx: number) => void;
  onClose: () => void;
  onToggle: (idx: number) => void;
  onRemove: (idx: number) => void;
}

function ErrorCard() {
  return <div className="sfg-error-card"><Icon name="alert-circle" size={40} /></div>;
}

function itemText(item: GridItem) {
  return item.type === "dict"
    ? ((item.value as string) || "{}")
    : (item.value !== undefined ? String(item.value) : (item.label || ""));
}

// Media identity only: a thumbnail or label arriving later shouldn't restart playback
function stageKey(item: GridItem) {
  return JSON.stringify([item.type, item.url, item.full_url, item.value]);
}

// Pauses the element's playback when it leaves the stage.
function usePauseOnUnmount(ref: RefObject<HTMLMediaElement | null>) {
  useEffect(() => {
    const media = ref.current;
    return () => media?.pause();
  }, [ref]);
}

function StageImage({ item }: { item: GridItem }) {
  const [src, setSrc] = useState(item.full_url || item.url || "");
  const [failed, setFailed] = useState(false);
  if (failed) return <ErrorCard />;
  return (
    <img
      src={src}
      alt={item.label || ""}
      onError={() => {
        // Fall back to the grid's preview before giving up
        if (item.full_url && item.url && src !== item.url) setSrc(item.url);
        else setFailed(true);
      }}
    />
  );
}

function StageVideo({ item }: { item: GridItem }) {
  const ref = useRef<HTMLVideoElement>(null);
  const [failed, setFailed] = useState(false);
  usePauseOnUnmount(ref);
  useEffect(() => { void ref.current?.play().catch(() => {}); }, []);
  if (failed) return <ErrorCard />;
  return (
    <video
      ref={ref}
      src={item.url || ""}
      poster={item.thumbnail || undefined}
      controls loop playsInline
      onError={() => setFailed(true)}
    />
  );
}

function StageAudio({ item }: { item: GridItem }) {
  const ref = useRef<HTMLAudioElement>(null);
  const [failed, setFailed] = useState(false);
  usePauseOnUnmount(ref);
  if (failed) return <ErrorCard />;
  return (
    <div className="sfg-viewer-audio">
      <Icon name="music" size={72} />
      {item.url ? <audio ref={ref} src={item.url} controls onError={() => setFailed(true)} /> : null}
    </div>
  );
}

function StageItem({ item }: { item: GridItem }) {
  switch (item.type) {
    case "image": return <StageImage item={item} />;
    case "video": return <StageVideo item={item} />;
    case "audio": return <StageAudio item={item} />;
    case "dict":  return <div className="sfg-viewer-dict">{itemText(item)}</div>;
    default:      return <div className="sfg-viewer-text">{itemText(item)}</div>;
  }
}

function ThumbVideoFrame({ url }: { url?: string }) {
  return <video src={url ? `${url}#t=0.1` : ""} muted preload="metadata" />;
}

function ThumbVideo({ item }: { item: GridItem }) {
  const [thumbFailed, setThumbFailed] = useState(false);
  if (!item.thumbnail || thumbFailed) return <ThumbVideoFrame url={item.url} />;
  return <img src={item.thumbnail} alt="" onError={() => setThumbFailed(true)} />;
}

interface ThumbProps {
  item: GridItem;
  idx: number;
  current: boolean;
  removable: boolean;
  onShow: (idx: number) => void;
  onRemove: (idx: number) => void;
}

function Thumb({ item, idx, current, removable, onShow, onRemove }: ThumbProps) {
  const isText = item.type !== "image" && item.type !== "video" && item.type !== "audio";
  return (
    <div
      className={
        "sfg-viewer-thumb" + (item.type === "audio" ? " sfg-viewer-thumb-icon" : "") + (current ? " current" : "")
      }
      title={item.label || `Item ${idx + 1}`}
      data-idx={String(idx)}
      style={isText ? { background: `hsla(${CARD_HUES[idx % CARD_HUES.length]}, 22%, 50%, 0.18)` } : undefined}
      onClick={() => onShow(idx)}
    >
      {item.type === "image" ? (
        <img src={item.url || ""} alt="" decoding="async" />
      ) : item.type === "video" ? (
        <ThumbVideo item={item} />
      ) : item.type === "audio" ? (
        <Icon name="music" size={22} />
      ) : (
        <div className="sfg-viewer-thumb-text">{itemText(item)}</div>
      )}
      {removable ? (
        <button
          type="button" className="sfg-viewer-thumb-remove" title="Remove from selection"
          onClick={(e) => {
            e.stopPropagation();
            onRemove(idx);
          }}
        >
          <Icon name="x" size={12} />
        </button>
      ) : null}
    </div>
  );
}

export function Viewer({
  items, selected, multiSelect, disabled, current,
  onShow, onClose, onToggle, onRemove,
}: ViewerProps) {
  const overlayRef = useRef<HTMLDivElement>(null);
  const swipe = useRef<{ x: number; y: number } | null>(null);

  const item  = items[current];
  const isSel = selected.includes(current);

  const step = (delta: number) => onShow(current + delta);
  const toggleCurrent = () => {
    if (disabled || !item) return;
    onToggle(current);
  };

  // Keys are taken in the capture phase on window so the editor's own shortcuts
  // (delete node, undo, …) don't fire underneath the modal.
  const onKeyDown = (e: KeyboardEvent) => {
    e.stopPropagation();
    switch (e.key) {
      case "ArrowLeft":  e.preventDefault(); step(-1);        break;
      case "ArrowRight": e.preventDefault(); step(1);         break;
      case " ":          e.preventDefault(); if (!e.repeat) toggleCurrent(); break;
      case "Escape":     e.preventDefault(); onClose();       break;
    }
  };
  const onKeyDownRef = useRef(onKeyDown);
  useLayoutEffect(() => { onKeyDownRef.current = onKeyDown; });

  useEffect(() => {
    const listener = (e: KeyboardEvent) => onKeyDownRef.current(e);
    window.addEventListener("keydown", listener, true);
    overlayRef.current?.focus();
    return () => window.removeEventListener("keydown", listener, true);
  }, []);

  let selectLabel: string;
  if (isSel)                                 selectLabel = "Selected";
  else if (!multiSelect && selected.length)  selectLabel = "Replace selection";
  else                                       selectLabel = "Select";

  let stripHint: string;
  if (!selected.length) stripHint = multiSelect ? "Nothing selected · press Space to select" : "Nothing selected";
  else if (multiSelect) stripHint = `${selected.length} selected`;
  else                  stripHint = "Selected · selecting another item replaces it";

  return createPortal(
    <div ref={overlayRef} className="sfg-viewer" tabIndex={-1} role="dialog" aria-modal="true">
      <div className="sfg-viewer-header">
        <div className="sfg-viewer-title">{item?.label || ""}</div>
        <span className="sfg-viewer-counter">{items.length ? `${current + 1} / ${items.length}` : ""}</span>
        <button
          type="button" className={"sfg-viewer-select" + (isSel ? " active" : "")}
          title="Toggle selection (Space)" disabled={disabled}
          onClick={toggleCurrent}
        >
          {isSel ? <Icon name="check" size={14} /> : null}
          {selectLabel}
        </button>
        <button type="button" className="sfg-viewer-icon-btn" title="Close (Esc)" onClick={onClose}>
          <Icon name="x" size={18} />
        </button>
      </div>
      <div className="sfg-viewer-body">
        <button
          type="button" className="sfg-viewer-nav prev" title="Previous (←)"
          disabled={current <= 0} onClick={() => step(-1)}
        >
          <Icon name="chevron-left" size={28} />
        </button>
        <div
          className="sfg-viewer-stage"
          onPointerDown={(e) => {
            // Leave media controls and text selection / scrolling to the element
            if ((e.target as Element).closest("video, audio, .sfg-viewer-text, .sfg-viewer-dict")) return;
            swipe.current = { x: e.clientX, y: e.clientY };
          }}
          onPointerUp={(e) => {
            const start = swipe.current;
            if (!start) return;
            const dx = e.clientX - start.x;
            const dy = e.clientY - start.y;
            swipe.current = null;
            if (Math.abs(dx) > SWIPE_THRESHOLD && Math.abs(dx) > Math.abs(dy)) step(dx < 0 ? 1 : -1);
          }}
          onPointerCancel={() => { swipe.current = null; }}
        >
          {item ? <StageItem key={`${current}:${stageKey(item)}`} item={item} /> : null}
        </div>
        <button
          type="button" className="sfg-viewer-nav next" title="Next (→)"
          disabled={current >= items.length - 1} onClick={() => step(1)}
        >
          <Icon name="chevron-right" size={28} />
        </button>
      </div>
      <div className="sfg-viewer-strip">
        <div className="sfg-viewer-strip-hint">{stripHint}</div>
        <div className="sfg-viewer-strip-items">
          {selected.map((idx) => {
            const selectedItem = items[idx];
            return selectedItem ? (
              <Thumb
                key={idx} item={selectedItem} idx={idx} current={idx === current} removable={!disabled}
                onShow={onShow} onRemove={onRemove}
              />
            ) : null;
          })}
        </div>
      </div>
    </div>,
    document.body,
  );
}
