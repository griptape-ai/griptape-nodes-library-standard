// _viewer.js — Fullscreen viewer for the SelectFromGrid widget.
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

import { CARD_HUES } from './_styles.js';
import { mkIcon } from './_icons.js';

const SWIPE_THRESHOLD = 50;

export function createViewer({
  getItems, getSelected, isMultiSelect, isDisabled,
  onToggle, onRemove,
}) {
  let current  = 0;
  let open     = false;
  let shownKey = null;   // what the stage is showing, so refresh() skips reloading unchanged media

  // ── DOM ────────────────────────────────────────────────────────────────────

  const overlay = document.createElement("div");
  overlay.className = "sfg-viewer";
  overlay.tabIndex  = -1;
  overlay.setAttribute("role", "dialog");
  overlay.setAttribute("aria-modal", "true");

  const header = document.createElement("div");
  header.className = "sfg-viewer-header";

  const titleEl = document.createElement("div");
  titleEl.className = "sfg-viewer-title";

  const counterEl = document.createElement("span");
  counterEl.className = "sfg-viewer-counter";

  const selectBtn = document.createElement("button");
  selectBtn.type = "button";
  selectBtn.className = "sfg-viewer-select";

  const closeBtn = document.createElement("button");
  closeBtn.type = "button";
  closeBtn.className = "sfg-viewer-icon-btn";
  closeBtn.title = "Close (Esc)";
  closeBtn.appendChild(mkIcon("x", 18));

  header.append(titleEl, counterEl, selectBtn, closeBtn);

  const body = document.createElement("div");
  body.className = "sfg-viewer-body";

  const prevBtn = document.createElement("button");
  prevBtn.type = "button";
  prevBtn.className = "sfg-viewer-nav prev";
  prevBtn.title = "Previous (←)";
  prevBtn.appendChild(mkIcon("chevron-left", 28));

  const nextBtn = document.createElement("button");
  nextBtn.type = "button";
  nextBtn.className = "sfg-viewer-nav next";
  nextBtn.title = "Next (→)";
  nextBtn.appendChild(mkIcon("chevron-right", 28));

  const stage = document.createElement("div");
  stage.className = "sfg-viewer-stage";

  body.append(prevBtn, stage, nextBtn);

  const strip = document.createElement("div");
  strip.className = "sfg-viewer-strip";

  const stripHint = document.createElement("div");
  stripHint.className = "sfg-viewer-strip-hint";

  const stripItems = document.createElement("div");
  stripItems.className = "sfg-viewer-strip-items";

  strip.append(stripHint, stripItems);

  overlay.append(header, body, strip);

  // ── Rendering ──────────────────────────────────────────────────────────────

  function mkTextBlock(className, text) {
    const el = document.createElement("div");
    el.className   = className;
    el.textContent = text;
    return el;
  }

  function mkErrorCard() {
    const card = document.createElement("div");
    card.className = "sfg-error-card";
    card.appendChild(mkIcon("alert-circle", 40));
    return card;
  }

  // Media identity only: a thumbnail or label arriving later shouldn't restart playback
  function stageKey(item) {
    return item ? JSON.stringify([item.type, item.url, item.full_url, item.value]) : null;
  }

  function renderStage() {
    stage.querySelectorAll("video, audio").forEach((m) => m.pause());
    stage.innerHTML = "";

    const items = getItems();
    const item  = items[current];
    shownKey = stageKey(item);
    if (!item) return;

    switch (item.type) {
      case "image": {
        const img = document.createElement("img");
        img.src = item.full_url || item.url || "";
        img.alt = item.label || "";
        let triedPreview = !item.full_url || !item.url;
        img.addEventListener("error", () => {
          // Fall back to the grid's preview before giving up
          if (triedPreview) { img.replaceWith(mkErrorCard()); return; }
          triedPreview = true;
          img.src = item.url;
        });
        stage.appendChild(img);
        break;
      }
      case "video": {
        const vid = document.createElement("video");
        vid.src         = item.url || "";
        if (item.thumbnail) vid.poster = item.thumbnail;
        vid.controls    = true;
        vid.loop        = true;
        vid.playsInline = true;
        vid.addEventListener("error", () => vid.replaceWith(mkErrorCard()));
        stage.appendChild(vid);
        void vid.play().catch(() => {});
        break;
      }
      case "audio": {
        const card = document.createElement("div");
        card.className = "sfg-viewer-audio";
        card.appendChild(mkIcon("music", 72));
        if (item.url) {
          const audio = document.createElement("audio");
          audio.src      = item.url;
          audio.controls = true;
          audio.addEventListener("error", () => card.replaceWith(mkErrorCard()));
          card.appendChild(audio);
        }
        stage.appendChild(card);
        break;
      }
      case "dict":
        stage.appendChild(mkTextBlock("sfg-viewer-dict", item.value || "{}"));
        break;
      default:
        stage.appendChild(mkTextBlock(
          "sfg-viewer-text",
          item.value !== undefined ? String(item.value) : (item.label || ""),
        ));
        break;
    }
  }

  function mkThumb(item, idx) {
    const thumb = document.createElement("div");
    thumb.className = "sfg-viewer-thumb" + (idx === current ? " current" : "");
    thumb.title = item.label || `Item ${idx + 1}`;
    thumb.dataset.idx = String(idx);

    switch (item.type) {
      case "image": {
        const img = document.createElement("img");
        img.src = item.url || "";
        img.alt = "";
        img.decoding = "async";
        thumb.appendChild(img);
        break;
      }
      case "video": {
        const mkFrame = () => {
          const vid = document.createElement("video");
          vid.src = item.url ? `${item.url}#t=0.1` : "";
          vid.muted = true;
          vid.preload = "metadata";
          return vid;
        };
        if (item.thumbnail) {
          const img = document.createElement("img");
          img.src = item.thumbnail;
          img.alt = "";
          img.addEventListener("error", () => img.replaceWith(mkFrame()), { once: true });
          thumb.appendChild(img);
        } else {
          thumb.appendChild(mkFrame());
        }
        break;
      }
      case "audio":
        thumb.classList.add("sfg-viewer-thumb-icon");
        thumb.appendChild(mkIcon("music", 22));
        break;
      default: {
        const text = item.type === "dict"
          ? (item.value || "{}")
          : (item.value !== undefined ? String(item.value) : (item.label || ""));
        const snippet = mkTextBlock("sfg-viewer-thumb-text", text);
        thumb.style.background = `hsla(${CARD_HUES[idx % CARD_HUES.length]}, 22%, 50%, 0.18)`;
        thumb.appendChild(snippet);
        break;
      }
    }

    thumb.addEventListener("click", () => show(idx));

    if (!isDisabled()) {
      const rm = document.createElement("button");
      rm.type = "button";
      rm.className = "sfg-viewer-thumb-remove";
      rm.title = "Remove from selection";
      rm.appendChild(mkIcon("x", 12));
      rm.addEventListener("click", (e) => {
        e.stopPropagation();
        onRemove(idx);
      });
      thumb.appendChild(rm);
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
    if (isSel) selectBtn.appendChild(mkIcon("check", 14));
    let label;
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
      thumb.classList.toggle("current", Number(thumb.dataset.idx) === current);
    }
  }

  // ── Navigation ─────────────────────────────────────────────────────────────

  function show(idx) {
    const items = getItems();
    if (!items.length) return;
    const next = Math.max(0, Math.min(items.length - 1, idx));
    const changed = next !== current;
    current = next;
    if (changed) renderStage();
    renderChrome();
    updateStripCurrent();
  }

  function step(delta) { show(current + delta); }

  function toggleCurrent() {
    if (isDisabled() || !getItems()[current]) return;
    onToggle(current);
  }

  // ── Events ─────────────────────────────────────────────────────────────────

  prevBtn.addEventListener("click", () => step(-1));
  nextBtn.addEventListener("click", () => step(1));
  closeBtn.addEventListener("click", () => close());
  selectBtn.addEventListener("click", () => toggleCurrent());

  // Keys are taken in the capture phase on window so the editor's own shortcuts
  // (delete node, undo, …) don't fire underneath the modal.
  function onKeyDown(e) {
    if (!open) return;
    e.stopPropagation();
    switch (e.key) {
      case "ArrowLeft":  e.preventDefault(); step(-1);        break;
      case "ArrowRight": e.preventDefault(); step(1);         break;
      case " ":          e.preventDefault(); if (!e.repeat) toggleCurrent(); break;
      case "Escape":     e.preventDefault(); close();         break;
    }
  }

  let swipe = null;
  stage.addEventListener("pointerdown", (e) => {
    // Leave media controls and text selection / scrolling to the element
    if (e.target.closest("video, audio, .sfg-viewer-text, .sfg-viewer-dict")) return;
    swipe = { x: e.clientX, y: e.clientY };
  });
  stage.addEventListener("pointerup", (e) => {
    if (!swipe) return;
    const dx = e.clientX - swipe.x;
    const dy = e.clientY - swipe.y;
    swipe = null;
    if (Math.abs(dx) > SWIPE_THRESHOLD && Math.abs(dx) > Math.abs(dy)) step(dx < 0 ? 1 : -1);
  });
  stage.addEventListener("pointercancel", () => { swipe = null; });

  // ── API ────────────────────────────────────────────────────────────────────

  function openAt(idx) {
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
    stage.querySelectorAll("video, audio").forEach((m) => m.pause());
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
      if (vid && items[current].thumbnail) vid.poster = items[current].thumbnail;
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
