var Ue=Object.create;var Q=Object.defineProperty;var Be=Object.getOwnPropertyDescriptor;var Ke=Object.getOwnPropertyNames;var Qe=Object.getPrototypeOf,Ze=Object.prototype.hasOwnProperty;var et=(e,t,r)=>()=>{if(r)throw r[0];try{return e&&(t=e(e=0)),t}catch(o){throw r=[o],o}};var Z=(e,t)=>()=>{try{return t||e((t={exports:{}}).exports,t),t.exports}catch(r){throw t=0,r}},tt=(e,t)=>{for(var r in t)Q(e,r,{get:t[r],enumerable:!0})},Se=(e,t,r,o)=>{if(t&&typeof t=="object"||typeof t=="function")for(let s of Ke(t))!Ze.call(e,s)&&s!==r&&Q(e,s,{get:()=>t[s],enumerable:!(o=Be(t,s))||o.enumerable});return e};var T=(e,t,r)=>(r=e!=null?Ue(Qe(e)):{},Se(t||!e||!e.__esModule?Q(r,"default",{value:e,enumerable:!0}):r,e)),ee=e=>Se(Q({},"__esModule",{value:!0}),e);var q={};tt(q,{REACT_MAJOR:()=>nt,WIDGET_HOST_KEY:()=>ie,provideWidgetHost:()=>rt,widgetHost:()=>ot});function rt(e){if(Ee(e.react.version)!==19)throw new Error(`Widgets are built against React 19, but the editor provides React ${e.react.version}.`);globalThis[ie]=e}function ot(){let e=globalThis[ie];if(!e)throw new Error("This widget renders with the editor's React, and this editor doesn't provide it. It needs an editor that shares React 19 with widgets.");if(Ee(e.react.version)!==19)throw new Error(`This widget is built for React 19, and this editor provides React ${e.react.version}.`);return e}var nt,ie,Ee,Y=et(()=>{nt=19,ie="__griptapeNodesWidgetHost",Ee=e=>Number.parseInt(e,10)});var J=Z((zt,Ce)=>{Ce.exports=(Y(),ee(q)).widgetHost().react});var Le=Z((Rt,Te)=>{Te.exports=(Y(),ee(q)).widgetHost().reactDomClient});var G=Z((Mt,Me)=>{Me.exports=(Y(),ee(q)).widgetHost().jsxRuntime});var _e=Z((Ht,He)=>{He.exports=(Y(),ee(q)).widgetHost().reactDom});var u=T(J(),1);var se=T(J(),1),ze=T(Le(),1);function Re(e){return(t,r)=>{let o=t.ownerDocument.createElement("div");o.style.display="contents",t.append(o);let s=!1,l=(0,ze.createRoot)(o,{onUncaughtError:b=>{s=!0,console.error("Runtime error in widget:",b);let y=o.ownerDocument.createElement("div");y.style.cssText="color: red; padding: 8px;",y.textContent="Widget error",o.replaceChildren(y)}});return l.render((0,se.createElement)(e,r)),{update:b=>{s||l.render((0,se.createElement)(e,b))},cleanup:()=>{o.remove(),queueMicrotask(()=>l.unmount())}}}}var Ne="1.4.0",O=[202,162,267,322,42,82,132,232],ae=[95,125,105,145,85,130,115,155],it=`
/* ── Theme-aware accent colour ────────────────────────────────── */
:root {
  --sfg-accent: #2563eb;
  --sfg-accent-rgb: 37, 99, 235;
}
.dark {
  --sfg-accent: #7a9db8;
  --sfg-accent-rgb: 122, 157, 184;
}

.sfg-widget {
  display: flex;
  flex-direction: column;
  gap: 8px;
  padding: 8px;
  height: 100%;
  box-sizing: border-box;
  overflow: hidden;
  user-select: none;
  -webkit-user-select: none;
}

.sfg-controls {
  display: flex;
  align-items: center;
  gap: 10px;
  flex-wrap: wrap;
  padding-bottom: 4px;
  border-bottom: 1px solid var(--border);
}

.sfg-label {
  font-size: 11px;
  color: var(--muted-foreground);
  display: flex;
  align-items: center;
  gap: 6px;
  white-space: nowrap;
}

.sfg-slider {
  width: 80px;
  accent-color: var(--sfg-accent);
  cursor: pointer;
  flex-shrink: 0;
}
.sfg-slider:disabled { opacity: 0.4; cursor: not-allowed; }

/* ── Layout toggle — segmented control ────────────────────────── */
.sfg-layout-btns { display: flex; gap: 0; }

.sfg-layout-btn {
  padding: 2px 8px;
  border: 1px solid var(--border);
  border-radius: 0;
  background: transparent;
  color: var(--muted-foreground);
  font-size: 10px;
  cursor: pointer;
  line-height: 18px;
  position: relative;
  transition: background 0.12s, color 0.12s, border-color 0.12s;
}
.sfg-layout-btns .sfg-layout-btn:first-child { border-radius: 4px 0 0 4px; }
.sfg-layout-btns .sfg-layout-btn + .sfg-layout-btn { border-radius: 0 4px 4px 0; margin-left: -1px; }
.sfg-layout-btn:hover { background: var(--muted); color: var(--foreground); z-index: 1; }
.sfg-layout-btn.active {
  background: rgba(var(--sfg-accent-rgb), 0.2);
  border-color: var(--sfg-accent);
  color: var(--foreground);
  z-index: 2;
}
.sfg-layout-btn:disabled { opacity: 0.4; cursor: not-allowed; }

/* ── Labels checkbox toggle ───────────────────────────────────── */
.sfg-labels-btn {
  display: flex;
  align-items: center;
  gap: 5px;
  padding: 2px 4px;
  border: none;
  border-radius: 4px;
  background: transparent;
  color: var(--muted-foreground);
  font-size: 10px;
  cursor: pointer;
  line-height: 18px;
  white-space: nowrap;
  transition: color 0.12s;
}
.sfg-labels-btn:hover { color: var(--foreground); }
.sfg-labels-btn.active { color: var(--foreground); }
.sfg-labels-btn:disabled { opacity: 0.4; cursor: not-allowed; }

.sfg-cb-box {
  width: 12px; height: 12px;
  border: 1.5px solid var(--border);
  border-radius: 2px;
  display: flex; align-items: center; justify-content: center;
  flex-shrink: 0;
  background: transparent;
  transition: background 0.12s, border-color 0.12s;
}
.sfg-cb-box svg { opacity: 0; transition: opacity 0.1s; color: #fff; }
.sfg-labels-btn.active .sfg-cb-box {
  background: var(--sfg-accent);
  border-color: var(--sfg-accent);
}
.sfg-labels-btn.active .sfg-cb-box svg { opacity: 1; }

/* ── Label size control ───────────────────────────────────────── */
.sfg-label-size-group {
  display: flex;
  align-items: center;
  gap: 4px;
  font-size: 10px;
  color: var(--muted-foreground);
  white-space: nowrap;
}
.sfg-label-size-group .sfg-slider { width: 52px; }
.sfg-hide-labels .sfg-label-size-group { display: none; }

.sfg-count { margin-left: auto; font-size: 10px; color: var(--muted-foreground); }
.sfg-count.active { color: var(--foreground); }

.sfg-clear-btn {
  padding: 2px 8px;
  border: 1px solid var(--border);
  border-radius: 4px;
  background: transparent;
  color: var(--muted-foreground);
  font-size: 10px;
  cursor: pointer;
  line-height: 18px;
  transition: background 0.12s, color 0.12s;
  flex-shrink: 0;
}
.sfg-clear-btn:hover { background: var(--muted); color: var(--foreground); }
.sfg-clear-btn.active { color: var(--foreground); border-color: rgba(var(--sfg-accent-rgb), 0.5); }
.sfg-clear-btn:disabled { opacity: 0.4; cursor: not-allowed; }

.sfg-fullscreen-btn {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 24px; height: 24px;
  padding: 0;
  border: 1px solid var(--border);
  border-radius: 4px;
  background: transparent;
  color: var(--muted-foreground);
  cursor: pointer;
  flex-shrink: 0;
  transition: background 0.12s, color 0.12s;
}
.sfg-fullscreen-btn:hover { background: var(--muted); color: var(--foreground); }

/* ── Grid layouts ─────────────────────────────────────────────── */
.sfg-grid {
  display: grid;
  gap: 5px;
  position: relative;
  flex: 1 1 0;
  min-height: 0;
  min-width: 0;
  overflow-y: auto;
  overflow-x: hidden;
  scrollbar-width: thin;
  scrollbar-color: var(--border) transparent;
  scrollbar-gutter: stable;
  align-content: start;
}
.sfg-grid::-webkit-scrollbar { width: 6px; }
.sfg-grid::-webkit-scrollbar-track { background: transparent; }
.sfg-grid::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }

.sfg-grid.layout-masonry { display: flex; align-items: flex-start; }
.sfg-masonry-col { display: flex; flex-direction: column; flex: 1; gap: 5px; min-width: 0; }

/* ── Loading spinner ──────────────────────────────────────────── */
@keyframes sfg-spin { to { transform: rotate(360deg); } }
.sfg-spinner {
  position: absolute;
  inset: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  background: var(--muted);
  z-index: 2;
  pointer-events: none;
}
.sfg-spinner svg {
  color: var(--sfg-accent);
  opacity: 0.7;
  animation: sfg-spin 0.9s linear infinite;
}

/* ── Lasso / box-select rect ──────────────────────────────────── */
.sfg-lasso {
  position: absolute;
  border: 1.5px dashed var(--sfg-accent);
  background: rgba(var(--sfg-accent-rgb), 0.10);
  border-radius: 3px;
  pointer-events: none;
  z-index: 10;
}

/* ── Cells ────────────────────────────────────────────────────── */
.sfg-cell {
  position: relative;
  border-radius: 5px;
  overflow: hidden;
  cursor: pointer;
  border: 2px solid transparent;
  background: var(--muted);
  transition: border-color 0.12s, opacity 0.15s, box-shadow 0.12s;
  box-sizing: border-box;
}
.sfg-cell:hover   { border-color: rgba(var(--sfg-accent-rgb), 0.45); }
.sfg-cell.pending { border-color: rgba(var(--sfg-accent-rgb), 0.7); background: rgba(var(--sfg-accent-rgb), 0.08); }
.sfg-cell.selected {
  border-color: var(--sfg-accent);
  box-shadow: inset 0 0 0 1px var(--sfg-accent);
}

/* Dim unselected cells when a selection exists */
.sfg-grid.has-selection .sfg-cell:not(.selected):not(.pending) {
  opacity: 0.45;
}

/* Checkmark badge */
.sfg-cell.selected::after {
  content: "";
  position: absolute;
  top: 4px; right: 4px;
  width: 20px; height: 20px;
  border-radius: 50%;
  background-color: var(--sfg-accent);
  background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16' fill='white'%3E%3Cpath d='M12.207 4.793a1 1 0 010 1.414l-5 5a1 1 0 01-1.414 0l-2-2a1 1 0 011.414-1.414L6.5 9.086l4.293-4.293a1 1 0 011.414 0z'/%3E%3C/svg%3E");
  background-repeat: no-repeat;
  background-position: center;
  background-size: 13px;
  z-index: 3;
  pointer-events: none;
  box-shadow: 0 1px 4px rgba(0,0,0,0.5);
}

/* Fullscreen button, shown on hover (top-left, clear of the checkmark badge) */
.sfg-expand-btn {
  position: absolute;
  top: 4px; left: 4px;
  width: 22px; height: 22px;
  padding: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  border: none;
  border-radius: 4px;
  background: rgba(0,0,0,0.6);
  color: #fff;
  cursor: pointer;
  z-index: 4;
  opacity: 0;
  transition: opacity 0.12s, background 0.12s;
}
.sfg-cell:hover .sfg-expand-btn,
.sfg-expand-btn:focus-visible { opacity: 1; }
.sfg-expand-btn:hover { background: rgba(var(--sfg-accent-rgb), 0.9); }

/* ── Grid inner frame ─────────────────────────────────────────── */
.sfg-cell-inner {
  width: 100%;
  overflow: hidden;
  display: flex;
  align-items: center;
  justify-content: center;
  background: #111;
}

/* Grid mode: grid-auto-rows (set via JS ResizeObserver) makes rows = column width.
   The inner fills the measured square cell completely. */
.sfg-grid:not(.layout-masonry) .sfg-cell-inner { height: 100%; }

/* Placeholder shape for masonry cells while their media is loading.
   Keeps the cell from collapsing to 0 so the spinner is visible and
   the eventual height change is from a reasonable size rather than zero. */
.layout-masonry .sfg-cell-inner.sfg-loading { aspect-ratio: 4 / 3; }

/* ── Media elements ───────────────────────────────────────────── */
.sfg-cell img {
  width: 100%; height: 100%; object-fit: contain; display: block; pointer-events: none;
  opacity: 0;
  transition: opacity 0.25s ease;
}
.sfg-cell img.sfg-loaded { opacity: 1; }
.layout-masonry .sfg-cell img { height: auto; object-fit: cover; }

.sfg-cell video {
  width: 100%; height: 100%; object-fit: contain; display: block; pointer-events: none;
  opacity: 0;
  transition: opacity 0.25s ease;
}
.sfg-cell video.sfg-loaded { opacity: 1; }
.layout-masonry .sfg-cell video { height: auto; object-fit: cover; }

.sfg-spinner { transition: opacity 0.2s ease; }

/* ── Error card (failed media load) ──────────────────────────── */
.sfg-error-card {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 100%;
  height: 100%;
  color: var(--muted-foreground);
  opacity: 0.35;
}

/* ── Audio card ───────────────────────────────────────────────── */
.sfg-audio-card {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 8px;
  padding: 12px 8px;
  width: 100%;
  box-sizing: border-box;
  background: rgba(0,0,0,0.3);
  color: var(--muted-foreground);
}
.sfg-audio-card audio { width: 100%; height: 28px; accent-color: var(--sfg-accent); }

/* ── Quote card (text items) ──────────────────────────────────── */
.sfg-quote-card {
  padding: 14px 12px;
  width: 100%;
  height: 100%;
  box-sizing: border-box;
  align-self: stretch;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  text-align: center;
}
.sfg-quote-text {
  font-size: 11px;
  color: var(--foreground);
  line-height: 1.55;
  word-break: break-word;
  display: -webkit-box;
  -webkit-line-clamp: 5;
  -webkit-box-orient: vertical;
  overflow: hidden;
}

/* ── Dict card ────────────────────────────────────────────────── */
.sfg-dict-card {
  padding: 8px;
  font-size: 9px;
  font-family: monospace;
  color: var(--muted-foreground);
  white-space: pre;
  overflow: hidden;
  text-overflow: ellipsis;
  max-height: 110px;
  display: -webkit-box;
  -webkit-line-clamp: 7;
  -webkit-box-orient: vertical;
}

/* ── Label overlay ────────────────────────────────────────────── */
.sfg-item-label {
  position: absolute;
  bottom: 0; left: 0; right: 0;
  padding: 3px 5px;
  font-size: var(--sfg-label-size, 10px);
  background: rgba(0,0,0,0.55);
  color: #fff;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  z-index: 2;
  pointer-events: none;
}
.sfg-hide-labels .sfg-item-label { display: none; }

/* ── Empty state ──────────────────────────────────────────────── */
.sfg-empty {
  padding: 24px 12px;
  text-align: center;
  color: var(--muted-foreground);
  font-size: 12px;
}

/* ── Fullscreen viewer ────────────────────────────────────────── */
.sfg-viewer {
  position: fixed;
  inset: 0;
  z-index: 10000;
  display: flex;
  flex-direction: column;
  background: #08080a;
  color: #eee;
  font: 13px/1.4 system-ui, -apple-system, "Segoe UI", sans-serif;
  letter-spacing: normal;
  word-spacing: normal;
  text-align: left;
  outline: none;
  user-select: none;
  -webkit-user-select: none;
}

.sfg-viewer-header {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 10px 14px;
  flex-shrink: 0;
}
.sfg-viewer-title {
  flex: 1;
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  color: #ccc;
}
.sfg-viewer-counter { color: #999; font-variant-numeric: tabular-nums; }

.sfg-viewer-select {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 5px 12px;
  border: 1px solid rgba(255,255,255,0.25);
  border-radius: 5px;
  background: transparent;
  color: #eee;
  font-size: 12px;
  cursor: pointer;
  transition: background 0.12s, border-color 0.12s;
}
.sfg-viewer-select:hover { background: rgba(255,255,255,0.08); }
.sfg-viewer-select.active { background: var(--sfg-accent); border-color: var(--sfg-accent); color: #fff; }
.sfg-viewer-select:disabled { opacity: 0.4; cursor: not-allowed; }

.sfg-viewer-icon-btn {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 32px; height: 32px;
  padding: 0;
  border: none;
  border-radius: 5px;
  background: transparent;
  color: #ccc;
  cursor: pointer;
}
.sfg-viewer-icon-btn:hover { background: rgba(255,255,255,0.1); color: #fff; }

.sfg-viewer-body {
  flex: 1;
  min-height: 0;
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 0 8px;
}
.sfg-viewer-nav {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 44px; height: 88px;
  padding: 0;
  border: none;
  border-radius: 6px;
  background: transparent;
  color: #ccc;
  cursor: pointer;
  flex-shrink: 0;
}
.sfg-viewer-nav:hover:not(:disabled) { background: rgba(255,255,255,0.08); color: #fff; }
.sfg-viewer-nav:disabled { opacity: 0.2; cursor: default; }

.sfg-viewer-stage {
  flex: 1;
  min-width: 0;
  height: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
  overflow: hidden;
  touch-action: pan-y;
}
.sfg-viewer-stage img,
.sfg-viewer-stage video {
  max-width: 100%;
  max-height: 100%;
  object-fit: contain;
  display: block;
}
.sfg-viewer-stage img { pointer-events: none; }
.sfg-viewer-stage .sfg-error-card { color: #999; }

.sfg-viewer-audio {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 24px;
  color: #888;
  width: min(560px, 100%);
}
.sfg-viewer-audio audio { width: 100%; accent-color: var(--sfg-accent); }

.sfg-viewer-text,
.sfg-viewer-dict {
  max-width: min(900px, 100%);
  max-height: 100%;
  overflow: auto;
  padding: 24px 28px;
  box-sizing: border-box;
  border-radius: 8px;
  background: rgba(255,255,255,0.05);
  user-select: text;
  -webkit-user-select: text;
  word-break: break-word;
}
.sfg-viewer-text { font-size: 18px; line-height: 1.6; white-space: pre-wrap; }
.sfg-viewer-dict { font-family: monospace; font-size: 13px; white-space: pre; color: #ccc; }

.sfg-viewer-strip {
  flex-shrink: 0;
  padding: 8px 14px 12px;
  border-top: 1px solid rgba(255,255,255,0.1);
}
.sfg-viewer-strip-hint { font-size: 11px; color: #999; margin-bottom: 6px; }
.sfg-viewer-strip-items {
  display: flex;
  gap: 8px;
  overflow-x: auto;
  min-height: 72px;
  scrollbar-width: thin;
}
.sfg-viewer-thumb {
  position: relative;
  width: 72px; height: 72px;
  flex-shrink: 0;
  border-radius: 5px;
  overflow: hidden;
  border: 2px solid transparent;
  background: #1a1a1d;
  cursor: pointer;
  box-sizing: border-box;
}
.sfg-viewer-thumb.current { border-color: var(--sfg-accent); }
.sfg-viewer-thumb img,
.sfg-viewer-thumb video {
  width: 100%; height: 100%; object-fit: cover; display: block; pointer-events: none;
}
.sfg-viewer-thumb-icon { display: flex; align-items: center; justify-content: center; color: #888; }
.sfg-viewer-thumb-text {
  padding: 5px;
  font-size: 9px;
  line-height: 1.35;
  color: #ddd;
  overflow: hidden;
  word-break: break-word;
  display: -webkit-box;
  -webkit-line-clamp: 5;
  -webkit-box-orient: vertical;
}
.sfg-viewer-thumb-remove {
  position: absolute;
  top: 3px; right: 3px;
  width: 18px; height: 18px;
  padding: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  border: none;
  border-radius: 50%;
  background: rgba(0,0,0,0.75);
  color: #fff;
  cursor: pointer;
}
.sfg-viewer-thumb-remove:hover { background: #c0392b; }
`;function Pe(){let e=document.getElementById("sfg-widget-styles");e?.dataset.version!==Ne&&(e||(e=document.createElement("style"),e.id="sfg-widget-styles",document.head.appendChild(e)),e.dataset.version=Ne,e.textContent=it)}var Ie=T(G(),1),st={music:'<path d="M9 18V5l12-2v13"/><circle cx="6" cy="18" r="3"/><circle cx="18" cy="16" r="3"/>',"loader-circle":'<path d="M21 12a9 9 0 1 1-6.219-8.56"/>',"alert-circle":'<circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/>',check:'<polyline points="20 6 9 17 4 12"/>',"maximize-2":'<polyline points="15 3 21 3 21 9"/><polyline points="9 21 3 21 3 15"/><line x1="21" x2="14" y1="3" y2="10"/><line x1="3" x2="10" y1="21" y2="14"/>',x:'<path d="M18 6 6 18"/><path d="m6 6 12 12"/>',"chevron-left":'<path d="m15 18-6-6 6-6"/>',"chevron-right":'<path d="m9 18 6-6-6-6"/>'},at={display:"block",flexShrink:0,pointerEvents:"none"};function S({name:e,size:t=14}){return(0,Ie.jsx)("svg",{viewBox:"0 0 24 24",width:t,height:t,fill:"none",stroke:"currentColor",strokeWidth:"2",strokeLinecap:"round",strokeLinejoin:"round",style:at,dangerouslySetInnerHTML:{__html:st[e]||""}})}var H=T(J(),1);var w=T(G(),1),X=e=>e.stopPropagation();function Ge({min:e,max:t,value:r,disabled:o,onInput:s,onCommit:l}){let b=(0,H.useRef)(null),y=(0,H.useRef)(l);return(0,H.useLayoutEffect)(()=>{y.current=l}),(0,H.useEffect)(()=>{let m=b.current;if(!m)return;let g=()=>y.current(parseInt(m.value,10));return m.addEventListener("change",g),()=>m.removeEventListener("change",g)},[]),(0,w.jsx)("input",{ref:b,type:"range",className:"sfg-slider",min:e,max:t,value:String(r),disabled:o,onPointerDown:X,onChange:m=>s(parseInt(m.currentTarget.value,10))})}function De({layout:e,columns:t,disabled:r,showLabels:o,labelSize:s,selectedCount:l,onColumnsChange:b,onColumnsCommit:y,onLayoutChange:m,onClear:g,onLabelsToggle:x,onLabelSizeChange:E,onLabelSizeCommit:C,onFullscreen:L}){let z=l>0,N=(h,P)=>(0,w.jsx)("button",{type:"button",className:"sfg-layout-btn"+(e===h?" active":""),disabled:r,onPointerDown:X,onClick:M=>{M.stopPropagation(),m(h)},children:P});return(0,w.jsxs)("div",{className:"sfg-controls",children:[(0,w.jsxs)("label",{className:"sfg-label",children:["Columns ",(0,w.jsx)(Ge,{min:1,max:8,value:t,disabled:r,onInput:b,onCommit:y}),(0,w.jsx)("span",{children:String(t)})]}),(0,w.jsxs)("div",{className:"sfg-layout-btns",children:[N("grid","Grid"),N("masonry","Masonry")]}),(0,w.jsxs)("button",{type:"button",className:"sfg-labels-btn"+(o?" active":""),disabled:r,onPointerDown:X,onClick:h=>{h.stopPropagation(),x(!o)},children:[(0,w.jsx)("span",{className:"sfg-cb-box",children:(0,w.jsx)(S,{name:"check",size:9})}),"Labels"]}),(0,w.jsxs)("label",{className:"sfg-label sfg-label-size-group",children:["Aa ",(0,w.jsx)(Ge,{min:8,max:32,value:s,disabled:r,onInput:E,onCommit:C}),(0,w.jsx)("span",{children:String(s)})]}),(0,w.jsx)("span",{className:"sfg-count"+(z?" active":""),children:z?`${l} selected`:""}),(0,w.jsx)("button",{type:"button",className:"sfg-clear-btn"+(z?" active":""),disabled:r,onPointerDown:X,onClick:h=>{h.stopPropagation(),g()},children:"Clear"}),(0,w.jsx)("button",{type:"button",className:"sfg-fullscreen-btn",title:"Fullscreen",onPointerDown:X,onClick:h=>{h.stopPropagation(),L()},children:(0,w.jsx)(S,{name:"maximize-2",size:12})})]})}var k=T(J(),1),Ae=T(_e(),1);var a=T(G(),1),lt=50;function ce(){return(0,a.jsx)("div",{className:"sfg-error-card",children:(0,a.jsx)(S,{name:"alert-circle",size:40})})}function le(e){return e.type==="dict"?e.value||"{}":e.value!==void 0?String(e.value):e.label||""}function ct(e){return JSON.stringify([e.type,e.url,e.full_url,e.value])}function Oe(e){(0,k.useEffect)(()=>{let t=e.current;return()=>t?.pause()},[e])}function dt({item:e}){let[t,r]=(0,k.useState)(e.full_url||e.url||""),[o,s]=(0,k.useState)(!1);return o?(0,a.jsx)(ce,{}):(0,a.jsx)("img",{src:t,alt:e.label||"",onError:()=>{e.full_url&&e.url&&t!==e.url?r(e.url):s(!0)}})}function ut({item:e}){let t=(0,k.useRef)(null),[r,o]=(0,k.useState)(!1);return Oe(t),(0,k.useEffect)(()=>{t.current?.play().catch(()=>{})},[]),r?(0,a.jsx)(ce,{}):(0,a.jsx)("video",{ref:t,src:e.url||"",poster:e.thumbnail||void 0,controls:!0,loop:!0,playsInline:!0,onError:()=>o(!0)})}function gt({item:e}){let t=(0,k.useRef)(null),[r,o]=(0,k.useState)(!1);return Oe(t),r?(0,a.jsx)(ce,{}):(0,a.jsxs)("div",{className:"sfg-viewer-audio",children:[(0,a.jsx)(S,{name:"music",size:72}),e.url?(0,a.jsx)("audio",{ref:t,src:e.url,controls:!0,onError:()=>o(!0)}):null]})}function ft({item:e}){switch(e.type){case"image":return(0,a.jsx)(dt,{item:e});case"video":return(0,a.jsx)(ut,{item:e});case"audio":return(0,a.jsx)(gt,{item:e});case"dict":return(0,a.jsx)("div",{className:"sfg-viewer-dict",children:le(e)});default:return(0,a.jsx)("div",{className:"sfg-viewer-text",children:le(e)})}}function pt({url:e}){return(0,a.jsx)("video",{src:e?`${e}#t=0.1`:"",muted:!0,preload:"metadata"})}function bt({item:e}){let[t,r]=(0,k.useState)(!1);return!e.thumbnail||t?(0,a.jsx)(pt,{url:e.url}):(0,a.jsx)("img",{src:e.thumbnail,alt:"",onError:()=>r(!0)})}function mt({item:e,idx:t,current:r,removable:o,onShow:s,onRemove:l}){let b=e.type!=="image"&&e.type!=="video"&&e.type!=="audio";return(0,a.jsxs)("div",{className:"sfg-viewer-thumb"+(e.type==="audio"?" sfg-viewer-thumb-icon":"")+(r?" current":""),title:e.label||`Item ${t+1}`,"data-idx":String(t),style:b?{background:`hsla(${O[t%O.length]}, 22%, 50%, 0.18)`}:void 0,onClick:()=>s(t),children:[e.type==="image"?(0,a.jsx)("img",{src:e.url||"",alt:"",decoding:"async"}):e.type==="video"?(0,a.jsx)(bt,{item:e}):e.type==="audio"?(0,a.jsx)(S,{name:"music",size:22}):(0,a.jsx)("div",{className:"sfg-viewer-thumb-text",children:le(e)}),o?(0,a.jsx)("button",{type:"button",className:"sfg-viewer-thumb-remove",title:"Remove from selection",onClick:y=>{y.stopPropagation(),l(t)},children:(0,a.jsx)(S,{name:"x",size:12})}):null]})}function $e({items:e,selected:t,multiSelect:r,disabled:o,current:s,onShow:l,onClose:b,onToggle:y,onRemove:m}){let g=(0,k.useRef)(null),x=(0,k.useRef)(null),E=e[s],C=t.includes(s),L=f=>l(s+f),z=()=>{o||!E||y(s)},N=f=>{switch(f.stopPropagation(),f.key){case"ArrowLeft":f.preventDefault(),L(-1);break;case"ArrowRight":f.preventDefault(),L(1);break;case" ":f.preventDefault(),f.repeat||z();break;case"Escape":f.preventDefault(),b();break}},h=(0,k.useRef)(N);(0,k.useLayoutEffect)(()=>{h.current=N}),(0,k.useEffect)(()=>{let f=R=>h.current(R);return window.addEventListener("keydown",f,!0),g.current?.focus(),()=>window.removeEventListener("keydown",f,!0)},[]);let P;C?P="Selected":!r&&t.length?P="Replace selection":P="Select";let M;return t.length?r?M=`${t.length} selected`:M="Selected · selecting another item replaces it":M=r?"Nothing selected · press Space to select":"Nothing selected",(0,Ae.createPortal)((0,a.jsxs)("div",{ref:g,className:"sfg-viewer",tabIndex:-1,role:"dialog","aria-modal":"true",children:[(0,a.jsxs)("div",{className:"sfg-viewer-header",children:[(0,a.jsx)("div",{className:"sfg-viewer-title",children:E?.label||""}),(0,a.jsx)("span",{className:"sfg-viewer-counter",children:e.length?`${s+1} / ${e.length}`:""}),(0,a.jsxs)("button",{type:"button",className:"sfg-viewer-select"+(C?" active":""),title:"Toggle selection (Space)",disabled:o,onClick:z,children:[C?(0,a.jsx)(S,{name:"check",size:14}):null,P]}),(0,a.jsx)("button",{type:"button",className:"sfg-viewer-icon-btn",title:"Close (Esc)",onClick:b,children:(0,a.jsx)(S,{name:"x",size:18})})]}),(0,a.jsxs)("div",{className:"sfg-viewer-body",children:[(0,a.jsx)("button",{type:"button",className:"sfg-viewer-nav prev",title:"Previous (←)",disabled:s<=0,onClick:()=>L(-1),children:(0,a.jsx)(S,{name:"chevron-left",size:28})}),(0,a.jsx)("div",{className:"sfg-viewer-stage",onPointerDown:f=>{f.target.closest("video, audio, .sfg-viewer-text, .sfg-viewer-dict")||(x.current={x:f.clientX,y:f.clientY})},onPointerUp:f=>{let R=x.current;if(!R)return;let $=f.clientX-R.x,U=f.clientY-R.y;x.current=null,Math.abs($)>lt&&Math.abs($)>Math.abs(U)&&L($<0?1:-1)},onPointerCancel:()=>{x.current=null},children:E?(0,a.jsx)(ft,{item:E},`${s}:${ct(E)}`):null}),(0,a.jsx)("button",{type:"button",className:"sfg-viewer-nav next",title:"Next (→)",disabled:s>=e.length-1,onClick:()=>L(1),children:(0,a.jsx)(S,{name:"chevron-right",size:28})})]}),(0,a.jsxs)("div",{className:"sfg-viewer-strip",children:[(0,a.jsx)("div",{className:"sfg-viewer-strip-hint",children:M}),(0,a.jsx)("div",{className:"sfg-viewer-strip-items",children:t.map(f=>{let R=e[f];return R?(0,a.jsx)(mt,{item:R,idx:f,current:f===s,removable:!o,onShow:l,onRemove:m},f):null})})]})]}),document.body)}var d=T(G(),1),ht=5,vt=5,de=110,te=e=>e.stopPropagation();function Ve(e){let t=e||{},r=t.layout||"grid";return{items:t.items||[],selected:t.selected_indices||[],columns:t.columns||3,layout:r,multiSelect:t.settings?.multi_select!==!1,showLabels:t.show_labels!==!1,labelSize:t.label_size||10,shownLayout:r,skeleton:!1}}function xt(e,t){let r=Ve(t),o=JSON.stringify(r.items)===JSON.stringify(e.items),s=JSON.stringify(r.selected)===JSON.stringify(e.selected),l=s?e.selected:r.selected,b=!o||!s||r.columns!==e.columns||r.layout!==e.layout;return r.multiSelect!==e.multiSelect&&!r.multiSelect&&l.length>1&&(l=[],b=!0),{...r,items:o?e.items:r.items,selected:l,shownLayout:b?r.layout:e.shownLayout,skeleton:b?!1:e.skeleton}}var yt=e=>e.type==="image"||e.type==="video"||e.type==="audio",je=(e,t)=>e.left<t.right&&e.left+e.width>t.left&&e.top<t.bottom&&e.top+e.height>t.top;function ue(e){return parseInt(e.dataset.idx,10)}function wt(e){let t=(0,u.useRef)(new Map),r=(0,u.useRef)(null);return(0,u.useEffect)(()=>{let o=new IntersectionObserver(s=>s.forEach(({isIntersecting:l,target:b})=>{l&&t.current.get(b)?.()}),{root:e.current,rootMargin:"300px 0px",threshold:0});return r.current=o,t.current.forEach((s,l)=>o.observe(l)),()=>{o.disconnect(),r.current=null}},[e]),(0,u.useCallback)((o,s)=>(t.current.set(o,s),r.current?.observe(o),()=>{t.current.delete(o),r.current?.unobserve(o)}),[])}function kt({fading:e}){let[t,r]=(0,u.useState)(!1);return t?null:(0,d.jsx)("div",{className:"sfg-spinner",style:e?{opacity:0}:void 0,onTransitionEnd:()=>{e&&r(!0)},children:(0,d.jsx)(S,{name:"loader-circle",size:22})})}function St(){return(0,d.jsx)("div",{className:"sfg-error-card",children:(0,d.jsx)(S,{name:"alert-circle",size:28})})}function Et({item:e,hydrated:t,videoRef:r}){let[o,s]=(0,u.useState)(!1),[l,b]=(0,u.useState)(!1),y=e.type==="audio",m=!l&&!(t&&y)&&!o,g=null;if(t)switch(e.type){case"image":g=(0,d.jsx)("img",{className:o?"sfg-loaded":void 0,src:e.url,alt:e.label||"",decoding:"async",onLoad:()=>s(!0),onError:()=>b(!0)});break;case"video":g=(0,d.jsx)("video",{ref:r,className:o?"sfg-loaded":void 0,src:e.url,poster:e.thumbnail||void 0,muted:!0,loop:!0,playsInline:!0,preload:"metadata",onLoadedMetadata:E=>{E.currentTarget.currentTime=.001,s(!0)},onError:()=>b(!0)});break;case"audio":g=l?null:(0,d.jsxs)("div",{className:"sfg-audio-card",children:[(0,d.jsx)(S,{name:"music",size:28}),e.url?(0,d.jsx)("audio",{src:e.url,controls:!0,onPointerDown:te,onClick:te,onError:()=>b(!0)}):null]});break}let x=!l&&!(t&&y);return(0,d.jsxs)("div",{className:"sfg-cell-inner"+(m?" sfg-loading":""),children:[x?(0,d.jsx)(kt,{fading:o}):null,g,l?(0,d.jsx)(St,{}):null]})}function Ct({item:e,idx:t,masonry:r,selected:o,pending:s,skeleton:l,watch:b,onExpand:y}){let m=(0,u.useRef)(null),g=(0,u.useRef)(null),[x,E]=(0,u.useState)(!1),C=yt(e),L=O[t%O.length],z=r?ae[t%ae.length]+"px":void 0;(0,u.useEffect)(()=>{let h=m.current;if(!(!h||!C||x||l))return b(h,()=>{(e.type==="image"||e.type==="video")&&!e.url||E(!0)})},[b,C,x,l,e.type,e.url]);let N;return l?N=(0,d.jsx)("div",{className:"sfg-cell-inner",style:{background:`hsla(${L}, 22%, 50%, 0.09)`,minHeight:z}}):N=(0,d.jsxs)(d.Fragment,{children:[C?(0,d.jsx)(Et,{item:e,hydrated:x,videoRef:g}):(0,d.jsx)("div",{className:"sfg-cell-inner",children:e.type==="dict"?(0,d.jsx)("div",{className:"sfg-dict-card",children:e.value||"{}"}):(0,d.jsx)("div",{className:"sfg-quote-card",style:{background:`hsla(${L}, 22%, 50%, 0.09)`,minHeight:z},children:(0,d.jsx)("div",{className:"sfg-quote-text",children:e.value!==void 0?String(e.value):e.label||""})})}),(0,d.jsx)("button",{type:"button",className:"sfg-expand-btn",title:"View fullscreen",onPointerDown:te,onClick:h=>{h.stopPropagation(),y(t)},children:(0,d.jsx)(S,{name:"maximize-2",size:12})}),e.label?(0,d.jsx)("div",{className:"sfg-item-label",children:e.label}):null]}),(0,d.jsx)("div",{ref:m,className:"sfg-cell"+(o?" selected":"")+(s?" pending":""),"data-idx":String(t),"data-hydrated":x?"true":void 0,onMouseEnter:()=>{g.current?.play().catch(()=>{})},onMouseLeave:()=>{let h=g.current;h&&(h.pause(),h.currentTime=0)},children:N})}function Tt(e){let{value:t,onChange:r}=e;(0,u.useInsertionEffect)(()=>Pe(),[]);let o=e.disabled||!1,[s,l]=(0,u.useState)(()=>Ve(t)),[b,y]=(0,u.useState)(e);b!==e&&(y(e),l(n=>xt(n,t)));let{items:m,selected:g,columns:x,layout:E,multiSelect:C,showLabels:L,labelSize:z,shownLayout:N,skeleton:h}=s,[P,M]=(0,u.useState)(null);P!==null&&m.length===0&&M(null);let[f,R]=(0,u.useState)(null),[$,U]=(0,u.useState)(()=>new Set),[We,ge]=(0,u.useState)(0),[fe,Fe]=(0,u.useState)(0),j=(0,u.useRef)(null),V=(0,u.useRef)(null),ne=(0,u.useRef)(void 0),qe=wt(j);function W(n){l(n),r({...t||{},items:n.items,selected_indices:n.selected,columns:n.columns,layout:n.layout,show_labels:n.showLabels,label_size:n.labelSize})}function F(n){W({...s,selected:n})}function pe(n){let i=g.includes(n);F(C?i?g.filter(c=>c!==n):[...g,n]:i?[]:[n])}function be(){let n=j.current,i=n?[...n.querySelectorAll(".sfg-cell")]:[];i.forEach((c,p)=>{let v=Math.min(p*6,40);c.style.transition=`opacity ${de}ms ease ${v}ms, transform ${de}ms ease ${v}ms`,c.style.opacity="0",c.style.transform="scale(0.82)"}),clearTimeout(ne.current),ne.current=setTimeout(()=>{l(c=>({...c,shownLayout:c.layout,skeleton:!1})),Fe(c=>c+1)},de+Math.min(i.length*6,40)+20)}(0,u.useEffect)(()=>()=>clearTimeout(ne.current),[]),(0,u.useLayoutEffect)(()=>{let n=j.current;if(!fe||!n)return;let i=[...n.querySelectorAll(".sfg-cell")];i.forEach(p=>{p.style.transition="none",p.style.opacity="0",p.style.transform="scale(0.88)"});let c=requestAnimationFrame(()=>{c=requestAnimationFrame(()=>{i.forEach((p,v)=>{let I=Math.min(v*12,90);p.style.transition=`opacity 0.2s ease ${I}ms, transform 0.22s cubic-bezier(0.34,1.56,0.64,1) ${I}ms`,p.style.opacity="",p.style.transform=""})})});return()=>cancelAnimationFrame(c)},[fe]),(0,u.useLayoutEffect)(()=>{let n=j.current;if(!n)return;ge(n.clientWidth);let i=new ResizeObserver(()=>ge(n.clientWidth));return i.observe(n),()=>i.disconnect()},[]);function re(n,i,c){let p=n.getBoundingClientRect(),v=p.width>0?p.width/n.offsetWidth:1;return{x:(i-p.left)/v+n.scrollLeft,y:(c-p.top)/v+n.scrollTop}}function me(n,i){let c=i.getBoundingClientRect(),p=n.getBoundingClientRect(),v=p.width>0?p.width/n.offsetWidth:1;return{left:(c.left-p.left)/v+n.scrollLeft,top:(c.top-p.top)/v+n.scrollTop,width:c.width/v,height:c.height/v}}function he(n,i,c){return{left:Math.min(n.startX,i),top:Math.min(n.startY,c),right:Math.max(n.startX,i),bottom:Math.max(n.startY,c)}}function Ye(n){if(o||n.button!==0)return;n.stopPropagation();let i=n.currentTarget,{x:c,y:p}=re(i,n.clientX,n.clientY);C&&R({left:c,top:p,width:0,height:0});let v=n.target.closest(".sfg-cell"),I=v&&i.contains(v)?ue(v):null;V.current={startX:c,startY:p,lasso:C,dragging:!1,startCell:I},i.setPointerCapture(n.pointerId)}function Je(n){let i=V.current;if(!i)return;let c=n.currentTarget,{x:p,y:v}=re(c,n.clientX,n.clientY),I=p-i.startX,D=v-i.startY;if(!i.dragging&&Math.hypot(I,D)>ht&&(i.dragging=!0),i.dragging&&i.lasso){let _=he(i,p,v);R({left:_.left,top:_.top,width:Math.abs(I),height:Math.abs(D)});let A=new Set;c.querySelectorAll(".sfg-cell").forEach(we=>{let ke=ue(we);je(me(c,we),_)&&!g.includes(ke)&&A.add(ke)}),U(A)}}function ve(){V.current=null,R(null),U(new Set)}function Xe(n){let i=V.current;if(!i)return;let c=n.currentTarget;if(ve(),i.dragging&&i.lasso){let{x:p,y:v}=re(c,n.clientX,n.clientY),I=he(i,p,v),D=[...g];c.querySelectorAll(".sfg-cell").forEach(_=>{if(!je(me(c,_),I))return;let A=ue(_);!isNaN(A)&&!D.includes(A)&&D.push(A)}),D.length!==g.length&&F(D)}else!i.dragging&&i.startCell!==null&&(isNaN(i.startCell)||pe(i.startCell))}function oe(n){m.length&&M(Math.max(0,Math.min(m.length-1,n)))}let B=N==="masonry",xe;if(!B){let n=(We-(x-1)*vt)/x;xe={gridTemplateColumns:`repeat(${x}, 1fr)`,gridAutoRows:n>0?`${n}px`:void 0}}let ye=(n,i)=>(0,d.jsx)(Ct,{item:n,idx:i,masonry:B,selected:g.includes(i),pending:$.has(i),skeleton:h,watch:qe,onExpand:oe},`${i}|${n.type}|${n.url??""}`),K;return m.length===0?K=(0,d.jsx)("div",{className:"sfg-empty",children:"Connect a list to display items here"}):B?K=Array.from({length:x},(n,i)=>(0,d.jsx)("div",{className:"sfg-masonry-col",children:m.map((c,p)=>p%x===i?ye(c,p):null)},i)):K=m.map(ye),(0,d.jsxs)("div",{className:"sfg-widget nodrag nowheel"+(L?"":" sfg-hide-labels"),style:{"--sfg-label-size":z+"px"},children:[(0,d.jsx)(De,{layout:E,columns:x,disabled:o,showLabels:L,labelSize:z,selectedCount:g.length,onColumnsChange:n=>l({...s,columns:n,shownLayout:E,skeleton:!0}),onColumnsCommit:n=>{be(),W({...s,columns:n})},onLayoutChange:n=>{be(),W({...s,layout:n})},onClear:()=>{g.length&&F([])},onLabelsToggle:n=>W({...s,showLabels:n}),onLabelSizeChange:n=>l({...s,labelSize:n}),onLabelSizeCommit:n=>W({...s,labelSize:n}),onFullscreen:()=>oe(g[0]??0)}),(0,d.jsxs)("div",{ref:j,className:"sfg-grid"+(B?" layout-masonry":"")+(g.length?" has-selection":""),style:xe,onPointerDown:Ye,onPointerMove:Je,onPointerUp:Xe,onPointerCancel:()=>{V.current&&ve()},onWheel:te,children:[K,f?(0,d.jsx)("div",{className:"sfg-lasso",style:{left:f.left+"px",top:f.top+"px",width:f.width+"px",height:f.height+"px"}}):null]}),P!==null&&m.length>0?(0,d.jsx)($e,{items:m,selected:g,multiSelect:C,disabled:o,current:Math.min(P,m.length-1),onShow:oe,onClose:()=>M(null),onToggle:pe,onRemove:n=>F(g.filter(i=>i!==n))}):null]})}var qt=Re(Tt);export{qt as default};
