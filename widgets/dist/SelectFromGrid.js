var Se="http://www.w3.org/2000/svg",Me=new Set(["animate","animateMotion","animateTransform","circle","clipPath","defs","desc","ellipse","feBlend","feColorMatrix","feComponentTransfer","feComposite","feConvolveMatrix","feDiffuseLighting","feDisplacementMap","feDistantLight","feDropShadow","feFlood","feFuncA","feFuncB","feFuncG","feFuncR","feGaussianBlur","feImage","feMerge","feMergeNode","feMorphology","feOffset","fePointLight","feSpecularLighting","feSpotLight","feTile","feTurbulence","filter","foreignObject","g","image","line","linearGradient","marker","mask","metadata","mpath","path","pattern","polygon","polyline","radialGradient","rect","set","stop","svg","switch","symbol","text","textPath","tspan","use","view"]),de=["value","checked","selected"];function fe(l,i){if(!(i==null||typeof i=="boolean")){if(Array.isArray(i)){for(let c of i)fe(l,c);return}l.appendChild(typeof i=="object"?i:document.createTextNode(String(i)))}}function Te(l,i){if(typeof i=="string"){l.style.cssText=i;return}if(!(!i||typeof i!="object"))for(let[c,h]of Object.entries(i))h!=null&&(c.startsWith("--")?l.style.setProperty(c,String(h)):l.style[c]=String(h))}function ge(l,i,c,h){if(i==="style"){Te(l,c);return}if(i==="class"||i==="className"){c!=null&&c!==!1&&l.setAttribute("class",String(c));return}if(i.length>2&&i.startsWith("on")&&typeof c=="function"){l.addEventListener(i.slice(2).toLowerCase(),c);return}if(c!=null){if(i==="innerHTML"||i==="textContent"||!h&&!i.includes("-")&&i in l){l[i]=c;return}c!==!1&&l.setAttribute(i,c===!0?"":String(c))}}function He(l,i){let c=Me.has(l),h=c?document.createElementNS(Se,l):document.createElement(l);fe(h,i.children);for(let[C,I]of Object.entries(i))C==="children"||C==="ref"||de.includes(C)||ge(h,C,I,c);for(let C of de)C in i&&ge(h,C,i[C],c);return i.ref?.(h),h}function e(l,i){return typeof l=="function"?l(i):He(l,i)}var pe="1.4.0",Z=[202,162,267,322,42,82,132,232],se=[95,125,105,145,85,130,115,155],ze=`
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
`;function ue(){let l=document.getElementById("sfg-widget-styles");l?.dataset.version!==pe&&(l||(l=e("style",{id:"sfg-widget-styles"}),document.head.appendChild(l)),l.dataset.version=pe,l.textContent=ze)}var Ie={music:'<path d="M9 18V5l12-2v13"/><circle cx="6" cy="18" r="3"/><circle cx="18" cy="16" r="3"/>',"loader-circle":'<path d="M21 12a9 9 0 1 1-6.219-8.56"/>',"alert-circle":'<circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/>',check:'<polyline points="20 6 9 17 4 12"/>',"maximize-2":'<polyline points="15 3 21 3 21 9"/><polyline points="9 21 3 21 3 15"/><line x1="21" x2="14" y1="3" y2="10"/><line x1="3" x2="10" y1="21" y2="14"/>',x:'<path d="M18 6 6 18"/><path d="m6 6 12 12"/>',"chevron-left":'<path d="m15 18-6-6 6-6"/>',"chevron-right":'<path d="m9 18 6-6-6-6"/>'};function E({name:l,size:i=14}){return e("svg",{viewBox:"0 0 24 24",width:i,height:i,fill:"none",stroke:"currentColor","stroke-width":"2","stroke-linecap":"round","stroke-linejoin":"round",style:"display:block;flex-shrink:0;pointer-events:none;",innerHTML:Ie[l]||""})}function be({layout:l,columns:i,isDisabled:c,showLabels:h,labelSize:C,onColumnsChange:I,onColumnsCommit:u,onLayoutChange:f,onClear:y,onLabelsToggle:L,onLabelSizeChange:P,onLabelSizeCommit:x,onFullscreen:M}){let D=p=>p.stopPropagation(),T=e("input",{type:"range",class:"sfg-slider",min:"1",max:"8",value:String(i),disabled:c,onPointerDown:D,onInput:()=>{k.textContent=T.value,I(parseInt(T.value,10))},onChange:()=>{u&&u(parseInt(T.value,10))}}),k=e("span",{children:String(i)}),G=e("button",{type:"button",class:"sfg-layout-btn"+(l==="grid"?" active":""),disabled:c,onPointerDown:D,onClick:p=>{p.stopPropagation(),G.classList.add("active"),B.classList.remove("active"),f("grid")},children:"Grid"}),B=e("button",{type:"button",class:"sfg-layout-btn"+(l==="masonry"?" active":""),disabled:c,onPointerDown:D,onClick:p=>{p.stopPropagation(),B.classList.add("active"),G.classList.remove("active"),f("masonry")},children:"Masonry"}),_=e("button",{type:"button",class:"sfg-labels-btn"+(h!==!1?" active":""),disabled:c,onPointerDown:D,onClick:p=>{p.stopPropagation(),_.classList.toggle("active"),L&&L(_.classList.contains("active"))},children:[e("span",{class:"sfg-cb-box",children:e(E,{name:"check",size:9})}),"Labels"]}),R=e("input",{type:"range",class:"sfg-slider",min:"8",max:"32",value:String(C||10),disabled:c,onPointerDown:D,onInput:()=>{let p=parseInt(R.value,10);V.textContent=String(p),P&&P(p)},onChange:()=>{x&&x(parseInt(R.value,10))}}),V=e("span",{children:String(C||10)}),X=e("span",{class:"sfg-count"}),O=e("button",{type:"button",class:"sfg-clear-btn",disabled:c,onPointerDown:D,onClick:p=>{p.stopPropagation(),y()},children:"Clear"}),K=e("button",{type:"button",class:"sfg-fullscreen-btn",title:"Fullscreen",onPointerDown:D,onClick:p=>{p.stopPropagation(),M()},children:e(E,{name:"maximize-2",size:12})}),A=e("div",{class:"sfg-controls",children:[e("label",{class:"sfg-label",children:["Columns ",T,k]}),e("div",{class:"sfg-layout-btns",children:[G,B]}),_,e("label",{class:"sfg-label sfg-label-size-group",children:["Aa ",R,V]}),X,O,K]});function d(p){T.disabled=p,G.disabled=p,B.disabled=p,O.disabled=p,_.disabled=p,R.disabled=p}return{controls:A,colSlider:T,colVal:k,gridBtn:G,masonryBtn:B,countEl:X,clearBtn:O,labelsBtn:_,labelSizeSlider:R,labelSizeVal:V,fullscreenBtn:K,setDisabled:d}}var Pe=50;function me({getItems:l,getSelected:i,isMultiSelect:c,isDisabled:h,onToggle:C,onRemove:I}){let u=0,f=!1,y=null,L=e("div",{class:"sfg-viewer-title"}),P=e("span",{class:"sfg-viewer-counter"}),x=e("button",{type:"button",class:"sfg-viewer-select",onClick:()=>U()}),M=e("button",{type:"button",class:"sfg-viewer-nav prev",title:"Previous (←)",onClick:()=>Y(-1),children:e(E,{name:"chevron-left",size:28})}),D=e("button",{type:"button",class:"sfg-viewer-nav next",title:"Next (→)",onClick:()=>Y(1),children:e(E,{name:"chevron-right",size:28})}),T=null,k=e("div",{class:"sfg-viewer-stage",onPointerDown:o=>{o.target.closest("video, audio, .sfg-viewer-text, .sfg-viewer-dict")||(T={x:o.clientX,y:o.clientY})},onPointerUp:o=>{if(!T)return;let r=o.clientX-T.x,g=o.clientY-T.y;T=null,Math.abs(r)>Pe&&Math.abs(r)>Math.abs(g)&&Y(r<0?1:-1)},onPointerCancel:()=>{T=null}}),G=e("div",{class:"sfg-viewer-strip-hint"}),B=e("div",{class:"sfg-viewer-strip-items"}),_=e("div",{class:"sfg-viewer",tabIndex:-1,role:"dialog","aria-modal":"true",children:[e("div",{class:"sfg-viewer-header",children:[L,P,x,e("button",{type:"button",class:"sfg-viewer-icon-btn",title:"Close (Esc)",onClick:()=>W(),children:e(E,{name:"x",size:18})})]}),e("div",{class:"sfg-viewer-body",children:[M,k,D]}),e("div",{class:"sfg-viewer-strip",children:[G,B]})]});function R({className:o,text:r}){return e("div",{class:o,textContent:r})}function V(){return e("div",{class:"sfg-error-card",children:e(E,{name:"alert-circle",size:40})})}function X(o){return o?JSON.stringify([o.type,o.url,o.full_url,o.value]):null}function O(){k.querySelectorAll("video, audio").forEach(g=>g.pause()),k.innerHTML="";let r=l()[u];if(y=X(r),!!r)switch(r.type){case"image":{let g=!r.full_url||!r.url,w=e("img",{src:r.full_url||r.url||"",alt:r.label||"",onError:()=>{if(g){w.replaceWith(e(V,{}));return}g=!0,w.src=r.url}});k.appendChild(w);break}case"video":{let g=e("video",{src:r.url||"",poster:r.thumbnail||null,controls:!0,loop:!0,playsInline:!0,onError:()=>g.replaceWith(e(V,{}))});k.appendChild(g),g.play().catch(()=>{});break}case"audio":{let g=e("div",{class:"sfg-viewer-audio",children:[e(E,{name:"music",size:72}),r.url?e("audio",{src:r.url,controls:!0,onError:()=>g.replaceWith(e(V,{}))}):null]});k.appendChild(g);break}case"dict":k.appendChild(e(R,{className:"sfg-viewer-dict",text:r.value||"{}"}));break;default:k.appendChild(e(R,{className:"sfg-viewer-text",text:r.value!==void 0?String(r.value):r.label||""}));break}}function K(o,r){let g=e("div",{class:"sfg-viewer-thumb"+(r===u?" current":""),title:o.label||`Item ${r+1}`,"data-idx":String(r),onClick:()=>j(r)});switch(o.type){case"image":g.appendChild(e("img",{src:o.url||"",alt:"",decoding:"async"}));break;case"video":{let w=()=>e("video",{src:o.url?`${o.url}#t=0.1`:"",muted:!0,preload:"metadata"});if(o.thumbnail){let N=e("img",{src:o.thumbnail,alt:""});N.addEventListener("error",()=>N.replaceWith(e(w,{})),{once:!0}),g.appendChild(N)}else g.appendChild(e(w,{}));break}case"audio":g.classList.add("sfg-viewer-thumb-icon"),g.appendChild(e(E,{name:"music",size:22}));break;default:{let w=o.type==="dict"?o.value||"{}":o.value!==void 0?String(o.value):o.label||"";g.style.background=`hsla(${Z[r%Z.length]}, 22%, 50%, 0.18)`,g.appendChild(e(R,{className:"sfg-viewer-thumb-text",text:w}));break}}return h()||g.appendChild(e("button",{type:"button",class:"sfg-viewer-thumb-remove",title:"Remove from selection",onClick:w=>{w.stopPropagation(),I(r)},children:e(E,{name:"x",size:12})})),g}function A(){let o=l(),r=i(),g=o[u],w=r.includes(u),N=c();L.textContent=g?.label||"",P.textContent=o.length?`${u+1} / ${o.length}`:"",M.disabled=u<=0,D.disabled=u>=o.length-1,x.disabled=h(),x.classList.toggle("active",w),x.replaceChildren(),w&&x.appendChild(e(E,{name:"check",size:14}));let F;w?F="Selected":!N&&r.length?F="Replace selection":F="Select",x.appendChild(document.createTextNode(F)),x.title="Toggle selection (Space)",r.length?N?G.textContent=`${r.length} selected`:G.textContent="Selected · selecting another item replaces it":G.textContent=N?"Nothing selected · press Space to select":"Nothing selected"}function d(){let o=l();B.replaceChildren(),i().forEach(r=>{o[r]&&B.appendChild(K(o[r],r))})}function p(){for(let o of B.children)o.classList.toggle("current",Number(o.dataset.idx)===u)}function j(o){let r=l();if(!r.length)return;let g=Math.max(0,Math.min(r.length-1,o)),w=g!==u;u=g,w&&O(),A(),p()}function Y(o){j(u+o)}function U(){h()||!l()[u]||C(u)}function ee(o){if(f)switch(o.stopPropagation(),o.key){case"ArrowLeft":o.preventDefault(),Y(-1);break;case"ArrowRight":o.preventDefault(),Y(1);break;case" ":o.preventDefault(),o.repeat||U();break;case"Escape":o.preventDefault(),W();break}}function Q(o){let r=l();r.length&&(u=Math.max(0,Math.min(r.length-1,o)),f||(f=!0,document.body.appendChild(_),window.addEventListener("keydown",ee,!0)),O(),A(),d(),_.focus())}function W(){f&&(f=!1,k.querySelectorAll("video, audio").forEach(o=>o.pause()),k.innerHTML="",y=null,window.removeEventListener("keydown",ee,!0),_.remove())}function ne(){if(!f)return;let o=l();if(!o.length){W();return}if(u=Math.min(u,o.length-1),X(o[u])!==y)O();else{let r=k.querySelector("video"),g=o[u]?.thumbnail;r&&g&&(r.poster=g)}A(),d()}return{open:Q,close:W,refresh:ne,destroy:W}}var ie=l=>l.stopPropagation();function De(){return e("div",{class:"sfg-spinner",children:e(E,{name:"loader-circle",size:22})})}function he(){return e("div",{class:"sfg-error-card",children:e(E,{name:"alert-circle",size:28})})}function Ge(l,i){let c=l;if(c._sfgInst?.wrapper?.isConnected)return c._sfgInst.handleUpdate(i),{cleanup:c._sfgInst.cleanup,update:c._sfgInst.handleUpdate};c._sfgInst&&c._sfgInst.cleanup(),ue();let h=i.value||{},C=i.onChange,I=i.disabled||!1,u=h.items||[],f=h.selected_indices||[],y=h.columns||3,L=h.layout||"grid",P=h.settings?.multi_select!==!1,x=h.show_labels!==!1,M=h.label_size||10,{controls:D,colSlider:T,colVal:k,gridBtn:G,masonryBtn:B,countEl:_,clearBtn:R,labelsBtn:V,labelSizeSlider:X,labelSizeVal:O,setDisabled:K}=be({layout:L,columns:y,isDisabled:I,showLabels:x,labelSize:M,onColumnsChange(t){y=t,F(!0)},onColumnsCommit(t){y=t,N(),j()},onLayoutChange(t){L=t,N(),j()},onClear(){f.length!==0&&U([])},onLabelsToggle(t){x=t,p.classList.toggle("sfg-hide-labels",!x),j()},onLabelSizeChange(t){M=t,p.style.setProperty("--sfg-label-size",t+"px")},onLabelSizeCommit(t){M=t,j()},onFullscreen(){A.open(f.length?f[0]:0)}}),A=me({getItems:()=>u,getSelected:()=>f,isMultiSelect:()=>P,isDisabled:()=>I,onToggle:t=>ee(t),onRemove:t=>U(f.filter(s=>s!==t))}),d=e("div",{class:"sfg-grid",onPointerDown:ye,onPointerMove:we,onPointerUp:Le,onPointerCancel:ke}),p=e("div",{class:"sfg-widget nodrag nowheel"+(x?"":" sfg-hide-labels"),style:{"--sfg-label-size":M+"px"},children:[D,d]});c.appendChild(p);function j(){C({...h||{},items:u,selected_indices:f,columns:y,layout:L,show_labels:x,label_size:M})}function Y(){d.querySelectorAll(".sfg-cell").forEach(t=>{t.classList.toggle("selected",f.includes(parseInt(t.dataset.idx,10)))})}function U(t){f=t,Y(),Q(),j(),A.refresh()}function ee(t){let s=f.includes(t);U(P?s?f.filter(n=>n!==t):[...f,t]:s?[]:[t])}function Q(){let t=f.length>0;_.textContent=t?`${f.length} selected`:"",_.classList.toggle("active",t),R.classList.toggle("active",t),d.classList.toggle("has-selection",t)}function W(){let n=(d.clientWidth-(y-1)*5)/y;d.style.gridAutoRows=n>0?`${n}px`:""}function ne(){let t=L==="masonry";d.classList.toggle("layout-masonry",t),t?(d.style.gridTemplateColumns="",d.style.gridAutoRows=""):(d.style.gridTemplateColumns=`repeat(${y}, 1fr)`,W())}let o=new ResizeObserver(()=>{L!=="masonry"&&W()});o.observe(d);let r=new IntersectionObserver(t=>t.forEach(({isIntersecting:s,target:n})=>{s&&w(n)}),{root:d,rootMargin:"300px 0px",threshold:0});function g(t,s,n,a=!1){t.dataset.idx=String(n);let m=Z[n%Z.length],S=L==="masonry"?se[n%se.length]+"px":null;if(a){t.appendChild(e("div",{class:"sfg-cell-inner",style:{background:`hsla(${m}, 22%, 50%, 0.09)`,minHeight:S}}));return}let H=s.type==="image"||s.type==="video"||s.type==="audio";t.appendChild(e("div",{class:"sfg-cell-inner"+(H?" sfg-loading":""),children:H?e(De,{}):s.type==="dict"?e("div",{class:"sfg-dict-card",textContent:s.value||"{}"}):e("div",{class:"sfg-quote-card",style:{background:`hsla(${m}, 22%, 50%, 0.09)`,minHeight:S},children:e("div",{class:"sfg-quote-text",textContent:s.value!==void 0?String(s.value):s.label||""})})})),t.appendChild(e("button",{type:"button",class:"sfg-expand-btn",title:"View fullscreen",onPointerDown:ie,onClick:v=>{v.stopPropagation(),A.open(n)},children:e(E,{name:"maximize-2",size:12})})),s.label&&t.appendChild(e("div",{class:"sfg-item-label",children:s.label})),H&&r.observe(t)}function w(t){if(t.dataset.hydrated==="true")return;let s=parseInt(t.dataset.idx,10),n=u[s];if(!n||(n.type==="image"||n.type==="video")&&!n.url)return;t.dataset.hydrated="true",r.unobserve(t);let a=t.querySelector(".sfg-cell-inner");if(!a)return;let m=a.querySelector(".sfg-spinner"),S=()=>{m&&(m.style.opacity="0",m.addEventListener("transitionend",()=>m.remove(),{once:!0}))},H=()=>{a.classList.remove("sfg-loading"),m&&m.remove(),a.appendChild(e(he,{}))};switch(n.type){case"image":{let v=e("img",{src:n.url,alt:n.label||"",decoding:"async",onLoad:()=>{a.classList.remove("sfg-loading"),v.classList.add("sfg-loaded"),S()},onError:H});a.appendChild(v);break}case"video":{let v=e("video",{src:n.url,poster:n.thumbnail||null,muted:!0,loop:!0,playsInline:!0,preload:"metadata",onLoadedMetadata:()=>{a.classList.remove("sfg-loading"),v.currentTime=.001,v.classList.add("sfg-loaded"),S()},onError:H});t.addEventListener("mouseenter",()=>{v.play().catch(()=>{})}),t.addEventListener("mouseleave",()=>{v.pause(),v.currentTime=0}),a.appendChild(v);break}case"audio":{a.classList.remove("sfg-loading"),m&&m.remove();let v=e("div",{class:"sfg-audio-card",children:[e(E,{name:"music",size:28}),n.url?e("audio",{src:n.url,controls:!0,onPointerDown:ie,onClick:ie,onError:()=>v.replaceWith(e(he,{}))}):null]});a.appendChild(v);break}}}function N(){let t=[...d.querySelectorAll(".sfg-cell")],s=110;t.forEach((n,a)=>{let m=Math.min(a*6,40);n.style.transition=`opacity ${s}ms ease ${m}ms, transform ${s}ms ease ${m}ms`,n.style.opacity="0",n.style.transform="scale(0.82)"}),setTimeout(()=>{F();let n=[...d.querySelectorAll(".sfg-cell")];n.forEach(a=>{a.style.transition="none",a.style.opacity="0",a.style.transform="scale(0.88)"}),requestAnimationFrame(()=>requestAnimationFrame(()=>{n.forEach((a,m)=>{let S=Math.min(m*12,90);a.style.transition=`opacity 0.2s ease ${S}ms, transform 0.22s cubic-bezier(0.34,1.56,0.64,1) ${S}ms`,a.style.opacity="",a.style.transform=""})}))},s+Math.min(t.length*6,40)+20)}function F(t=!1){if(d.innerHTML="",ne(),u.length===0){d.appendChild(e("div",{class:"sfg-empty",children:"Connect a list to display items here"})),Q();return}let s=(n,a)=>{let m=e("div",{class:"sfg-cell"+(f.includes(a)?" selected":"")});return g(m,n,a,t),m};if(L==="masonry"){let n=Array.from({length:y},()=>e("div",{class:"sfg-masonry-col"}));d.append(...n),u.forEach((a,m)=>n[m%y].appendChild(s(a,m)))}else d.append(...u.map(s));Q()}d.addEventListener("wheel",t=>{t.stopPropagation()},{passive:!0});let b=null,ve=5;function xe(t){for(;t&&t!==d;){if(t.classList.contains("sfg-cell"))return t;t=t.parentElement}return null}function oe(t,s){let n=d.getBoundingClientRect(),a=n.width>0?n.width/d.offsetWidth:1;return{x:(t-n.left)/a+d.scrollLeft,y:(s-n.top)/a+d.scrollTop}}function ae(t){let s=t.getBoundingClientRect(),n=d.getBoundingClientRect(),a=n.width>0?n.width/d.offsetWidth:1;return{left:(s.left-n.left)/a+d.scrollLeft,top:(s.top-n.top)/a+d.scrollTop,width:s.width/a,height:s.height/a}}function ye(t){if(I||t.button!==0)return;t.stopPropagation();let{x:s,y:n}=oe(t.clientX,t.clientY),a=null;P&&(a=e("div",{class:"sfg-lasso",style:{left:s+"px",top:n+"px",width:"0",height:"0"}}),d.appendChild(a)),b={startX:s,startY:n,lasso:a,dragging:!1,startCell:xe(t.target)},d.setPointerCapture(t.pointerId)}function we(t){if(!b)return;let{x:s,y:n}=oe(t.clientX,t.clientY),a=s-b.startX,m=n-b.startY;if(!b.dragging&&Math.hypot(a,m)>ve&&(b.dragging=!0),b.dragging&&b.lasso){let S=Math.min(b.startX,s),H=Math.min(b.startY,n),v=Math.max(b.startX,s),$=Math.max(b.startY,n);Object.assign(b.lasso.style,{left:S+"px",top:H+"px",width:Math.abs(a)+"px",height:Math.abs(m)+"px"}),d.querySelectorAll(".sfg-cell").forEach(q=>{let{left:z,top:J,width:re,height:te}=ae(q),Ee=z<v&&z+re>S&&J<$&&J+te>H,Ce=parseInt(q.dataset.idx,10);q.classList.toggle("pending",Ee&&!f.includes(Ce))})}}function Le(t){if(b){if(b.lasso&&b.lasso.remove(),d.querySelectorAll(".sfg-cell.pending").forEach(s=>s.classList.remove("pending")),b.dragging&&b.lasso){let{x:s,y:n}=oe(t.clientX,t.clientY),a=Math.min(b.startX,s),m=Math.min(b.startY,n),S=Math.max(b.startX,s),H=Math.max(b.startY,n),v=!1;d.querySelectorAll(".sfg-cell").forEach($=>{let{left:q,top:z,width:J,height:re}=ae($);if(q<S&&q+J>a&&z<H&&z+re>m){let te=parseInt($.dataset.idx,10);!isNaN(te)&&!f.includes(te)&&(f=[...f,te],$.classList.add("selected"),v=!0)}}),v&&(Q(),j(),A.refresh())}else if(!b.dragging&&b.startCell){let s=parseInt(b.startCell.dataset.idx,10);isNaN(s)||ee(s)}b=null}}function ke(){b&&(b.lasso&&b.lasso.remove(),d.querySelectorAll(".sfg-cell.pending").forEach(t=>t.classList.remove("pending")),b=null)}F();function le(t){C=t.onChange;let s=(t.disabled||!1)!==I;I=t.disabled||!1,K(I);let n=t.value||{},a=n.items||[],m=n.selected_indices||[],S=n.columns||3,H=n.layout||"grid",v=n.settings?.multi_select!==!1,$=n.show_labels!==!1,q=n.label_size||10,z=!1;JSON.stringify(a)!==JSON.stringify(u)&&(u=a,z=!0),JSON.stringify(m)!==JSON.stringify(f)&&(f=m,z=!0),S!==y&&(y=S,T.value=String(y),k.textContent=String(y),z=!0),H!==L&&(L=H,G.classList.toggle("active",L==="grid"),B.classList.toggle("active",L==="masonry"),z=!0);let J=v!==P;J&&(P=v,!P&&f.length>1&&(f=[],z=!0)),$!==x&&(x=$,V.classList.toggle("active",x),p.classList.toggle("sfg-hide-labels",!x)),q!==M&&(M=q,X.value=String(M),O.textContent=String(M),p.style.setProperty("--sfg-label-size",M+"px")),h=n,z&&F(),(z||s||J)&&A.refresh()}function ce(){o.disconnect(),r.disconnect(),A.destroy(),p.remove(),delete c._sfgInst}return c._sfgInst={wrapper:p,handleUpdate:le,cleanup:ce},{cleanup:ce,update:le}}export{Ge as default};
