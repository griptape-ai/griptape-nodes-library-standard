// Steps through a PDF's pages in the browser with pdf.js, drawing one page at a time.
// The node supplies the PDF's URL in `value.url`, and its `rotation` and `password` parameters arrive as
// synced_parameters. The page lives here, so stepping never reaches the engine.

const WIDGET_VERSION = "0.6.0";

// The node's rotation parameter, in clockwise degrees. It arrives in `value.rotation`.
const ROTATION_DEGREES = {
  "none": 0,
  "90° clockwise": 90,
  "180°": 180,
  "90° counter-clockwise": 270,
};
const PDFJS_URL = new URL("./vendor/pdf.min.mjs", import.meta.url).href;
const WORKER_URL = new URL("./vendor/pdf.worker.min.mjs", import.meta.url).href;

let pdfjsPromise = null;

// A Worker has to come from the page's own origin, and the static server is another origin, so the
// worker script is fetched and started from a blob URL. If that fails pdf.js is left to its own fallback.
function loadPdfjs() {
  if (!pdfjsPromise) {
    pdfjsPromise = (async () => {
      const pdfjs = await import(PDFJS_URL);
      let workerMode = "blob worker";
      try {
        const response = await fetch(WORKER_URL);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const blob = new Blob([await response.text()], { type: "text/javascript" });
        pdfjs.GlobalWorkerOptions.workerSrc = URL.createObjectURL(blob);
      } catch (e) {
        workerMode = `direct worker URL (blob failed: ${e.message})`;
        pdfjs.GlobalWorkerOptions.workerSrc = WORKER_URL;
      }
      return { pdfjs, workerMode };
    })();
    pdfjsPromise.catch(() => { pdfjsPromise = null; });
  }
  return pdfjsPromise;
}

// A document is closed through its loading task. The document itself has no destroy().
function closeDocument(document) {
  return document.loadingTask.destroy().catch((e) => console.debug(`[PdfPageViewer] could not close a document: ${e.message}`));
}

export default function PdfPageViewer(container, props) {
  if (container._pdfViewer?.wrapper?.isConnected) {
    container._pdfViewer.handleUpdate(props);
    return { cleanup: container._pdfViewer.cleanup, update: container._pdfViewer.handleUpdate };
  }
  if (container._pdfViewer) delete container._pdfViewer;

  let url = "";
  let password = "";
  let doc = null;
  let page = 1;
  let rendering = false;
  let pendingPage = null;
  let workerMode = "";
  let rotation = 0;
  let loadToken = 0;
  // pdf.js wants a document's destroy awaited before the next one opens, so a swap waits for this.
  let teardown = Promise.resolve();

  const wrapper = document.createElement("div");
  wrapper.className = "pdf-page-viewer nodrag nowheel";
  wrapper.style.cssText = "display:flex;flex-direction:column;gap:6px;height:100%;user-select:none;-webkit-user-select:none;";

  const canvasWrap = document.createElement("div");
  canvasWrap.style.cssText = "background:#111;border-radius:6px;overflow:hidden;flex:1 1 0;min-height:240px;display:flex;align-items:center;justify-content:center;";
  const canvas = document.createElement("canvas");
  canvas.style.cssText = "display:block;";
  canvasWrap.appendChild(canvas);

  const row = document.createElement("div");
  row.style.cssText = "display:flex;align-items:center;justify-content:center;gap:8px;";
  const button = (label) => {
    const b = document.createElement("button");
    b.textContent = label;
    b.style.cssText = "padding:2px 10px;border-radius:4px;border:1px solid var(--border,#444);background:var(--secondary,#222);color:inherit;cursor:pointer;";
    for (const type of ["pointerdown", "mousedown"]) b.addEventListener(type, (e) => e.stopPropagation());
    return b;
  };
  const prev = button("‹");
  const next = button("›");
  const label = document.createElement("span");
  label.style.cssText = "font-size:12px;min-width:90px;text-align:center;";
  row.append(prev, label, next);

  const status = document.createElement("div");
  status.style.cssText = "font-size:10px;font-family:monospace;color:var(--muted-foreground);word-break:break-all;";

  wrapper.append(canvasWrap, row, status);
  container.appendChild(wrapper);

  // Only a problem is worth a line of text. The version is in the page row's tooltip for bug reports.
  row.title = `PdfPageViewer v${WIDGET_VERSION}`;
  console.debug(`[PdfPageViewer] v${WIDGET_VERSION} mounted`);
  function setStatus(text) {
    status.textContent = text;
  }

  function updateControls() {
    const count = doc ? doc.numPages : 0;
    label.textContent = count ? `Page ${page} of ${count}` : "No PDF";
    prev.disabled = page <= 1;
    next.disabled = !count || page >= count;
  }

  async function draw(target) {
    if (!doc) return;
    if (rendering) {
      pendingPage = target;
      return;
    }
    rendering = true;
    const shown = doc;
    try {
      const pdfPage = await shown.getPage(target);
      if (shown !== doc) return;
      // The rotation adds to any the PDF already applies to the page, as it does when the images are extracted.
      const turn = (pdfPage.rotate + rotation) % 360;
      const base = pdfPage.getViewport({ scale: 1, rotation: turn });
      // The whole page fits in the box, so the canvas is sized on both sides to keep the page's proportions.
      const fit = Math.min(Math.max(canvasWrap.clientWidth, 240) / base.width, Math.max(canvasWrap.clientHeight, 240) / base.height);
      const dpr = window.devicePixelRatio || 1;
      const viewport = pdfPage.getViewport({ scale: fit * dpr, rotation: turn });
      canvas.width = Math.round(viewport.width);
      canvas.height = Math.round(viewport.height);
      canvas.style.width = `${base.width * fit}px`;
      canvas.style.height = `${base.height * fit}px`;
      await pdfPage.render({ canvasContext: canvas.getContext("2d"), viewport }).promise;
      // A render of a document that has since been replaced says nothing about the new one.
      if (shown === doc) setStatus("");
    } catch (e) {
      if (shown === doc) setStatus(`render failed: ${e.message}`);
    } finally {
      rendering = false;
      if (pendingPage !== null) {
        const again = pendingPage;
        pendingPage = null;
        draw(again);
      }
    }
  }

  function go(target) {
    if (!doc) return;
    page = Math.min(Math.max(target, 1), doc.numPages);
    updateControls();
    draw(page);
  }

  prev.addEventListener("click", () => go(page - 1));
  next.addEventListener("click", () => go(page + 1));

  async function load(newUrl, newPassword) {
    const token = ++loadToken;
    url = newUrl;
    password = newPassword;
    const old = doc;
    doc = null;
    page = 1;
    updateControls();
    if (old) teardown = teardown.then(() => closeDocument(old));
    // The old page goes at once, so what is on the canvas is always the PDF the node has now.
    canvas.width = canvas.height = 0;
    canvas.style.width = canvas.style.height = "0px";
    console.debug(`[PdfPageViewer] loading ${newUrl || "(no PDF)"}`);
    if (!newUrl) {
      setStatus("");
      return;
    }
    setStatus("Loading...");
    try {
      const loaded = await loadPdfjs();
      await teardown;
      if (token !== loadToken) return;
      workerMode = loaded.workerMode;
      setStatus("Opening PDF...");
      const task = loaded.pdfjs.getDocument({ url: newUrl, password: newPassword || undefined });
      // A large PDF can take a while to arrive, and a blank box with no word looks like a hang.
      task.onProgress = ({ loaded: bytes, total }) => {
        if (token !== loadToken) return;
        const percent = total ? ` ${Math.round((bytes / total) * 100)}%` : "";
        setStatus(`Opening PDF...${percent}`);
      };
      const opened = await task.promise;
      if (token !== loadToken) {
        console.debug(`[PdfPageViewer] dropped ${newUrl}, a newer load replaced it`);
        teardown = teardown.then(() => closeDocument(opened));
        return;
      }
      doc = opened;
      updateControls();
      setStatus("");
      draw(page);
    } catch (e) {
      console.debug(`[PdfPageViewer] ${newUrl} failed: ${e.name}: ${e.message}`);
      if (token !== loadToken) return;
      if (e.name === "PasswordException") {
        setStatus(newPassword ? "The password is wrong." : "This PDF is password protected. Set 'password'.");
      } else {
        setStatus(`Could not open the PDF: ${e.message}`);
      }
    }
  }

  function handleUpdate(newProps) {
    const newUrl = (newProps.value && newProps.value.url) || "";
    const newRotation = ROTATION_DEGREES[newProps.value && newProps.value.rotation] ?? 0;
    if (newRotation !== rotation) {
      rotation = newRotation;
      draw(page);
    }
    const newPassword = (newProps.value && newProps.value.password) || "";
    if (newUrl !== url || newPassword !== password) load(newUrl, newPassword);
  }

  // The box changes size with the node, and the page is drawn to fit it.
  const resizeObserver = new ResizeObserver(() => draw(page));
  resizeObserver.observe(canvasWrap);

  function cleanup() {
    resizeObserver.disconnect();
    loadToken++;
    if (doc) closeDocument(doc);
    doc = null;
    wrapper.remove();
    delete container._pdfViewer;
  }

  container._pdfViewer = { wrapper, handleUpdate, cleanup };
  updateControls();
  handleUpdate(props);
  return { cleanup, update: handleUpdate };
}
