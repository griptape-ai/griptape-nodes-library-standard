---
name: griptape-nodes-widget-dev
description: Building and debugging custom parameter widgets for Griptape Nodes Library. Use when creating, fixing, or debugging JS widgets that display in node parameters — including widget lifecycle issues, value update flow, video/media display, URL resolution for the static file server, and the Python ↔ JS data bridge.
---

# Griptape Nodes Custom Widget Development

## Quick Setup

### 1. Register in `griptape_nodes_library.json`
```json
{
  "name": "My Library",
  "widgets": [
    { "name": "MyWidget", "path": "widgets/MyWidget.js" }
  ]
}
```
The `name` here must match what you pass to `Widget(name=..., library=...)` in Python.

### 2. Attach to a Python parameter
```python
from griptape_nodes.exe_types.param_types.parameter_dict import ParameterDict
from griptape_nodes.traits.widget import Widget

self.add_parameter(
    ParameterDict(
        name="my_data",
        default_value={"adjustment": 0, "url": ""},
        tooltip="...",
        allowed_modes={ParameterMode.PROPERTY},
        traits={Widget(name="MyWidget", library="My Library")},
    )
)
```

### 3. Write `widgets/MyWidget.js`
```js
export default function MyWidget(container, props) {
  const { value, onChange, disabled } = props;

  const wrapper = document.createElement("div");
  wrapper.className = "nodrag nowheel";

  const btn = document.createElement("button");
  btn.textContent = `Value: ${value?.adjustment ?? 0}`;
  btn.addEventListener("pointerdown", (e) => {
    e.stopPropagation();
    onChange({ ...value, adjustment: (value?.adjustment ?? 0) + 1 });
  });

  wrapper.appendChild(btn);
  container.appendChild(wrapper);

  function cleanup() { wrapper.remove(); }
  function handleUpdate(newProps) {
    btn.textContent = `Value: ${newProps.value?.adjustment ?? 0}`;
  }

  container._instance = { handleUpdate, cleanup, wrapper };
  return { cleanup, update: handleUpdate };
}
```

---

## Critical: Return `{ cleanup, update }` — not just `cleanup`

The framework mounts the widget **once**. When the parameter value changes server-side, it calls `handleRef.current.update(props)`. A plain cleanup return means updates are silently dropped.

```js
// WRONG — value changes after mount are ignored
return cleanup;

// CORRECT — framework forwards value changes via update()
function handleUpdate(newProps) {
  const v = newProps.value;
  internalUpdate(v?.url || "", v?.adjustment ?? 0, newProps.onChange);
}
container._instance = { handleUpdate, cleanup, wrapper };
return { cleanup, update: handleUpdate };
```

The early-return path (when the widget is already mounted) must also return the handle:
```js
if (container._instance?.wrapper?.isConnected) {
  container._instance.handleUpdate(props);
  return { cleanup: container._instance.cleanup, update: container._instance.handleUpdate };
}
```

---

## Pushing data from Python into the widget

React to input changes with `after_value_set`, then write into the widget's dict parameter. The frontend picks up the new value and calls `update`.

```python
def after_value_set(self, parameter, value):
    if parameter.name == "my_input":
        self.set_parameter_value("my_data", {"url": self._resolve_url(value), "adjustment": 0})
    return super().after_value_set(parameter, value)
```

Do **not** call `set_parameter_value` on the widget param from `__init__` — unnecessary and causes double-render on placement.

---

## Preventing sync loops between Python and the widget

When the widget emits a change and Python writes individual parameters back, `after_value_set` fires for each one. Without guards these trigger another widget push, which triggers another emit, and so on.

Use two flags — one for each direction — to break the loop:

```python
self._syncing_to_widget = False  # Python → widget in progress
self._syncing_to_params = False  # widget → params in progress

def after_value_set(self, parameter, value):
    if parameter.name == "my_widget_param" and not self._syncing_to_widget:
        # widget changed → sync to individual params
        self._syncing_to_params = True
        try:
            self.set_parameter_value("x", value.get("x", 0))
            self.publish_update_to_parameter("x", value.get("x", 0))
        finally:
            self._syncing_to_params = False

    elif not self._syncing_to_params and parameter.name == "x":
        # individual param changed → push to widget
        self._syncing_to_widget = True
        try:
            self._push_widget({...})
        finally:
            self._syncing_to_widget = False
```

---

## Pre-publish the widget dict before individual param publishes

**Problem (snap-then-jump):** When the widget emits and Python calls `publish_update_to_parameter("x", 300)` for each individual param, FlowEditor re-renders *all* widgets using the currently-stored widget dict value. If the stored dict still has the old value (`x: 100`), the widget snaps back to `x: 100` on every individual param publish.

**Fix:** publish the full widget dict first, before any individual param publishes. This ensures the stored value is authoritative before the re-render fires.

```python
def _sync_params_from_widget(self, widget_dict: dict) -> None:
    # ✅ pre-publish the widget dict FIRST
    self.publish_update_to_parameter("my_widget_param", widget_dict)

    self._syncing_to_params = True
    try:
        for key, val in widget_dict.items():
            self.set_parameter_value(key, val)
            self.publish_update_to_parameter(key, val)
    finally:
        self._syncing_to_params = False
```

---

## Detecting wire connections to lock widget controls

When an upstream node wires into a parameter, the widget should show that parameter as read-only (don't let the user drag a handle that would override a connected value).

```python
from griptape_nodes.retained_mode.events.connection_events import (
    ListConnectionsForNodeRequest,
    ListConnectionsForNodeResultSuccess,
)

def _get_locked_params(self) -> list[str]:
    watched = {"x", "y", "width", "height"}
    try:
        result = GriptapeNodes.handle_request(ListConnectionsForNodeRequest(node_name=self.name))
        if isinstance(result, ListConnectionsForNodeResultSuccess):
            return [c.target_parameter_name for c in result.incoming_connections
                    if c.target_parameter_name in watched]
    except Exception:
        pass
    return []

def after_incoming_connection(self, source_node, source_parameter, target_parameter):
    if target_parameter.name in self._watched_params:
        self._push_widget({**self._current_widget_dict(), "locked": self._get_locked_params()})
    return super().after_incoming_connection(source_node, source_parameter, target_parameter)

def after_incoming_connection_removed(self, source_node, source_parameter, target_parameter):
    if target_parameter.name in self._watched_params:
        self._push_widget({**self._current_widget_dict(), "locked": self._get_locked_params()})
    return super().after_incoming_connection_removed(source_node, source_parameter, target_parameter)
```

In the widget JS, read `value.locked` and use it to disable the relevant interactive elements.

---

## Hiding a parameter's property row (keep socket only)

Use `hide_property=True` on the parameter to hide it from the property panel while keeping the input socket visible for wiring:

```python
self.add_parameter(
    ParameterImage(
        name="input_image",
        default_value=None,
        tooltip="...",
        hide_property=True,   # hides the row; input socket still visible
    )
)
```

---

## Resolving file paths to browser-accessible URLs

```python
from griptape_nodes.retained_mode.events.static_file_events import (
    CreateStaticFileDownloadUrlFromPathRequest,
    CreateStaticFileDownloadUrlFromPathResultSuccess,
)

resolved = File(artifact.value).resolve()  # handles macro:// paths
result = GriptapeNodes.handle_request(CreateStaticFileDownloadUrlFromPathRequest(file_path=resolved))
if isinstance(result, CreateStaticFileDownloadUrlFromPathResultSuccess):
    return result.url
```

---

## Canvas sizing: fill the node, letterbox the image

To make a canvas widget that scales with the node and never pushes controls off-screen:

```js
// wrapper fills the node container
wrapper.style.cssText = "display:flex;flex-direction:column;height:100%;gap:6px;";

// canvasWrap takes all remaining vertical space
canvasWrap.style.cssText = [
  "flex:1 1 0", "min-height:180px",
  "display:flex", "align-items:center", "justify-content:center",
  "background:#111", "border-radius:6px", "overflow:hidden",
].join(";");

// size the canvas to letterbox within the available area
function resizeCanvas() {
  if (!imageLoaded || !imgNatW || !imgNatH) return;
  const areaW = canvasWrap.clientWidth  || 480;
  const areaH = canvasWrap.clientHeight || 360;
  scale = Math.min(areaW / imgNatW, areaH / imgNatH);
  canvas.width  = Math.round(imgNatW * scale);
  canvas.height = Math.round(imgNatH * scale);
  render();
}

// fire on resize
const ro = new ResizeObserver(() => { if (imageLoaded) resizeCanvas(); });
ro.observe(canvasWrap);
```

- `flex:1 1 0; min-height:0` lets `canvasWrap` shrink; `min-height:180px` gives a sensible floor.
- `Math.min(areaW/imgNatW, areaH/imgNatH)` letterboxes — the canvas never exceeds either dimension.
- Canvas element has no CSS `width:100%`; its pixel dimensions are set directly.
- The flex-centering (`align-items/justify-content: center`) keeps the canvas centered when it doesn't fill the full area (e.g., portrait image in wide node).

---

## Multi-file widget organization (subdirectory pattern)

For complex widgets, split into a subdirectory so the main file stays readable:

```
widgets/MyWidget/
  MyWidget.js     ← main entry point (registered in library JSON)
  _styles.js      ← visual constants (colors, sizes)
  _sidebar.js     ← preset controls, toolbars
  _footer.js      ← sliders, status bar
```

Update the library JSON path:
```json
{ "name": "MyWidget", "path": "widgets/MyWidget/MyWidget.js" }
```

Import from siblings with relative paths — these are standard ES module imports:
```js
import { HANDLE_R, HANDLE_FILL } from './_styles.js';
import { createSidebar } from './_sidebar.js';
import { createFooter }  from './_footer.js';
```

Each sub-module exports a factory function that returns DOM elements and a `sync` method:
```js
// _sidebar.js
export function createSidebar({ getState, onAction }) {
  const el = document.createElement("div");
  // ... build DOM ...
  function syncDisabled(locked, disabled) { /* update button states */ }
  return { el, syncDisabled };
}
```

---

## Icons: use Lucide SVGs via a `_icons.js` file

Widgets use Lucide icons (MIT licensed). Store only the inner SVG path markup in a shared `_icons.js` — not the full `<svg>` tag — and wrap it with a `mkIcon` factory that applies the standard Lucide attributes.

**`_icons.js`**
```js
// Lucide SVG icon paths (MIT licensed)

export const ICON_PATHS = {
  "rotate-ccw": `<path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/><path d="M3 3v5h5"/>`,
  trash:        `<polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v2"/>`,
  // add more as needed...
};

export function mkIcon(name, size = 15) {
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("viewBox", "0 0 24 24");
  svg.setAttribute("width", size);
  svg.setAttribute("height", size);
  svg.setAttribute("fill", "none");
  svg.setAttribute("stroke", "currentColor");
  svg.setAttribute("stroke-width", "2");
  svg.setAttribute("stroke-linecap", "round");
  svg.setAttribute("stroke-linejoin", "round");
  svg.style.cssText = "display:block;flex-shrink:0;pointer-events:none;";
  svg.innerHTML = ICON_PATHS[name] || "";
  return svg;
}
```

**Usage:**
```js
import { mkIcon } from './_icons.js';

const btn = document.createElement("button");
btn.appendChild(mkIcon("rotate-ccw", 14));
btn.title = "Reset rotation";
```

**How to get path data for a new icon:**
1. Find the icon on [lucide.dev](https://lucide.dev)
2. Copy the SVG source
3. Strip the outer `<svg ...>` tag — keep only the inner `<path>`, `<circle>`, `<line>`, etc. elements as a single string
4. Add it to `ICON_PATHS`

`mkIcon` sets `stroke="currentColor"` so icons inherit the button/label text color automatically, and they adapt to both light and dark themes. Set `pointer-events:none` so icon clicks don't block the parent button's handler.

---

## Theming: use Tailwind CSS variables, not hardcoded colors

The app uses Tailwind CSS / shadcn UI. Always use CSS variables so widgets render correctly in both light and dark modes. Never use hardcoded hex colors like `#1a1a1a` or `#888` for UI chrome.

| Variable | Purpose |
|---|---|
| `var(--background)` | Page/node background |
| `var(--card)` | Card/panel surface |
| `var(--muted)` | Subtle panel background (for toolbars, status bars, etc.) |
| `var(--muted-foreground)` | De-emphasised text and icons |
| `var(--foreground)` | Primary text |
| `var(--border)` | Borders and dividers |
| `var(--accent)` | Hover highlight background |
| `var(--accent-foreground)` | Text on accent |
| `var(--destructive)` | Error / destructive action color |

```js
// WRONG — breaks in light mode
panel.style.background = "#1a1a1a";
label.style.color = "#888";

// CORRECT — adapts automatically
panel.style.background = "var(--muted)";
label.style.color = "var(--muted-foreground)";
```

The canvas/image area can keep a hardcoded dark background (`#111`) since it's always showing media content.

---

## Common gotchas

- `nodrag nowheel` class on wrapper — prevents canvas drag/scroll stealing events
- Stop `pointerdown`/`mousedown`/`keydown` propagation on interactive elements
- Call `onChange` on `blur`, not on every `input` event — avoids focus-stealing
- Pass a copy to `onChange`, never internal state: `onChange({ ...value, field: newVal })`
- Cleanup must remove DOM elements and delete `container._instance`
