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

## Common gotchas

- `nodrag nowheel` class on wrapper — prevents canvas drag/scroll stealing events
- Stop `pointerdown`/`mousedown`/`keydown` propagation on interactive elements
- Call `onChange` on `blur`, not on every `input` event — avoids focus-stealing
- Pass a copy to `onChange`, never internal state: `onChange({ ...value, field: newVal })`
- Cleanup must remove DOM elements and delete `container._instance`

---

## Splitting a Large Widget into ES Modules

Widgets are served as static files and loaded with `import(widgetUrl)`. The browser resolves **relative imports** relative to that served URL — so sibling files in `widgets/` work with no build step.

```
widgets/
  MyWidget.js        ← main entry point
  _geometry.js       ← pure math helpers
  _drawing.js        ← canvas drawing factory
  _hotkeys.js        ← document keyboard handler setup/teardown
  _tooltip.js        ← tooltip factory
  _styles.js         ← CSS injection + constants
  _icons.js          ← SVG path strings + icon builder
```

Convention: prefix helper files with `_` to distinguish them from widget entry points.

**Entry point imports:**
```js
import { ICON_PATHS, mkIcon } from './_icons.js';
import { injectStyles, defaultData, DEFAULT_CANVAS_WIDTH, DEFAULT_CANVAS_HEIGHT } from './_styles.js';
import { decimatePoints, paintCenter, frameRotHandle } from './_geometry.js';
import { createDrawing } from './_drawing.js';
import { createTooltip } from './_tooltip.js';
import { setupHotkeys } from './_hotkeys.js';
```

No CDN, no bundler, no build step. The browser handles it.

### Module type guide

| Module | Pattern | What goes in it |
|--------|---------|-----------------|
| `_geometry.js` | Pure functions | Math, coordinate transforms, bounds calculations — no DOM, no canvas |
| `_drawing.js` | Factory | `createDrawing(getState)` — canvas draw calls that need shared mutable state |
| `_tooltip.js` | Factory | `createTooltip()` — creates DOM element, returns `{ addTooltip, cleanup }` |
| `_hotkeys.js` | Setup/teardown | `setupHotkeys(getState, actions)` — registers document listeners, returns `cleanup()` |
| `_styles.js` | Constants + side-effect | CSS string injection, default data shapes, size constants |
| `_icons.js` | Data + helper | SVG path strings, `mkIcon(name, size)` builder |

---

## Factory Pattern for Stateful Canvas Functions

When drawing functions need access to mutable widget state (like `ctx`, `displayScale`, `hoverId`), use a factory that takes a `getState` lambda instead of passing state as parameters on every call.

```js
// _drawing.js
export function createDrawing(getState) {
  // getState() returns { ctx, displayScale, hoverId }

  function drawText(ann, selected) {
    const { ctx, displayScale, hoverId } = getState();  // reads current values
    // ...
  }

  return { drawText, drawRect, drawEllipse, drawPaint, drawArrowAnnotation };
}
```

```js
// Widget main file
let ctx, displayScale, hoverId;  // mutable — updated as the widget runs

const drawing = createDrawing(() => ({ ctx, displayScale, hoverId }));
const { drawText, drawRect } = drawing;
```

The lambda `() => ({ ctx, displayScale, hoverId })` is evaluated fresh on each call, so the drawing functions always see the current values without the widget needing to push updates.

Same pattern applies to `setupHotkeys(getState, actions)`:
```js
const _cleanupHotkeys = setupHotkeys(
  () => ({ mouseIsOver: _mouseIsOver, textEditId, activeTool, currentValue, toolSettings }),
  { setTool, resetView, deleteAnnotations, /* ... */ }
);
```

---

## Scoping Hotkeys to the Widget (Mouse-Over Guard)

`document.addEventListener` handlers fire globally. To prevent widget keyboard shortcuts from hijacking the rest of the app, guard with a `_mouseIsOver` flag set by the container's `mouseenter`/`mouseleave` events.

```js
let _mouseIsOver = false;
container.addEventListener("mouseenter", () => { _mouseIsOver = true; });
container.addEventListener("mouseleave", () => { _mouseIsOver = false; });
```

Use `container` (not `canvas`) so the guard covers the toolbar, settings panels, and canvas area.

In each document-level key handler:
```js
function _deleteInterceptor(e) {
  if (!getState().mouseIsOver) return;  // ← guard first
  // ... rest of handler
}
```

**Exception: skip the guard on `keyup` for any modifier you track as a persistent boolean.** If the user holds a modifier key, moves the mouse out of the widget, and releases, the boolean must still reset — otherwise the widget gets stuck in the wrong state (wrong cursor, wrong tool behavior, etc.).

```js
function _onAltDown(e) {
  if (!getState().mouseIsOver) return;  // guard: only activate if over widget
  if (e.key === "Alt") actions.onAltDown();
}

function _onAltUp(e) {
  // NO guard — always release alt state regardless of mouse position
  if (e.key === "Alt") actions.onAltUp();
}
```

This applies to **any** modifier you store as state — alt, shift, ctrl, meta. The pattern is always the same: guard `keydown`, skip the guard on `keyup`. If you're not storing a boolean (just reading `e.shiftKey` inline at event time), no special handling is needed.

---

## Cleanup Consolidation

Each factory returns its own cleanup. Collect them all in the widget's `cleanup()`:

```js
const _tooltip = createTooltip();
const _cleanupHotkeys = setupHotkeys(getState, actions);

function cleanup() {
  _tooltip.cleanup();       // removes tooltip DOM element, clears timer
  _cleanupHotkeys();        // removes all document keydown/keyup listeners
  resizeObserver.disconnect();
  wrapper.remove();
  delete container._instance;
}
```

This keeps cleanup traceable — each factory owns its own teardown, the widget just calls them in sequence.
