---
name: griptape-nodes-widget-dev
description: Building and debugging custom parameter widgets for Griptape Nodes Library. Use when creating, fixing, or debugging TypeScript widgets that display in node parameters — including widget lifecycle issues, value update flow, syncing a widget with node parameters (synced_parameters), video/media display, URL resolution for the static file server, and the Python ↔ JS data bridge.
---

# Griptape Nodes Custom Widget Development

## Quick Setup

Widgets are TypeScript sources in `widgets/src/<Name>/`, built into one self-contained bundle each in `widgets/dist/`, which is committed. `widgets/README.md` covers the toolchain (`make widgets/install`, `widgets/watch`, `widgets/test`, `widgets/build`, `widgets/check-dist`).

### 1. Register in `griptape_nodes_library.json` and the build
```json
{
  "name": "My Library",
  "widgets": [
    { "name": "MyWidget", "path": "widgets/dist/MyWidget.js" }
  ]
}
```
The `name` here must match what you pass to `Widget(name=..., library=...)` in Python. Add the entry to `ENTRIES` in `widgets/scripts/build.mjs` too: `MyWidget: "src/MyWidget/MyWidget.ts"`.

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

### 3. Write `widgets/src/MyWidget/MyWidget.ts`
The editor's contract types come from `@griptape-ai/nodes-widget-api` (see `widgets/.npmrc` for where it's installed from). Import them with `import type`; the package has no runtime code.

```ts
import type { WidgetHandle, WidgetProps } from "@griptape-ai/nodes-widget-api";

interface MyValue { adjustment?: number; url?: string }
type MyWidgetProps = WidgetProps<MyValue>;
type MyContainer = HTMLElement & { _instance?: { handleUpdate: (p: MyWidgetProps) => void; cleanup: () => void; wrapper: HTMLElement } };

export default function MyWidget(host: HTMLElement, props: MyWidgetProps): WidgetHandle<MyWidgetProps> {
  const container = host as MyContainer;
  const { value, onChange } = props;

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
  function handleUpdate(newProps: MyWidgetProps) {
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

## Pushing display data from Python into the widget

The widget's own parameter holds data that isn't a node parameter, such as an image URL the node resolves for display. React to input changes with `after_value_set`, write the new value, and publish it; the editor calls `update`.

```python
def after_value_set(self, parameter, value):
    if parameter.name == "input_image":
        preview = {"image_url": self._resolve_url(value)}
        self.set_parameter_value("my_data", preview)
        self.publish_update_to_parameter("my_data", preview)
    return super().after_value_set(parameter, value)
```

Do **not** call `set_parameter_value` on the widget param from `__init__`. It's unnecessary and causes a double render on placement; put the starting value in `default_value` instead.

**Loading a workflow skips this hook.** Values load with `initial_setup=True`, which stores them without calling `before_value_set` or `after_value_set`. A widget that relies on `after_value_set` to receive its display data gets nothing on load, unless the saved widget value already holds it. If the widget must be rebuilt from other values on load, override `set_parameter_value` and recompute when `initial_setup` is true (see `ImagePreviewMixin.set_parameter_value`).

Never copy a node parameter into the widget's value; sync it instead (next section).

---

## Syncing the widget with node parameters (`synced_parameters`)

When a widget edits values that are also node parameters (a crop box and its `left`/`top`/`width`/`height` sliders, a rotation preview and its `angle`), list them in `synced_parameters`. The widget then reads and writes those parameters directly. Each value has one copy, the parameter, so the node needs **no sync code at all**: no copying into the widget dict, no re-entrancy flags, no "pre-publish" ordering, no connection hooks for locking. Older widgets did all of that (and some still do); don't copy that pattern.

Requires engine 0.107.0+ and the matching editor. The contract is documented in the engine's `docs/development/custom_nodes/custom_widgets.md` ("Live Parameter Sync").

```python
self.add_parameter(
    ParameterDict(
        name="crop_editor",
        default_value={"image_url": ""},          # only what isn't a parameter
        allowed_modes={ParameterMode.PROPERTY},
        traits={
            Widget(
                name="CropImageEditor",
                library="Griptape Nodes Library",
                synced_parameters=["left", "top", "width", "height"],
            )
        },
    )
)
```

What the widget gets:

- `value`: its own value with each synced parameter's current value merged in under its name (a live value from a drag wins). Only a dict value takes the merge.
- `onChange(value)`: pass the **whole** value. Keys that name a synced parameter are set on that parameter, as if its own control moved; everything else replaces the widget's own value, so a key you leave out is removed from it. Unchanged values and locked parameters are skipped.
- `onLiveChange({ name: value })`: in-progress values during a drag. The parameters' controls on the node move with it; nothing is sent to the engine until `onChange`.
- `locked`: `string[]` of synced parameters with an incoming connection. Disable direct manipulation of them. It comes from the editor, so the node needs no connection hooks, and it's never saved.

```js
export default function CropEditor(container, props) {
  let value = props.value || {};
  let { onChange, onLiveChange } = props;
  let locked = props.locked || [];
  let dragging = false;

  function handleUpdate(newProps) {
    value = newProps.value || {};
    onChange = newProps.onChange;
    onLiveChange = newProps.onLiveChange;
    locked = newProps.locked || [];
    if (!dragging) render();   // during a drag the incoming value is our own live value
  }

  function onPointerMove(rect) {
    onLiveChange?.({ left: rect.left, top: rect.top });
  }

  function onPointerUp(rect) {
    dragging = false;
    onChange({ ...value, left: rect.left, top: rect.top });   // whole value
  }
  // ...
  return { cleanup, update: handleUpdate };
}
```

If a synced parameter's name differs from what the widget calls it internally, the widget maps names; the editor routes by parameter name. Examples: `image/crop_image.py` with `widgets/src/CropImageEditor/`, `video/crop_video.py` with `widgets/src/CropVideoEditor/` (presets plus custom fields), `video/adjust_mask_size.py` with `widgets/src/MaskAdjustmentPreview/`.

**Migrating an old widget:** add `synced_parameters`, drop the parameter copies from the widget dict's default and from every push, delete the `_syncing_*` flags, the dict-to-parameter and parameter-to-dict sync methods, `_get_locked_params` and the connection hooks that only refreshed `locked`, and have the JS read `props.locked`. A saved workflow may still hold the old copies in the widget dict; that's harmless because the parameters win.

For a live preview of a node's result (an image or video operation drawn in the browser), use `ImagePreviewMixin` instead of a custom widget; see the `add-node-preview` skill.

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

`hide_property` hides the editor's whole parameter component, and for image and video inputs that component holds the file browser. The user can then only provide the media by connecting it. That's deliberate on the preview nodes (the preview shows the media), but decide it on purpose. An item in a `ParameterList` inherits the list's `ui_options`, so setting `hide_property` in the list's `ui_options` hides every item's thumbnail too (`image/combine_masks.py`).

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

## Multi-file widget organization

Every widget has its own directory; split a complex one into helper modules so the entry stays readable:

```
widgets/src/MyWidget/
  MyWidget.ts     ← entry point (listed in ENTRIES in scripts/build.mjs)
  _types.ts       ← the value shapes it exchanges with the node
  _styles.ts      ← visual constants (colors, sizes)
  _sidebar.ts     ← preset controls, toolbars
  _footer.ts      ← sliders, status bar
```

Import siblings with relative paths and no extension. The build bundles them, and any npm package the widget imports, into `widgets/dist/MyWidget.js`, so the editor fetches one file:
```ts
import { HANDLE_R, HANDLE_FILL } from './_styles';
import { createSidebar } from './_sidebar';
import { createFooter }  from './_footer';
```

Each sub-module exports a factory function that returns DOM elements and a `sync` method:
```ts
// _sidebar.ts
export function createSidebar({ getState, onAction }: SidebarOptions) {
  const el = document.createElement("div");
  // ... build DOM ...
  function syncDisabled(locked, disabled) { /* update button states */ }
  return { el, syncDisabled };
}
```

---

## Icons: use Lucide SVGs via a `_icons.ts` file

Widgets use Lucide icons (MIT licensed). Store only the inner SVG path markup in the widget's `_icons.ts` — not the full `<svg>` tag — and wrap it with a `mkIcon` factory that applies the standard Lucide attributes.

**`_icons.ts`**
```ts
// Lucide SVG icon paths (MIT licensed)

export const ICON_PATHS: Record<string, string> = {
  "rotate-ccw": `<path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/><path d="M3 3v5h5"/>`,
  trash:        `<polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v2"/>`,
  // add more as needed...
};

export function mkIcon(name: string, size = 15): SVGSVGElement {
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("viewBox", "0 0 24 24");
  svg.setAttribute("width", String(size));
  svg.setAttribute("height", String(size));
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
```ts
import { mkIcon } from './_icons';

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
- Scrollable containers that drive a `ResizeObserver` layout callback need `scrollbar-gutter: stable` to prevent jitter (scrollbar appearance otherwise shifts `clientWidth`, re-triggering the observer)
- Don't copy node parameters into the widget's value; list them in `synced_parameters`
- With `synced_parameters`, `onChange` must carry the whole value; keys left out are removed from the widget's own value
- Loading a workflow (`initial_setup=True`) skips `after_value_set`, so display data pushed only from that hook is missing on load unless it was saved
- Browsers cap how many media players a page holds; a widget with `<video>` must pause it, remove `src` and call `load()` in cleanup
