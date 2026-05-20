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
