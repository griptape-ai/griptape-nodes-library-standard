---
name: dropdown-row-subtitles
description: Add per-row subtitle text and/or icons to an Options dropdown via update_ui_options, so each choice shows a name, a secondary description line, and optionally a Lucide icon.
argument-hint: <parameter-name> <where-descriptions-come-from>
allowed-tools: Read Edit Grep Glob
disable-model-invocation: false
---

# Dropdown Row Subtitles (and Icons)

A pattern for enriching an `Options` dropdown so each row shows a **name**, a secondary **subtitle** (description), and/or a **Lucide icon**. These are pushed via `update_ui_options` after the parameter is registered — they are not part of the `Options` trait itself.

Real examples:
- Subtitles: `griptape_nodes_library/utils/situation_utils.py` — situation dropdown on save nodes
- Icons + subtitles: `griptape_nodes_library/variables/set_variable.py` — variable picker with download-state icons
- Icons + subtitles: `griptape_nodes/exe_types/param_components/huggingface/huggingface_model_parameter.py`

---

## How It Works

The UI renders rich rows by combining feature flags with a `"data"` list:

| Flag | Effect |
|------|--------|
| `"dropdown_row_subtitles": True` | Shows a secondary description line under each choice |
| `"dropdown_row_icons": True` | Shows a Lucide icon to the left of each choice |

Both flags can be used together. Each entry in `"data"` is a dict with some or all of:

```python
{"name": str, "subtitle": str, "icon": str}
# "icon" is a Lucide icon name, e.g. "check-circle", "download", "loader"
```

The `"name"` values must match the `Options(choices=[...])` list exactly — the UI pairs them by name.

---

## Imports

```python
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString
from griptape_nodes.traits.options import Options
```

No special import is needed for `update_ui_options` — it is a method on every `Parameter`.

---

## Core Pattern

### 1. Build the data list

Include whichever keys you need — all three are optional but `"name"` should always be present:

```python
def _build_row_data(
    names: list[str],
    descriptions: dict[str, str] | None = None,
    icons: dict[str, str] | None = None,
) -> list[dict[str, str]]:
    rows = []
    for n in names:
        row: dict[str, str] = {"name": n}
        if descriptions:
            row["subtitle"] = descriptions.get(n, "")
        if icons:
            row["icon"] = icons.get(n, "")
        rows.append(row)
    return rows
```

### 2. Create the parameter with `Options`

```python
options_trait = Options(choices=names)

self.my_param = ParameterString(
    name="my_param",
    default_value=names[0],
    traits={options_trait},
    settable=True,
)
self.add_parameter(self.my_param)
```

### 3. Push subtitle data **after** `add_parameter`

`update_ui_options` must be called after the parameter is registered with the node — pushing it before has no effect.

```python
self.my_param.update_ui_options({
    "data": _build_row_data(names, descriptions=descriptions, icons=icons),
    "dropdown_row_subtitles": True,   # omit if no subtitles
    "dropdown_row_icons": True,       # omit if no icons
})
```

---

## Updating Subtitles at Runtime

When choices change (e.g. in a refresh callback), update both the `Options` trait **and** `ui_options`. Both are needed: the trait governs value validation; `update_ui_options` drives the rendered dropdown.

```python
def _on_refresh(self, _button, button_details):
    fresh_names, fresh_descriptions = _fetch_names_and_descriptions()

    options_trait.choices = fresh_names          # keeps validation in sync
    self.my_param.update_ui_options({
        "data": _build_row_data(fresh_names, descriptions=fresh_descriptions, icons=fresh_icons),
        "dropdown_row_subtitles": True,
        "dropdown_row_icons": True,
    })
```

---

## What NOT to Do

- **Don't** put descriptions or icons inside `Options(choices=...)` — `Options` only accepts a flat `list[str]`.
- **Don't** call `update_ui_options` before `add_parameter` — the parameter must be registered first.
- **Don't** forget the feature flags (`"dropdown_row_subtitles": True`, `"dropdown_row_icons": True`) — without them the `"data"` key is silently ignored and rows render as plain text.
- **Don't** skip updating `options_trait.choices` on refresh — stale choices cause validation failures when the user selects a newly-added option.

---

## Complete Minimal Example

```python
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString
from griptape_nodes.traits.options import Options


ITEMS = {
    "alpha": "The first option",
    "beta":  "The second option",
    "gamma": "The third option",
}


class MyNode(BaseNode):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        names = sorted(ITEMS.keys())
        self._options_trait = Options(choices=names)

        self.my_param = ParameterString(
            name="my_param",
            default_value=names[0],
            tooltip="Choose an item.",
            traits={self._options_trait},
            settable=True,
        )
        self.add_parameter(self.my_param)
        # Must follow add_parameter:
        self.my_param.update_ui_options({
            "data": [{"name": n, "subtitle": ITEMS[n]} for n in names],
            "dropdown_row_subtitles": True,
        })
```

---

## Real Examples

**Subtitles only** — `griptape_nodes_library/utils/situation_utils.py`:
`add_situation_parameter()` and `build_situation_data()` show the full pattern including
a refresh button that updates both the trait choices and the subtitle data in one callback.

**Icons + subtitles** — `griptape_nodes_library/variables/set_variable.py`:
Variable picker that shows a Lucide icon per row indicating download state
(`"check-circle"`, `"download"`, `"loader"`) alongside a subtitle.
