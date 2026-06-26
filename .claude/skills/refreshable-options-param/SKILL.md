---
name: refreshable-options-param
description: Add a ParameterString with an Options dropdown and a Button that refreshes the choices at runtime (e.g. querying a live API for available models, voices, etc.).
argument-hint: <parameter-name> <what-the-list-contains>
allowed-tools: Read Edit Grep Glob
disable-model-invocation: false
---

# Refreshable Options Parameter

A pattern for a dropdown (`Options`) whose choices are fetched at runtime, with a refresh button the user can click to re-query the source.

The canonical example is the Ollama model picker in
`griptape_nodes_library/config/prompt/ollama_prompt_driver.py`.

---

## Imports

```python
from typing import Any

from griptape_nodes.exe_types.core_types import NodeMessageResult, Parameter
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString
from griptape_nodes.traits.button import Button, ButtonDetailsMessagePayload
from griptape_nodes.traits.options import Options
```

---

## Core Pattern

### 1. Fetch the initial list before creating the parameter

Call your data-source function inside `__init__`, before creating the parameter, so the dropdown is pre-populated on first render.

```python
available_items = self._get_available_items()   # may return empty list or error sentinel

if not available_items:
    available_items = ["⚠️ No items found"]
```

### 2. Create the parameter with both traits

Pass `Options` and `Button` together in the `traits` set.  The `Button` wires the refresh callback via `on_click`.

```python
self.items_param = ParameterString(
    name="item",
    default_value=available_items[0],
    tooltip="Select an item from the list.",
    traits={
        Options(choices=available_items),
        Button(
            icon="list-restart",   # Lucide icon name
            size="icon",
            variant="secondary",
            on_click=self._refresh_items,
        ),
    },
)
self.add_parameter(self.items_param)
```

Common icon choices: `"list-restart"` (refresh list), `"refresh-cw"` (generic refresh), `"search"` (fetch/search).

---

## Fetching Choices

### Low-level fetch (raises on error)

```python
def _get_items(self, *, raise_on_error: bool = True) -> list[str]:
    try:
        items = _query_your_source()   # replace with real call
        if items:
            items.sort()
            return items
        return []
    except Exception as e:
        if raise_on_error:
            raise
        return []
```

### User-facing wrapper (returns sentinel on failure)

```python
WARNING_EMOJI = "⚠️"

def _get_available_items(self) -> list[str]:
    try:
        return self._get_items(raise_on_error=True)
    except Exception as e:
        return [f"{WARNING_EMOJI} Could not load items: {e}"]
```

---

## Refresh Callback

`on_click` receives the `Button` instance and a `ButtonDetailsMessagePayload`.  The return value is `NodeMessageResult | None`; return `None` unless you need to send a message back to the UI.

```python
def _refresh_items(
    self,
    button: Button,
    button_details: ButtonDetailsMessagePayload,
) -> NodeMessageResult | None:  # noqa: ARG002
    try:
        fresh_items = self._get_items(raise_on_error=False)

        if not fresh_items:
            fresh_items = [f"{WARNING_EMOJI} No items found"]

        current_value = self.get_parameter_value("item")

        self._update_option_choices(
            param="item",
            choices=fresh_items,
            default=fresh_items[0],
        )

        # Preserve the user's current selection if it is still valid
        if current_value and current_value in fresh_items:
            self.set_parameter_value("item", current_value)
        else:
            # Pick the first real (non-sentinel) value
            first_real = next(
                (v for v in fresh_items if not v.startswith(WARNING_EMOJI)),
                None,
            )
            self.set_parameter_value("item", first_real or fresh_items[0])

    except Exception:
        fallback = [f"{WARNING_EMOJI} No items found"]
        self._update_option_choices(param="item", choices=fallback, default=fallback[0])
        self.set_parameter_value("item", fallback[0])

    return None
```

Key helper: `self._update_option_choices(param, choices, default)` — replaces the `Options` trait's choices list in-place and updates the parameter's default.

---

## Auto-refresh on Related Parameter Changes

If the list depends on another parameter (e.g. a server URL), re-fetch whenever that parameter changes:

```python
def after_value_set(self, parameter: Parameter, value: Any) -> None:
    if parameter.name in ("server_url", "port"):
        self._refresh_item_list()   # internal helper, not the button callback
    return super().after_value_set(parameter, value)
```

Internal helper (same logic as the button callback, but called programmatically):

```python
def _refresh_item_list(self) -> None:
    fresh_items = self._get_items(raise_on_error=False) or [f"{WARNING_EMOJI} No items found"]
    current_value = self.get_parameter_value("item")
    self._update_option_choices(param="item", choices=fresh_items, default=fresh_items[0])
    if current_value and current_value in fresh_items:
        self.set_parameter_value("item", current_value)
    else:
        first_real = next((v for v in fresh_items if not v.startswith(WARNING_EMOJI)), None)
        self.set_parameter_value("item", first_real or fresh_items[0])
```

---

## Validation

Skip validation when the selected value is an error sentinel, then check the real values:

```python
def validate_before_node_run(self) -> list[Exception] | None:
    exceptions = []
    selected = self.get_parameter_value("item")

    if not selected:
        exceptions.append(ValueError("No item selected"))
        return exceptions

    # Don't validate sentinel values — they are UI state, not real selections
    if selected.startswith(WARNING_EMOJI):
        return None

    available = self._get_items(raise_on_error=False)
    if not available:
        exceptions.append(ValueError("No items available — check your connection"))
    elif selected not in available:
        exceptions.append(ValueError(f"'{selected}' is no longer available"))

    return exceptions or None
```

---

## Optional: Status Messages

Show/hide a `ParameterMessage` based on whether the source is reachable (see Ollama pattern):

```python
from griptape_nodes.exe_types.core_types import ParameterMessage

self.error_message = ParameterMessage(
    name="source_error_message",
    title="Source Unavailable",
    value="Cannot reach the item source. Check your connection settings.",
    variant="warning",
)
self.add_node_element(self.error_message)
self.move_element_to_position(self.error_message.name, "first")
self.hide_message_by_name("source_error_message")   # hidden by default

# Then show/hide after each fetch attempt:
def _update_message_visibility(self) -> None:
    try:
        items = self._get_items(raise_on_error=True)
        self.hide_message_by_name("source_error_message")
    except Exception:
        self.show_message_by_name("source_error_message")
```

Call `_update_message_visibility()` at the end of `__init__`, and again at the end of `_refresh_items` / `_refresh_item_list`.

---

## Complete Minimal Example

```python
from typing import Any

from griptape_nodes.exe_types.core_types import NodeMessageResult, Parameter
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString
from griptape_nodes.traits.button import Button, ButtonDetailsMessagePayload
from griptape_nodes.traits.options import Options

WARNING_EMOJI = "⚠️"


class MyNode(BaseNode):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        available = self._get_available_items()
        if not available:
            available = [f"{WARNING_EMOJI} No items found"]

        self.items_param = ParameterString(
            name="item",
            default_value=available[0],
            tooltip="Choose an item.",
            traits={
                Options(choices=available),
                Button(icon="list-restart", size="icon", variant="secondary", on_click=self._refresh_items),
            },
        )
        self.add_parameter(self.items_param)

    def _get_items(self, *, raise_on_error: bool = True) -> list[str]:
        # Replace with real query
        return sorted(["alpha", "beta", "gamma"])

    def _get_available_items(self) -> list[str]:
        try:
            return self._get_items(raise_on_error=True)
        except Exception as e:
            return [f"{WARNING_EMOJI} Error: {e}"]

    def _refresh_items(self, button: Button, button_details: ButtonDetailsMessagePayload) -> NodeMessageResult | None:  # noqa: ARG002
        fresh = self._get_items(raise_on_error=False) or [f"{WARNING_EMOJI} No items found"]
        current = self.get_parameter_value("item")
        self._update_option_choices(param="item", choices=fresh, default=fresh[0])
        if current and current in fresh:
            self.set_parameter_value("item", current)
        else:
            first_real = next((v for v in fresh if not v.startswith(WARNING_EMOJI)), None)
            self.set_parameter_value("item", first_real or fresh[0])
        return None

    def validate_before_node_run(self) -> list[Exception] | None:
        selected = self.get_parameter_value("item")
        if not selected or selected.startswith(WARNING_EMOJI):
            return [ValueError("No valid item selected")] if not selected else None
        available = self._get_items(raise_on_error=False)
        if available and selected not in available:
            return [ValueError(f"'{selected}' is no longer available")]
        return None

    def process(self) -> None:
        item = self.get_parameter_value("item")
        # use item ...
```

---

## Real Example

`griptape_nodes_library/config/prompt/ollama_prompt_driver.py` — full working implementation with:
- Live Ollama model list via the `ollama` Python package
- Connection-settings group (`base_url`, `port`) that trigger auto-refresh via `after_value_set`
- Two conditional `ParameterMessage` widgets (server not running / no models installed)
- `validate_before_node_run` that checks model availability before execution
