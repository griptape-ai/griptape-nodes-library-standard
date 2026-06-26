---
name: hide-show-parameter
description: Hide or show node parameters by name at runtime — in __init__ for initial state, or in after_value_set to toggle visibility based on another parameter's value.
argument-hint: <parameter-name> <condition-parameter-name>
allowed-tools: Read Edit Grep Glob
disable-model-invocation: false
---

# Hide and Show Parameters

Use `hide_parameter_by_name` / `show_parameter_by_name` to control parameter visibility at runtime. Both accept a single name string or a list of names.

```python
self.hide_parameter_by_name("my_param")
self.show_parameter_by_name("my_param")

# Multiple at once
self.hide_parameter_by_name(["param_a", "param_b"])
self.show_parameter_by_name(["param_a", "param_b"])
```

---

## Pattern 1 — Hidden by default, shown conditionally

Hide in `__init__` right after `add_parameter`, then toggle in `after_value_set` when the controlling parameter changes.

```python
def __init__(self, **kwargs) -> None:
    super().__init__(**kwargs)

    self.enable_param = Parameter(
        name="enable_thing",
        type="bool",
        default_value=False,
        allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
        tooltip="Enable the thing",
    )
    self.add_parameter(self.enable_param)

    self.thing_options_param = Parameter(
        name="thing_options",
        type="str",
        default_value="default",
        allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
        tooltip="Options for the thing (only relevant when enable_thing is on)",
    )
    self.add_parameter(self.thing_options_param)
    self.hide_parameter_by_name(self.thing_options_param.name)   # hidden until needed

def after_value_set(self, parameter: Parameter, value: Any) -> None:
    if parameter is self.enable_param:
        if value:
            self.show_parameter_by_name(self.thing_options_param.name)
        else:
            self.hide_parameter_by_name(self.thing_options_param.name)
    super().after_value_set(parameter, value)
```

---

## Pattern 2 — Shown by default, hidden conditionally

Start visible, hide when the controlling value makes it irrelevant.

```python
self.add_parameter(self.detail_param)
# No hide call — visible by default

def after_value_set(self, parameter: Parameter, value: Any) -> None:
    if parameter is self.mode_param:
        if value == "simple":
            self.hide_parameter_by_name(self.detail_param.name)
        else:
            self.show_parameter_by_name(self.detail_param.name)
    super().after_value_set(parameter, value)
```

---

## Pattern 3 — Hide a group of related parameters together

```python
ADVANCED_PARAMS = ["threshold", "iterations", "seed"]

# In __init__, after adding all of them:
self.hide_parameter_by_name(ADVANCED_PARAMS)

# In after_value_set:
if parameter is self.show_advanced_param:
    if value:
        self.show_parameter_by_name(ADVANCED_PARAMS)
    else:
        self.hide_parameter_by_name(ADVANCED_PARAMS)
```

---

## Notes

- Always call `add_parameter` **before** `hide_parameter_by_name` — you can't hide a parameter that hasn't been added yet.
- Use `parameter is self.foo_param` (identity) in `after_value_set`, not `parameter.name == "foo"` — it's faster and avoids string bugs.
- Always call `super().after_value_set(parameter, value)` at the end of the method.
- Hidden parameters still exist and hold their value — hiding is purely visual. The value is preserved and can still be read/written programmatically.
- If you need to hide a `ParameterMessage` (not a regular parameter), use `hide_message_by_name` / `show_message_by_name` instead.

---

## Real example in this repo

`griptape_nodes_library/variables/create_variable.py` — `auto_name_case` is hidden on init and shown/hidden in `after_value_set` when `auto_name` toggles.
