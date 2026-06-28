---
name: add-parameter-badge
description: Add an info/warning/error badge to a Griptape Nodes parameter to surface docs links, examples, or contextual warnings in the UI.
argument-hint: <parameter-name> <variant> <message>
allowed-tools: Read Edit Grep Glob
disable-model-invocation: false
---

# Add a Badge to a Parameter

Badges appear inline on a parameter in the node UI. Use them for:
- **Static info badges**: docs links, expression syntax examples (XPath, JMESPath, regex, etc.)
- **Conditional warning/error badges**: set/clear dynamically in `after_value_set` based on the parameter's current value

## Badge API

`BadgeData` lives in `griptape_nodes.exe_types.core_types`. Every `Parameter` (and `ParameterGroup`) inherits three methods:

```python
parameter.set_badge(variant, title, message, *, icon, color, hide, hide_clear_button)
parameter.clear_badge()
parameter.get_badge()
```

### `set_badge` arguments

| arg | type | notes |
|---|---|---|
| `variant` | `str` | See variant table below |
| `title` | `str \| None` | Short bold header shown above the message |
| `message` | `str \| None` | Body text — **renders as Markdown** |
| `icon` | `str \| None` | Lucide icon name, overrides the variant default |
| `color` | `str \| None` | Hex `"#3b82f6"` or `rgb(...)`, overrides variant default |
| `hide` | `bool \| None` | Hides the badge indicator without removing it |
| `hide_clear_button` | `bool \| None` | `True` (default) hides the dismiss button |

### Variants

| variant | when to use |
|---|---|
| `"info"` | General informational context that doesn't fit a more specific type |
| `"warning"` | Something may go wrong or produce unexpected results — user should take note |
| `"error"` | Something is wrong and will likely fail |
| `"success"` | Operation completed or value is valid |
| `"tip"` | A helpful shortcut, trick, or non-obvious workflow suggestion |
| `"link"` | Pointing to a related resource or external URL |
| `"docs"` | Reference to documentation (API docs, spec, etc.) |
| `"help"` | Syntax examples, expression formats, field reference — "how do I use this?" |
| `"note"` | A caveat, side-effect, or extra detail the user should be aware of |
| `"cloud-upload"` | Upload / cloud-transfer context (billing, size limits, remote storage) |

### Message markdown

The `message` field renders as **CommonMark markdown**:

```
`//h1` — all h1 elements
`//a/@href` — link hrefs

[XPath Docs](https://www.w3schools.com/xml/xpath_syntax.asp)
```

Use `\n` for line breaks in Python string literals, or triple-quoted strings.

## Pattern 1 — Static badge (set once in `__init__`)

Assign the parameter to a local variable, call `set_badge`, then `add_parameter`:

```python
path_param = ParameterString(
    name="path",
    allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
    default_value="",
    tooltip="JMESPath expression to extract data",
    placeholder_text="ex: user.name, items[0].title",
)
path_param.set_badge(
    variant="help",  # use "help" for syntax/docs badges; "info" for general info; "warning"/"error" for problems
    title="JMESPath Syntax",
    message="`user.name` — nested key\n`items[0].title` — array index\n`[*].name` — all names\n\n[JMESPath Docs](https://jmespath.org/)",
)
self.add_parameter(path_param)
```

You can also pass `badge=BadgeData(...)` directly to the `Parameter` constructor — the base `Parameter.__init__` accepts it.

## Pattern 2 — Conditional badge (set/clear in `after_value_set`)

```python
def _update_my_badge(self, value: Any) -> None:
    param = self.get_parameter_by_name("my_param")
    if param is None:
        return
    if <condition_is_bad>:
        param.set_badge(
            variant="warning",
            message="Explanation of the problem.",
        )
    else:
        param.clear_badge()

def after_value_set(self, parameter: Parameter, value: Any) -> None:
    super().after_value_set(parameter, value)
    if parameter.name == "my_param":
        self._update_my_badge(value)
```

Call `_update_my_badge(default_value)` at the end of `__init__` so the badge reflects the initial state.

## Real examples in this repo

| File | Badge type | What it does |
|---|---|---|
| `griptape_nodes_library/html/html_extract_value.py` | static info | XPath syntax + W3Schools link |
| `griptape_nodes_library/xml/xml_extract_value.py` | static info | XPath syntax + W3Schools link |
| `griptape_nodes_library/yaml/yaml_extract_value.py` | static info | JMESPath syntax + jmespath.org link |
| `griptape_nodes_library/json/json_extract_value.py` | static info | JMESPath syntax + jmespath.org link |
| `griptape_nodes_library/video/ltx_video_extend.py` | conditional warning | warns when `context=0` changes billing |
| `griptape_nodes_library/filesystem/file_output_settings.py` | conditional info | explains filename collision policy |
