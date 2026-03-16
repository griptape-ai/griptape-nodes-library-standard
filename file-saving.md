# Saving Files from Nodes

This guide explains how to save files from nodes using the project-aware file saving system. It covers the `ProjectFileParameter` component and the `FileDestination`/`File` classes, and shows how to migrate from the legacy `StaticFilesManager.save_static_file()` pattern.

## Why migrate from `StaticFilesManager`

The legacy pattern saves files to a flat static files folder and returns a localhost URL. It has several limitations:

- Files go into a flat directory with no project-relative structure.
- No collision handling — callers must manage naming conflicts manually.
- Absolute/relative path detection is manual and error-prone.
- No user-configurable output locations or naming templates.

The new system solves these problems:

- **Project-aware paths** — files are saved relative to the project's configured directories using macro templates like `{outputs}/image.png`.
- **Configurable collision policies** — overwrite, create-new (auto-increment), or fail.
- **User-configurable situations** — project templates define named "situations" (e.g. `save_node_output`) that control where and how files are saved. Users can change the policy without touching node code.
- **`FileOutputSettings` integration** — a cog button on the parameter lets users wire in a `FileOutputSettings` node for per-connection control.

## Quick start

Three steps to add a file-saving parameter to a node.

### Step 1: Create the component in `__init__`

```python
from griptape_nodes.exe_types.param_components.project_file_parameter import ProjectFileParameter
from griptape_nodes.exe_types.node_types import DataNode


class MyImageNode(DataNode):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        # ... add other parameters ...

        self._output_file = ProjectFileParameter(
            node=self,
            name="output_file",
            default_filename="output.png",
        )
```

### Step 2: Register the parameter

Call `add_parameter()` after constructing the component. This adds the file path
parameter to the node and attaches the file picker and cog-button traits.

```python
        self._output_file.add_parameter()
```

### Step 3: Call `build_file()` in `process()` and write content

```python
    def process(self) -> None:
        image_bytes: bytes = self.get_parameter_value("image_bytes")

        dest = self._output_file.build_file()
        saved = dest.write_bytes(image_bytes)

        # saved.location is the portable macro path, e.g. "{outputs}/output.png"
        self.parameter_output_values["saved_file"] = saved.location
```

That's it. Path resolution, directory creation, and collision handling are all automatic.

## API reference

### `ProjectFileParameter`

**Location:** `griptape_nodes.exe_types.param_components.project_file_parameter`

```python
ProjectFileParameter(
    node: BaseNode,
    name: str,
    *,
    default_filename: str,
    situation: str = "save_node_output",
    allowed_modes: set[ParameterMode] | None = None,
)
```

| Argument           | Description                                                                                                                                    |
| ------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| `node`             | The parent node instance (`self`).                                                                                                             |
| `name`             | Parameter name as it appears in the UI and in `get_parameter_value()`.                                                                         |
| `default_filename` | Fallback filename when the parameter is empty (e.g. `"image.png"`).                                                                            |
| `situation`        | Situation name to look up in the current project template. Controls the macro template and collision policy. Defaults to `"save_node_output"`. |
| `allowed_modes`    | Which `ParameterMode` values are allowed. Defaults to `{INPUT, PROPERTY}`.                                                                     |

**Methods:**

- `add_parameter()` — Creates the `Parameter` and adds it to the node. Call once in `__init__` after constructing the component.
- `build_file(**extra_vars) -> FileDestination` — Resolves the current parameter value (or any connected `FileOutputSettings` node) into a `FileDestination`. Pass extra macro variables as keyword arguments (e.g. `build_file(sub_dirs="renders")`).

### `FileDestination`

**Location:** `griptape_nodes.files.file`

A pre-configured write handle. It bundles the target path with the write policy so callers don't need to know the policy details.

```python
dest.write_bytes(content: bytes) -> File
dest.write_text(content: str, encoding: str = "utf-8") -> File

await dest.awrite_bytes(content: bytes) -> File
await dest.awrite_text(content: str, encoding: str = "utf-8") -> File

dest.resolve() -> str  # absolute path string (resolves macros)
```

`write_bytes` and `write_text` both return a `File` pointing to the path where the content was actually written. Under `CREATE_NEW` policy, the written path may differ from the requested path (e.g. `output_1.png` instead of `output.png`).

### `File`

**Location:** `griptape_nodes.files.file`

A path-like object for reading and writing files. Returned by `FileDestination.write_bytes()` and `FileDestination.write_text()`.

```python
file.location -> str      # portable macro path, e.g. "{outputs}/output.png"
file.name     -> str      # filename only, e.g. "output.png"
file.resolve() -> str     # absolute path string

file.read_bytes() -> bytes
file.read_text(encoding: str = "utf-8") -> str
file.read_data_uri(fallback_mime: str = "application/octet-stream") -> str

await file.aread_bytes() -> bytes
await file.aread_text(encoding: str = "utf-8") -> str
await file.aread_data_uri(...) -> str
```

`file.location` returns the macro template string when the file is inside a project directory (e.g. `"{outputs}/output.png"`), or the plain absolute path string otherwise. Store `location` when you want a portable reference that survives project moves.

### `ProjectFileDestination`

**Location:** `griptape_nodes.files.project_file`

This is what `build_file()` creates under the hood when the parameter holds a plain string filename. It:

1. Looks up the named situation in the current project to get the macro template and collision policy.
1. Parses the filename into `file_name_base` and `file_extension` components.
1. Builds a `MacroPath` with variables `file_name_base`, `file_extension`, `node_name`, plus any `**extra_vars`.
1. After writing, maps the absolute written path back to its portable macro form.

You rarely construct `ProjectFileDestination` directly. Use `build_file()` instead.

### `ExistingFilePolicy`

**Location:** `griptape_nodes.retained_mode.events.os_events`

```python
ExistingFilePolicy.OVERWRITE    # replace existing file
ExistingFilePolicy.CREATE_NEW   # auto-increment (output_1.png, output_2.png, ...)
ExistingFilePolicy.FAIL         # raise FileWriteError if file exists
```

The policy is determined by the situation's `on_collision` setting. Node authors don't pass it directly when using `ProjectFileParameter`.

## Before and after

### Before (legacy pattern)

```python
from pathlib import Path

from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.exe_types.node_types import DataNode
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes


class SaveImageNode(DataNode):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self.add_parameter(
            Parameter(
                name="image_bytes",
                type="bytes",
                input_types=["bytes"],
                allowed_modes={ParameterMode.INPUT},
                tooltip="Image bytes to save",
            )
        )
        self.add_parameter(
            Parameter(
                name="filename",
                type="str",
                default_value="output.png",
                input_types=["str"],
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                tooltip="Output filename",
            )
        )
        self.add_parameter(
            Parameter(
                name="output_url",
                type="str",
                output_type="str",
                allowed_modes={ParameterMode.OUTPUT},
                tooltip="URL of saved file",
            )
        )

    def process(self) -> None:
        image_bytes = self.get_parameter_value("image_bytes")
        filename = self.get_parameter_value("filename") or "output.png"

        if Path(filename).is_absolute():
            Path(filename).write_bytes(image_bytes)
            url = filename
        else:
            url = GriptapeNodes.StaticFilesManager().save_static_file(
                image_bytes, filename
            )

        self.parameter_output_values["output_url"] = url
```

### After (new pattern)

```python
from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.exe_types.node_types import DataNode
from griptape_nodes.exe_types.param_components.project_file_parameter import ProjectFileParameter


class SaveImageNode(DataNode):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self.add_parameter(
            Parameter(
                name="image_bytes",
                type="bytes",
                input_types=["bytes"],
                allowed_modes={ParameterMode.INPUT},
                tooltip="Image bytes to save",
            )
        )

        self._output_file = ProjectFileParameter(
            node=self,
            name="output_file",
            default_filename="output.png",
        )
        self._output_file.add_parameter()

        self.add_parameter(
            Parameter(
                name="saved_file",
                type="str",
                output_type="str",
                allowed_modes={ParameterMode.OUTPUT},
                tooltip="Portable path of saved file",
            )
        )

    def process(self) -> None:
        image_bytes = self.get_parameter_value("image_bytes")

        dest = self._output_file.build_file()
        saved = dest.write_bytes(image_bytes)

        self.parameter_output_values["saved_file"] = saved.location
```

Key differences:

- The manual `Path.is_absolute()` check and `StaticFilesManager` call are replaced by a single `build_file()` + `write_bytes()`.
- The output value is `saved.location` (a portable macro path) instead of a localhost URL.
- Path resolution, directory creation, and collision handling are automatic.

## Advanced usage

### Sub-directories via `build_file()`

Pass extra macro variables to organize output files into sub-directories. The variable names must match placeholders in the situation's macro template.

```python
dest = self._output_file.build_file(sub_dirs="renders/pass_1")
saved = dest.write_bytes(image_bytes)
```

### Custom situations

Use a non-default `situation` when you want different output locations or policies for different node types.

```python
self._output_file = ProjectFileParameter(
    node=self,
    name="output_file",
    default_filename="report.txt",
    situation="save_report_output",
)
```

The situation must be defined in the project template. If it isn't found, `build_file()` logs an error and falls back to a default macro template with `CREATE_NEW` policy.

### Async nodes

Use `awrite_bytes` and `awrite_text` in async `process()` implementations.

```python
async def process(self) -> None:
    image_bytes = self.get_parameter_value("image_bytes")

    dest = self._output_file.build_file()
    saved = await dest.awrite_bytes(image_bytes)

    self.parameter_output_values["saved_file"] = saved.location
```

### `FileOutputSettings` node

When the parameter is in `INPUT` mode, a cog button appears in the UI. Clicking it creates a `FileOutputSettings` node and connects it to the parameter automatically. The `FileOutputSettings` node lets users configure the situation, filename template, and collision policy without editing node code.

When `build_file()` detects an incoming connection from a `FileDestinationProvider` (such as `FileOutputSettings`), it retrieves the `FileDestination` from that node directly instead of constructing a new one from the parameter string.

## Migration checklist

Use this checklist when converting an existing node:

1. **Remove the `StaticFilesManager` import** — delete `from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes` if it is only used for `StaticFilesManager`.

1. **Add the `ProjectFileParameter` import:**

    ```python
    from griptape_nodes.exe_types.param_components.project_file_parameter import ProjectFileParameter
    ```

1. **Replace the manual filename `Parameter`** — remove the plain `str` parameter that held the filename and replace it with a `ProjectFileParameter`:

    ```python
    self._output_file = ProjectFileParameter(
        node=self,
        name="output_file",
        default_filename="output.png",
    )
    self._output_file.add_parameter()
    ```

1. **Replace the `process()` file-saving logic** — remove the `is_absolute()` check and `save_static_file()` call:

    ```python
    # Remove:
    if Path(filename).is_absolute():
        Path(filename).write_bytes(content)
        url = filename
    else:
        url = GriptapeNodes.StaticFilesManager().save_static_file(content, filename)

    # Replace with:
    dest = self._output_file.build_file()
    saved = dest.write_bytes(content)
    ```

1. **Update the output parameter value** — store `saved.location` instead of a URL:

    ```python
    self.parameter_output_values["output_file"] = saved.location
    ```

1. **Remove unused imports** — `from pathlib import Path` may no longer be needed.

1. **Run `make check`** — verify no linting or type errors were introduced.
