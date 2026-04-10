---
name: impl-proxy-node
description: Implement a Griptape Nodes Library node for a proxied service from a spec file.
argument-hint: <spec-file-path>
allowed-tools: Bash Read Write Edit Grep Glob
disable-model-invocation: false
---

# Implement a Proxy Node

Implement a node in this repo (griptape-nodes-library-standard) based on the specification file produced by `/api-research` (in the griptape-cloud repo). The node extends `GriptapeProxyNode` and communicates with the upstream API through the Griptape Cloud proxy.

## 1. Read the Spec and API Key

Read the spec file from `$ARGUMENTS`. The path may be absolute or relative to this repo (e.g., `../../griptape-cloud/.scratch/proxy-spec-<name>/spec.md`). Also read the API key from `.api_key` in the same directory as the spec (needed for integration testing).

Extract from the spec:
- Service name, model IDs
- Classification (sync/async, media type, result format)
- Endpoint request/response schemas
- Suggested node parameters
- Quirks

## 2. Create a Feature Branch

```bash
git checkout main && git pull
git checkout -b feat/<service-name>-proxy-node
```

## 3. Read the Base Class

Read `griptape_nodes_library/griptape_proxy_node.py` to understand the interface. The three abstract methods you must implement:

- `async _build_payload(self) -> dict[str, Any]` - Build the request JSON
- `async _parse_result(self, result_json: dict[str, Any], generation_id: str) -> None` - Parse result and set outputs
- `_set_safe_defaults(self) -> None` - Clear outputs on error

Optional overrides:
- `_get_api_model_id(self) -> str` - Map friendly name to API model ID
- `_extract_error_message(self, response_json) -> str` - Custom error extraction

## 4. Read a Reference Node

Based on the spec's media type, read the closest existing node:

- **Audio**: `griptape_nodes_library/audio/eleven_labs_music_generation.py` (simple, raw bytes result)
- **Image**: `griptape_nodes_library/image/flux_2_image_generation.py` (model mapping, URL-based result, image download)
- **Video**: `griptape_nodes_library/video/kling_text_to_video_generation.py` (async, video result)

Study the patterns for:
- Parameter definition in `__init__`
- How `_build_payload()` gathers param values
- How `_parse_result()` handles different result formats (URLs, base64, raw bytes)
- How `ProjectFileParameter` is used for output files
- How `_create_status_parameters()` is called at the end of `__init__`

## 5. Create the Node File

Create `griptape_nodes_library/<category>/<provider>_<modality>.py` where:
- `<category>` matches the media type (audio, image, video, etc.)
- `<provider>` is the service name (lowercase, underscores)
- `<modality>` describes what it does (e.g. `image_generation`, `text_to_speech`)

### Imports

```python
from __future__ import annotations

import asyncio
import base64
import json as _json
import logging
from contextlib import suppress
from typing import Any

from griptape.artifacts import ImageUrlArtifact  # or AudioUrlArtifact, VideoUrlArtifact
from griptape_nodes.exe_types.core_types import Parameter, ParameterGroup, ParameterMode
from griptape_nodes.exe_types.param_components.project_file_parameter import ProjectFileParameter
from griptape_nodes.exe_types.param_types.parameter_bool import ParameterBool
from griptape_nodes.exe_types.param_types.parameter_dict import ParameterDict
from griptape_nodes.exe_types.param_types.parameter_float import ParameterFloat
from griptape_nodes.exe_types.param_types.parameter_int import ParameterInt
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString
from griptape_nodes.exe_types.param_types.parameter_image import ParameterImage  # if needed
from griptape_nodes.traits.options import Options
from griptape_nodes.traits.slider import Slider

from griptape_nodes_library.griptape_proxy_node import GriptapeProxyNode
```

Only import what you actually need.

### Class Structure

```python
class <ClassName>(GriptapeProxyNode):
    """<Description from spec>

    Inputs:
        - <param>: <description>

    Outputs:
        - generation_id (str): Generation ID from the API
        - <media_output>: Generated media artifact
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.category = "API Nodes"
        self.description = "<description>"

        # --- INPUT PARAMETERS ---
        # Add parameters based on the spec's "Suggested Node Parameters" section.
        # Use ParameterString, ParameterInt, ParameterFloat, ParameterBool as appropriate.
        # Use Options(choices=[...]) for enum fields.
        # Use Slider(min_val=..., max_val=...) for bounded numeric fields.
        # Group advanced/optional params in ParameterGroup(name="...", ui_options={"collapsed": True}).
        #
        # Parameter design guidelines:
        # - Prefer arbitrary inputs over hardcoded presets when the API supports a range.
        #   For example, use width/height ParameterInt fields instead of a size dropdown
        #   with a fixed list of resolutions. You can offer named presets as a separate
        #   dropdown that sets the other fields.
        # - Match parameter defaults to the API's documented defaults, not arbitrary values.
        # - Match parameter ranges (min/max) to what the API actually accepts.

        # --- OUTPUT PARAMETERS ---
        self.add_parameter(
            ParameterString(
                name="generation_id",
                tooltip="Generation ID from the API",
                allowed_modes={ParameterMode.OUTPUT},
                hide_property=True,
                hide=True,
            )
        )

        self.add_parameter(
            ParameterDict(
                name="provider_response",
                tooltip="Verbatim response from Griptape model proxy",
                allowed_modes={ParameterMode.OUTPUT},
                hide_property=True,
                hide=True,
            )
        )

        # Add the media output parameter (ParameterImage, ParameterAudio, or ParameterVideo)
        # with allowed_modes={ParameterMode.OUTPUT, ParameterMode.PROPERTY}, settable=False,
        # and pulse_on_run=True in ui_options.

        # Add ProjectFileParameter for file output
        self._output_file = ProjectFileParameter(
            node=self,
            name="output_file",
            default_filename="<sensible_default>.<ext>",
        )
        self._output_file.add_parameter()

        # Status parameters MUST be last
        self._create_status_parameters(
            result_details_tooltip="Details about the generation result or any errors",
            result_details_placeholder="Generation status will appear here...",
            parameter_group_initially_collapsed=True,
        )
```

### Model ID Mapping

If the node offers multiple models via a dropdown, create a mapping dict:

```python
MODEL_MAPPING = {
    "Friendly Name": "api-model-id",
}

def _get_api_model_id(self) -> str:
    model = self.get_parameter_value("model") or DEFAULT_MODEL
    return MODEL_MAPPING.get(str(model), str(model))
```

If there's only one model, override `_get_api_model_id()` to return the fixed ID:

```python
def _get_api_model_id(self) -> str:
    return "the-model-id"
```

The model ID MUST match what `get_model_ids()` returns in the proxy client.

### _build_payload()

```python
async def _build_payload(self) -> dict[str, Any]:
    # Gather parameter values
    prompt = self.get_parameter_value("prompt") or ""
    # ... other params ...

    payload = {
        "prompt": prompt,
        # ... map to the request schema from the spec ...
    }

    # For image inputs, convert to data URI:
    # data_uri = await File(image_url).aread_data_uri(fallback_mime="image/png")

    return payload
```

Do NOT include the `model` field in the payload - the base class handles routing via `_get_api_model_id()`.

### _parse_result()

Handle the result based on the spec's "Result Format":

**For JSON with URL** (e.g. `{"result": {"sample": "https://..."}}`):**
```python
async def _parse_result(self, result_json: dict[str, Any], generation_id: str) -> None:
    url = result_json.get("result", {}).get("sample")  # adjust path per spec
    if not url:
        self._set_safe_defaults()
        self._set_status_results(was_successful=False, result_details="No URL in response.")
        return

    from griptape_nodes.files.file import File
    image_bytes = await File(url).aread_bytes()
    if image_bytes:
        dest = self._output_file.build_file()
        saved = await dest.awrite_bytes(image_bytes)
        self.parameter_output_values["<media_output>"] = ImageUrlArtifact(saved.location)
        self._set_status_results(was_successful=True, result_details="Generated successfully.")
```

**For raw binary bytes** (e.g. audio/video returned directly):
```python
async def _parse_result(self, result_json: dict[str, Any], generation_id: str) -> None:
    audio_bytes = result_json.get("raw_bytes")
    if not audio_bytes:
        # Fall back to base64 if present
        b64 = result_json.get("audio_base64")
        if b64:
            audio_bytes = await asyncio.to_thread(base64.b64decode, b64)

    if not audio_bytes:
        self._set_safe_defaults()
        self._set_status_results(was_successful=False, result_details="No data in response.")
        return

    dest = self._output_file.build_file()
    saved = await dest.awrite_bytes(audio_bytes)
    self.parameter_output_values["<media_output>"] = AudioUrlArtifact(value=saved.location, name=saved.name)
    self._set_status_results(was_successful=True, result_details="Generated successfully.")
```

### _set_safe_defaults()

```python
def _set_safe_defaults(self) -> None:
    self.parameter_output_values["generation_id"] = ""
    self.parameter_output_values["provider_response"] = None
    self.parameter_output_values["<media_output>"] = None
```

### _extract_error_message() (optional)

Override only if the spec's "Error Format" or "Quirks" section indicates the provider uses a non-standard error structure:

```python
def _extract_error_message(self, response_json: dict[str, Any]) -> str:
    # Try provider-specific error path first
    # Fall back to super()._extract_error_message(response_json)
    return super()._extract_error_message(response_json)
```

## 6. Register in the Manifest

Edit `griptape_nodes_library.json`. Add a new entry to the `"nodes"` array:

```json
{
    "class_name": "<ClassName>",
    "file_path": "griptape_nodes_library/<category>/<filename>.py",
    "metadata": {
        "category": "<category>",
        "description": "<One-line description>",
        "display_name": "<Human-Friendly Name>",
        "icon": "<lucide-icon-name>",
        "group": "<optional group name>"
    }
}
```

Choose an appropriate [Lucide icon](https://lucide.dev/icons/) name. Common choices:
- Image: `"image"`
- Video: `"video"`
- Audio: `"volume-2"`
- Music: `"music"`
- 3D: `"box"`

## 7. Update __init__.py (if applicable)

Check if `griptape_nodes_library/<category>/__init__.py` exists and has explicit imports. If so, add the new class:

```python
from griptape_nodes_library.<category>.<filename> import <ClassName>
```

And add it to `__all__` if the file uses one.

Note: This repo uses manifest-based discovery via `griptape_nodes_library.json`, so the `__init__.py` update may not be strictly required. Check the existing `__init__.py` to see if other nodes in the same category are imported there. If they are, add yours too for consistency. If the file only has an empty `__all__` or doesn't exist, skip this step.

## 8. Write an Integration Test

**Prerequisites:** The integration test requires:
- "Griptape Nodes Testing Library" installed (provides `AssertFileExists` node). Verify it's available before writing the test.
- `GT_CLOUD_API_KEY` environment variable set. Even when running against localhost with auth disabled, the node's secret resolution still requires this to be set. Any non-empty value works for local testing.

Create `tests/integration/test_<provider>_<modality>.py` following the existing pattern from `tests/integration/test_elevenlabs_text_to_speech_generation.py`.

The test should:
1. Register the library
2. Create a workflow with `StartFlow` -> `<YourNode>` -> `AssertFileExists` -> `EndFlow`
3. Wire up connections between nodes
4. Provide minimal valid input via `StartFlow` parameters
5. Execute and verify a file was produced

Study the existing test carefully and replicate its structure. Key patterns:
- Script metadata header with `# /// script` block
- Library registration via `RegisterLibraryFromFileRequest`
- Node creation via `CreateNodeRequest` with `resolution="resolved"` and `initial_setup=True`
- Connection creation via `CreateConnectionRequest`
- Workflow execution via `LocalWorkflowExecutor`

## 9. Run the Integration Test

The local griptape-cloud infrastructure should still be running from the `/impl-proxy-client` phase.

**Important:** Use `uv run python` (not bare `python`) to ensure the correct virtual environment and dependencies are available:

```bash
GT_CLOUD_PROXY_BASE_URL=http://localhost:8000 GT_CLOUD_PROXY_API_KEY=local uv run python -m pytest tests/integration/test_<provider>_<modality>.py -v
```

These proxy-specific env vars override only the proxy endpoint and API key without affecting other engine systems (file storage, user auth, etc.) that use `GT_CLOUD_BASE_URL` / `GT_CLOUD_API_KEY`. The `GT_CLOUD_PROXY_API_KEY` must NOT start with `gt-` when targeting local infra, because the local server's `ApiKeyAuthenticator` would reject any `gt-` token not in its DB. A non-`gt-` value like `local` falls through to `LocalUserAuthenticator` which auto-authenticates.

**Test all model IDs:** If the node supports multiple models, run the test once for each model ID to verify they all work through the proxy. You can parameterize the test or run it multiple times with different `--json-input` values.

If the test fails:
- Check the node's `_build_payload()` output matches what the proxy client expects
- Check the proxy logs: `cd ../../griptape-cloud && docker compose logs web -f`
- Check Celery worker logs: `docker compose logs workers -f`
- Verify the model ID in `_get_api_model_id()` matches `get_model_ids()` in the proxy client

Fix issues and re-run until the test passes.

## 10. Run Linting

```bash
make check
```

Fix any linting or type errors.

## 11. Review: Verify Node Against Spec and Docs

Re-read the spec file and use `/agent-browser` to open the API documentation page. Compare both against the node implementation.

**Parameters:**
- [ ] Every node input parameter has a corresponding field in the API spec
- [ ] Parameter types match (int vs float vs string vs bool)
- [ ] Default values match the API's documented defaults
- [ ] Min/max ranges and valid choices match the API constraints
- [ ] No documented API parameters are missing that a user would reasonably want to configure
- [ ] Arbitrary inputs are preferred over hardcoded presets when the API supports a range (e.g. width/height instead of a fixed size dropdown)

**Payload building:**
- [ ] `_build_payload()` produces JSON that matches the spec's request schema exactly
- [ ] Field names and nesting match (e.g. `input.messages` vs `input.prompt`)
- [ ] Optional fields are only included when the user sets them

**Response parsing:**
- [ ] `_parse_result()` extracts data from the correct JSON path in the proxy response
- [ ] Account for the proxy's response wrapping: the proxy client's `fetch_completed_generation()` may extract a sub-object before returning, so the node receives a different structure than the raw API response

**Model mapping:**
- [ ] `_get_api_model_id()` returns IDs that match `get_model_ids()` in the proxy client
- [ ] The friendly names in the dropdown are accurate

Fix any discrepancies before proceeding.

## 12. Commit and Create a PR

```bash
git add -A
git commit -m "feat: add <service-name> <modality> node"
git push -u origin feat/<service-name>-proxy-node
gh pr create \
  --title "feat: add <service-name> <modality> node" \
  --body "$(cat <<'EOF'
Adds a `<ClassName>` node for <service name> <modality> generation via the Griptape Cloud proxy.

<Describe what the node does, what models it supports, and the key parameters it exposes.>

Depends on the proxy client PR: <link to griptape-cloud PR>

## Sources

- <link to the API documentation page>
- <any other references consulted (SDK repos, blog posts, etc.)>
- <note any nuances, quirks, or workarounds discovered during implementation>

## Testing with the engine

To test the node end-to-end in the Griptape Nodes UI:

1. Check out both branches:
   - This repo: `feat/<service-name>-proxy-node`
   - griptape-cloud: the proxy client branch (see linked PR above)
2. Start the local griptape-cloud: `cd griptape-cloud && make up/debug`
3. Create DB records for the model config (see proxy client PR for setup steps)
4. Start the engine with proxy overrides:
   ```bash
   GT_CLOUD_PROXY_BASE_URL=http://localhost:8000 GT_CLOUD_PROXY_API_KEY=local make run/watch
   ```
5. In the Nodes UI, add a `<Display Name>` node and connect it to a workflow
6. Set the prompt and any other parameters, then run the workflow
7. Verify the output image/audio/video appears in the node's output

## Integration test

To run the automated integration test:

```bash
GT_CLOUD_PROXY_BASE_URL=http://localhost:8000 GT_CLOUD_PROXY_API_KEY=local \
  python -m pytest tests/integration/test_<provider>_<modality>.py -v
```

Closes <issue-reference>
EOF
)"
```

Print the PR URL.
