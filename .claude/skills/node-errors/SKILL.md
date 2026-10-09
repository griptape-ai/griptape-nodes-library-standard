---
name: node-errors
description: Write a node's error handling in the Griptape Nodes Library - error messages, NodeError with fields/response/links, missing API keys, provider and HTTP failures, validation errors, and routing failures through the Failed output. Use whenever you add or change a raise, a validate_before_node_run exception, a _set_status_results(was_successful=False) call, or a proxy node's _extract_error_message.
---

# Node Error Handling

The editor shows a failed node's name and exception type next to its message. A `NodeError` can also
carry `fields`, `response`, and `links`, which the editor shows under **More info**.

Reference: the engine's `docs/development/custom_nodes/example_node_error_node.py`.

## Writing the message

A good message lets an artist fix the problem without asking anyone. Include:

- **Which parameter**, using the label the UI shows.
- **What's wrong with it**, including the value when it helps.
- **What's allowed**, such as a range or the supported formats.
- **What to do next.**

| Instead of | Write |
| --- | --- |
| "Invalid input." | "'Width' must be between 256 and 2048. It's set to 4000." |
| "Could not read the source image." | "Could not read the image connected to 'Input Image'. Connect a PNG or JPEG." |
| "No model ID provided" | "No model is selected. Choose one in 'Model' and run the node again." |
| "Request failed: 429" | "The provider is rate-limiting requests. Wait a minute and run the node again." |

Don't start with the node's name; the editor shows it. Don't paste a response or `repr()` into the
message; attach it instead.

**Link to the fix when a page explains it**: the provider's docs for a limit or content policy, a
supported-formats page, or a place in the editor such as
`#settings-secrets?filter=KEY`. Only link pages that exist. A link to a generic home page doesn't help.

## Pick the exception

**Bad input:** a plain exception. It keeps its type, which the editor shows.

```python
raise ValueError("Connect an image to 'Input Image'.")
```

**Something to attach** (a provider response, status code, ID, or link): `NodeError`.

```python
from griptape_nodes.exe_types.core_types import NodeError, NodeErrorLink

raise NodeError(
    "The prompt contains blocked words. Rephrase it and try again.",
    links=[NodeErrorLink(label="Content policy", url="https://example.com/content-policy")],
)
```

You can attach any of three optional parts:

- `fields`: labelled values a user may quote to support, such as a status code or request ID.
- `links`: up to three pages that explain the fix.
- `response`: the provider's response body.

```python
raise NodeError(
    "The prompt contains blocked words. Rephrase it and try again.",
    fields={"status_code": 400, "request_id": "req_8f2c41d0"},
    links=[NodeErrorLink(label="Content policy", url="https://example.com/content-policy")],
    response={"error": {"code": "content_policy", "message": "The prompt contains blocked words."}},
)
```

When re-raising a caught exception, add `from e`.

Use these field names where they apply: `status_code`, `request_id`, `error_code`, `parameter`,
`generation_id`, `status`, `job_id`, `task_id`.

**Missing API key:** return it from validation, so the node fails before it runs.

```python
def validate_before_node_run(self) -> list[Exception] | None:
    if not GriptapeNodes.SecretsManager().get_secret(API_KEY_NAME):
        return [missing_secret_error(API_KEY_NAME, key_url="https://example.com/api-keys")]
    return None
```

It says "`KEY` is not set. Add it in Settings → API Keys & Secrets, then run the node again." and
links to the secret. For a missing Griptape Cloud credential, use `missing_credential_error(attempted)`.

The helpers `error_response`, `missing_secret_error`, and others live in
`griptape_nodes_library/utils/node_error_utils.py`.

**Don't re-wrap** an exception just to add a prefix (`raise RuntimeError(f"Failed: {e}") from e`).
It hides the real type. Let it propagate.

## Every failure must reach the editor

In a `SuccessFailureNode`, route every failure through `_handle_failure_exception`. It follows
**Failed** if that's wired, and raises otherwise. Setting the status and returning is a silent
failure: with **Failed** unwired, nothing shows up.

```python
error = ValueError("Connect a file to 'Source'.")
self._set_status_results(was_successful=False, result_details=str(error))
self._handle_failure_exception(error)
return
```

- Don't `raise` again after it, or a wired **Failed** output can't catch the failure.
- Nodes without a **Failed** output just raise.
- Never raise from `after_value_set`, button callbacks, or previews. Only from `process`.
- **Exception: the Engine Node** never raises. A failed request is often the answer a workflow asked
  for, read from `was_successful`.
- **Exception: Math Expression** returns 0.0 for an expression it can't evaluate, so a typo doesn't
  stop the run.

## Proxy nodes

The base class `GriptapeProxyNode` attaches the response, IDs, and status for you.

- `_extract_error_message` returns only the provider's reason, or `""`. No node name, no prefix.
- In `_parse_result`, a result you can't use: set the status to failed and return. The base fails the node.

## Testing

```python
def test_raises_when_failed_is_unwired(node):
    node._has_outgoing_connections = lambda _param: False
    with pytest.raises(ValueError, match="Connect an image"):
        node.process()

def test_follows_failed_when_wired(node):
    node._has_outgoing_connections = lambda _param: True
    node.process()
    assert node.parameter_output_values["was_successful"] is False
```
