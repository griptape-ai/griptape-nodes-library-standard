---
name: success-failure-node
description: Implement a Griptape Nodes Library node using SuccessFailureNode instead of ControlNode. Use whenever a node does something that can fail in a way a user might want to handle — file I/O, API calls, variable creation, JSON parsing, external services, or any operation where a partial failure is meaningful. SuccessFailureNode adds a failure output edge, a Status group with was_successful and result_details outputs, and re-raise behavior when no failure edge is wired.
---

# SuccessFailureNode Implementation Guide

Use `SuccessFailureNode` instead of `ControlNode` whenever a node does work that can legitimately fail at runtime and the user should be able to route around it — file operations, API calls, variable creation, data parsing, external services.

## When to use SuccessFailureNode vs ControlNode

| Situation | Use |
|---|---|
| Node reads/writes files or external state | `SuccessFailureNode` |
| Node makes API or engine requests that can be rejected | `SuccessFailureNode` |
| Node parses user-supplied data (JSON, CSV, etc.) | `SuccessFailureNode` |
| Node creates or mutates workflow variables | `SuccessFailureNode` |
| Node does pure in-memory computation that cannot fail | `ControlNode` |
| Node is a control-flow helper (merge, branch, loop counter) | `ControlNode` |

## Minimal implementation

```python
from griptape_nodes.exe_types.node_types import (
    NodeDependencies,
    NodeResolutionState,
    SuccessFailureNode,
)

class MyNode(SuccessFailureNode):
    def __init__(self, name: str, metadata: dict | None = None) -> None:
        super().__init__(name, metadata)

        # ... add parameters ...

        self._create_status_parameters(
            result_details_tooltip="Details about what happened.",
            result_details_placeholder="Results will appear here after the node runs.",
        )

    async def aprocess(self) -> None:
        self._clear_execution_status()
        try:
            # ... do work ...
            self._set_status_results(was_successful=True, result_details="Done: ...")
        except Exception as exc:
            self._set_status_results(was_successful=False, result_details=str(exc))
            self._handle_failure_exception(exc)
```

## The three required calls in `aprocess`

### `self._clear_execution_status()`
Call first, before any work. Resets `was_successful` and `result_details` from the previous run. Without this, stale results from the last run are visible until the node finishes.

### `self._set_status_results(was_successful, result_details)`
Call on every exit path — both success and failure. Sets the `was_successful` (bool) and `result_details` (str) output parameters that appear in the collapsed Status group.

```python
# Success — include useful summary
self._set_status_results(
    was_successful=True,
    result_details=f"Processed {len(items)} item(s).",
)

# Failure — include the exception message
self._set_status_results(
    was_successful=False,
    result_details=str(exc),
)
```

### `self._handle_failure_exception(exc)`
Call after `_set_status_results` on the failure path. Behavior depends on wiring:
- **No failure edge connected** → re-raises `exc`, halting the workflow (same as an unhandled exception in `ControlNode`).
- **Failure edge connected** → swallows the exception and routes flow through the failure output. Downstream nodes on that edge run; the workflow continues.

Always call `_set_status_results` *before* `_handle_failure_exception` so the Status group is populated even when the exception is swallowed.

## Emitting empty outputs on failure

When a failure edge is wired, downstream nodes can still read this node's output parameters. If those outputs hold values from a *previous* successful run, stale data flows downstream — usually not what you want.

Emit safe empty values for each output parameter in the `except` block, before `_handle_failure_exception`:

```python
async def aprocess(self) -> None:
    self._clear_execution_status()
    try:
        result = do_work()
        self.parameter_output_values["items"] = result
        self.parameter_output_values["count"] = len(result)
        self._set_status_results(was_successful=True, result_details=f"Done: {len(result)} item(s).")
    except Exception as exc:
        self._set_status_results(was_successful=False, result_details=str(exc))
        self.parameter_output_values["items"] = []   # don't let stale values flow downstream
        self.parameter_output_values["count"] = 0
        self._handle_failure_exception(exc)
```

Use the natural empty value for the type: `[]` for lists, `{}` for dicts, `""` for strings, `0` for numbers, `None` for any/unknown. You only need to clear outputs that a failure-path downstream node might actually use.

## `_create_status_parameters` options

```python
self._create_status_parameters(
    result_details_tooltip="...",          # tooltip on the result_details output
    result_details_placeholder="...",      # placeholder text before the node runs
    parameter_group_initially_collapsed=True,   # default True — keep Status collapsed
)
```

The method adds a "Status" `ParameterGroup` containing:
- `was_successful` — `bool` OUTPUT
- `result_details` — `str` OUTPUT

Call this **after** all other `add_parameter` calls so the Status group appears at the bottom.

## Using `process()` instead of `aprocess()`

Both work. Use `async def aprocess()` when the node makes `await GriptapeNodes.ahandle_request(...)` calls. Use `def process()` for synchronous work.

```python
def process(self) -> None:
    self._clear_execution_status()
    try:
        result = do_sync_work()
        self._set_status_results(was_successful=True, result_details=f"Done: {result}")
    except Exception as exc:
        self._set_status_results(was_successful=False, result_details=str(exc))
        self._handle_failure_exception(exc)
```

## Variable nodes: also reset resolution state

Variable nodes have side effects on every run — add these two methods so the engine doesn't skip them:

```python
def _reset_resolution_state(self) -> None:
    self.make_node_unresolved(
        current_states_to_trigger_change_event={NodeResolutionState.RESOLVED, NodeResolutionState.RESOLVING}
    )

def validate_before_workflow_run(self) -> list[Exception] | None:
    self._reset_resolution_state()
    return None

def validate_before_node_run(self) -> list[Exception] | None:
    self._reset_resolution_state()
    return None
```

## Import checklist

```python
from griptape_nodes.exe_types.node_types import (
    NodeDependencies,        # if you override get_node_dependencies
    NodeResolutionState,     # if you add validate_before_*_run
    SuccessFailureNode,      # the base class
    VariableAccess,          # if you declare variable refs
    VariableReference,       # if you declare variable refs
)
```

## Testing

```python
@pytest.mark.asyncio
async def test_success_sets_was_successful(node):
    node.set_parameter_value("my_input", "some value")
    await node.aprocess()
    assert node.parameter_output_values["was_successful"] is True

@pytest.mark.asyncio
async def test_failure_sets_was_successful(node):
    # No failure edge wired → _handle_failure_exception re-raises
    with pytest.raises(ValueError, match="..."):
        await node.aprocess()
    assert node.parameter_output_values["was_successful"] is False
```

When no failure edge is connected (the normal test setup), `_handle_failure_exception` re-raises the original exception, so `pytest.raises` works unchanged. The `was_successful` assertion verifies the Status group was populated before the exception propagated.
