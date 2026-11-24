---
description: Calls any engine request, such as listing nodes or checking a workflow, and exposes its inputs and results as parameters.
when_to_use: Use it when you need to drive the engine itself from a workflow and no regular node does it.
---

## When to use it

Most work has a dedicated node. This is the escape hatch for engine operations that don't. Pick a request type and the node builds the matching input and output parameters for you. It is aimed at advanced workflows and tooling.

## Parameters

### request_type

The engine request to run. Choosing one replaces the input and output parameters with the ones that request uses. Connections to parameters that exist on both requests are kept.

Types marked with `*` have no matching success and failure result in the engine. They can be selected but the node refuses to run them.

### Input parameters

Inputs are named `input_` followed by the request field name. Empty strings on optional fields are treated as not set.

### Output parameters

Results are named `output_success_` or `output_failure_` followed by the result field name. Only the set that matches the outcome is filled in. The values from the previous request type are cleared when you switch.

### Failure handling

A failed request does not stop the workflow. It follows the Failed control output and sets `was_successful` to false, so you can branch on the answer. The result details show the engine's messages.

> [!NOTE]
> The node runs on every workflow run. It is never skipped as already resolved.
