---
description: Marks where a workflow finishes, records whether it succeeded, and returns values from the flow.
when_to_use: Use it at the end of a workflow that is called from another workflow or published, so the result and outputs are returned.
---

## When to use it

You want an explicit exit point. Connect the last node of the chain to the `Succeeded` input. Connect a failure path, such as an error branch, to the `Failed` input. Add parameters to the node (the editor shows an add-parameter control) to define the values the workflow returns.

Pair it with [Start Flow](StartFlow.md).

## Example

A render check workflow ends in two branches. The branch that confirms all frames exist connects to `Succeeded`. The branch that finds missing frames connects to `Failed` and puts the list of frames into `result_details`.

## Parameters

### Succeeded and failed

Two control inputs, `exec_in` (displayed as `Succeeded`) and `failed` (displayed as `Failed`). The one that is used to enter the node decides `was_successful`. If the node is entered some other way, it counts as a success.

### result_details

In the collapsed Status group. Connect or write text to describe the result. The node prefixes it with `[SUCCEEDED]` or `[FAILED]`, and writes `No details supplied by flow` if it is empty.

### was_successful

In the Status group. Set by the node when it runs.

Any parameters you add are copied to the node's outputs when it runs. You can have more than one End Flow in a workflow.
