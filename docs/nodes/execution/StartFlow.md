---
description: Marks where a workflow starts and defines the values passed into it.
when_to_use: Use it when the workflow will run as a unit, for example when called from another workflow, and needs named inputs.
---

## When to use it

You want an explicit entry point. Connect `Flow Out` to the first node of the control chain. Add parameters to the node (the editor shows an add-parameter control on it) to define inputs that the workflow receives when it is called as a subflow or run as a published workflow.

## Example

A shot-processing workflow takes a `shot_code` input.

1. Add **Start Flow** and add a text parameter named `shot_code`.
1. Connect `Flow Out` to an **Agent** and connect `shot_code` into its prompt.
1. Connect the end of the chain to an [End Flow](EndFlow.md).

## Parameters

### exec_out

Displayed as `Flow Out`. The control output that starts the chain.

Any other parameters are ones you add to the node. The node has no inputs of its own and its `process` does nothing.
