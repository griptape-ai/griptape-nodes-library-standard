---
description: Runs another registered workflow as a single node, using that workflow's inputs and outputs as parameters.
when_to_use: Use it when you want to reuse a finished workflow inside a bigger one without copying its nodes.
---

## When to use it

Pick a saved workflow and this node takes on its inputs and outputs. Set the inputs, run the node, and read the results from the outputs. The inner nodes stay out of your canvas.

## Parameters

### workflow_file

The workflow to run. Only workflows registered with the engine are listed. Changing the choice rebuilds the node's parameters from the new workflow, so connections to old parameters are removed. Parameters with the same name as a built-in one, such as `was_successful`, are not added.

The workflow's own control parameters are skipped. Inputs and outputs are matched by parameter name.

### Refresh Workflow

Use this after you edit the chosen workflow. It reloads the workflow's inputs and outputs and keeps values and connections for parameters whose names still exist. The next run imports the latest version.

### Failure handling

The node fails if no workflow is chosen, if the workflow is not registered, if it can't be loaded, or if the run fails. The details appear in the status output. When Failed is wired, the flow continues down it. Otherwise the error is raised.

### Outputs

Only outputs that have a value are set. An output the inner workflow left empty is not updated.

> [!NOTE]
> The inner workflow is imported when the node runs and is not saved with your workflow. It must be registered when you open and run the parent workflow.
