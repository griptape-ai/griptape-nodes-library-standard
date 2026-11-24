---
description: Stops the workflow when it is reached, with a reason you choose.
when_to_use: Use it when a branch of your workflow should halt the whole run, for example when a check fails.
---

## When to use it

Connect it at the end of a branch that should not carry on. When it runs, it raises an error carrying your reason and the workflow stops.

## Parameters

### cancellation_reason

The message shown when the workflow stops. If it is empty, the default message "Cancelled running workflow via Cancel Workflow Node." is used.

> [!NOTE]
> The source stops the run by raising a RuntimeError. It is reported as a node error, not as a separate cancelled state.
