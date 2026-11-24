---
description: Merges several branches into one output by passing along whichever input received a value.
when_to_use: Use it when different branches of your workflow produce the same kind of result and you want one place to continue from.
---

## When to use it

After an [IfElse](IfElse.md) or other branch, only one path runs. Connect each path's result to a `select_inputs` slot. `output` gives you the value from the path that actually ran, so the rest of the workflow doesn't need to care which one it was.

## Parameters

### select_inputs

A list of inputs. Add one for each branch. The node does not wait for all of them, so inputs from branches that did not run are fine.

### output

The value from the input that received one. If more than one input has a value, the last one in the list wins. Every input that had a value is cleared afterward, so a later run starts fresh. Its type follows the first connection you make. If you remove that connection the type resets and the stored output is dropped.
