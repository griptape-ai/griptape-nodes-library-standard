---
description: Adds a text note to the canvas. It does not run or connect to anything.
when_to_use: Use it to document a section of a workflow, leave instructions for the next person, or record a decision.
---

## When to use it

You want to explain part of a workflow where people will see it. For example, label a group of nodes "Plate prep, do not change the colorspace" or leave a hint on how to use a node.

The note is stored in the workflow file and does nothing when the workflow runs.

## Example

Place a Note next to an Agent and enter:

```
To use the prompt_context, create a new KeyValuePair node.
Give it a key of "shot" and a value of "SH010".
Connect the dictionary output to the prompt_context on your Agent.
In the Agent prompt, use {{shot}} where you want the value to appear.
```

## Parameters

### note

The note text. The field renders Markdown, so headings, lists, bold text and links are formatted. The field is a property only, so it has no input or output connections.
