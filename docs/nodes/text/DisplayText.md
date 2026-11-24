---
description: Shows text in the workflow and passes the same text on to other nodes.
when_to_use: Use it when you want to read a text result at a point in the workflow, such as an Agent's reply, or leave a visible note on the canvas.
---

## When to use it

DisplayText shows a piece of text on the canvas. Connect the output of an earlier step to see what it produced, or type text into it to use it as a label or description. It does not trim or reformat the text, and whitespace and newlines are kept. Text values pass through as received. A value that is not text is converted to text first (`5` becomes `5`, a list becomes a JSON string), and an empty or missing value becomes an empty string. You can connect `text` to any node that takes text.

## Example

To read an Agent's answer on the canvas:

1. Add a **DisplayText** node.
1. Connect the output of an [Agent](../agents/Agent.md) node to its `text` parameter.
1. When the workflow runs, the generated reply replaces the initial text.

## Parameters

### text

Accepts multiline text, defaulting to an empty string. It is the input, the editable property and the output, so the displayed text is the same text that downstream nodes receive. The output is updated as soon as the value is set, and again when the node runs.
