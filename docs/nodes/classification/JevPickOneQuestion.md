---
description: Describes a pick-one question for a Questions (JEV) node to ask along with other questions.
when_to_use: Use it to add a question with several possible answers to a Questions (JEV) node, such as which department a note is for.
---

## When to use it

This node builds one pick-one question. It does not call JEV. Connect its `question` output to a row of `questions` on a [Questions (JEV)](JevAsk.md) node, or click **+ Pick One** on that node to add this node already connected.

To ask a single pick-one question and send the flow down the option's path, use [Pick One (JEV)](JevPickOne.md).

## How the answer comes back

The answer appears in the Questions (JEV) node's `answers` under this node's **Name**, as the label of the option JEV picked. In `details` it appears as `label`, `description`, `confidence` and `probabilities`.

## Parameters

### Name

What this question's answer is called in `answers` and `details`, for example `department`. Each question needs its own name. A node added by the **+ Pick One** button is named `pick_one_1`, `pick_one_2` and so on. If you leave it empty, the node's own name is used, and renaming the node changes the answer's name.

### Question

Optional. Tells JEV what to decide, for example `What department does this belong to?`.

### Options

One option per row, up to 255. Write a label, or `Label: description` to tell JEV what the option means. The text before the first colon is the label. Without a colon, the whole row is the label and JEV decides from the label alone.

- You need at least two options, and each label must be unique.
- Empty rows are ignored.

### Question (output)

The question, for the `questions` list of a Questions (JEV) node.
