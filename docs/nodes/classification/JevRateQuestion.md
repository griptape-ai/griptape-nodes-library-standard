---
description: Describes a rating question for a Questions (JEV) node to ask along with other questions.
when_to_use: Use it to add a graded question to a Questions (JEV) node, such as how big a change is.
---

## When to use it

This node builds one rating question. It does not call JEV. Connect its `question` output to a row of `questions` on a [Questions (JEV)](JevAsk.md) node, or click **+ Rate** on that node to add this node already connected.

To rate a single thing and send the flow down the matching level's path, use [Rate (JEV)](JevRate.md).

## How the answer comes back

The answer appears in the Questions (JEV) node's `answers` under this node's **Name**, as the label of the level JEV scored. In `details` it appears as `level`, `score`, `label`, `description`, `confidence` and `probabilities`.

## Parameters

### Name

What this question's answer is called in `answers` and `details`, for example `size`. Each question needs its own name. A node added by the **+ Rate** button is named `rate_1`, `rate_2` and so on. If you leave it empty, the node's own name is used, and renaming the node changes the answer's name.

### Question

Optional. Tells JEV what to rate, for example `How big is this note?`.

### Levels

2 to 10 levels, lowest first. Describe the situation at each level. Write a label before a colon to name it, such as `small: tiny note`. A level with no label is called `Level 1`, `Level 2` and so on, counting from the lowest.

Empty rows are ignored.

### Question (output)

The question, for the `questions` list of a Questions (JEV) node.
