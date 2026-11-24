---
description: Describes a yes/no question for a Questions (JEV) node to ask along with other questions.
when_to_use: Use it to add a yes/no question to a Questions (JEV) node, such as whether a note is clear.
---

## When to use it

This node builds one yes/no question. It does not call JEV. Connect its `question` output to a row of `questions` on a [Questions (JEV)](JevAsk.md) node, or click **+ Yes/No** on that node to add this node already connected.

To ask a single yes/no question and branch on the answer, use [Ask Yes/No (JEV)](JevAskYesNo.md).

## How the answer comes back

The answer appears in the Questions (JEV) node's `answers` under this node's **Name**, as `true` or `false`. In `details` it appears as `answer` and `probability`.

## Parameters

### Name

What this question's answer is called in `answers` and `details`, for example `director_clarity`. Each question needs its own name. A node added by the **+ Yes/No** button is named `yes_no_1`, `yes_no_2` and so on. If you leave it empty, the node's own name is used, and renaming the node changes the answer's name.

### Question

A yes/no question about the text, for example `Is this a clear note from the director?`. It is required.

### Criteria

In a collapsed group.

- **Yes means:** optional. Describes what should count as Yes.
- **No means:** optional. Describes what should count as No.

### Say Yes at or above

JEV returns the probability that the answer is yes. The answer is `true` when the probability is at least this value, from 0 to 1. The default is 0.5. Raise it to require more certainty, and lower it to accept weaker signals.

### Question (output)

The question, for the `questions` list of a Questions (JEV) node.
