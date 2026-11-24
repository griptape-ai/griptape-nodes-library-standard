---
description: A calculator you talk to in plain language. A model works out the sum with a calculator tool and returns the answer and its reasoning.
when_to_use: Use it for quick arithmetic written as a sentence, such as frame counts, durations or render budgets, when you don't want to build the expression by hand.
---

## When to use it

You have a calculation described in words and want a number back, plus an explanation of how it was reached. For exact expressions with known inputs, use [Math](Math.md) or [Math Expression](MathExpression.md) instead.

## What it needs

- A Griptape Nodes license or a Griptape Cloud API key (`GT_CLOUD_API_KEY`).
- Model access: a model your license does not allow stops the run with an error.

## Example

1. Add an **Askulator** node.
1. Enter `A 12 second shot at 24 frames per second, plus 8 frames of handles on each end. How many frames?` in `instruction`.
1. Run the workflow. `result` is `304` and `output` explains the steps.

## Parameters

### instruction

The question, in plain language. If it is empty, the node does nothing.

### model

Hidden by default. One of `gpt-4.1`, `gpt-4.1-mini` (the default), `gpt-4.1-nano` or `gpt-5`. The list marks models your license does not allow. Older saved display names such as `GPT-4.1 mini` load as the matching id.

### result

The final answer, as short text. It is a number with a unit where one applies.

### output

The model's reasoning, with a line `Using a <tool name>` for each tool call. Both outputs fill in as the answer streams. The final validated answer replaces them when the run finishes.

## Important notes

- The model is told to use its Calculator tool for the arithmetic.
- If the question is missing a value, the model is instructed to assume a likely number and say so in its reasoning. Read `output` before you rely on `result`.
- `result` is text, not a number. Convert it with [To Float](../convert/ToFloat.md) or [To Integer](../convert/ToInteger.md) to use it in later math.
