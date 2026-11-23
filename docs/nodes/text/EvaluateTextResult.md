---
description: Scores a piece of text against criteria you write, using a model, and explains the score.
when_to_use: Use it when you want to check generated text, such as a shot description from an Agent, against a standard before moving on.
---

## When to use it

You have an actual output and want to know how well it meets criteria you define, optionally compared with an expected output. For example, check that an Agent's summary of a review session keeps every shot code from the notes.

The evaluation runs on a model through Griptape Cloud and needs a Griptape key. The model first writes three or four evaluation steps from your criteria. It then scores the input, expected output and actual output against those steps. The score is the model's judgment, so it can change between runs.

## Example

1. Add **Evaluate Text Result**.
1. Set `criteria` to "Does the output list every shot code that appears in the input?"
1. Put the original review notes in `input` and the Agent's summary in `actual_output`. `expected_output` can stay empty.
1. Run. `score` is a number from 0 to 1 and `reason` explains it.

## Parameters

### Examples

Fills `criteria`, `input`, `expected_output` and `actual_output` from a preset: Paraphrase, Factual or Analogy. Editing any of those fields resets this back to "Choose a preset..". Use it to see how the node works.

### input

The original text that the actual output is based on.

### expected_output

The reference output to compare against.

### actual_output

The text being evaluated.

### criteria

How to judge the output. Required. The node fails if it is empty.

### model

Offers `gpt-4.1` (default), `gpt-4.1-mini`, `gpt-4.1-nano` and `gpt-5`. The parameter is hidden by default in the editor.

### score

A number from 0 to 1, where 1 means the criteria are fully met. The model scores from 0 to 10 and the node divides by 10.

### reason

The model's short explanation for the score.
