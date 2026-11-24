---
description: Summarizes a block of text with a model on Griptape Cloud.
when_to_use: Use it when you have long text, such as a set of review notes or a vendor report, and want a short version.
---

## When to use it

You have text and want a shorter summary of it with no extra instructions. For a custom summary, such as one grouped by shot or limited to action items, use an [Agent](../agents/Agent.md) with your own prompt.

## What it needs

- A Griptape Nodes license or a Griptape Cloud API key (`GT_CLOUD_API_KEY`).
- The model call runs through Griptape Cloud.
- Non-empty text in `prompt`. An empty or whitespace-only `prompt` does nothing.
- Text longer than about 100,000 tokens is summarized in chunks, and the chunk summaries are then summarized together.

## Example

1. Add a **Load Text** node and pick a long notes file from a dailies review.
1. Add **Summarize Text** and connect the text to its `prompt` input.
1. Run. The summary is in `output`.

## Parameters

### prompt

The text to summarize. Despite the name, this is the content, not an instruction.

### model

Offers `gpt-4.1`, `gpt-4.1-mini`, `gpt-4.1-nano` (default) and `gpt-5`. The parameter is hidden by default in the editor.

### output

The summary.
