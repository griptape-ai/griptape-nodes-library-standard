---
description: Turns a plain-language description of a date or time into text in a format you pick.
when_to_use: Use it when you want a date string for a file name, a report header or a prompt, such as "next Friday" or "two weeks from today".
---

## When to use it

You want a date or time as text. Describe it in `prompt` ("Friday of next week", "the delivery date, 10 days from today") and choose the output style with `format`.

A model works out the answer using a date and time tool and writes it in the requested format. Check the output when exact formatting matters. An empty `prompt` does nothing.

## What it needs

- A Griptape Nodes license or a Griptape Cloud API key (`GT_CLOUD_API_KEY`).
- The model call runs on Griptape Cloud. A model your license does not allow stops the run with an error.

## Example

1. Add **Date and Time** and set `prompt` to "ten days from today".
1. Set `format` to `2024-06-15` to get a year-month-day string.
1. Run. The text, such as `2026-12-04`, is in `output`.

Use it to stamp a delivery date into a shot-status report built with a [Merge Texts](MergeTexts.md) node.

## Parameters

### prompt

The date or time to look up, in plain language.

### format

The style to write the result in. The choices are examples that the model follows: `2024-06-15`, `2024-06-15 12:00:00`, `Jun 15, 2024` (default), `Friday, June 15 at 2pm`, `June 18th, 2024, in the afternoon`, `This Saturday at 7 PM` and `Custom format`.

### custom_format

Shown when `format` is `Custom format`. Describe the format you want, for example `YYYYMMDD` or `%Y-%m-%d %H:%M`. The text is passed to the model as written. It is not run as a strftime pattern.

### model

Offers `gpt-4.1`, `gpt-4.1-mini` (default), `gpt-4.1-nano` and `gpt-5`. The parameter is hidden by default in the editor.

### output

The resulting date or time.
