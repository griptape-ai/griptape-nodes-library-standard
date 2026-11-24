---
description: Asks a yes/no question about some text or images and sends the flow down a Yes or No path.
when_to_use: Use it when a workflow should branch on a judgment about text, such as whether a message asks for a refund.
---

## When to use it

You have text, and optionally images, and want to branch on a question about it. JEV returns the probability that the answer is yes. The node compares it to a threshold and continues from the Yes or No flow output. A Failed output is taken on errors.

For a graded judgment, use [Rate (JEV)](JevRate.md). For more than two outcomes, use [Pick One (JEV)](JevPickOne.md).

## What it needs

- A Griptape Nodes license or a Griptape Cloud API key (`GT_CLOUD_API_KEY`). To use your own TypeSafe key (`TYPESAFE_API_KEY`) instead, set **API Key Provider** to **Customer**.
- The node waits for the result. `timeout` sets how long, 600 seconds by default, or 0 for no limit. If a run times out, the **Refresh** button on the node fetches the result later.
- Text in `context`, images, or both. The node fails if both are empty.
- Images are resized to fit JEV's limits of 280 to 1568 pixels per side. They are saved as JPEG, and transparency is flattened onto white. The encoded images must total under 30 MiB, counting an image once for each time the context or question names it. Otherwise the node fails.

## Parameters

### context

The text JEV reads. JSON also works. It can be empty if you connect images.

### images

Optional images. Name them in the question or context as `[image 1]`, `[image 2]` and so on. Numbering counts connected images only, so empty rows are skipped.

An image whose aspect ratio is too extreme to fit the size limits raises an error that tells you to crop it.

### question

Required. Ask one narrow thing, for example `Does the customer ask for a refund?`.

### Define Yes and No (optional)

A collapsed group with `yes_means` and `no_means`. Describe what should count as each answer when the question alone is ambiguous.

### threshold

The minimum probability for a Yes. The default is 0.5. Raise it to require more certainty.

### answer and probability

`probability` is JEV's chance of Yes, from 0 to 1. `answer` is true when it is at or above `threshold`. The flow output follows `answer`.

### model

In the collapsed Advanced group. `jev-latest` is the newest stable model and `jev-preview` is the newest release. Models your license does not allow are marked.
