---
description: Asks a yes/no question about some text or images and sends the flow down a Yes or No path.
when_to_use: Use it when a workflow should branch on a judgment about text, such as whether a message asks for a refund.
---

## When to use it

You have text, and optionally images, and want to branch on a question about it. JEV returns the probability that the answer is yes. The node compares it to a threshold and continues from the Yes or No flow output. A Failed output is taken on errors.

The node runs through Griptape Cloud and needs a Griptape Cloud API key (`GT_CLOUD_API_KEY`).

## Parameters

### context

The text JEV reads. JSON also works. It can be empty if you connect images. The node fails if both are empty.

### images

Optional images. Name them in the question or context as `[image 1]`, `[image 2]` and so on. They are numbered among connected images only, so empty rows are skipped. Images are resized to fit JEV's limits of 280 to 1568 pixels per side and saved as JPEG, and transparency is flattened onto white. An image with an aspect ratio too extreme to fit the limits raises an error that tells you to crop it.

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
