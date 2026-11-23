---
description: Picks the option that best fits some text or images and sends the flow down that option's path.
when_to_use: Use it when a workflow should branch into one of several routes based on what some text is about.
---

## When to use it

You have text, and optionally images, and want to route it to one of several outcomes, such as which department should handle a message. Write each option as a row. Each row becomes its own flow output, and the flow continues from the one JEV picks. A Failed output is taken on errors.

The node runs through Griptape Cloud and needs a Griptape Cloud API key (`GT_CLOUD_API_KEY`).

## Parameters

### context

The text JEV reads. JSON also works. It can be empty if you connect images. The node fails if both are empty.

### images

Optional images. Name them in the question or context as `[image 1]`, `[image 2]` and so on. They are numbered among connected images only, so empty rows are skipped. Images are resized to fit JEV's limits of 280 to 1568 pixels per side and saved as JPEG, and transparency is flattened onto white. An image with an aspect ratio too extreme to fit the limits raises an error that tells you to crop it.

### question

Optional. Tells JEV what to decide, for example `Which department should handle this?`.

### options

One option per row, up to 255. Write a label, or `Label: description` to tell JEV what the option means. The text before the first colon is the label. Without a colon, the whole row is the label and JEV decides from the label alone.

- You need at least two options, and each label must be unique.
- Empty rows are ignored.
- The outputs follow the list as you edit it. Deleting a row, or clearing its text, removes its output and any wires on it. Renaming a row keeps its output and wires and only changes the label. Reordering keeps wires on their rows.

### label and description

The label and description of the picked option. `description` is empty if the option had none.

### confidence

How sure JEV is, from 0 to 1. A low value means the text could fit another option too.

### probabilities

JEV's probability for every option, keyed by label. They add up to about 1.

### model

In the collapsed Advanced group. `jev-latest` is the newest stable model and `jev-preview` is the newest release. Models your license does not allow are marked.
