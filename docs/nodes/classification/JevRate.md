---
description: Rates some text or images against levels you describe and sends the flow down the matching level's path.
when_to_use: Use it when you want a graded judgment, such as how urgent a message is, and a different route for each grade.
---

## When to use it

You have text, and optionally images, and want to place it on a scale. Describe each level in a row, lowest first. Each row becomes its own flow output, and the flow continues from the level JEV scores. A Failed output is taken on errors.

The node runs through Griptape Cloud and needs a Griptape Cloud API key (`GT_CLOUD_API_KEY`).

For a two-way decision, use [Ask Yes/No (JEV)](JevAskYesNo.md). To pick between unrelated options, use [Pick One (JEV)](JevPickOne.md).

## Parameters

### context

The text JEV reads. JSON also works. It can be empty if you connect images. The node fails if both are empty.

### images

Optional images. Name them in the question or context as `[image 1]`, `[image 2]` and so on. They are numbered among connected images only, so empty rows are skipped. Images are resized to fit JEV's limits of 280 to 1568 pixels per side and saved as JPEG, and transparency is flattened onto white. An image with an aspect ratio too extreme to fit the limits raises an error that tells you to crop it.

### question

Optional. Tells JEV what to rate, for example `How urgent is this message?`.

### levels

One level per row, lowest first. You need 2 to 10. Describe the situation at each level. To name a level, put a label before a colon, as in `Minor: broken but a workaround exists`.

- A row with a colon is named by its label. A row without one is named `Level N`, counting non-empty rows from 1, and the whole row is its description.
- Empty rows are ignored.
- Deleting or renaming a row removes its output and any wires on it. Reordering keeps wires on their rows.

### score

JEV's score from 1 to the number of levels. It weighs every level by its probability, so it can fall between levels.

### level

The score rounded to the nearest level, starting at 1. The flow follows this level.

### label and description

The label and description of the level that was chosen.

### confidence

How sure JEV is, from 0 to 1. A low value means the probability was spread across levels.

### probabilities

JEV's probability for every level, keyed by level number starting at 1.

### model

In the collapsed Advanced group. `jev-latest` is the newest stable model and `jev-preview` is the newest release. Models your license does not allow are marked.
