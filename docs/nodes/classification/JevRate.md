---
description: Rates some text or images against levels you describe and sends the flow down the matching level's path.
when_to_use: Use it when you want a graded judgment, such as how urgent a message is, and a different route for each grade.
---

## When to use it

You have text, and optionally images, and want to place it on a scale. Describe each level in a row, lowest first. Each row becomes its own flow output, and the flow continues from the level JEV scores. A Failed output is taken on errors.

For a two-way decision, use [Ask Yes/No (JEV)](JevAskYesNo.md). To pick between unrelated options, use [Pick One (JEV)](JevPickOne.md).

## What it needs

- A Griptape Nodes license or a Griptape Cloud API key (`GT_CLOUD_API_KEY`). To use your own TypeSafe key (`TYPESAFE_API_KEY`) instead, set **API Key Provider** to **Customer**.
- The node waits for the result. `timeout` sets how long, 600 seconds by default. If a run times out, the **Refresh** button on the node fetches the result later.
- Text in `context`, images, or both. The node fails if both are empty.
- Images are resized to fit JEV's limits of 280 to 1568 pixels per side. They are saved as JPEG, and transparency is flattened onto white. The encoded images must total under 30 MiB, counting an image once for each time the context or question names it. Otherwise the node fails.
- 2 to 10 levels.

## Parameters

### context

The text JEV reads. JSON also works. It can be empty if you connect images.

### images

Optional. Name them in the question or context as `[image 1]`, `[image 2]` and so on. Numbering counts connected images only, so empty rows are skipped. An aspect ratio too extreme to fit the size limits raises an error that asks you to crop.

### question

Optional. Tells JEV what to rate, for example `How urgent is this message?`.

### levels

One level per row, lowest first. Describe the situation at each level. To name a level, put a label before a colon, as in `Minor: broken but a workaround exists`.

- A row with a colon is named by its label. A row without one is named `Level N`, counting non-empty rows from 1, and the whole row is its description.
- Empty rows are ignored.
- Deleting a row, or clearing its text, removes its output and any wires on it. Renaming a row keeps its output and wires and only changes the label. Reordering keeps wires on their rows.

### score

JEV's score from 1 to the number of levels. It weighs every level by its probability, so it can fall between levels.

### level

The score rounded to the nearest level. The flow follows it.

### label and description

The label and description of the chosen level.

### confidence

From 0 to 1. A low value means the probability was spread across levels.

### probabilities

Each level's probability, keyed by level number starting at 1.

### model

In the collapsed Advanced group. `jev-latest` is the newest stable model and `jev-preview` is the newest release. Models your license does not allow are marked.
