---
description: Shows two images in one compare view in the editor so you can check them against each other.
when_to_use: Use it to check a change by eye, such as a graded frame against the original plate or two versions of a shot.
---

## When to use it

You want to judge the difference between two images inside the workflow. Connect one image to `Image_1` and the other to `Image_2`. The compare view in the `Compare` parameter updates when either image changes, and again when the node runs. The node does not change or save either image.

## Example

1. Load a plate and connect it to `Image_1`.
1. Pass the plate through [Adjust Image EQ](AdjustImageEQ.md) and connect the result to `Image_2`.
1. Look at `Compare` to see the grade against the original.

## Parameters

### Compare

A dictionary with the two images under the keys `input_image_1` and `input_image_2`. The editor draws it as the compare view. It is also available as an output for downstream connections.
