---
description: Flips a mask so white becomes black and black becomes white.
when_to_use: Use it when you have a matte for the opposite region from the one you need, such as turning a subject matte into a background matte.
---

## When to use it

You have a mask that selects one region and need the complement. For example, invert a roto matte of a character to get a mask of everything else in the shot.

## Example

1. Add a **Paint Mask** or [Display Mask](DisplayMask.md) node to produce a matte.
1. Add **Invert Mask** and connect the matte to `input_mask`.
1. Run the workflow. The inverted matte is in `output_mask`.

## Parameters

### input_mask

The mask to invert. The node reads it according to its mode:

- RGBA: only the alpha channel is used. The RGB values are ignored.
- Grayscale (L): used as is.
- Any other mode, including RGB: converted to grayscale first.

Every value is replaced by `255 - value`.

### output_mask

The inverted mask as a grayscale PNG, saved to the project location set in `output_file`. The node has a live preview above its settings. The preview is a display only and the saved result is produced when the node runs.

If `input_mask` is empty, the node fails with a message asking you to connect a mask.
