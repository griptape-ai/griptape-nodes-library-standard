---
description: Merges a list of masks into one by taking the highest value at each pixel.
when_to_use: Use it when a node produces several mattes and you want one combined matte, such as merging per-object roto mattes into a single subject matte.
---

## When to use it

You have several masks of the same size and want their union. For example, combine the individual mattes for each character in a shot into one matte of everything that should stay in the plate.

## Example

1. Produce several mattes, for example from separate **Paint Mask** or [Display Mask](DisplayMask.md) nodes.
1. Add **Combine Masks** and connect each matte to `masks`.
1. Run the workflow. The union is in `output_mask`. Pass it to [Apply Mask](ApplyMask.md) or [Invert Mask](InvertMask.md).

## Parameters

### masks

A list of masks. At least one is required. All masks must be the same width and height, and validation fails with the size of the first mask and the index of the first mismatch.

Each mask is read as a single grayscale channel:

- RGBA or LA: the alpha channel is used. If the alpha is almost entirely opaque (its minimum is 200 or more and its range is 32 or less), the node assumes the mask is stored in the color instead and uses the red channel (RGBA) or the gray channel (LA).
- Grayscale (L): used as is.
- RGB and other modes: converted to grayscale.

At each pixel the output takes the maximum value across all masks.

### output_mask

The combined mask as a grayscale PNG, saved to the project location set in `output_file`. The node shows a live preview above its settings. The saved result is produced when the node runs.
