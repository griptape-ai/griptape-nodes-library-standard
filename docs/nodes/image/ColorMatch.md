---
description: Transfers the color characteristics of a reference image onto a target image with a choice of color matching algorithms.
when_to_use: Use it when you need one image to take on the colors of another, such as matching a CG render to a plate.
---

## When to use it

You have a target image and a reference with the look you want. Use it to match a CG render to a plate, correct lighting differences between shots or stop-motion frames, or apply one reference's grade across many images.

## Example

1. Connect the render to `target_image` and the plate to `reference_image`.
1. Set `method` to `mkl` and `strength` to 1.0 for a full transfer.
1. Connect `output` to [Display Image](DisplayImage.md) to view the result.

Other setups:

- **Photo grading**: Use a graded photo as the reference. Use `strength` below 1.0 for subtle results.
- **Stop-motion correction**: Use a frame with correct lighting as the reference and the frames to fix as targets.
- **Batch style transfer**: Run each target through the node with the same reference. If the simple methods do not match well, try `hm-mvgd-hm`.

## Parameters

### reference_image

Required. The image whose colors are copied. It can be a different size from the target.

### target_image

Required. The image that is modified. The output keeps its dimensions.

### method

Default is `mkl`. Results differ between methods, so try several.

- `mkl`: Monge-Kantorovich Linearization. Uses optimal transport.
- `hm`: Histogram Matching, per channel.
- `reinhard`: Classic color transfer.
- `mvgd`: Matches the mean and covariance of the color distributions.
- `hm-mvgd-hm` and `hm-mkl-hm`: Compound methods that run `hm`, then `mvgd` or `mkl`, then `hm` again.

### strength

0.0 to 10.0, default 1.0. The node computes `result = target + strength * (matched - target)`. At 0 the target is returned unchanged, 1.0 is the full transfer and values above 1.0 exaggerate the effect.

### output

The color-matched image, a PNG written to the location set by `output_file`. When `strength` is above 0 the output is RGB and any alpha channel is dropped, so re-apply alpha downstream if you need it. At `strength` 0 the target's alpha is kept.

## Common issues

- **No change visible**: `strength` is 0.
- **Unexpected colors**: Try a different `method`, or lower `strength`.
