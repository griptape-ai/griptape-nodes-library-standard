---
description: Transfers the color characteristics of a reference image onto a target image with a choice of color matching algorithms.
when_to_use: Use it when you need one image to take on the colors of another, such as matching a CG render to a plate.
---

## When to use it

You have a target image and a reference with the look you want. Typical uses are matching a CG render to a plate, correcting lighting differences between shots or stop-motion frames, applying one reference's grade across many images, and transferring the mood of a reference photo.

## Example

1. Load the plate and the render with two **Load Image** nodes.
1. Connect the render to `target_image` and the plate to `reference_image`.
1. Set `method` to `mkl` and `strength` to 1.0 for a full transfer.
1. Connect `output` to [Display Image](DisplayImage.md) to view the result.
1. Lower `strength` if the effect is too strong, or raise it if too subtle.

Other setups:

- **Photo grading**: Use a graded photo as the reference and try `reinhard` or `mkl`. Use `strength` below 1.0 for subtle results and above 1.0 for dramatic ones.
- **Stop-motion correction**: Use a frame with correct lighting as the reference and the frames to fix as targets, with `mkl` and `strength` 1.0.
- **Batch style transfer**: Run each target through the node with the same style reference. If the simple methods do not match well, try a compound method such as `hm-mvgd-hm`.

## Parameters

### reference_image

The image whose colors are copied. It can be a different size from the target.

### target_image

The image that is modified. The output keeps its dimensions.

### method

The algorithm. Default is `mkl`. Results differ between methods, so try several.

- `mkl`: Monge-Kantorovich Linearization. Uses optimal transport to match color distributions.
- `hm`: Histogram Matching. Matches the histogram of each channel independently.
- `reinhard`: Classic color transfer based on decorrelated color spaces.
- `mvgd`: Multi-Variate Gaussian Distribution. Matches the mean and covariance of the color distributions.
- `hm-mvgd-hm` and `hm-mkl-hm`: Compound methods that chain two of the above.

### strength

0.0 to 10.0, default 1.0. The node computes `result = target + strength * (matched - target)`. At 0 the target is returned unchanged, 1.0 is the full transfer and values above 1.0 exaggerate the effect.

### output

The color-matched image. Images are converted to RGB for processing, so any alpha channel is dropped and the output is a PNG written to the location set by `output_file`. Re-apply the alpha downstream if you need it.

## Notes

Both images must be connected and `strength` must be between 0 and 10. The node uses the color-matcher library.

## Common issues

- **No change visible**: `strength` is 0.
- **Effect too strong**: Reduce `strength` below 1.0.
- **Unexpected colors**: Try a different `method`.
