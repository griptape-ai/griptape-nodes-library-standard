---
description: Adds a soft glow to an image by blending it with a brightened, blurred copy of itself.
when_to_use: Use it when you want a soft halo or dreamy look on a render or still, such as a glow around lights in a sci-fi plate.
---

## When to use it

You want a soft, luminous glow. Typical uses are atmospheric or romantic lighting, fantasy and sci-fi looks, and a lens-flare style bloom on bright areas.

## Example

1. Load or generate an image with **Load Image** or **Generate Image** and connect it to `input_image`.
1. Set `bloom_amount` to 0.8 for a moderate glow and `bloom_radius` to 8 for a medium-sized bloom.
1. Connect `output` to [Display Image](DisplayImage.md) to view the result.

## Parameters

### bloom_amount

0.0 to 2.0, default 0.5. At 0 the image is returned unchanged. The blend weight stops at 1.0, so values above 1.0 only make the blurred layer brighter.

### bloom_radius

1 to 20, default 5. The blur radius. Lower values give a tighter glow and higher values a larger, more diffuse one.

### output

The image with the bloom applied. The node works in RGB, so any alpha channel is dropped. Re-apply the alpha downstream if you need it.

## Notes

The glow is built from the whole image, not just the bright areas. The node does this in three steps:

1. The whole image is brightened by a factor of `1 + bloom_amount`.
1. The brightened copy is blurred with a Gaussian blur of radius `bloom_radius`.
1. The original is blended with the blurred copy, with a weight of `bloom_amount` capped at 1.0.

The node shows a preview as you change the sliders.

## Common issues

- **No visible effect**: `bloom_amount` is 0.
- **Too subtle or too strong**: Raise or lower `bloom_amount`.
