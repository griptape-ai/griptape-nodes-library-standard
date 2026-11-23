---
description: Divides an image's color by its alpha to recover straight alpha from a premultiplied image.
when_to_use: Use it when semi-transparent edges look too dark, such as in 3D renders or EXR files, before color grading or compositing.
---

## When to use it

Renders from 3D software and EXR files usually come premultiplied. The color has already been scaled by alpha, so semi-transparent areas look darker than they should. Run this node first, before color grading, keying or compositing with other nodes in this library, which expect straight alpha.

To go the other way, use [Premultiply Image](PremultiplyImage.md).

## Parameters

### invert_alpha

Divides by `1 - alpha` instead of alpha. The alpha channel itself is never changed.

### output

Always saved as a PNG, named by `output_file`. Pixels where the divisor is zero, such as fully transparent pixels, become black.
