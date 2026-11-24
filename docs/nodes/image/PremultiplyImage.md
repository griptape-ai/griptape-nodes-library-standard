---
description: Multiplies an image's color by its alpha, converting straight alpha to premultiplied alpha.
when_to_use: Use it when a downstream tool or file pipeline expects premultiplied images, such as a compositing package.
---

## When to use it

Nodes in this library work with straight alpha. Some outside tools expect premultiplied alpha, where the color has already been scaled by the alpha. Run this node right before handing the image to one of those tools.

Don't premultiply before other nodes in this library. If you feed the result into [Merge Images](MergeImages.md), set its `input_alpha` to premultiplied. Otherwise soft edges get a dark fringe.

To go the other way, use [Unpremultiply Image](UnpremultiplyImage.md).

## Parameters

### input_image

Images without an alpha channel are treated as fully opaque.

### invert_alpha

Multiplies by `1 - alpha` instead of alpha. Opaque areas go dark and transparent areas keep their color. This is useful for edge matting. The alpha channel itself is never changed.

### output

Always saved as a PNG, named by `output_file`.
