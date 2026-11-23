---
description: Makes every pixel that matches a chosen color transparent, with controls to refine the edge.
when_to_use: Use it when you need to key out a flat background, such as a green screen, or remove a single color from an image.
---

## When to use it

You have an image with a solid color you want gone, such as a green screen or a white backdrop. Pick the color, raise `tolerance` until the background is clear, then tidy the edge with `grow_shrink` and `blur_edges`. The result is a PNG with an alpha channel.

Pixels that were already transparent stay transparent. The node only adds transparency.

## Parameters

### color

The color to remove, as a hex value. Defaults to green (`#00ff00`).

### tolerance

A pixel matches when each of its red, green and blue values is within `tolerance` of the color. The range is 0 to 255 and the default is 10. Higher values remove more shades near the color.

### grow_shrink

Moves the edge of the transparent area outward when positive and inward when negative, in pixels. Use it to remove a leftover fringe or to bring back lost edge detail.

### blur_edges

Softens the edge of the transparent area. Higher values give a wider, smoother transition.
