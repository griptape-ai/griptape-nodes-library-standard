---
description: Applies a Gaussian blur to an image.
when_to_use: Use it when you want to soften an image, hide detail, or make a background for something placed on top.
---

## When to use it

Blurs the whole image by a radius you choose. Higher values give a stronger blur. A radius of 0 returns the image unchanged.

## Parameters

### radius

Blur radius from 0 to 50, default 5. Values outside that range fail validation.

### output_format

`auto` saves the result in the same format as the input.

### quality

Only affects lossy formats such as JPEG.
