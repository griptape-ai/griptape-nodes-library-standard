---
description: Rotates an image by a quarter turn or by any angle.
when_to_use: Use it when an image is sideways, upside down or slightly crooked, or when you want to tilt it on purpose.
---

## When to use it

Pick a quarter turn (90° clockwise, 180° or 90° counter-clockwise) for a lossless rotation. Pick `custom angle` to rotate by any number of degrees. The extra settings for custom rotation appear only when `rotation` is set to `custom angle`.

The node first applies any EXIF orientation the image carries, because the saved output drops that tag. The result is rotated as the image is normally displayed.

## Parameters

### angle

Degrees from -360 to 360. Positive turns clockwise and negative turns counter-clockwise.

### expand_canvas

On by default. The canvas grows so no corners are cut off. Turn it off to keep the original size, which crops the corners.

### background_color

Fills the empty corners left by a custom angle. Default is transparent. Formats without transparency, such as JPEG, fill the corners with white. Images that aren't RGB are converted to RGBA so the fill can be transparent.

### output_format

`auto` saves the result in the same format as the input.
