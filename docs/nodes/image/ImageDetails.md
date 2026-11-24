---
description: Reads an image's size, aspect ratio, color mode, channel count and file format.
when_to_use: Use it when a later step needs to know the size or shape of an image, such as picking a matching aspect ratio.
---

## When to use it

Connect an image and read its properties as outputs. The values update as soon as an image is set, without running the workflow.

## Parameters

### ratio_str

The aspect ratio reduced to lowest terms, such as `16:9`. It is `0:0` if the size can't be read.

### color_space

The image mode, such as `RGB`, `RGBA` or `L`.

### channels

How many channels the mode has. `RGB` has 3 and `RGBA` has 4.

### format

The file format, such as `JPEG`, `PNG` or `WEBP`.

If there is no image, or it can't be read, all values fall back to 0, `0:0` or `UNKNOWN`.
