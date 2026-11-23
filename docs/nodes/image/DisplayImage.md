---
description: Shows an image in the workflow, passes it through unchanged, and reports its width and height.
when_to_use: Use it when you want to see an image at a point in the workflow or read its pixel dimensions, such as checking a plate's resolution.
---

## When to use it

You want to look at an image at some point in the workflow, pass it on to other nodes, or read its size. It also helps when debugging, to see which image is being processed.

## Example

1. Connect an image source such as **Generate Image** or **Load Image** to `image`.
1. Check in the node that it is the image you expected.
1. Use `width` and `height` if other nodes need the dimensions.
1. Connect the `image` output to other nodes that need the image.

## Parameters

### image

The image to display. The same image is passed through unchanged on the `image` output.

### width, height

The image size in pixels, as integers. They are hidden by default. Show them in the Properties panel on the right.

For image URL inputs, the node reads the whole image at its location to measure it, so a large remote image can take time to load. For image artifact inputs, the size is read from the artifact's properties. The node cannot measure SVG files and reports 0 for both.

## Common issues

- **Image not displaying**: Check that the source is connected and produces valid image data.
- **Dimensions are 0**: No image is connected, the file could not be read, or the image is an SVG.
