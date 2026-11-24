---
description: Composes several images as layers on one canvas that you arrange in a visual editor.
when_to_use: Use it when you want to lay out images by hand, such as a layout board of reference frames, a collage of elements on a plate-sized canvas, or a quick paint-over.
---

## When to use it

You have a few images and want to place, scale, rotate and stack them on a canvas of a chosen size. Connect the images to `input_images`, then click **Open Image Bash Editor** to arrange them. Each connected image becomes a layer, centered on the canvas and scaled down if it is bigger than the canvas. The editor also has brush layers.

## Example

1. Connect a background plate and two elements to `input_images`.
1. Set `canvas_size` to `HD (16:9)`.
1. Open the editor, put the plate at the back, and position and scale the two elements.
1. Run the workflow. The composite is in `output_image`.

## Parameters

### canvas_size

A preset (for example `2K Full`, `2K Scope`, `4K Flat`, `HD (16:9)`, `UHD (4K, 16:9)`, `Instagram Square`) or `Custom`. A preset fills in `width` and `height` and locks them. `Custom` unlocks them. The default is `Custom` at 1920 by 1080.

### background_color

The canvas color. Default is `#ffffff`.

### input_images

The images to place. If you remove all connections, the image layers are removed and brush layers stay.

### comp_on_run

On by default. The node rebuilds the composite from the editor's layers when it runs, using the current canvas size and background color. When off, the node passes on the last image the editor produced and ignores the canvas size and background settings.

### output_image

The composite, saved as PNG with the name in `output_file` (default `image_bash.png`). With `comp_on_run` off, nothing is output until the editor has produced an image.
