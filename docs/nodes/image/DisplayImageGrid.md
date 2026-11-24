---
description: Arranges a list of images into one contact-sheet image, as a uniform grid or a masonry layout.
when_to_use: Use it when you want one image that shows several results together, such as a contact sheet of takes or a review board of shot frames.
---

## When to use it

You have several images and want a single image that shows all of them. For example, a contact sheet of the latest frame from each shot, or a board comparing several versions of the same plate. Connect the images to the `images` input. If the list is empty, the node outputs a plain placeholder image.

## Example

1. Generate or load several frames and collect them into a list connected to `images`.
1. Set `layout_style` to `grid` and `columns` to 4.
1. Set `output_image_size` to `preset` and `output_preset` to `1080p (1920x1080)` to get a review board at exactly 1920x1080.
1. Set `output_format` to `jpeg` for a small file, or leave it as `png`.
1. Run the workflow. The sheet is in `output`.

## Parameters

### layout_style

`grid` places the images in equal square cells. `masonry` keeps each image's aspect ratio and lets the heights vary. `crop_to_fit` and `grid_justification` apply to `grid` only.

### grid_justification

Where an incomplete last row sits: `left`, `center` or `right`. Hidden when `layout_style` is `masonry`. With a preset output size, it also sets where the finished grid sits horizontally on the preset canvas.

### columns

From 1 to 10. Default is 4.

### crop_to_fit

On by default. Each image is cropped to fill its square cell. Turn it off to scale each image to fit inside its cell without cropping.

### transparent_bg and background_color

`background_color` is hidden while `transparent_bg` is on. Choosing `jpeg` as the output format turns `transparent_bg` off, because JPEG has no transparency.

### output_image_size

`custom` uses `output_image_width`, which is the width of the output in pixels (default 1200). The height follows from the layout.

`preset` uses `output_preset`: `4K (3840x2160)`, `1440p (2560x1440)`, `1080p (1920x1080)` or `720p (1280x720)`. The grid is built at the preset width, scaled down if it is taller than the preset, never scaled up, then placed on a canvas of exactly the preset size. The grid is centered vertically and padded with the background color.

### output_format

`png`, `jpeg` or `webp`. The file is saved with the name in `output_file` (default `grid.png`).

### output

The finished image. It also updates in the editor when you change any layout or style setting, before you run the node.
