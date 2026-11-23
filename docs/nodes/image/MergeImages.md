---
description: Combines up to four images into one, side by side, stacked, in a grid, or layered on top of each other.
when_to_use: Use it to build a comparison strip, a contact sheet, or a quick layered composite from a few images.
---

## When to use it

You have two to four images and want a single image. For example, you can put a plate, a first render and a final render side by side for a dailies review, or stack a render over a plate at partial opacity to check alignment.

## Example

1. Add a **Merge Images** node.
1. Connect the plate to the first `Images` row, the first render to the second and the final render to the third. Use the add button on the list for more rows.
1. Set `layout` to `horizontal`.
1. Set `background` to `black` so any gaps are black instead of white.
1. Run the workflow. The strip is in `output`.

## Parameters

### Images

The images to merge. The list holds 1 to 4 images. Rows beyond the fourth are ignored.

### layout

- `horizontal`: images side by side. They are all scaled to the tallest image's height, keeping aspect ratio.
- `vertical`: images stacked. They are all scaled to the widest image's width, keeping aspect ratio.
- `grid`: 1 to 3 images go in one row. Four images go in two rows of two. Each cell is the average width and height of the inputs, and each image is fitted inside its cell and centered.
- `quick composite`: images are layered. The last image is the bottom layer and the first image is on top. Each upper image is scaled to fit inside the bottom image and centered on it.

### input_alpha

How the input images store transparency. Use `straight` for most generated images. Use `premultiplied` for renders and keyer output that are premultiplied. A wrong choice fringes soft edges, dark if premultiplied images are treated as straight and bright the other way. This only matters for images that have an alpha channel.

### opacity

From 0 to 1, default 1. In `quick composite` the bottom image stays fully opaque and this fades the images over it. In the other layouts it fades every image toward the background.

### background

- `transparent`: empty areas stay transparent and the result keeps its alpha channel.
- `white` (default) and `black`: the result is flattened onto that color and has no alpha channel.

### output

The merged image, saved as a PNG to the location set by `output_file`. If no images are connected, `output` is empty and nothing is saved.
