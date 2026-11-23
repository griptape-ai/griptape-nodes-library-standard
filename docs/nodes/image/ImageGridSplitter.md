---
description: Splits an image that contains a regular grid of images into separate images, with automatic or manual grid detection.
when_to_use: Use it when you have a contact sheet, collage or grid render and need each tile as its own image.
---

## When to use it

You have one image that holds a regular grid of images, such as a 2x2 or 3x3 collage or a contact sheet of shot thumbnails. The node gives you each tile as its own image, both as a list and as named per-cell outputs, for wiring into other nodes.

## Example

1. Add **Image Grid Splitter** and connect the grid image to `input_image`.
1. Set `grid_detection_mode` to `auto`, or to `manual` and set `rows` and `cols`.
1. Check the `preview` to confirm the split lines.
1. Run the node.

For a 2x2 grid, `r1c1` is top-left, `r1c2` is top-right, `r2c1` is bottom-left and `r2c2` is bottom-right. `images` contains `[r1c1, r1c2, r2c1, r2c2]`.

## Parameters

### grid_detection_mode

- `auto`: estimates a regular grid of equal-sized cells. This is best-effort and can be less accurate when neighboring cells look alike.
- `manual`: uses `rows` and `cols`.

### rows, cols

Manual mode only. Sliders from 1 to 12, default 2 each.

### preview

The original image with dotted yellow grid lines. The editor shows the lines live as you change the settings. The output file is written when the node runs.

### detected_rows, detected_cols, detected_count

The `detections_auto` group. Filled on each run in auto mode and hidden in manual mode.

### images

A list of image URL artifacts in row-major order, left to right and top to bottom. Cells are saved as JPEG. The node trims uniform gap strips and edge strips between cells, so cells can be a few pixels smaller than an even split.

### grid_cells

One image output per cell, named `r1c1`, `r1c2`, up to `r{row}c{col}`. These outputs are created or updated when you run the node, based on the chosen or detected grid size.

## Common issues

- **Preview lines do not match the grid**: Switch to `manual` and set the correct `rows` and `cols`.
- **Unexpected auto detection**: Some grids have weak visual boundaries. Manual mode is the reliable fallback.
