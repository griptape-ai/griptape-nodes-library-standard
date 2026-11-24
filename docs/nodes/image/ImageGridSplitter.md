---
description: Splits an image that contains a regular grid of images into separate images, with automatic or manual grid detection.
when_to_use: Use it when you have a contact sheet, collage or grid render and need each tile as its own image.
---

## When to use it

You have one image that holds a regular grid of images, such as a 2x2 or 3x3 collage or a contact sheet of shot thumbnails. The node gives you each tile as its own image, both as a list and as named per-cell outputs, for wiring into other nodes.

## Example

1. Add **Image Grid Splitter** and connect the grid image to `input_image`.
1. Set `grid_detection_mode` to `auto`, or to `manual` and set `rows` and `cols`.
1. Check the `preview`, then run the node.

For a 2x2 grid, `r1c1` is top-left, `r1c2` is top-right, `r2c1` is bottom-left and `r2c2` is bottom-right. `images` contains `[r1c1, r1c2, r2c1, r2c2]`.

## Parameters

### grid_detection_mode

- `auto`: estimates a regular grid of equal-sized cells from edge strength, trying 1 to 12 rows and columns. It falls back to 1 when no count stands out. This is best-effort and can pick the wrong count, for example reading a 1x4 grid as 5x4. Check `detected_rows` and `detected_cols`, or use `manual`.
- `manual`: uses `rows` and `cols`. Dividers are not needed.

### rows, cols

Manual mode only. Sliders from 1 to 12, default 2 each.

### preview

The original image with dotted yellow grid lines, shown live as you change settings.

### detected_rows, detected_cols, detected_count

The grid that was used, filled on every run. Hidden in manual mode.

### images

A list of image URL artifacts in row-major order, left to right and top to bottom. Cells are saved as JPEG, so alpha is dropped.

Cells are not an exact even split. The node removes divider strips between cells. It also trims up to 12 pixels from each cell edge where the rows or columns are nearly one color, even without a divider. Cells with a flat background can lose a lot, for example 100x80 cells come out 76x56. A smooth gradient is not trimmed.

### grid_cells

One JPEG output per cell, named `r1c1`, `r1c2`, up to `r{row}c{col}`. They are created on each run from the chosen or detected grid size.

## Common issues

- **Preview lines or auto detection do not match the grid**: Some grids have weak visual boundaries. Switch to `manual` and set the correct `rows` and `cols`.

## Edge cases

- With no `input_image` the node does nothing and `images` is an empty list.
