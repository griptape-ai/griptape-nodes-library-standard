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

- `auto`: estimates a regular grid of equal-sized cells from edge strength at evenly spaced positions, trying 1 to 12 rows and columns. It falls back to 1 when no count stands out. This is best-effort. It can pick the wrong count: a 1x4 grid can be read as 5x4, a 4x1 grid as 1x1 and a 6x6 grid as 3x2. Check `detected_rows` and `detected_cols`, or use `manual`.
- `manual`: uses `rows` and `cols`. Dividers are not needed.

### rows, cols

Manual mode only. Sliders from 1 to 12, default 2 each. Values outside 1 to 12 are rejected when you set them.

### preview

The original image with dotted yellow grid lines, saved as JPEG. The editor shows the lines live as you change the settings. The `preview` output is written when the node runs, and is empty if there is no input image.

### detected_rows, detected_cols, detected_count

The `detections_auto` group. Filled on every run with the grid that was used, in auto and manual mode. The group is hidden in manual mode.

### images

A list of image URL artifacts in row-major order, left to right and top to bottom. Cells are saved as JPEG, so alpha is dropped. With no `input_image` the node does nothing and `images` is an empty list.

Cells are not an exact even split. The node removes divider strips between cells. It also trims up to 12 pixels from each cell edge where the rows or columns are nearly one color, even when there is no divider. A cell with a flat background can lose a lot: 100x80 cells with flat edges come out 76x56, and a 200x200 image with a flat black background comes out 172x172 as a single cell. A smooth gradient with no flat edges is not trimmed.

### grid_cells

One image output per cell, named `r1c1`, `r1c2`, up to `r{row}c{col}`, saved as JPEG. These outputs are created when you run the node, based on the chosen or detected grid size. Each run removes the previous cell outputs and creates new ones.

## Common issues

- **Preview lines do not match the grid**: Switch to `manual` and set the correct `rows` and `cols`.
- **Unexpected auto detection**: Some grids have weak visual boundaries. Manual mode is the reliable fallback.
