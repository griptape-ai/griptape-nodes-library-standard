---
description: Finds the most prominent colors in an image and returns them as hex values.
when_to_use: Use it to pull a palette from a reference image, such as the dominant colors of a concept frame or a plate.
---

## When to use it

You want the main colors of an image as values you can use elsewhere. For example, you can extract a five-color palette from a concept frame and use it to set the colors for a title card or a set of annotation overlays.

## Example

1. Add a **Load Image** node and pick a concept frame.
1. Add **Extract Key Colors** and connect the image to `input_image`.
1. Set `num_colors` to 5.
1. Run the workflow. `color_1` to `color_5` hold the colors, most prominent first, and `colors` holds them as a list.

## Parameters

### input_image

The image to analyze. The node converts it to RGB first, so transparency is ignored.

### num_colors

The target number of colors, from 2 to 12. The default is 3. Only that many `color_N` parameters are shown.

### algorithm

- `kmeans` (default): clusters the pixels with scikit-learn's KMeans and returns the cluster centers.
- `median_cut`: repeatedly splits the group of pixels with the largest color range at its median, then averages each group.

With both, colors are sorted by how many pixels they cover, largest first. With `median_cut` the node can return fewer colors than requested when the image has too few distinct pixels to split. The unused `color_N` values are then empty.

### color_1 to color_12

Each color as a hex value like `#a1b2c3`, with a color picker. They are filled when the node runs.

### colors

All the extracted colors as a list of hex strings, in the same order.

## Notes

- Every pixel is analyzed, so a very large image takes longer.
- On failure, such as with no image connected, the node reports the error in `result_details` and sets `was_successful` to false.
