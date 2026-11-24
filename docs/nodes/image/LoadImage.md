---
description: Brings an image into the workflow from a file, a URL, or another node, and can extract one channel as a mask.
when_to_use: Use it at the start of a workflow to load a plate, render, reference frame or other image.
---

## When to use it

You need an image as the input of a workflow. For example, load a plate for a shot, or a render pass you want to inspect or composite.

## Example

1. Add a **Load Image** node.
1. Click the image area to browse for a file, drop a file on it, or paste a path or URL into `path`.
1. Run the workflow. The image is in `image` and its location is in `path`.
1. Connect `image` to the nodes that should use it.

## Parameters

### image

Shown in the editor as "Image or Path to Image". Accepts an image from another node, an image file, or a string path or URL. The image and `path` stay in sync: choosing a file fills `path`, and entering a path loads the image.

### path

Shown as "File Path or URL". A local path or a URL. Supported file types are png, jpg, jpeg, gif, bmp, webp, tiff and tif. If `image` has a value, it is used and `path` is not read. The node fails if both are empty or the file cannot be opened as an image.

If the file is outside the project, the node shows a warning and a **Copy to Project** button. The button copies the file into the project's inputs folder and points the node at the copy.

### mask_channel

Hidden by default. `none` (default), `red`, `green`, `blue` or `alpha`. When set to a channel, that channel is saved as a grayscale PNG and sent to `output_mask`. Setting it to `none` clears `output_mask`. `output_mask` is also hidden by default.

For an RGB image, `alpha` gives a fully white mask. For a grayscale image, the image itself is the mask. Other modes, such as palette images, fail the mask extraction, which is logged as a warning and leaves `output_mask` empty. The mask is saved to the project location set in `mask_output_file`.

### Status

The status group shows whether the load succeeded and why it failed if not.
