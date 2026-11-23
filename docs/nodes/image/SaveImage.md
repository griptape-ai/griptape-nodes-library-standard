---
description: Saves an image to a file, in the format given by the file extension.
when_to_use: Use it at the end of an image branch to write a result to disk, such as a graded frame or a render pass, with a name and folder you control.
---

## When to use it

You want an image written to a file you can find later. The node re-encodes the image to match the extension in `output_file`, for example `.png`, `.jpg` or `.webp`. It also passes the saved image on through its `image` output, now pointing at the file on disk, so you can connect more nodes after it.

The folder and naming rules come from the `situation` you pick. To control them with more precision, connect a [File Output Settings](../filesystem/FileOutputSettings.md) node to `output_file`.

## Example

1. Process a frame with [Adjust Image EQ](AdjustImageEQ.md).
1. Connect the result to the `image` input of **Save Image**.
1. Set `output_file` to `SH010_comp_v001.png`.
1. Run the workflow. `result_details` shows the saved path.

## Parameters

### image

The image to save. If nothing is connected, the node finishes with a warning, creates no file, and `was_successful` is true.

### output_file

The file name. The extension picks the format, and an unsupported extension fails the save. `jpg` is treated as JPEG. The default name is `griptape_nodes.png`.

JPEG has no transparency. Transparent pixels are composited over black.

### situation

Selects the project situation, which sets the output folder and naming conventions. Use the refresh button to reload the list from the current project.

### was_successful and result_details

`result_details` has the saved path on success, or the reason for a failure. When the failure output is not connected, a failed save also stops the workflow.
