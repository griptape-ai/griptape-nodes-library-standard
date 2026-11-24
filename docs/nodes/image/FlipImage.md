---
description: Mirrors an image left to right, top to bottom, or both.
when_to_use: Use it when an image is mirrored the wrong way, or when you want a flipped variation of a plate or element.
---

## When to use it

You want a mirrored copy of an image. Connect the image to `input_image` and pick a `direction`. The preview in the editor shows the result as you change it. The output file is created when the node runs.

## Parameters

### direction

`horizontal` mirrors left to right (the default). `vertical` flips top to bottom. `both` does both, which is the same as a 180 degree rotation.

### output_format

`auto` saves in the same format as the input. Otherwise `PNG`, `JPEG` or `WEBP`. `quality` applies to `JPEG` and `WEBP`.

### was_successful and result_details

`was_successful` says whether the flip worked. `result_details` has the reason when it did not. When the failure output is not connected, a failure also stops the workflow.

### output

The flipped image, saved with the name in `output_file` (default `output.png`).
