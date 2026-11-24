---
description: Generates a 3D model from two to four views of an object (front, left, back, right) using Tripo.
when_to_use: Use it when you have several angles of the same object and want more accurate geometry than a single image gives.
---

## When to use it

Connect the front view and at least one other view. Tripo uses all of them to build the model. With a single image, use [Tripo Image to 3D](TripoImageTo3DGeneration.md).

The model is saved as GLB, or as FBX when `quad` is on. A preview image is saved next to it when Tripo provides one.

## What it needs

- A Griptape Nodes license or a Griptape Cloud API key (`GT_CLOUD_API_KEY`). To use your own Tripo 3D key (`TRIPO_API_KEY`) instead, set **API Key Provider** to **Customer**.
- The node waits for the result. `timeout` sets how long, 600 seconds by default. If a run times out, the **Refresh** button on the node fetches the result later.
- Texturing and detailed quality settings can use more credits. The status message reports the credits Tripo charged when Tripo returns them.
- A front image and at least one more view. The node tooltip gives PNG, JPEG or WEBP, up to 20 MB for each. The node does not check this.

## Parameters

### front_image

Required. The node fails if it is missing, even when other views are connected.

### left_image, back_image, right_image

Optional views. You need at least two in total, counting the front. Connect each view to the matching input.

### model_version

Which Tripo model to use. The node shows only the options that version supports and hides the rest. The multiview node offers P2, P1, H3 v3.1 (default), H3 v3.0 and H2 v2.5. The legacy v1.4 is not available here.

Which versions offer each option:

| Option | Versions |
| --- | --- |
| `texture`, `pbr`, `texture_quality`, `texture_alignment` | all |
| `geometry_quality` | H3 v3.0 and v3.1 |
| `face_limit` | P1 and P2 |
| `quad` | P2 |

### texture_alignment

`original_image` lines the textures up with your images. `geometry` lines them up with the generated shape. It only matters when `texture` is on.

### face_limit

The maximum polygon count. 0 lets Tripo decide. On P2, the value must be 48 to 50,000 for triangles or 48 to 25,000 for quads. Any other value fails before the job is sent.

### quad

Makes a quad mesh and saves an FBX file instead of GLB. P2 only.

### output_file

Where the model is saved. The extension changes to match the real format, and the preview is saved beside it as `.webp`.
