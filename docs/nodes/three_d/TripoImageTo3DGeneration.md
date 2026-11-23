---
description: Generates a 3D model from one reference image using Tripo.
when_to_use: Use it when you have a picture of an object and want a 3D model of it.
---

## When to use it

Connect one image and Tripo builds a model from it. The node runs through the Griptape model proxy and needs a Griptape Cloud API key (`GT_CLOUD_API_KEY`). It polls until the job finishes. The default timeout is 10 minutes. If a run times out, the Refresh button in the Status group retrieves the result. Texturing and detailed quality settings cost more, and the status message reports the credits Tripo charged.

The model is saved as GLB, or as FBX when `quad` is on. A preview image is saved next to it when Tripo provides one.

For better geometry from several angles, use [Tripo Multiview to 3D](TripoMultiviewTo3DGeneration.md). To start from text, use [Tripo Text to 3D](TripoTextTo3DGeneration.md).

## Parameters

### image

Required. PNG, JPEG or WEBP, up to 10 MB.

### model_version

Which Tripo model to use. The node shows only the options that version supports and hides the rest.

| Version | Notes |
| --- | --- |
| P2 | Preview. Low-poly with quad mesh output. |
| P1 | Premium. Low-poly with stable topology. |
| H3 v3.1 (default) | High detail with geometry control. |
| H3 v3.0 | High detail with geometry control. |
| H2 v2.5 | Standard quality. |
| v1.4 | Legacy. Basic parameters only, with no texture options. |

Which versions offer each option:

| Option | Versions |
| --- | --- |
| `texture`, `pbr`, `texture_quality`, `texture_alignment` | all except v1.4 |
| `geometry_quality` | H3 v3.0 and v3.1 |
| `face_limit` | P1 and P2 |
| `quad` | P2 |

### texture_alignment

`original_image` lines the textures up with your image. `geometry` lines them up with the generated shape. It only matters when `texture` is on.

### enable_image_autofix

Hidden by default. Lets Tripo correct the input image before generating.

### face_limit

The maximum polygon count. 0 lets Tripo decide. On P2, the value must be 48 to 50,000 for triangles or 48 to 25,000 for quads. Any other value fails before the job is sent.

### quad

Makes a quad mesh and saves an FBX file instead of GLB. P2 only.

### output_file

Where the model is saved. The extension changes to match the real format, and the preview is saved beside it as `.webp`.
