---
description: Generates a 3D model from a text prompt using Tripo.
when_to_use: Use it when you want a 3D model from a description alone.
---

## When to use it

Describe the object and Tripo builds a model. To start from a picture, use [Tripo Image to 3D](TripoImageTo3DGeneration.md). For several views, use [Tripo Multiview to 3D](TripoMultiviewTo3DGeneration.md).

The model is saved as GLB, or as FBX when `quad` is on. A preview image is saved next to it when Tripo provides one.

## What it needs

- A Griptape Nodes license or a Griptape Cloud API key (`GT_CLOUD_API_KEY`). To use your own Tripo 3D key (`TRIPO_API_KEY`) instead, set **API Key Provider** to **Customer**.
- The node waits for the result. `timeout` sets how long, 600 seconds by default. If a run times out, the **Refresh** button on the node fetches the result later.
- Texturing and detailed quality settings can use more credits. The status message reports the credits Tripo charged when Tripo returns them.
- A prompt. The node tooltip gives a limit of 1024 characters. The node does not check this.

## Parameters

### prompt

Required.

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
| `texture`, `pbr`, `texture_quality`, `negative_prompt` | all except v1.4 |
| `geometry_quality` | H3 v3.0 and v3.1 |
| `face_limit` | P1 and P2 |
| `quad` | P2 |

### negative_prompt

Things to avoid. Up to 255 characters. It is sent only when it is not empty.

### face_limit

The maximum polygon count. 0 lets Tripo decide. On P2, the value must be 48 to 50,000 for triangles or 48 to 25,000 for quads. Any other value fails before the job is sent.

### quad

Makes a quad mesh and saves an FBX file instead of GLB. P2 only.

### output_file

Where the model is saved. The extension changes to match the real format, and the preview is saved beside it as `.webp`.
