---
description: Upscales, sharpens, denoises, restores, relights or cuts out an image with Topaz Labs models.
when_to_use: Use it when you need to improve a photo or generated image, such as upscaling it, fixing blur or noise, repairing scratches or removing the background.
---

## When to use it

You have a photo or generated image that needs work: upscaling, blur or noise, scratches, lighting or a background cutout. Pick an operation, then a model for it, and connect the image. The settings shown change with the operation and model.

## What it needs

- A Griptape Nodes license or a Griptape Cloud API key (`GT_CLOUD_API_KEY`). To use your own Topaz Labs key (`TOPAZ_API_KEY`) instead, set **API Key Provider** to **Customer**.
- The node waits for the result. `timeout` sets how long, 600 seconds by default, or 0 for no limit. If a run times out, the **Refresh** button on the node fetches the result later.
- Topaz charges credits by output size. See the note under `resize_mode`.

## Parameters

### operation

Changing it resets `model` to the first model for that operation.

| Operation | Models |
| --- | --- |
| `enhance` | Standard V2, Low Resolution V2, CGI, High Fidelity V2, Text Refine |
| `enhance-generative` | Redefine, Recovery V2, Standard MAX, Wonder |
| `sharpen` | Standard, Strong, Lens Blur, Lens Blur V2, Motion Blur, Natural, Refocus, Wildlife, Portrait |
| `sharpen-generative` | Super Focus V2 |
| `denoise` | Normal, Strong, Extreme |
| `restore-generative` | Dust-Scratch, Dust-Scratch V2 |
| `lighting` | Adjust, Adjust V2, White Balance, Colorize |
| `matting` | Object |
| `tool` | Transparency Upscale |

### model

Only the settings the chosen model supports are shown.

- `prompt`, `autoprompt`, `creativity` and `texture` belong to Redefine.
- `grain` and its settings belong to Dust-Scratch V2.
- `mode` (`alpha` or `segmentation`) belongs to the matting model.

### face_enhancement

`face_enhancement_strength` and `face_enhancement_creativity` are only sent when this is on.

### autoprompt

Redefine only. When on, Topaz writes the prompt and your `prompt` is ignored.

### resize_mode

Shown only for `enhance`, `enhance-generative` and `tool`. Other operations keep the input resolution.

| Mode | What you set |
| --- | --- |
| `none` | Nothing. Topaz picks the size. |
| `width` | `output_width`. The height follows the source aspect ratio. |
| `height` | `output_height`. The width follows the source aspect ratio. |
| `width and height` | Both. `crop_to_fill` crops to fill instead of letterboxing when the ratio differs. |
| `percentage` | `percentage` of the source size, from 1 to 500. |

Sizes can be 1 to 32,000 pixels.

> [!NOTE]
> For `enhance` and `tool`, Topaz bills 1 credit per 24 megapixels of output. For `enhance-generative` it is 1 credit per 4 megapixels. Each run costs at least 1 credit. A badge on `resize_mode` shows the cost when the size is known up front.

### output_format

`jpeg` (default), `png` or `webp`.
