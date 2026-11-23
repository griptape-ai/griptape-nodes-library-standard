---
description: Generates images from a prompt, and optionally from reference images, with ByteDance Seedream models.
when_to_use: Use it when you want high-resolution images from Seedream, including several related images from one prompt or edits guided by reference images.
---

## When to use it

Seedream generates images from text and can take reference images to guide the result. The node offers Seedream 5.0 Pro, Seedream 5.0 Lite and Seedream 4.5. The default is Seedream 4.5.

It runs through the Griptape model proxy. You need a Griptape Cloud credential, or you can switch `api_key_provider` to use your own ByteDance Seed key (`SEED_API_KEY` in Settings, Secrets). Generations are billed. The run waits up to `timeout` seconds, 600 by default.

## Parameters

### model

The other settings change with the model.

| Model | Reference images | Batch (`max_images`) | `output_format` | `optimize_prompt_mode` |
| --- | --- | --- | --- | --- |
| Seedream 5.0 Pro | Up to 10 | No, always one image | Yes | `standard` or `fast` |
| Seedream 5.0 Lite | Up to 14 | Yes | Yes | `standard` only |
| Seedream 4.5 | Up to 14 | Yes | No | Not available |

Older Seedream 4.0 and 3.0 selections in saved workflows are migrated to Seedream 5.0 Lite.

### images

Reference images for the model. The run fails validation if you connect more than the model allows. Seedream 5.0 Pro bills every reference image after the first. The other models do not.

### size

The choices depend on the model. A resolution level such as `2K` lets the model pick the aspect ratio from your prompt. An explicit `WxH` size fixes it.

> [!WARNING]
> On Seedream 5.0 Pro, `1K` and `1.5K` cost the same, so `1.5K` is the better value. `2K` and any explicit size above 2,360,000 pixels cost twice as much. Switching to Pro always resets `size` to `1.5K` for this reason.

### max_images

The most images the model may generate, from 1 to 15. The node asks the model to generate a related set automatically, so you can get fewer than this number. Not used by Seedream 5.0 Pro.

### image_url

The first generated image. When the model returns more, `image_url_2`, `image_url_3` and so on appear, up to `image_url_15`. Each output holds the image in the same position the provider returned. If one image can't be downloaded, its slot stays empty and the others are kept.

### provider_response

The provider's response, unchanged. It is hidden by default.
