---
description: Generates or edits images with OpenAI GPT Image models, with optional reference images, custom sizes and animated GIF output on the 2.6 models.
when_to_use: Use it when you want OpenAI image results, including edits from reference images, exact sizes, or short animated GIFs.
---

## When to use it

You want an image from OpenAI's GPT Image models, from a prompt alone or built from up to 16 reference images. Pick it for exact sizes, a transparent background or a short animated GIF. Several settings depend on the model.

## What it needs

- A Griptape Nodes license or a Griptape Cloud API key (`GT_CLOUD_API_KEY`). To use your own OpenAI key (`OPENAI_API_KEY`) instead, set **API Key Provider** to **Customer**.
- The node waits for the result. `timeout` sets how long, 600 seconds by default. If a run times out, the **Refresh** button on the node fetches the result later.
- Reference images are uploaded to a short-lived public link, deleted after the run. Images that are already public URLs are used as they are.
- More GIF frames cost more.

## Parameters

### model

The default is `gpt-image-2.5-flare`. Other choices are `gpt-image-1`, `gpt-image-1.5`, `gpt-image-2`, `gpt-image-2.5-sunburst`, `gpt-image-2.6-sunburst` and `gpt-image-2.6-flare`.

| Setting | gpt-image-1 | gpt-image-1.5 | gpt-image-2 and 2.5 | gpt-image-2.6 |
| --- | --- | --- | --- | --- |
| Size | Three fixed sizes | Three fixed sizes | Presets or `custom` | Presets or `custom` |
| Transparent background | Yes | No | No | No |
| `complexity` | No | No | No | Yes |
| GIF output | No | No | No | Yes |

Switching model changes the choices for `size`, `background` and `output_format`. A value the new model doesn't support is reset.

### size

Fixed sizes are `1024x1024`, `1024x1536` and `1536x1024`. The 2 family adds presets up to `3840x2160` and `custom`.

### custom_width and custom_height

Each snaps to the nearest multiple of 16, up to 3840. For still images the aspect ratio can be at most 3:1 and the total must be 655,360 to 8,294,400 pixels. These limits don't apply to GIF frames.

### background

`transparent` works only on `gpt-image-1` and needs `png` or `webp` output.

### moderation

`low` applies less strict filtering than the default, `auto`.

### output_format

`png`, `jpeg` or `webp`. The 2.6 models add `gif`. `output_compression`, 0 to 100, appears for `jpeg` and `webp`.

### complexity

2.6 models only. Controls how much the model plans before it renders. Lower values suit focused edits. Higher values help with dense text and complex layouts but add latency. The default is `high`.

### frame_count and frame_duration_ms

Shown for GIF output.

- 2 to 16 frames, 9 by default, each shown for 20 to 2000 ms, 200 by default.
- `size` is one frame and only has to be a multiple of 16.
- Frames, size and `n` must fit the model's pixel budget, so a large size limits how many frames fit.

### input_images

Up to 16 reference images.

### n

How many images to generate, 1 to 10, each with its own output. If one image can't be downloaded, its slot stays empty and the others are kept.
