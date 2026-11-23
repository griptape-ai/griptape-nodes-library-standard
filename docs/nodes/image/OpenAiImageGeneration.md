---
description: Generates or edits images with OpenAI GPT Image models, with optional reference images, custom sizes and animated GIF output on the 2.6 models.
when_to_use: Use it when you want OpenAI image results, including edits from reference images, exact sizes, or short animated GIFs.
---

## When to use it

Write a prompt to generate an image. Add up to 16 reference images to edit or build on them. The default model is `gpt-image-2.5-flare`. The list also has `gpt-image-1`, `gpt-image-1.5`, `gpt-image-2`, `gpt-image-2.5-sunburst`, `gpt-image-2.6-sunburst` and `gpt-image-2.6-flare`. Several settings depend on the model.

It runs through the Griptape model proxy. You need a Griptape Cloud credential, or you can switch `api_key_provider` to use your own OpenAI key (`OPENAI_API_KEY` in Settings, Secrets). The run waits up to `timeout` seconds, 600 by default.

## Parameters

### model

| Setting | gpt-image-1 | gpt-image-1.5 | gpt-image-2 and 2.5 | gpt-image-2.6 |
| --- | --- | --- | --- | --- |
| Size | Three fixed sizes | Three fixed sizes | Presets or `custom` | Presets or `custom` |
| Transparent background | Yes | No | No | No |
| `complexity` | No | No | No | Yes |
| GIF output | No | No | No | Yes |

Switching model changes the choices for `size`, `background` and `output_format`. A value the new model doesn't support is reset.

### size

The fixed sizes are `1024x1024`, `1024x1536` and `1536x1024`. Models in the 2 family add larger presets up to `3840x2160` and a `custom` option. Choosing `custom` shows `custom_width` and `custom_height`.

### custom_width and custom_height

Each is snapped to the nearest multiple of 16 when you enter it, up to 3840. For still images, the aspect ratio can be at most 3:1 and the total must be between 655,360 and 8,294,400 pixels. These limits are checked when you run. They don't apply to GIF frames.

### background

`transparent` works only on `gpt-image-1` and needs `png` or `webp` output.

### moderation

`auto` is the default. `low` applies less strict filtering.

### output_format

`png`, `jpeg` or `webp`. The 2.6 models add `gif`, which produces an animated GIF. `output_compression`, from 0 to 100, appears for `jpeg` and `webp`.

### complexity

2.6 models only. Controls how much the model plans before it renders. Lower values suit focused edits. Higher values help with dense text and complex layouts, and can add latency. The default is `high`.

### frame_count and frame_duration_ms

Shown for GIF output. A GIF has 2 to 16 frames, 9 by default, and each frame shows for 20 to 2000 ms, 200 by default. `size` is the size of one frame, which only has to be a multiple of 16. The frames, their size and `n` must fit the model's pixel budget, so a large size can limit how many frames fit. More frames cost more.

### input_images

Up to 16 images. They are uploaded to Griptape Cloud so the proxy can fetch them, and the uploads are deleted after the run. Images that are already public URLs are used as they are.

### n

How many images to generate, 1 to 10. Each gets its own output. `image_url_2` and up appear as needed. If one image can't be downloaded, its slot stays empty and the others are kept.
