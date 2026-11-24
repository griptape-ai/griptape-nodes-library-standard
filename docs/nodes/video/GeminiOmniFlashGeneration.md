---
description: Generates a short 720p video with audio from a text prompt, optionally guided by reference images, using Google Gemini Omni Flash.
when_to_use: Use it when you want a quick clip from text, or want to steer the look with one or more reference images.
---

## When to use it

You want a short clip from a description. Write a `prompt` and run the node. `aspect_ratio` is the only output control.

## What it needs

- A Griptape Nodes license or a Griptape Cloud API key (`GT_CLOUD_API_KEY`). To use your own Google Cloud service account JSON (`GOOGLE_SERVICE_ACCOUNT_JSON`) instead, set **Credentials Provider** to **Customer**.
- The node waits for the result. `timeout` sets how long, 600 seconds by default, and 0 means no limit. If a run times out, the **Refresh** button on the node fetches the result later.
- The output is 720p video with audio. The model chooses the clip length itself, between 3 and 10 seconds, so there is no duration setting.

## Parameters

### reference_images

Optional images to guide the video. Refer to them in the prompt as `<IMAGE_REF_0>`, `<IMAGE_REF_1>` and so on, starting at 0, to say how each one is used. For example: "in the style of `<IMAGE_REF_0>` a woman `<IMAGE_REF_1>` is walking". Audio references are not supported.

The node picks the task from what you connect. With reference images it makes a reference-to-video clip. With only `image` it makes image-to-video. With neither it makes text-to-video.

An image that cannot be loaded is skipped without a message.

### image

Deprecated. It is hidden unless an older workflow already uses it. Use `reference_images` instead. If both are set, `image` is ignored and a warning is logged.

### aspect_ratio

`16:9` or `9:16`. The default is `16:9`.

### video_url

The generated video. It is also saved to the project at `output_file`. If a generation returns more than one video, only the first is available here and a warning is logged.
