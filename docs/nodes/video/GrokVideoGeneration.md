---
description: Generates a video from a text prompt, with an optional first-frame image, using xAI's Grok video model.
when_to_use: Use it when you want a short clip of 1 to 15 seconds from a prompt, or to animate a still image into a clip.
---

## When to use it

You want to turn a description into a video. Write a `prompt`, choose an aspect ratio, duration and resolution, and run the node. Connect an image to `image` to use it as the first frame.

## What it needs

- A Griptape Nodes license or a Griptape Cloud API key (`GT_CLOUD_API_KEY`). To use your own xAI key (`GROK_API_KEY`) instead, set **API Key Provider** to **Customer**.
- The node waits for the result. `timeout` sets how long, 600 seconds by default. If a run times out, the **Refresh** button on the node fetches the result later.
- Set `timeout` to 0 to wait with no limit.
- Generation can take a few minutes.

## Parameters

### image

Optional first frame. When it is connected, the clip starts from this image and the prompt describes what happens next.

### aspect_ratio

One of `1:1`, `16:9`, `9:16`, `4:3`, `3:4`, `3:2` or `2:3`. The default is `16:9`.

### duration

Length in seconds, 1 to 15. The default is 6.

### resolution

`480p` or `720p`. The default is `720p`.

### video_url

The generated video. It is also saved to the project at `output_file`.
