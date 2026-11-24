---
description: Generates a video driven by an audio clip and a text prompt using LTX models, with an optional image.
when_to_use: Use it when you have a voice, music or sound clip and want a video that goes with it.
---

## When to use it

You have audio and want visuals for it. Connect the audio to `audio`, describe the video in `prompt`, and run the node. Add an image to `image` to guide how the video looks.

## What it needs

- A Griptape Nodes license or a Griptape Cloud API key (`GT_CLOUD_API_KEY`). To use your own LTX Studio key (`LTX_API_KEY`) instead, set **API Key Provider** to **Customer**.
- The node waits for the result. `timeout` sets how long, 1200 seconds by default, or 0 for no limit. If a run times out, the **Refresh** button on the node fetches the result later.

## Parameters

### model

`ltx-2-pro`, `ltx-2-3-pro`, `ltx-2-5-pro` or `ltx-2-5-fast`. The default is `ltx-2-5-fast`.

### audio

Required. The node converts the audio to a format LTX accepts. WAV files are re-encoded to MP3 first.

> [!NOTE]
> If the audio cannot be loaded, the node reports "Connect audio to 'Input Audio'", the same message as for a missing connection.

### image

Optional image to guide the video.

### resolution

`1920x1080` or `1080x1920`. This node offers no other sizes and has no duration or frame rate setting.

### guidance_scale

Optional, from 1 to 50. Higher values follow the prompt more closely. Leave it empty to use the LTX default.

### video_url

The generated video. It is also saved to the project at `output_file`.
