---
description: Regenerates one segment of a video, its picture, its audio or both, from a new prompt using LTX.
when_to_use: Use it when most of a clip is right and you want to redo a short part of it without regenerating the whole thing.
---

## When to use it

Most of a clip is right and one moment needs to change. Connect the video to `video`, mark the part to redo in `retake_segment`, describe what should happen there in `prompt`, and run the node. The rest of the clip is kept.

## What it needs

- A Griptape Nodes license or a Griptape Cloud API key (`GT_CLOUD_API_KEY`). To use your own LTX Studio key (`LTX_API_KEY`) instead, set **API Key Provider** to **Customer**.
- The node waits for the result. `timeout` sets how long, 1200 seconds by default, or 0 for no limit. If a run times out, the **Refresh** button on the node fetches the result later.
- A local video is uploaded to Griptape Cloud as a short-lived public link, which is deleted after the run. A video that is already a public https link is sent as is. If the upload is not possible, the video is embedded in the request instead. That only works for files up to about 11 MB.
- The clip can be up to 21 seconds long. The node checks the length before it submits.

## Parameters

### video

The clip to edit.

### retake_segment

The start and end time, in seconds, of the part to regenerate. The segment must be at least 2 seconds long and end no later than 21 seconds. When you connect a video, the slider's maximum follows its length and the segment is pulled back to fit.

### mode

`replace_audio_and_video` (the default), `replace_video` or `replace_audio`.

### resolution

`1920x1080` or `1080x1920`. Retake renders at 1080p only. When a video is connected, the orientation is picked from it. Square videos use `1920x1080`.

### prompt

What should happen in the segment. Up to 5000 characters.

### model

Only `ltx-2-3-pro` is offered. Older saved values of `ltx-2-pro` move to it.
