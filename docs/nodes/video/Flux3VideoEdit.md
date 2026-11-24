---
description: Edits an existing MP4 clip from a text instruction using FLUX Video Edit [fast].
when_to_use: Use it when you have a clip of up to 15 seconds and want to change it with a prompt.
---

## When to use it

You have a short clip and want a changed version. Connect it to `video`, describe the change in `prompt`, and run the node.

## What it needs

- A Griptape Nodes license or a Griptape Cloud API key (`GT_CLOUD_API_KEY`). To use your own Black Forest Labs key (`BFL_API_KEY`) instead, set **API Key Provider** to **Customer**.
- The node waits for the result. `timeout` sets how long, 600 seconds by default, or 0 for no limit. If a run times out, the **Refresh** button on the node fetches the result later.
- The input must be an MP4 of 15 seconds or less, at least 160 x 160 pixels, and no more than 50 MB. The node checks the file type and length before it submits and stops with a message if the clip does not qualify. It does not check the pixel size or file size, and it skips the length check for a video passed as an inline data link. The clip is sent inline with the request.

## Parameters

### prompt

The edit instruction. It must be between 1 and 4096 characters.

### video

The clip to edit, up to 50 MB.

### safety_tolerance

`least restrictive`, `moderate` or `most restrictive`. The default is `moderate`.
