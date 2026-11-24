---
description: Edits an existing video from a text instruction using xAI's Grok video model.
when_to_use: Use it when you have a short MP4 and want to change it with a prompt, such as adding an object or changing the style.
---

## When to use it

You have a clip and want a changed version. Connect the clip to `video`, describe the change in `prompt`, and run the node. Use a Trim Video node to convert or shorten a clip that does not fit the limits.

## What it needs

- A Griptape Nodes license or a Griptape Cloud API key (`GT_CLOUD_API_KEY`). To use your own xAI key (`GROK_API_KEY`) instead, set **API Key Provider** to **Customer**.
- The node waits for the result. `timeout` sets how long, 600 seconds by default, or 0 for no limit. If a run times out, the **Refresh** button on the node fetches the result later.
- The input must be an MP4 of 8.7 seconds or less. The node checks the file type and length before it submits, because xAI accepts a request that breaks these limits and then fails the job without a reason. The duration of a video passed as an inline data link is not checked here, so xAI enforces it.

## Parameters

### video

The clip to edit. It is sent to the service as part of the request.

### prompt

What to change. Both `prompt` and `video` are required.

### video_url

The edited video. It is also saved to the project at `output_file`.
