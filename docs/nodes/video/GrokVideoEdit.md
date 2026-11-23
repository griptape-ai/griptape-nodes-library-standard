---
description: Edits an existing video from a text instruction using xAI's Grok video model.
when_to_use: Use it when you have a short MP4 and want to change it with a prompt, such as adding an object or changing the style.
---

## When to use it

You have a clip and want a changed version. Connect the clip to `video`, describe the change in `prompt`, and run the node.

The input must be an MP4 of 8.7 seconds or less. The node checks this before it submits, because xAI accepts a request that breaks these limits and then fails the job without a reason. Use a Trim Video node to convert or shorten a clip that does not fit. The duration of a video passed as an inline data link is not checked here, so xAI enforces it.

The node runs through the Griptape model proxy and needs a Griptape Cloud API key. Switch `api_key_provider` to **Customer** to use your own xAI key instead, stored in the `GROK_API_KEY` secret. `timeout` sets how long the node waits, in seconds, and 0 means no limit. If the wait ends first, the Refresh button in the Status group retrieves a finished result.

## Parameters

### video

The clip to edit. It is sent to the service as part of the request.

### prompt

What to change. Both `prompt` and `video` are required.

### video_url

The edited video. It is also saved to the project at `output_file`.
