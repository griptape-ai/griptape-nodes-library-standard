---
description: Edits an existing MP4 clip from a text instruction using FLUX Video Edit [fast].
when_to_use: Use it when you have a clip of up to 15 seconds and want to change it with a prompt.
---

## When to use it

You have a short clip and want a changed version. Connect it to `video`, describe the change in `prompt`, and run the node.

The input must be an MP4 of 15 seconds or less. The node checks this before it submits and stops with a message if the clip does not qualify. The requirements badge on `video` also lists a minimum size of 160 x 160 pixels.

The node runs through the Griptape model proxy and needs a Griptape Cloud API key. Switch `api_key_provider` to **Customer** to use your own Black Forest Labs key instead, stored in the `BFL_API_KEY` secret. `timeout` sets how long the node waits, in seconds, and 0 means no limit. If the wait ends first, the Refresh button in the Status group retrieves a finished result.

## Parameters

### prompt

The edit instruction. It must be between 1 and 4096 characters.

### video

The clip to edit. The tooltip also lists a 50 MB limit, which this node does not check itself.

### safety_tolerance

`least restrictive`, `moderate` or `most restrictive`. The default is `moderate`.
