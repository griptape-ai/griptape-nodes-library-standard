---
description: Deprecated. Extended a Kling video by 4 to 5 seconds; Griptape Cloud no longer serves Kling video extension.
when_to_use: "Don't use it for new work. Use LTX Video Extend or Kling Omni Video Generation instead."
---

## When to use it

> [!WARNING]
> This node is deprecated. Kling no longer serves video extension through Griptape Cloud, so this node can no longer run. Saved workflows that use it still load. To extend a clip, replace it with [LTX Video Extend](LTXVideoExtend.md), or with [Kling Omni Video Generation](KlingOmniVideoGeneration.md) using the clip as a reference video.

You made a clip with a Kling node and want more of it. Connect the `kling_video_id` output of that node to `video_id`, describe how the clip should continue, and run the node.

## What it needs

- A Griptape Nodes license or a Griptape Cloud API key (`GT_CLOUD_API_KEY`). To use your own Kling AI key (`KLING_API_KEY`) instead, set **API Key Provider** to **Customer**.
- The node waits for the result. `timeout` sets how long, 600 seconds by default, or 0 for no limit. If a run times out, the **Refresh** button on the node fetches the result later.
- Each run adds 4 to 5 seconds. The longest total video is 3 minutes.
- It works on Kling's own ID, not on a video file, so a clip that did not come from Kling cannot be extended here. [Kling Text to Video Generation](KlingTextToVideoGeneration.md) and [Kling Image to Video Generation](KlingImageToVideoGeneration.md) both output the ID.

## Parameters

### video_id

Required. The ID of the Kling video to continue.

### prompt

Optional. Up to 2500 characters, describing how the video continues.

### negative_prompt

Optional. Up to 2500 characters, describing what to avoid.

### cfg_scale

From 0 to 1. Higher values follow the prompt more strictly and give the model less freedom.

### kling_video_id

The ID of the extended video. You can connect it to another Kling Video Extension node to keep going.
