---
description: Upscales a video to 1080p, 4K or 8K using LTX.
when_to_use: Use it when you need a clip at a higher resolution, up to 8K, and can supply H.264 or HEVC footage.
---

## When to use it

You have a clip and need it at a higher resolution, up to 8K, from H.264 or HEVC footage. Connect it to `video`, choose a `resolution`, and run the node.

## What it needs

- A Griptape Nodes license or a Griptape Cloud API key (`GT_CLOUD_API_KEY`). To use your own LTX Studio key (`LTX_API_KEY`) instead, set **API Key Provider** to **Customer**.
- The node waits for the result. `timeout` sets how long, 1200 seconds by default, or 0 for no limit. If a run times out, the **Refresh** button on the node fetches the result later.
- A local video is uploaded to Griptape Cloud as a short-lived public link, which is deleted after the run. If the video is a link to another website, Griptape Cloud downloads it first, which can take up to about 2.5 minutes for a large file. A local file starts faster.
  - It must be H.264 or HEVC.
  - The short side must be at least 480 pixels, and the video can be at most 7680 x 4320.
  - It needs at least 2 frames and a frame rate of at least 1 fps.
  - The frame limit depends on the output `resolution`.
- Billing is per second of input video, by output resolution.

## Parameters

### resolution

The output size: `1080p`, `4k` (the default) or `8k`. Higher resolutions allow fewer input frames.

| Output | Maximum input frames |
| --- | --- |
| `1080p` | 2401 |
| `4k` | 601 |
| `8k` | 241 |

### direct_download

Downloads the result straight from LTX instead of through Griptape Cloud hosting. Use it for very large outputs. Griptape Cloud does not keep the file in this mode, so **Refresh** fails once LTX has removed it.
