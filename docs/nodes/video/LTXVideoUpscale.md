---
description: Upscales a video to 1080p, 4K or 8K using LTX.
when_to_use: Use it when you need a clip at a higher resolution, up to 8K, and can supply H.264 or HEVC footage.
---

## When to use it

You have a clip and need it larger. Connect it to `video`, choose a `resolution`, and run the node.

The input is checked before it submits.

- It must be H.264 or HEVC.
- The short side must be at least 480 pixels, and the video can be at most 7680 x 4320.
- It needs at least 2 frames and a frame rate of at least 1 fps.
- The frame limit depends on the output resolution.

To send the video, the node uploads it to Griptape Cloud as a short-lived public link and deletes it after the run. If the video is a link to another website, Griptape Cloud downloads it first, which can take up to about 2.5 minutes for a large file. A local file starts faster.

The node runs through the Griptape model proxy and needs a Griptape Cloud API key. Switch `api_key_provider` to **Customer** to use your own LTX Studio key instead, stored in the `LTX_API_KEY` secret. `timeout` sets how long the node waits, in seconds, and defaults to 1200. 0 means no limit. If the wait ends first, the Refresh button in the Status group retrieves a finished result.

## Parameters

### resolution

The output size: `1080p`, `4k` (the default) or `8k`. Higher resolutions allow fewer input frames.

| Output | Maximum input frames |
| --- | --- |
| `1080p` | 2401 |
| `4k` | 601 |
| `8k` | 241 |

Billing is per second of input video, by output resolution.

### direct_download

Downloads the result straight from LTX to this machine instead of through Griptape Cloud hosting. Use it for very large outputs. Griptape Cloud does not keep the file in this mode, so Refresh fetches it from LTX again and fails once LTX has removed it.
