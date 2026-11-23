---
description: Changes a video's frame to a new resolution or aspect ratio and fills the new space with generated content using LTX.
when_to_use: Use it when you need a clip in another shape, such as turning a 16:9 video into 9:16 or square, without cropping the subject.
---

## When to use it

You have a clip in one shape and need another. Choose the target `resolution`, run the node, and LTX extends the picture around the source so it fills the new frame. The new areas are generated to match the video.

The input can be up to 60 seconds and 1800 frames, with each side between 64 and 4096 pixels. The node checks these before it submits.

To send the video, the node uploads it to Griptape Cloud as a short-lived public link and deletes it after the run. If that is not possible it embeds the video in the request instead, which only works for files up to about 11 MB.

The node runs through the Griptape model proxy and needs a Griptape Cloud API key. Switch `api_key_provider` to **Customer** to use your own LTX Studio key instead, stored in the `LTX_API_KEY` secret. `timeout` sets how long the node waits, in seconds, and defaults to 1200. 0 means no limit. If the wait ends first, the Refresh button in the Status group retrieves a finished result.

## Parameters

### resolution

The output size. Each option shows its aspect ratio: 1:1, 4:5, 5:4, 9:16 and 16:9, each at a 720 and a 1080 size. The default is `1080x1920`. Billing is per second of input video, at the 720p or 1080p tier of the output.

### placement_mode

Where the source video sits in the new frame.

- `center` fits it at the largest size that fits.
- `anchor` aligns it to an edge or corner chosen in `anchor`.
- `rectangle` places it exactly using `x`, `y`, `width` and `height`.

Placements other than `center` cannot enlarge the video more than 2x, so a small input can be rejected.

### x and y

Rectangle mode only. The left and top edge of the placed video as a fraction of the output size, from 0 up to but not including 1.

### width and height

Rectangle mode only. The size of the placed video as a fraction of the output size, above 0 and up to 1. Set at least one. If you leave the other blank, it follows the video's aspect ratio.
