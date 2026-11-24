---
description: Changes a video's frame to a new resolution or aspect ratio and fills the new space with generated content using LTX.
when_to_use: Use it when you need a clip in another shape, such as turning a 16:9 video into 9:16 or square, without cropping the subject.
---

## When to use it

You have a clip in one shape and need another, such as 16:9 to 9:16 or square, without cropping the subject. Choose the target `resolution` and LTX extends the picture around the source to fill the new frame. The new areas are generated to match the video.

## What it needs

- A Griptape Nodes license or a Griptape Cloud API key (`GT_CLOUD_API_KEY`). To use your own LTX Studio key (`LTX_API_KEY`) instead, set **API Key Provider** to **Customer**.
- The node waits for the result. `timeout` sets how long, 1200 seconds by default, or 0 for no limit. If a run times out, the **Refresh** button on the node fetches the result later.
- A local file is uploaded to Griptape Cloud as a short-lived public link, which is deleted after the run. A public link to another website is sent as it is. If the upload is not possible, the video is embedded in the request instead. That only works for files up to about 11 MB.
- The input can be up to 60 seconds and 1800 frames, with each side between 64 and 4096 pixels. The node checks this before it submits, when it can read the video.
- Billing is per second of input video, at the 720p or 1080p tier of the output.

## Parameters

### resolution

The output size. Each option shows its aspect ratio: 1:1, 4:5, 5:4, 9:16 and 16:9, each at a 720 and a 1080 size. The default is `1080x1920`.

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
