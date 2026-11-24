---
description: Animates a person in a still image using the motion from a reference video, with WAN Animate.
when_to_use: Use it when you want a character in an image to move the way a person does in another video.
---

## When to use it

You have a picture of a person and a video of someone moving, and you want the person in the picture to perform that motion. The length of the result follows the reference video.

## What it needs

- A Griptape Nodes license or a Griptape Cloud API key (`GT_CLOUD_API_KEY`). To use your own Alibaba Cloud Model Studio key (`DASHSCOPE_API_KEY`) instead, set **API Key Provider** to **Customer**.
- The node waits for the result. `timeout` sets how long, 600 seconds by default, or 0 for no limit. If a run times out, the **Refresh** button on the node fetches the result later.
- The image and video are uploaded to Griptape Cloud storage as a short-lived public link, which is deleted after the run.
- Inputs outside these limits fail at the provider.
- Image: JPG, JPEG, PNG, BMP or WEBP, 200 to 4096 px, aspect ratio 1:3 to 3:1, up to 5 MB.
- Video: MP4, AVI or MOV, 2 to 30 seconds, 200 to 2048 px, aspect ratio 1:3 to 3:1, up to 200 MB.
- Both should show a single person facing the camera with the full face visible.

## Parameters

### model

`wan2.2-animate-mix` or `wan2.2-animate-move`.

### mode

`wan-std` costs less. `wan-pro` gives higher quality.

### image_url

Required. The image of the person to animate.

### video_url

Required. The reference video that supplies the motion. The node reads its length, rounds it up to whole seconds and sends that as the duration, so there is no duration setting.
