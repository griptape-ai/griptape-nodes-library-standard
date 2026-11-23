---
description: Animates a person in a still image using the motion from a reference video, with WAN Animate.
when_to_use: Use it when you want a character in an image to move the way a person does in another video.
---

## When to use it

You have a picture of a person and a video of someone moving, and you want the person in the picture to perform that motion. The length of the result follows the reference video.

The node runs through Griptape Cloud and uses your Griptape key. Turn on the API Key Provider option to use your own Alibaba Cloud Model Studio key instead. The image and video are uploaded to Griptape Cloud storage so the service can fetch them. The uploads are deleted when the run ends. The node waits for the result. Use `timeout` (600 seconds by default) to change how long it waits. The Refresh button fetches a result after a timeout.

## Parameters

### model

`wan2.2-animate-mix` or `wan2.2-animate-move`.

### mode

`wan-std` costs less. `wan-pro` gives higher quality.

### image_url

Required. The image of the person to animate. The model docs list a single person facing the camera, with the full face visible, as the expected content. Images are 200 to 4096 px, aspect ratio 1:3 to 3:1, up to 5 MB.

### video_url

Required. The reference video that supplies the motion. Same expectations for content: one person facing the camera, full face visible. Videos are MP4, AVI or MOV, 2 to 30 seconds, 200 to 2048 px, aspect ratio 1:3 to 3:1, up to 200 MB.

The node reads the video's length and rounds it up to whole seconds. It sends that as the duration, so there is no duration setting.
