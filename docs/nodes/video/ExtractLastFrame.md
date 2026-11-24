---
description: Saves the last frame of a video as a PNG image.
when_to_use: Use it when you want to continue a shot from where a clip ended, such as feeding the last frame of one generated clip into the next.
---

## When to use it

You need a still of the final frame. For example, take the last frame of a generated clip and use it as the start image for the next clip, such as with [Kling Image To Video Generation](KlingImageToVideoGeneration.md), to chain short clips into a longer shot.

The node asks ffmpeg to seek to 3 seconds before the end of the file and keep the last frame it decodes. The ffmpeg run times out after 300 seconds.

## Parameters

### last_frame_image

The extracted frame as an image. The PNG is saved to the project using the `output_file` setting, whose default name is `last_frame.png`.

This node has no `output_frame_rate` or `processing_speed` setting. Those are hidden because it does not encode a video.
