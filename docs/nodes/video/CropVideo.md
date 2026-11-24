---
description: Crops a video to a size and position you pick by dragging in an editor, by preset, or by numbers.
when_to_use: Use it when you want to cut a video down to a smaller frame, such as a square or a 16:9 region, without scaling it.
---

## When to use it

You need only part of the frame. Connect a video, then drag the crop box in the editor on the node, or set the size and position with the controls under `crop_coordinates`. The editor and the controls stay in sync.

Cropping keeps the pixels as they are. It does not scale. To change the resolution, use [Resize Video](ResizeVideo.md) afterwards.

## Parameters

### crop_size

The size of the crop box in pixels. The presets (480p up to 2160p, Square, Portrait, 4:3) set a fixed width and height. They crop to that size, they do not resize to it. Choose **Custom** to type `custom_width` and `custom_height`.

If the size is larger than the video, it is cut down to the video size. Widths and heights are also rounded down to an even number, with a minimum of 2.

> [!WARNING]
> With **Custom** and a width or height of 0, no crop is applied and the output is the full frame.

### crop_position

Where the box sits in the frame. Presets such as `center` and `top-left` place it automatically. **Custom** shows `custom_left` and `custom_top`, which are the pixel position of the box's top-left corner. The box is kept inside the frame.

### output_frame_rate

`auto` keeps the frame rate of the input. A specific rate converts the video to it.

### processing_speed

Trades encode time against quality: `fast`, `balanced` or `quality`. The audio track is copied unchanged. If the video has no audio, the output has none.
