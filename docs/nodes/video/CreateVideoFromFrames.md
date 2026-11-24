---
description: Combines image frames or a numbered image sequence into an MP4, MOV or GIF.
when_to_use: Use it when you have a folder or list of images, or a rendered frame sequence, and want a video or animated GIF from them.
---

## When to use it

You have frames from a render, an [Extract Frames](ExtractFrames.md) run or an image generator, and you want to play them back. Connect them to `frames_input`, set `frame_rate`, and pick a format. You can also add an audio track to MP4 and MOV output.

Only `png`, `jpg`, `jpeg` and `webp` files are used. Empty or corrupt images are skipped.

## Parameters

### frames_input

Accepts any of these:

- A sequence, such as the `sequence` output of Extract Frames. Its order is kept.
- A pattern such as `frame.####.png`. Matching files are used in number order.
- A folder. Its images are used in filename order.
- A list of image paths or image artifacts, used in filename order.
- A single image.

Use the file browser to pick a folder or a sequence.

### ordering_mode

`Sequential` plays the frames in the order found. `Respect frame numbers` reads the number in each filename and repeats the previous frame to fill any gap. For example, frames 1, 2 and 5 play as 1, 2, 2, 2, 5.

> [!NOTE]
> `Respect frame numbers` is applied to every input type, including sequences and patterns.

### format

`mp4` and `mov` are encoded as H.264. `gif` uses a generated palette and hides `processing_speed` and the video output. The result then appears on `image`.

### processing_speed

Trades encode time against quality: `fast`, `balanced` or `quality`. It applies to MP4 and MOV only.

### audio

Optional. For MP4 and MOV, the audio is added as AAC and the video ends when the shorter of the two ends. GIFs have no audio.

### image

The GIF result. It is only shown when `format` is `gif`.
