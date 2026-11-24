---
description: Joins two or more videos end to end into one video, resizing mismatched clips to match the first.
when_to_use: Use it when you want to assemble shots into a sequence, or chain several short generated clips into a longer one.
---

## When to use it

You have clips that should play one after another. For example, join the approved shots of a sequence into one clip for a review, or chain several generated clips into a longer shot. Connect the videos to `video_inputs` in the order they should play. To cut a section out of a single clip, use [Trim Video](TrimVideo.md) or [Split Video](SplitVideo.md).

The node needs at least two videos. Each input is copied to a temporary file first. The size of the first video is the target for the whole sequence. Any later video with a different size is scaled down to fit inside that size with its aspect ratio kept, and padded with black bars. The audio is copied during that resize.

The join ffmpeg run times out after 600 seconds. Each resize times out after 300 seconds.

## Parameters

### video_inputs

Shown as "Videos to Concatenate". Accepts individual videos, lists of videos or file paths.

### output_format

`mp4` (default), `avi`, `mov`, `mkv` or `webm`. Changing it also changes the extension of `output_file`.

### video_codec

`libx264` (default), `libx265`, `libvpx-vp9` or `copy`. `copy` skips re-encoding, which is fastest, but needs the clips to share a codec and format. With `copy`, `processing_speed` is not applied.

### audio_codec

`aac` (default), `mp3`, `libmp3lame` or `copy`. Avoid `copy` when the clips' audio formats differ.

### output_frame_rate

`auto` keeps the frame rate of the first video. A specific rate converts the result to it. Use `auto` with `video_codec` set to `copy`, because a frame rate change needs re-encoding.

### processing_speed

Trades encode time against quality: `fast`, `balanced` or `quality`. Not used with `copy`.

### output

The joined video. It is saved to the project using the `output_file` setting, whose default name is `concatenated.mp4`.
