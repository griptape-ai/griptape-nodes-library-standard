---
description: Cuts a video down to a start and end point, given as timecodes or as frame numbers.
when_to_use: Use it when you want to keep one section of a video and drop the rest.
---

## When to use it

You have a longer clip and only need part of it. Set where the part starts and ends, and the node saves it as a new MP4. To cut a video into several pieces at once, use [Split Video](SplitVideo.md).

## Parameters

### trim_by

Choose `timecode` or `frame range`. Only the matching start and end inputs are shown.

### start_timecode / end_timecode

Timecodes use SMPTE format `HH:MM:SS:FF`, where `FF` is the frame number within the second. For example, `00:00:02:12` is 2 seconds and 12 frames in. The node reads the video's frame rate to turn them into seconds, and handles drop-frame timecode.

### start_frame / end_frame

Frame numbers start at 0. `end_frame` is exclusive, so the frame it names is not in the result. `end_frame` must be greater than `start_frame`.

### Cut accuracy

Frame range trims are always re-encoded, so the frame count is exact. Timecode trims that start at 0:00 and are long enough may copy the video without re-encoding. That is fast, but sources with B-frames can end up with a few extra frames after the end point. All other timecode trims are re-encoded.

### End past the video

If the end point is beyond the video's length, the node trims to the end of the video instead of failing.

### video_out

The trimmed video, saved with the name in `output_file` (default `trimmed.mp4`). Check `result_details` after the run for the exact range that was cut.
