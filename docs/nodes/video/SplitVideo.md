---
description: Cuts a video into several clips from a list of timecode ranges or frame ranges.
when_to_use: Use it when you want to break a long take or an edit into separate shots or segments.
---

## When to use it

You have one video and want several clips from it. For example, split a rough assembly into one clip per shot using a list of frame ranges, then send each clip to its own processing branch. To keep a single section, use [Trim Video](TrimVideo.md).

Each range becomes one item in the `split_videos` list. The list is cleared and refilled on every run.

## Parameters

### split_by

`timecode` (default) shows `timecodes`. `frame range` shows `frame_ranges`.

### timecodes

One range per line as `HH:MM:SS:FF-HH:MM:SS:FF`, optionally followed by `|Title`. The default is `00:00:00:00-00:01:00:00`. This input also accepts JSON.

The text is not parsed by fixed rules. The node sends it to the `gpt-4.1-mini` model through Griptape Cloud, which rewrites it into a list of ranges. This needs a Griptape Cloud credential, either your `GT_CLOUD_API_KEY` or a Griptape Nodes license. Without one the node fails with a missing-credential error. Check the log to see what the model returned. Frame numbers are converted using the frame rate the node detects in the video, including drop frame for 29.97 and 59.94 fps.

### frame_ranges

One range per line as `start-end` or `start-end|Title`, using frame numbers. The default is `0-100`. Lines that cannot be read, or where the end is not after the start, are skipped and noted in the log. This mode does not call a model and needs no credential. Frame range cuts are always re-encoded so the frame count is exact.

### split_videos

The clips, as a list of videos. Each clip is saved to the project using the `output_file` setting, whose default name is `segment.mp4`. The titles you give are used for temporary working files only, not for the saved file names.

## Behavior

- A range that starts after the end of the video is skipped. A range that ends after the end of the video is shortened to fit.
- The node fails if no valid range remains.
- A timecode cut that starts at 0 and is long enough is made by copying the streams. Other cuts are re-encoded with libx264 (CRF 18) and AAC audio.
- Each ffmpeg run times out after 300 seconds.
