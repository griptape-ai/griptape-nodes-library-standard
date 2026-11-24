---
description: Cuts a video into several clips from a list of timecode ranges or frame ranges.
when_to_use: Use it when you want to break a long take or an edit into separate shots or segments.
---

## When to use it

You have one video and want several clips from it, for example one clip per shot from a rough assembly, each sent to its own processing branch. To keep a single section, use [Trim Video](TrimVideo.md).

## What it needs

**Timecode mode needs credentials. Frame-range mode needs none.**

- A Griptape Nodes license or a Griptape Cloud API key (`GT_CLOUD_API_KEY`), for timecode mode only. The node sends your text to the `gpt-4.1-mini` model through Griptape Cloud, which rewrites it into a list of ranges. Without a credential it fails with a missing-credential error.

## Parameters

### split_by

`timecode` (default) shows `timecodes`. `frame range` shows `frame_ranges`.

### timecodes

One range per line as `HH:MM:SS:FF-HH:MM:SS:FF`, optionally followed by `|Title`. The default is `00:00:00:00-00:01:00:00`. This input also accepts JSON. A model parses the text, so check the log to see what it returned. Drop frame is handled for 29.97 and 59.94 fps.

### frame_ranges

One range per line as `start-end` or `start-end|Title`, using frame numbers. The default is `0-100`. Unreadable lines, and lines where the end is not after the start, are skipped and noted in the log. Cuts are re-encoded so the frame count is exact.

### split_videos

The clips, as a list of videos, cleared and refilled on every run. Each clip is saved using the `output_file` setting, default `segment.mp4`. Titles do not affect the saved file names.

## Edge cases

- A range that starts after the end of the video is skipped. A range that ends after the end is shortened to fit.
- The node fails if no valid range remains.
- A timecode cut that starts at 0 and is long enough is made by copying the streams, so it can run a few frames past the end on sources with B-frames. Other cuts are re-encoded with libx264 (CRF 18) and AAC audio.
- Each ffmpeg run times out after 300 seconds, so a very long segment can fail.
