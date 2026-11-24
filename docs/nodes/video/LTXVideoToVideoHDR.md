---
description: Upgrades a standard-range video to HDR with LTX and returns the result as a ZIP and a sequence of EXR frames.
when_to_use: Use it when you need HDR frames from an SDR clip for grading or compositing in a linear or ACES workflow.
---

## When to use it

You have an SDR clip and want HDR frames for grading or compositing in a linear or ACES workflow. The result is a set of EXR images, not a video file.

## What it needs

- A Griptape Nodes license or a Griptape Cloud API key (`GT_CLOUD_API_KEY`). To use your own LTX Studio key (`LTX_API_KEY`) instead, set **API Key Provider** to **Customer**.
- The node waits for the result. `timeout` sets how long, 900 seconds by default, or 0 for no limit. If a run times out, the **Refresh** button on the node fetches the result later.
- A local video is uploaded to Griptape Cloud as a short-lived public link, which is deleted after the run. A video that is already a public https link is sent as is. If the upload is not possible, the video is embedded in the request instead. That only works for files up to about 11 MB.
- The input can be up to 4K. A video over the frame limit stops the run with a message.

| Input size | Maximum frames |
| --- | --- |
| Up to 1920 x 1080 | 273 |
| Up to 2560 x 1440 | 193 |
| Up to 3840 x 2160 | 65 |

## Parameters

### model

`ltx-hdr-2` is the current model. `ltx-2-3-pro` is deprecated, always returns ACEScg and ignores `output_colorspace`.

### output_colorspace

The colour space of the EXR frames: `acescg` (the default), `acescct`, `srgb_linear` or `aces2065-1`. Shown for `ltx-hdr-2` only.

### direct_download

Downloads the results straight from LTX instead of through Griptape Cloud hosting. Use it for very large outputs. Griptape Cloud does not keep the files in this mode, so **Refresh** fails once LTX has removed them.

### output_file and exr_sequence

`output_file` is where the ZIP of EXR frames is saved. `exr_sequence` is the same frames unpacked into a sequence for a node that reads EXR sequences. `colour_space` gives their OCIO name.

### preview_video

A 10-bit HLG MP4 preview, saved at `preview_video_file`. Empty when the generation has no preview. A missing preview does not fail the run.

## Edge cases

- If unpacking fails, the ZIP is still saved and the status says so.
