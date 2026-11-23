---
description: Upgrades a standard-range video to HDR with LTX and returns the result as a ZIP and a sequence of EXR frames.
when_to_use: Use it when you need HDR frames from an SDR clip for grading or compositing in a linear or ACES workflow.
---

## When to use it

You have an SDR clip and want HDR frames to work with in a compositing or grading tool. Connect the video to `video`, choose an output colour space, and run the node. The result is a set of EXR images, not a video file. A small 10-bit HLG MP4 preview is also saved when the generation has one.

The input is checked before it submits. It can be up to 4K, and the frame limit depends on its size.

| Input size | Maximum frames |
| --- | --- |
| Up to 1920 x 1080 | 273 |
| Up to 2560 x 1440 | 193 |
| Up to 3840 x 2160 | 65 |

To send the video, the node uploads it to Griptape Cloud as a short-lived public link and deletes it after the run. If that is not possible it embeds the video in the request instead, which only works for files up to about 11 MB.

The node runs through the Griptape model proxy and needs a Griptape Cloud API key. Switch `api_key_provider` to **Customer** to use your own LTX Studio key instead, stored in the `LTX_API_KEY` secret. A largest-size job takes about ten minutes, so `timeout` defaults to 900 seconds. 0 means no limit. If the wait ends first, the Refresh button in the Status group retrieves a finished result.

## Parameters

### model

`ltx-hdr-2` is the current model. `ltx-2-3-pro` is marked deprecated and goes to the same service. It always returns ACEScg and ignores `output_colorspace`.

### output_colorspace

The colour space of the EXR frames: `acescg` (the default), `acescct`, `srgb_linear` or `aces2065-1`. It is only shown for `ltx-hdr-2`.

### direct_download

Downloads the results straight from LTX to this machine instead of through Griptape Cloud hosting. Use it for very large outputs. Griptape Cloud does not keep the files in this mode, so Refresh fetches them from LTX again and fails once LTX has removed them.

### output_file

Where the ZIP of EXR frames is saved. `project_path` returns the project path of that ZIP after a run.

### exr_sequence

The EXR frames unpacked into a sequence. Connect it to a node that reads EXR sequences. `colour_space` gives the OCIO name of their colour space. If unpacking fails, the ZIP is still saved and the status says so.

### preview_video

The HLG MP4 preview, saved at `preview_video_file`. It is empty when the generation has no preview. A missing preview does not fail the run.
