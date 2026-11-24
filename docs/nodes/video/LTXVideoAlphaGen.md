---
description: Generates a greyscale alpha matte for a video with LTX, with an optional ProRes 4444 copy that carries the matte as its alpha channel.
when_to_use: Use it when you need to cut the subject out of a video without a green screen, as a matte or as a transparent clip.
---

## When to use it

You want a moving matte for a clip without a green screen. Connect the video to `video` and run the node. `mask_video` is a greyscale video where white is the subject, black is transparent and grey is partly transparent. Turn on `output_rgba_prores` to also get the original footage with that matte as its alpha.

## What it needs

- A Griptape Nodes license or a Griptape Cloud API key (`GT_CLOUD_API_KEY`). To use your own LTX Studio key (`LTX_API_KEY`) instead, set **API Key Provider** to **Customer**.
- The node waits for the result. `timeout` sets how long, 600 seconds by default, or 0 for no limit. If a run times out, the **Refresh** button on the node fetches the result later.
- Input limits: up to 3,072 frames, a long side of up to 3840 pixels, a short side of up to 2160 pixels, an aspect ratio no wider than 32:9, and 200 MB. The node checks the frames, sides and aspect ratio before it submits, when it can read the video. It does not check the 200 MB limit.
- Billing is per second of video, by its short side. Hover over `video` in the editor to see the rates.
- A local file is uploaded to Griptape Cloud as a short-lived public link, which is deleted after the run. A public link to another website is sent as it is, and Griptape Cloud downloads it first, which can take about 2.5 minutes for a large file. A local file starts faster. If the upload is not possible, the video is embedded in the request instead, which only works for files up to about 11 MB.

## Parameters

### output_rgba_prores

Also saves the input as a ProRes 4444 `.mov` with the matte as straight alpha. The files are very large, about 42 MB per second at 1080p and 172 MB per second at 4K. `rgba_video` and `rgba_video_file` only show when this is on. When it is off, the matte is fetched straight from LTX.

### direct_download

Downloads the results straight from LTX to this machine instead of through Griptape Cloud hosting. Always on when `output_rgba_prores` is off. Griptape Cloud keeps no copy in this mode, so **Refresh** fails once LTX has removed the files.

### mask_video

The matte as an MP4, saved at `output_file`.

### rgba_video

The ProRes 4444 `.mov`, saved at `rgba_video_file`. Empty unless `output_rgba_prores` is on.
