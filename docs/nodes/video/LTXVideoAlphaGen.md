---
description: Generates a greyscale alpha matte for a video with LTX, with an optional ProRes 4444 copy that carries the matte as its alpha channel.
when_to_use: Use it when you need to cut the subject out of a video without a green screen, as a matte or as a transparent clip.
---

## When to use it

You want a moving matte for a clip. Connect the video to `video` and run the node. `mask_video` is a greyscale video where white is the subject, black is transparent and grey is partly transparent. Turn on `output_rgba_prores` to also get the original footage with that matte as its alpha.

The input is checked before it submits. It can have up to 3,072 frames, a long side of up to 3840 pixels, a short side of up to 2160 pixels, and an aspect ratio no wider than 32:9. The tooltip on `video` also lists a 200 MB limit. Billing is per second of video, by its short side, and the tooltip lists the rates.

To send the video, the node uploads it to Griptape Cloud as a short-lived public link and deletes it after the run. If the video is a link to another website, Griptape Cloud downloads it first, which can take up to about 2.5 minutes for a large file. A local file starts faster.

The node runs through the Griptape model proxy and needs a Griptape Cloud API key. Switch `api_key_provider` to **Customer** to use your own LTX Studio key instead, stored in the `LTX_API_KEY` secret. `timeout` sets how long the node waits, in seconds, and defaults to 600. 0 means no limit. If the wait ends first, the Refresh button in the Status group retrieves a finished result.

## Parameters

### output_rgba_prores

Also saves the input as a ProRes 4444 `.mov` with the matte as straight alpha. The files are very large, about 42 MB per second at 1080p and 172 MB per second at 4K. The node only shows `rgba_video` and `rgba_video_file` when this is on.

When it is off, the ProRes file is never downloaded and the matte is fetched straight from LTX.

### direct_download

Downloads the results straight from LTX to this machine instead of through Griptape Cloud hosting. It is always on when `output_rgba_prores` is off. Griptape Cloud does not keep the files in this mode, so Refresh fetches them from LTX again and fails once LTX has removed them.

### mask_video

The matte as an MP4. It is saved at `output_file`.

### rgba_video

The ProRes 4444 `.mov`. It is empty unless `output_rgba_prores` is on. It is saved at `rgba_video_file`.
