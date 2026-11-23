---
description: Generates a talking-head video from one image and an audio file with OmniHuman.
when_to_use: Use it when you want a still portrait to speak or sing along to an audio track.
---

## When to use it

You have an image of a person and a recording, and you want a video where the person performs the audio. The node sends both to the OmniHuman service, waits for the result and saves the video.

The node runs through Griptape Cloud and uses your Griptape key. Turn on the API Key Provider option to use your own ByteDance Seed key instead. Generation is not instant. Use `timeout` (600 seconds by default) to change how long the node waits. The Refresh button fetches the result after a timeout.

The service downloads the image and audio itself, so both are uploaded to Griptape Cloud storage for a temporary public URL. The uploads are deleted when the run ends.

## Parameters

### model_id

`omnihuman-1-5` is the default. With `omnihuman-1-0` the `prompt`, `seed` and `fast_mode` inputs are hidden. Values you set on them earlier are still sent.

### mask_image_urls

A list of mask image URLs that tell OmniHuman which subject to animate. Leave it empty to let the node detect the subject (see `auto_detect_masks`). If more than one mask is given, they are sent together.

### auto_detect_masks

On by default, and hidden in the editor. When `mask_image_urls` is empty, the node calls the OmniHuman subject-detection model first and uses the masks it finds. If detection fails, the node carries on without masks.

### auto_image_resize

OmniHuman accepts images up to 5 MB and 4096 x 4096 pixels. When this is on, a larger image is shrunk to fit. When it is off, a larger image stops the run with an error.

### seed

Use -1 for a random seed. A seed of 0 is not sent.

### fast_mode

Trades some effects for speed.
