---
description: Generates a talking-head video from one image and an audio file with OmniHuman.
when_to_use: Use it when you want a still portrait to speak or sing along to an audio track.
---

## When to use it

You have an image of a person and a recording, and you want a video where the person speaks or sings the audio. The node sends both to the OmniHuman service and saves the video.

## What it needs

- A Griptape Nodes license or a Griptape Cloud API key (`GT_CLOUD_API_KEY`). To use your own ByteDance Seed key (`SEED_API_KEY`) instead, set **API Key Provider** to **Customer**.
- The node waits for the result. `timeout` sets how long, 600 seconds by default, or 0 for no limit. If a run times out, the **Refresh** button on the node fetches the result later.
- Both `image_url` and `audio_url` are required. They are uploaded to Griptape Cloud storage as a short-lived public link, which is deleted after the run, because the service downloads them itself.
- The image can be up to 5 MB and 4096 x 4096 pixels. See `auto_image_resize`.

## Parameters

### model_id

`omnihuman-1-5` is the default. With `omnihuman-1-0` the `prompt`, `seed` and `fast_mode` inputs are hidden, but values you set on them earlier are still sent.

### mask_image_urls

Mask image URLs that tell OmniHuman which subject to animate. Leave it empty to let the node detect the subject (see `auto_detect_masks`). Several masks are sent together.

### auto_detect_masks

On by default, and hidden in the editor. When `mask_image_urls` is empty, the node runs OmniHuman subject detection first and uses the masks it finds. If detection fails, the node carries on without masks.

### auto_image_resize

When on, an image over the size limit is shrunk to fit. When off, it stops the run with an error.

### seed

Use -1 for a random seed. A seed of 0 is not sent.

### fast_mode

Trades some effects for speed.
