---
description: Moves a character from a reference image using the actions in a reference video, with Kling Motion Control.
when_to_use: Use it when you have a picture of a character and a clip of someone performing an action, and you want the character to perform it.
---

## When to use it

You want a character to copy a performance. Connect the character picture to `reference_image` and a clip with the movement to `reference_video`. The result is a new video where the character from the image performs the actions from the clip. A `prompt` is optional and adds guidance.

To give Kling access to the files, the node uploads both to Griptape Cloud as short-lived public links and deletes them after the run.

The node runs through the Griptape model proxy and needs a Griptape Cloud API key. Switch `api_key_provider` to **Customer** to use your own Kling key instead, stored in the `KLING_API_KEY` secret. `timeout` sets how long the node waits, in seconds, and 0 means no limit. If the wait ends first, the Refresh button in the Status group retrieves a finished result.

## Parameters

### reference_image

Required. The character. The tooltip lists JPG, JPEG or PNG, up to 10 MB.

### reference_video

Required. The action to copy. The tooltip lists MP4 or MOV, up to 100 MB.

### character_orientation

Decides whose orientation the result follows.

- `image` follows the picture. The video can be up to 10 seconds.
- `video` follows the clip. The video can be up to 30 seconds. This is the default.

### keep_original_sound

On by default. Keeps the sound of the reference video in the result.

### mode

`std` is more cost-effective. `pro` is higher quality and is the default.

### prompt

Optional. Up to 2500 characters.

### kling_video_id

Kling's ID for the result. Pass it to [Kling Video Extension](KlingVideoExtension.md) to continue the clip.
