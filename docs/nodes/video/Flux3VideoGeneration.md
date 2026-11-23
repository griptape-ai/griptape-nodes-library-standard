---
description: Generates a video with synchronized audio from text, keyframe images or a starting clip using FLUX 3 Video.
when_to_use: Use it when you want a 5 to 20 second clip with sound from a prompt, from one or more still images, or as a continuation of an existing clip.
---

## When to use it

You want a generated clip that includes audio. Pick a `mode`, write a `prompt` that describes the scene, motion and sound, and run the node.

The node runs through the Griptape model proxy and needs a Griptape Cloud API key. Switch `api_key_provider` to **Customer** to use your own Black Forest Labs key instead, stored in the `BFL_API_KEY` secret. `timeout` sets how long the node waits, in seconds, and 0 means no limit. If the wait ends first, the Refresh button in the Status group retrieves a finished result.

## Parameters

### mode

Chooses the input the node uses.

| Mode | Input | Duration |
| --- | --- | --- |
| `t2v` | Prompt only. | `auto`, or 5 to 20 seconds. |
| `i2v` | One to ten images in `keyframes`. | `auto`, or 5 to 20 seconds. |
| `v2v` | A clip in `start_video`, which the video continues from. | `auto`, or 5 to 15 seconds. |

Only the input for the chosen mode is shown. Switching mode updates the `duration` choices.

### keyframes

Images for `i2v`. One image sets the first frame. Two set the first and last frames. With three or more, the images are spaced evenly through the clip. You must pick a number of seconds in `duration` when you connect three or more. `auto` is rejected in that case.

### start_video

The clip to continue in `v2v`. It must be an MP4 of 15 seconds or less. If it is not, the node stops with a message before submitting.

### duration

Seconds, or `auto` to let the model decide.

### aspect_ratio

`auto` lets the model choose from the prompt and any reference media.

### resolution

`hd`, `fhd`, `qhd` or `uhd`.

### draft

Renders a fast preview at a fraction of the cost of a full render. It only works with `resolution` set to `hd`. The node stops with a message for any other combination.

### generate_audio

On by default. Turn it off for a silent clip.

### safety_tolerance

`least restrictive`, `moderate` or `most restrictive`. The default is `moderate`.
