---
description: Generates a video with Seedance 2.0, from text, first and last frames, or a mix of reference images, videos and audio.
when_to_use: Use it when you want Seedance 2.0 output, including 1080p or 4k, or when you want to steer a clip with reference media.
---

## When to use it

You want a clip from a prompt, from start and end frames, or guided by reference images, videos and audio. Pick the work with `input_mode`. For editing or extending an existing video, use [Seedance 2.5 Video Generation](Seedance25VideoGeneration.md).

## What it needs

- A Griptape Nodes license or a Griptape Cloud API key (`GT_CLOUD_API_KEY`). To use your own ByteDance Seed key (`SEED_API_KEY`) instead, set **API Key Provider** to **Customer**.
- The node waits for the result. `timeout` sets how long, 600 seconds by default, and 0 means no limit. If a run times out, the **Refresh** button on the node fetches the result later.
- Seedance does not accept video as base64. Local reference videos are uploaded to Griptape Cloud storage as a short-lived public link, which is deleted after the run.
- To use a human likeness or voice, pass the media through [Seedance Human Reference Asset](SeedanceHumanReferenceAsset.md). That needs a Griptape Nodes license or Griptape Cloud API key. The node stops with an error if a private-asset reference is connected while **API Key Provider** is set to **Customer**.

## Parameters

### model_id

Seedance 2.0, 2.0 Fast and 2.0 Mini share the same features. Fast and Mini stop at 720p. Only Seedance 2.0 offers 1080p and 4k. Changing the model updates the `resolution` choices.

### input_mode

Each mode accepts only its own inputs. Connecting media that belongs to another mode stops the run with an error.

| Mode | Inputs used |
|------|-------------|
| Text Only | None. Any connected media is an error. |
| First/Last Frame | `first_frame`, `last_frame` |
| Multimodal References | `reference_images`, `reference_video_1` to `reference_video_3`, `reference_audio` |

### prompt

In Multimodal References mode, refer to media in the order you connected it, for example `[Image 1]`, `[Video 2]`, `[Audio 1]`.

### Reference inputs

- Up to 9 images, 3 videos and 3 audio clips. Reference audio can be 2 to 15 seconds each, 15 seconds in total.
- Audio cannot be used alone. Add at least one image or video.
- Videos are filled in order. `reference_video_2` needs `reference_video_1`, and `reference_video_3` needs `reference_video_2`. Each slot appears once the one before it is filled.

### duration

4 to 15 seconds. Choose -1 to let the model decide.

### ratio

`adaptive` lets the model choose the aspect ratio.

### generate_audio

Adds generated audio to the video.
