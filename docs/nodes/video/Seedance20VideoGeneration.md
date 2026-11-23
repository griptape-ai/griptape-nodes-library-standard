---
description: Generates a video with Seedance 2.0, from text, first and last frames, or a mix of reference images, videos and audio.
when_to_use: Use it when you want Seedance 2.0 output, including 1080p or 4k, or when you want to steer a clip with reference media.
---

## When to use it

You want a clip from a prompt, from start and end frames, or guided by reference images, videos and audio. Pick the work with `input_mode`. For the older models, use [Seedance Video Generation](SeedanceVideoGeneration.md). For editing or extending an existing video, use [Seedance 2.5 Video Generation](Seedance25VideoGeneration.md).

The node runs through Griptape Cloud and uses your Griptape key. Turn on the API Key Provider option to use your own ByteDance Seed key instead. It waits for the result and saves the video. Use `timeout` (600 seconds by default) to change how long it waits. The Refresh button fetches a result after a timeout.

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
- Seedance does not accept video as base64. Local videos are uploaded to Griptape Cloud storage so the service can fetch them. The upload is deleted when the run ends.

To use a human likeness or voice, pass the media through [Seedance Human Reference Asset](SeedanceHumanReferenceAsset.md). That needs Griptape authentication and does not work with your own key.

### duration

4 to 15 seconds. Choose -1 to let the model decide.

### ratio

`adaptive` lets the model choose the aspect ratio.

### generate_audio

Adds generated audio to the video.
