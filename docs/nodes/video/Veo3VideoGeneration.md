---
description: Generates one to four videos, with audio, from a prompt and optional images using Google Veo 3.1.
when_to_use: Use it when you want high-quality clips with generated sound, or want to steer a clip with start and end frames or reference images.
---

## When to use it

You want a short video from a prompt, or from a starting image. Veo can generate audio with the video. You can generate up to four variations in one run.

The node runs through Griptape Cloud and uses your Griptape key. Turn on the Credentials Provider option to use your own Google Cloud service account JSON instead. It waits for the result and saves the videos. Use `timeout` (600 seconds by default) to change how long it waits. The Refresh button fetches a result after a timeout.

## Parameters

### model_id

| Model | `start_frame` | `last_frame` | `reference_images` |
|-------|---------------|--------------|--------------------|
| `veo-3.1-generate-001` (default) | Yes | Yes | Yes |
| `veo-3.1-fast-generate-001` | Yes | Yes | No |

Saved workflows that used Veo 3.0 are switched to the matching 3.1 model.

### reference_images and reference_type

Only for `veo-3.1-generate-001`. `asset` keeps objects or characters from the images and accepts up to 3. `style` keeps the artistic style and uses 1. Extra images are ignored. Connecting reference images forces `duration_seconds` to 8. A run with reference images and another duration stops with an error.

### duration_seconds

4, 6 or 8 seconds.

### resolution

The tooltip says 1080p only supports 8 seconds. The node only logs a warning when you pick 1080p with another duration. It does not stop the run, so the provider decides.

### person_generation

`allow_all`, `allow_adult` (default) or `dont_allow`. Controls whether the model generates people.

### sample_count

1 to 4. Each video is saved to its own output: `video_url`, then `video_url_2` to `video_url_4`. Only the outputs that were produced are shown. If one video cannot be saved, its slot stays empty and the others keep their places. Videos filtered by Google's safety checks are logged.

### seed

Use `randomize_seed` to get a new seed on each run.
