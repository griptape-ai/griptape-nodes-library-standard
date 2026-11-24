---
description: Generates or edits a video with Kling Omni from a prompt that refers to reference images, videos, elements and voices.
when_to_use: Use it when one clip needs several references, such as a character image, a style video, first and last frames, or when you want to edit an existing video with a prompt.
---

## When to use it

You want to steer a clip with more than text. Connect any mix of first frame, end frame, reference images and a reference video, then name them in the prompt. For example: "keep the face from @image_1".

## What it needs

- A Griptape Nodes license or a Griptape Cloud API key (`GT_CLOUD_API_KEY`). To use your own Kling AI key (`KLING_API_KEY`) instead, set **API Key Provider** to **Customer**.
- The node waits for the result. `timeout` sets how long, 600 seconds by default. If a run times out, the **Refresh** button on the node fetches the result later.
- Set `timeout` to 0 to wait with no limit.
- The node uploads reference videos to Griptape Cloud as short-lived public links and deletes them after the run.

## Parameters

### model_name

The default is `kling-v3-omni`. The other choices are `kling-video-o1`, `kling-4.0` and `kling-4.0-flash`. The controls shown depend on the model.

### prompt

Reference numbers follow input order, with `reference_video` as video 1.

| Models | Reference syntax |
| --- | --- |
| `kling-v3-omni`, `kling-video-o1` | `<<<image_1>>>`, `<<<video_1>>>`, `<<<element_1>>>`. |
| `kling-4.0`, `kling-4.0-flash` | `@first_frame`, `@last_frame`, `@middle_frame_1`, `@image_1`, `@video_1`, `@element_1`, `@voice_1`. |

The limit is 2500 characters on `kling-v3-omni` and `kling-video-o1`, and 26,000 English characters or 11,000 of other text on Kling 4.0.

### reference_video and video_refer_type

`base` edits the video. `feature` only uses its style or motion. `video_keep_sound` keeps the original audio and only applies to `base`. Kling 4.0 models also take up to 4 more reference videos in `additional_reference_videos`, which are never edited and need `reference_video` to be set.

### first_frame_image, end_frame_image and middle_frames

On the older models, an end frame needs a first frame. `middle_frames` is for Kling 4.0 only and sits between the first and end frames. Kling 4.0 Flash takes a first frame only.

### element_ids and voice_ids

Comma-separated IDs, such as `123,456`. Older models need whole numbers in `element_ids`. `voice_ids` is for Kling 4.0 only, and Kling 4.0 Flash does not take voices.

### mode

`std`, `pro` (the default) or `4k`. Not shown for Kling 4.0. `4k` is not offered on `kling-video-o1`.

### duration

- `kling-v3-omni`: 3 to 15 s, or 10 s at most with a reference video in `std` or `pro` mode.
- `kling-video-o1`: 3 to 10 s. A prompt-only or first-frame-only clip must be 5 or 10 s.
- `kling-4.0`: 3 to 30 s. `kling-4.0-flash`: 3 to 20 s.

### multi_shot and shot_count

Older models only. Up to 6 shots, each with its own prompt and a whole-second duration. The durations must add up to `duration`, and `prompt` is hidden.

### aspect_ratio, resolution and audio

Kling 4.0 models only, except as noted.

- `aspect_ratio` also offers `21:9` and `adaptive`. `adaptive` follows the prompt, then the base video, keyframes or reference video.
- `resolution` on Kling 4.0 Flash is 720p only.
- `audio` `native` generates audio with the video. `original` keeps the reference video's audio, needs a reference video and cannot be used with `voice_ids`.

### kling_video_id

Kling's own ID for the generated video.

## Edge cases

- Reference limits on `kling-v3-omni` and `kling-video-o1`: images and elements together up to 7, or 4 with a reference video. An end frame is not allowed with more than 2 images.
- Reference limits on Kling 4.0: up to 10 images, 7 elements, 3 voices and 5 videos, 15 in total.
- On the older models, a base reference video cannot be combined with a first or end frame.
- `kling-v3-omni` has no `4k` mode when a reference video is set.
