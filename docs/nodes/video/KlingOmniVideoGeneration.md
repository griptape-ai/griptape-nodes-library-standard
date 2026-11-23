---
description: Generates or edits a video with Kling Omni from a prompt that refers to reference images, videos, elements and voices.
when_to_use: Use it when one clip needs several references, such as a character image, a style video, first and last frames, or when you want to edit an existing video with a prompt.
---

## When to use it

You want to steer a clip with more than text. Connect any mix of first frame, end frame, reference images and a reference video, then name them in the prompt. For example: "keep the face from @image_1".

The node runs through the Griptape model proxy and needs a Griptape Cloud API key. Switch `api_key_provider` to **Customer** to use your own Kling key instead, stored in the `KLING_API_KEY` secret. `timeout` sets how long the node waits, in seconds, and 0 means no limit. If the wait ends first, the Refresh button in the Status group retrieves a finished result.

Reference videos are uploaded to Griptape Cloud as short-lived public links and deleted after the run.

## Parameters

### model_name

The default is `kling-v3-omni`. The other choices are `kling-video-o1`, `kling-4.0` and `kling-4.0-flash`. The controls shown depend on the model.

### prompt

Numbers in references follow input order, with `reference_video` as video 1.

| Models | Reference syntax |
| --- | --- |
| `kling-v3-omni`, `kling-video-o1` | `<<<image_1>>>`, `<<<video_1>>>`, `<<<element_1>>>`. |
| `kling-4.0`, `kling-4.0-flash` | `@first_frame`, `@last_frame`, `@middle_frame_1`, `@image_1`, `@video_1`, `@element_1`, `@voice_1`. |

The prompt is up to 2500 characters on `kling-v3-omni` and `kling-video-o1`. Kling 4.0 accepts up to 26,000 English characters or 11,000 of other text, in proportion when mixed.

### reference_video and video_refer_type

`base` edits the video. `feature` only uses its style or motion. On the older models, editing a base video cannot be combined with a first or end frame. `video_keep_sound` keeps the original audio, and only applies to `base`.

### additional_reference_videos

Kling 4.0 models only. Up to 4 more videos that are used as references and never edited. They are `@video_2`, `@video_3` and so on in list order. They need `reference_video` to be set.

### first_frame_image, end_frame_image and middle_frames

On the older models, an end frame needs a first frame. `middle_frames` is for Kling 4.0 only and sits between the first and end frames. Kling 4.0 Flash takes a first frame only.

### Limits on the number of references

- `kling-v3-omni` and `kling-video-o1`: images and elements together can be up to 7, or up to 4 when a reference video is set. An end frame is not allowed with more than 2 images.
- Kling 4.0: up to 10 images, 7 elements, 3 voices and 5 videos, and up to 15 references in total.

### element_ids and voice_ids

Comma-separated IDs, such as `123,456`. Older models need whole numbers in `element_ids`. `voice_ids` is for Kling 4.0 only, and Kling 4.0 Flash does not take voices.

### mode

`std`, `pro` (the default) or `4k`. Not shown for Kling 4.0. `4k` is not available on `kling-v3-omni` when a reference video is set, and `kling-video-o1` has no `4k`.

### duration

Models differ.

- `kling-v3-omni`: 3 to 15 s. With a reference video in `std` or `pro` mode, 10 s at most.
- `kling-video-o1`: 3 to 10 s. A prompt-only or first-frame-only clip must be 5 or 10 s.
- `kling-4.0`: 3 to 30 s. `kling-4.0-flash`: 3 to 20 s.

### multi_shot and shot_count

Older models only. With `multi_shot` on, you fill in a prompt and a whole-second duration for each shot, up to 6. The shot durations must add up to `duration`, and `prompt` is hidden.

### aspect_ratio

Kling 4.0 models also offer `21:9` and `adaptive`. `adaptive` follows the prompt, then the base video, keyframes or reference video.

### resolution

Kling 4.0 only. Kling 4.0 Flash is 720p only.

### audio

Kling 4.0 only. `native` generates audio with the video. `original` keeps the reference video's audio. It needs a reference video and cannot be used with `voice_ids`.

### kling_video_id

Kling's own ID for the generated video.
