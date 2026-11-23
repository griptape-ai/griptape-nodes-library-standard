---
description: Generates, edits or extends a video with Seedance 2.5, using text, frames, or reference images, videos and audio.
when_to_use: Use it when you want to edit or extend an existing clip, work with many references, render a cheap Draft before a 1080p or 4k Final, or generate clips up to 30 seconds.
---

## When to use it

You want one node for the full Seedance 2.5 range: text-to-video, first and last frames, reference-guided generation, editing a video and extending a video. Pick the work with `task`. The node shows only the inputs that task uses and rejects a mismatched request before it is queued.

The node runs through Griptape Cloud and uses your Griptape key. Turn on the API Key Provider option to use your own ByteDance Seed key instead. It waits for the result and saves the video. Use `timeout` (600 seconds by default) to change how long it waits. The Refresh button fetches a result after a timeout.

## Tasks

| Task | Inputs | Locked settings |
|------|--------|-----------------|
| Text to Video | `prompt` | None |
| First/Last Frame | `first_frame`, `last_frame` | `ratio` is `adaptive` |
| Reference to Video | At least one reference image, video or audio | None |
| Video Editing | At least one reference video | `ratio` is `adaptive`, `duration` is -1 |
| Video Extension | At least one reference video | `ratio` is `adaptive` |

For Video Editing and Video Extension the prompt must say what to do. Editing needs one of: edit, add, delete, remove, modify, replace, change. Extension needs extend or continue. The node checks for these words before it runs.

## Parameters

### mode

- Normal renders the video in one step at `resolution`.
- Draft renders a 480p preview.
- Final renders a 1080p or 4k video from a finished Draft.

Each step is billed separately. A Draft ID expires after 14 days.

### draft_task_id

Used in Final mode only. Paste the value from the `draft_task_id_output` of a finished Draft run. It is not the `generation_id`. In Final mode the prompt, task, ratio, duration and audio settings are hidden and not sent.

### final_resolution

1080p or 4k for a Final. 4k jobs can queue for a while.

### resolution

Used in Normal mode. 1080p and 4k output are 10-bit H.265/HEVC. Some players cannot open them. Try VLC or mpv.

### Reference inputs

- Up to 30 images, 10 videos and 10 audio clips. Videos and audio are 2 to 30 seconds each and no more than 30 seconds in total.
- In the prompt, refer to media as `@Image 1`, `@Video 1`, `@Audio 1`, in list order.
- Audio can be used on its own.
- Audio cues in the prompt use `()` for music, `<>` for sound effects, `{}` for dialogue and `【】` for subtitles.
- Seedance does not accept video as base64. Local videos are uploaded to Griptape Cloud storage so the service can fetch them. The upload is deleted when the run ends.

To use a human likeness or voice, pass the media through [Seedance Human Reference Asset](SeedanceHumanReferenceAsset.md). That needs Griptape authentication and does not work with your own key.

### duration

4 to 30 seconds. -1 lets the model choose.

### output_format

`mp4` plays almost everywhere. `mov` keeps more color precision and is recommended for editing and extension. Changing the format also changes the extension of `output_file`, unless you typed your own file name.

### return_last_frame

Also saves the last frame as a PNG on `last_frame_url`. Feed it into the `first_frame` of the next run to continue a sequence.

### draft_task_id_output

Filled in when a Draft finishes. Connect it to `draft_task_id` for the Final run.
