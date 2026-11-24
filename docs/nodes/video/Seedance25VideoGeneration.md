---
description: Generates, edits or extends a video with Seedance 2.5, using text, frames, or reference images, videos and audio.
when_to_use: Use it when you want to edit or extend an existing clip, work with many references, render a cheap Draft before a 1080p or 4k Final, or generate clips up to 30 seconds.
---

## When to use it

You want one node for Seedance 2.5: text-to-video, first and last frames, references, editing and extending a video. Pick the work with `task`.

## What it needs

- A Griptape Nodes license or a Griptape Cloud API key (`GT_CLOUD_API_KEY`). To use your own ByteDance Seed key (`SEED_API_KEY`) instead, set **API Key Provider** to **Customer**.
- The node waits for the result. `timeout` sets how long, 600 seconds by default. If a run times out, the **Refresh** button on the node fetches the result later.
- Set `timeout` to 0 to wait with no limit.
- Seedance does not accept video as base64. Local reference videos are uploaded to Griptape Cloud storage as a short-lived public link, which is deleted after the run.
- Human likeness or voice references must go through [Seedance Human Reference Asset](SeedanceHumanReferenceAsset.md). That needs a Griptape Nodes license or Griptape Cloud API key and does not work with your own key.
- Each Draft and Final step is billed separately.
- Reference limits: up to 30 images, 10 videos and 10 audio clips. The node enforces these counts. The provider limits videos and audio to 2 to 30 seconds each and 30 seconds in total.

## Tasks

| Task | Inputs | Locked settings |
|------|--------|-----------------|
| Text to Video | `prompt` | None |
| First/Last Frame | `first_frame`, `last_frame` | `ratio` is `adaptive` |
| Reference to Video | At least one reference image, video or audio | None |
| Video Editing | At least one reference video | `ratio` is `adaptive`, `duration` is -1 |
| Video Extension | At least one reference video | `ratio` is `adaptive` |

For Video Editing and Video Extension the prompt must say what to do. Editing needs one of: edit, add, delete, remove, modify, replace, change. Extension needs extend or continue.

## Parameters

### mode

- Normal renders the video in one step at `resolution`.
- Draft renders a 480p preview.
- Final renders a 1080p or 4k video from a finished Draft.

Run a Final with the same organization and compatible credentials as the Draft. A Draft ID expires after 14 days. `draft_task_id_output` is filled in when a Draft finishes.

### draft_task_id

Final mode only. Paste the `draft_task_id_output` of a finished Draft run, not its `generation_id`. In Final mode the other settings are hidden and not sent.

### resolution and final_resolution

`resolution` applies in Normal mode, `final_resolution` in Final mode. 1080p and 4k output are 10-bit H.265/HEVC, which some players cannot open. Try VLC or mpv.

### Reference inputs

- In the prompt, refer to media as `@Image 1`, `@Video 1`, `@Audio 1`, in list order.
- Audio can be used on its own.
- Audio cues use `()` for music, `<>` for sound effects, `{}` for dialogue and `【】` for subtitles.

### duration

4 to 30 seconds. -1 lets the model choose.

### output_format

`mov` keeps more color precision and is recommended for editing and extension. Otherwise `mp4` plays almost everywhere. Changing the format also changes the extension of `output_file`, unless you typed your own file name.

### return_last_frame

Also saves the last frame as a PNG on `last_frame_url`. Feed it into the `first_frame` of the next run to continue a sequence.
