---
description: Generates a new video that features the subjects from up to three reference videos, guided by a prompt.
when_to_use: Use it when you want to put characters from existing clips into a new scene described by a prompt.
---

## When to use it

You have clips of a character, or of up to three characters, and you want them in a new scene described by a prompt. The model takes the subject and voice from each reference video.

## What it needs

- A Griptape Nodes license or a Griptape Cloud API key (`GT_CLOUD_API_KEY`). To use your own Alibaba Cloud Model Studio key (`DASHSCOPE_API_KEY`) instead, set **API Key Provider** to **Customer**.
- The node waits for the result. `timeout` sets how long, 600 seconds by default, or 0 for no limit. If a run times out, the **Refresh** button on the node fetches the result later.
- Nothing is uploaded. The reference videos and audio are sent inline in the request as base64 data, so large clips make a large request.
- Inputs outside these limits fail at the provider. Reference videos are MP4 or MOV, 2 to 30 seconds, up to 100 MB.
- Your own audio can be WAV or MP3, 3 to 30 seconds, up to 15 MB. See `audio and input_audio`.

## Parameters

### prompt

Required, up to 1500 characters. Refer to the subjects in the reference videos as `character1`, `character2` and `character3`, in the order of `reference_video_1`, `reference_video_2` and `reference_video_3`. For example:

```text
character1 is happily watching a movie on the sofa
```

### reference_video_1 / reference_video_2 / reference_video_3

`reference_video_1` is required. The other two are optional.

### audio and input_audio

`audio` is on by default and asks the model to generate sound. To use your own audio instead, turn `audio` off and `input_audio` appears. Audio longer than the video is cut off. Shorter audio leaves the rest of the video silent.

### size

The size is written as `width*height` and covers 16:9, 9:16, 1:1, 4:3 and 3:4.

- 1080p tier: 1920*1080, 1080*1920, 1440*1440, 1632*1248, 1248*1632.
- 720p tier: 1280*720, 720*1280, 960*960, 1088*832, 832*1088.

### duration

5 or 10 seconds.

### shot_type

`single` is one continuous shot. `multi` switches between shots.
