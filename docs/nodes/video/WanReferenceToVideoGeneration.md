---
description: Generates a new video that features the subjects from up to three reference videos, guided by a prompt.
when_to_use: Use it when you want to put characters from existing clips into a new scene described by a prompt.
---

## When to use it

You have clips of a character, or of up to three characters, and you want them in a new scene. The model takes the subject and voice from each reference video. The prompt describes the new scene.

The node runs through Griptape Cloud and uses your Griptape key. Turn on the API Key Provider option to use your own Alibaba Cloud Model Studio key instead. The reference videos and audio are sent inline in the request as base64 data, so nothing is uploaded, and large clips make a large request. The node waits for the result. Use `timeout` (600 seconds by default) to change how long it waits. The Refresh button fetches a result after a timeout.

## Parameters

### prompt

Required, up to 1500 characters. Refer to the subjects in the reference videos as `character1`, `character2` and `character3`, in the order of `reference_video_1`, `reference_video_2` and `reference_video_3`. For example:

```text
character1 is happily watching a movie on the sofa
```

### reference_video_1 / reference_video_2 / reference_video_3

`reference_video_1` is required. The other two are optional. Videos are MP4 or MOV, 2 to 30 seconds, up to 100 MB.

### audio and input_audio

`audio` is on by default and asks the model to generate sound. To use your own audio instead, turn `audio` off. `input_audio` then appears. It takes WAV or MP3, 3 to 30 seconds, up to 15 MB. Audio longer than the video is cut off. Shorter audio leaves the rest of the video silent.

### size

The size is written as `width*height`. The 1080p tier is 1920*1080, 1080*1920, 1440*1440, 1632*1248 and 1248*1632. The 720p tier is 1280*720, 720*1280, 960*960, 1088*832 and 832*1088. The pairs cover 16:9, 9:16, 1:1, 4:3 and 3:4.

### duration

5 or 10 seconds.

### shot_type

`single` is one continuous shot. `multi` switches between shots.
