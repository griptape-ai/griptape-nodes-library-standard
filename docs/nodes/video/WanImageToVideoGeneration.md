---
description: Generates a video from a starting image and an optional prompt with Alibaba's WAN image-to-video models.
when_to_use: Use it when you want to animate a still image into a short clip.
---

## When to use it

You have an image and want it to move. The image is the starting frame, and the prompt describes the motion and any other elements. To work from text alone, use [WAN Text to Video Generation](WanTextToVideoGeneration.md).

The node runs through Griptape Cloud and uses your Griptape key. Turn on the API Key Provider option to use your own Alibaba Cloud Model Studio key instead. It waits for the result and saves the video. Use `timeout` (600 seconds by default) to change how long it waits. The Refresh button fetches a result after a timeout.

## Parameters

### model

The model decides which resolutions, durations and extras are available. Changing it updates the `resolution` and `duration` choices and resets them to the first option if the current value is not supported.

| Model | Resolutions | Durations (s) | Audio | `shot_type` |
|-------|-------------|---------------|-------|-------------|
| `wan2.6-i2v` (default) | 720P, 1080P | 5, 10, 15 | Yes | Yes |
| `wan2.5-i2v-preview` | 480P, 720P, 1080P | 5, 10 | Yes | No |
| `wan2.2-i2v-flash` | 480P, 720P | 5 | No | No |
| `wan2.2-i2v-plus` | 480P, 1080P | 5 | No | No |
| `wanx2.1-i2v-plus` | 720P | 5 | No | No |
| `wanx2.1-i2v-turbo` | 480P, 720P | 3, 4, 5 | No | No |

### input_image

Required. The tooltip lists JPG, PNG, BMP and WEBP, 360 to 2000 px, up to 10 MB.

### audio and input_audio

`audio` is on by default and asks the model to generate sound. To use your own audio instead, turn `audio` off. `input_audio` then appears. It takes WAV or MP3, 3 to 30 seconds, up to 15 MB. Audio longer than the video is cut off. Shorter audio leaves the rest of the video silent. Only `wan2.6-i2v` and `wan2.5-i2v-preview` use either setting.

### shot_type

`single` is one continuous shot. `multi` switches between shots. It only has an effect when `prompt_extend` is on. `wan2.6-i2v` only.

### prompt_extend

Lets the service rewrite your prompt to improve the result.

### prompt and negative_prompt

`prompt` is optional. Its length limit depends on the model, from 800 to 2000 characters. `negative_prompt` is limited to 500 characters.
