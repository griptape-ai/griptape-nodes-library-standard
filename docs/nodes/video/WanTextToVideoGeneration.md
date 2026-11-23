---
description: Generates a video from a text prompt with Alibaba's WAN text-to-video models.
when_to_use: Use it when you want a short clip from a description alone, with optional generated or supplied audio.
---

## When to use it

You want a video from a prompt, with no starting image. To start from a picture, use [WAN Image to Video Generation](WanImageToVideoGeneration.md).

The node runs through Griptape Cloud and uses your Griptape key. Turn on the API Key Provider option to use your own Alibaba Cloud Model Studio key instead. It waits for the result and saves the video. Use `timeout` (600 seconds by default) to change how long it waits. The Refresh button fetches a result after a timeout.

## Parameters

### model

The model decides which sizes, durations and extras are available. Changing it updates the `size` and `duration` choices and resets them to the first option if the current value is not supported.

| Model | Sizes | Durations (s) | Audio | `shot_type` |
|-------|-------|---------------|-------|-------------|
| `wan2.6-t2v` (default) | 720P, 1080P | 5, 10, 15 | Yes | Yes |
| `wan2.5-t2v-preview` | 480P, 720P, 1080P | 5, 10 | Yes | No |
| `wan2.2-t2v-plus` | 480P, 720P | 5 | No | No |
| `wanx2.1-t2v-turbo` | 480P, 720P | 5 | No | No |
| `wanx2.1-t2v-plus` | 480P, 720P | 5 | No | No |

### audio and input_audio

`audio` is on by default and asks the model to generate sound. To use your own audio instead, turn `audio` off. `input_audio` then appears. It takes WAV or MP3, 3 to 30 seconds, up to 15 MB. Audio longer than the video is cut off. Shorter audio leaves the rest of the video silent. Only `wan2.6-t2v` and `wan2.5-t2v-preview` use either setting.

### shot_type

`single` is one continuous shot. `multi` switches between shots. It only has an effect when `prompt_extend` is on. `wan2.6-t2v` only.

### prompt_extend

Lets the service rewrite your prompt to improve the result.

### prompt and negative_prompt

Each is limited to 2000 characters.
