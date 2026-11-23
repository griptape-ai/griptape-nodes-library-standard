---
description: Generates a sound effect from a text description with Eleven Labs.
when_to_use: Use it when you want a sound effect such as footsteps, rain or a door slam, without searching a library.
---

## When to use it

Describe the sound, choose a length and run. The result is saved as an audio file and returned as `audio_url`.

The node runs through Griptape Cloud and uses your Griptape key. Turn on the API Key Provider option to use your own Eleven Labs key instead. It waits for the result for up to `timeout` seconds (600 by default). If a run times out, the Refresh button in the Status group fetches the result later.

## Parameters

### loop

Asks for a sound that repeats smoothly, such as ambience or an engine hum.

### sound_duration_seconds

From 0.5 to 30 seconds. The default is 6.

### prompt_influence

From 0.0 to 1.0. Higher values follow the description more closely and leave the model less room to vary. The default is 0.3.
