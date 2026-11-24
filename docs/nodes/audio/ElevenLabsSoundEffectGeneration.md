---
description: Generates a sound effect from a text description with Eleven Labs.
when_to_use: Use it when you want a sound effect such as footsteps, rain or a door slam, without searching a library.
---

## When to use it

Describe the sound, choose a length and run. The result is saved as an audio file and returned as `audio_url`.

## What it needs

- A Griptape Nodes license or a Griptape Cloud API key (`GT_CLOUD_API_KEY`). To use your own ElevenLabs key (`ELEVENLABS_API_KEY`) instead, set **API Key Provider** to **Customer**.
- The node waits for the result. `timeout` sets how long, 600 seconds by default, or 0 for no limit. If a run times out, the **Refresh** button on the node fetches the result later.

## Parameters

### loop

Asks for a sound that repeats smoothly, such as ambience or an engine hum.

### sound_duration_seconds

From 0.5 to 30 seconds. The default is 6.

### prompt_influence

From 0.0 to 1.0. Higher values follow the description more closely and leave the model less room to vary. The default is 0.3.
