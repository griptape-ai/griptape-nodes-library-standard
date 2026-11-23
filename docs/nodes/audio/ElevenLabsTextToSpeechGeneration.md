---
description: Turns text into speech with an Eleven Labs voice, and returns timing for each character.
when_to_use: Use it when you want a voice over or spoken line from text.
---

## When to use it

Write the text, pick a voice and run. The speech is saved as an audio file and returned as `audio_url`. The node also returns `alignment` data with the start and end time of each character.

The node runs through Griptape Cloud and uses your Griptape key. Turn on the API Key Provider option to use your own Eleven Labs key instead. It waits for the result for up to `timeout` seconds (600 by default). If a run times out, the Refresh button in the Status group fetches the result later.

## Parameters

### voice_preset

Choose one of the preset voices, or `Custom...` to enter a voice ID in `custom_voice_id`. The custom voice must be publicly accessible to your Eleven Labs account.

### language_code

Optional ISO 639-1 code such as `es` or `fr`. It hints at the pronunciation.

### seed

Use -1 for a random result. Pick a number from 0 to 10 for repeatable results.

### previous_text / next_text

Optional text that comes before or after this line. Fill them in when you generate a script line by line, so the voice stays consistent across lines.

### stability

`Creative` is more variable and emotional, `Natural` is balanced and `Robust` is the most consistent.

### speed

From 0.7 to 1.2. Values at the extremes can reduce quality.

### alignment / normalized_alignment

Character timing as returned by Eleven Labs. `normalized_alignment` is the normalized version of the same data. Both are empty if the run fails.
