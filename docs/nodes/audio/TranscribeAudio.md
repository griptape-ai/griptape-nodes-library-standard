---
description: Turns speech in an audio file into text with OpenAI's Whisper model, through Griptape Cloud.
when_to_use: Use it when you need a written transcript of a recording, such as a dailies review session or a director's voice notes.
---

## When to use it

You have a recording and want the words as text. For example, a supervisor's voice notes from a review session can become written notes that an [Agent](../agents/Agent.md) sorts by shot.

The node runs through Griptape Cloud and uses your Griptape key. Turn on the API Key Provider option to use your own OpenAI key instead. It waits for the result. Use `timeout` (600 seconds by default) to change how long it waits. The Refresh button fetches a result after a timeout.

## Example

1. Add a **Load Audio** node and pick the recording of a review session.
1. Add **Transcribe Audio** and connect the audio to its `audio` input.
1. Set `language` to `en` if you know it. This improves accuracy and speed.
1. Put the studio's terms in `prompt`, such as "Shots are named SH010, SH020. Departments: roto, paint, comp." The node uses it as context, so shot codes and department names are more likely to be spelled right.
1. Run the workflow. The text is in `output`.

## Parameters

### model

The only choice is `whisper-1`. Workflows saved with `gpt-4o-mini-transcribe` or `gpt-4o-transcribe` load as `whisper-1`.

### audio

mp3, mp4, mpeg, mpga, m4a, wav, webm or flac, up to 25 MB according to the tooltip. The node does not check the size before it sends the request, so a larger file fails at the provider. Split a long recording first, for example with [Trim Audio](TrimAudio.md).

### language

An ISO-639-1 code such as `en`, `es` or `fr`. Leave it empty to let the model detect the language.

### prompt

Optional text that guides the style or gives context, such as names and jargon. Write it in the same language as the audio.

### response_format

`json` returns the text only. `verbose_json` also fills `segments` (text with start and end times), `detected_language` and `duration`, which are hidden until you pick it.

### temperature

From 0 to 1. At 0 (the default) the node leaves it out of the request. Higher values make the output less predictable. Keep it at 0 for transcripts.

### agent

Optional. Connect an Agent and the transcript is added to its conversation, so a later Agent node can answer questions about it without being passed the text again.

### output

The transcribed text.
