---
description: Cuts audio down to a start and end time, chosen by dragging handles on the waveform.
when_to_use: Use it when you want to keep just one part of a clip.
---

## When to use it

Connect audio, then drag the in and out handles on the waveform to pick what to keep. You can also type the times. The node writes the trimmed clip to a new file and outputs it as `audio_out`.

The trim is sample accurate. The output format comes from the extension of `output_file`, and `.wav` is used if there is none.

## Parameters

### audio

Must be a file or an http(s) URL. Raw audio data with no file behind it is rejected. Save it to a file first with [Save Audio](SaveAudio.md).

### start_time

Where the kept part begins, in seconds. It must be before the end of the clip.

### end_time

Where the kept part ends, in seconds. Use 0 to keep everything after the start. A value above 0 must be greater than `start_time`.

### trim_editor

The waveform widget. Clips longer than 30 minutes get a warning badge and a flat waveform. Playing, marking and trimming still work.

### output_file

The file to save the trimmed audio to. The default is `trimmed.wav`.
