---
description: Mixes up to four audio tracks into one file, with volume and pan for each track.
when_to_use: Use it when you want to layer audio, such as a voice over music and sound effects.
---

## When to use it

Connect audio to one to four tracks, set each track's level, and the node mixes them with ffmpeg. The mix is as long as the longest track. Tracks you leave empty are ignored. At least one track is required.

The default volumes suit a layered mix: track 1 at 1.0, track 2 at 0.5, tracks 3 and 4 at 0.3.

## Parameters

### track1_volume, track2_volume, track3_volume, track4_volume

From 0.0 to 1.0.

### track1_pan, track2_pan, track3_pan, track4_pan

From -1.0 (left) to 1.0 (right), with 0 as center. Panning lowers the opposite channel and keeps the nearer one at full level. A pan of 0 leaves the track as it is.

### output_format

Picks the audio codec. Only `mp3` and `wav` set a codec explicitly.

> [!WARNING]
> The mix is built in a temporary `.mp3` file, so `flac`, `aac` and `ogg` do not appear to produce those formats. The saved file name comes from `output_file`, not from this setting.

### quality

Sets the bitrate: `low` is 128 kbps, `medium` 192 kbps and `high` 320 kbps. `lossless` sets no bitrate. It has no practical effect on `wav`.

### output_file

Where the mix is saved. The default is `mix.mp3`.
