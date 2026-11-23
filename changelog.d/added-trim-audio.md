- `Trim Audio` trims audio to a start and end time. Its waveform player has draggable in and out
  markers that set `start_time` and `end_time`, a strip along the bottom that shifts the whole range,
  and plays while you scrub. Mark In and Mark Out buttons (also the I and O keys) set a point at the
  current time. You can also type the times or wire them in. An `end_time` of 0 keeps everything after
  the start. For clips over 30 minutes the waveform is not drawn, and a badge on the player says so.
  https://github.com/griptape-ai/griptape-nodes-library-standard/issues/643
