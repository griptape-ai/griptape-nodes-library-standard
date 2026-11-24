- `Audio Details` reads audio with the FFmpeg bundled with the library, so it works without FFmpeg
  installed. When the audio can't be read, the node fails with the reason instead of reporting 0
  and `UNKNOWN` as if the clip were empty.
  [#1058](https://github.com/griptape-ai/griptape-nodes-library-standard/issues/1058)
