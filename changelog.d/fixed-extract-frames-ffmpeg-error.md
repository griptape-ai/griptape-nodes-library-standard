- `Extract Frames` shows ffmpeg's actual error when extraction fails, not just its version banner.
  `Extract Frames` and `Extract Last Frame` use `-fps_mode`, replacing the `-vsync` flag ffmpeg
  deprecated.
  [#1131](https://github.com/griptape-ai/griptape-nodes-library-standard/issues/1131)
