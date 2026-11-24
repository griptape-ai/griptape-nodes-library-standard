- `Flux 3 Video Edit` and `Flux 3 Video Upscale` stop before the run when the input video's base64-encoded
  data URI is over the 50 MiB request limit (about 37.5 MiB of MP4), instead of sending it.
  https://github.com/griptape-ai/griptape-nodes-library-standard/issues/1052
