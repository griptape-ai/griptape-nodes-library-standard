- Veo 3 Video Generation refuses to run when `resolution` is 1080p or `reference images` are
  connected and `duration_seconds` is not 8, or when `reference images` are connected with Veo 3.1
  Fast. The 1080p case used to send the request anyway; the reference-image cases failed partway
  through the run instead of before it.
  [#1052](https://github.com/griptape-ai/griptape-nodes-library-standard/issues/1052)
