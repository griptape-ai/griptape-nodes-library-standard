- Image Bash falls back to `HD (16:9)` when `canvas_size` holds an unknown value, such as a stale
  saved one, instead of failing with `KeyError: 'HD'`.
  [#1067](https://github.com/griptape-ai/griptape-nodes-library-standard/issues/1067)
