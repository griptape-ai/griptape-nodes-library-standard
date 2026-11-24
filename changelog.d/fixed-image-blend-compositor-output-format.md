- **Breaking:** `Image Blend Compositor` saves in its `output_format` instead of always PNG. The
  default `auto` keeps the base image's format, or PNG when that format can't be written, so a JPEG
  base image now gives a JPEG output with transparency flattened onto white. Set `output_format` to
  `PNG` to keep the old output.
  [#1060](https://github.com/griptape-ai/griptape-nodes-library-standard/issues/1060)
