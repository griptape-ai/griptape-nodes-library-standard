- **Breaking:** Crop Video fails before the run when the crop size is 0, including at its default
  settings, instead of re-encoding the full frame. A `crop_size` that isn't a preset or a
  `WIDTHxHEIGHT` size above 0 fails the same way. A saved workflow that relied on the full frame
  now stops at this node: set `custom_width` and `custom_height` above 0, drag a crop area in the
  editor, or choose a `crop_size` preset.
  [#1041](https://github.com/griptape-ai/griptape-nodes-library-standard/issues/1041)
