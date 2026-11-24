- **Breaking:** Image processing nodes with a `quality` setting, such as `Flip Image` and
  `Adjust Image EQ`, apply it when saving JPEG or WebP. Before, the setting was ignored and lossy
  outputs saved at Pillow's default quality, so saved workflows now give larger, sharper files at the
  default of 95. Set `quality` lower for smaller files.
  [#1060](https://github.com/griptape-ai/griptape-nodes-library-standard/issues/1060)
