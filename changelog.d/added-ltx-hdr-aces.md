- `LTX Video to Video HDR` gains an `LTX HDR (ACES)` model, now the default for new nodes, with an
  `output colour space` choice (ACEScg, ACEScct, linear sRGB or ACES2065-1) shown only for it. The node
  also outputs a `preview_video` (10-bit HLG MP4), an `EXR Sequence` and its `Colour Space` alongside
  the EXR ZIP. `LTX 2.3 Pro` is deprecated but still works in saved workflows. The node has a
  `Direct Download` option for outputs too large to host, and stops a run before submitting when the
  estimated output is over 4 GiB (about 4.3 GB) and that option is off.
  [#915](https://github.com/griptape-ai/griptape-nodes-library-standard/issues/915)
