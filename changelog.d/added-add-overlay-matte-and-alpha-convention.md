- `Add Overlay` has an optional `matte_video` input. Its brightness (white is visible, black is
  transparent) replaces the overlay's alpha, so an overlay can be paired with a separate mask video
  or image. It also has an `alpha_convention` option, `straight` (default) or `premultiplied`, for
  overlays whose own alpha is premultiplied, which removes the dark fringe on their soft edges.
  [#985](https://github.com/griptape-ai/griptape-nodes-library-standard/issues/985)
