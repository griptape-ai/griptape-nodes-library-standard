- `Add Overlay` has `choke` and `edge_softness` settings that trim a halo off a soft-edged overlay.
  `choke` shrinks the overlay's alpha inward by that many pixels, which drops edge pixels still
  carrying the footage's original background color. `edge_softness` blurs the choked edge so it does
  not look hard. Both default to 0, which leaves the overlay unchanged.
  [#985](https://github.com/griptape-ai/griptape-nodes-library-standard/issues/985)
