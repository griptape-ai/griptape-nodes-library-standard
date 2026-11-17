- **Breaking:** `Add Overlay` blend modes other than `overlay` now respect the overlay's
  transparency. Before, transparent pixels were blended as black, which produced dark fringes and
  boxes around PNG and alpha-video overlays. A saved workflow that uses a transparent overlay with
  one of those modes now renders the base video through the transparent areas. Overlays without
  transparency look the same as before.
  [#985](https://github.com/griptape-ai/griptape-nodes-library-standard/issues/985)
