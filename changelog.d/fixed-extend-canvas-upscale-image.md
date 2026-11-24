- **Breaking:** Extend Canvas `upscale_factor` scales the image and the custom margins along with
  the canvas, so the image fills the same share of the result at every factor. Saved workflows with
  a factor other than 1 give a different image: the image is larger above 1 and smaller below 1. To
  keep the image at its original size, set the factor to 1, choose `custom`, and set the margins to
  the space you want around the image. `upscale_factor` must be greater than 0, and the
  `canvas_mask` tooltip says the extended area is white, or transparent on a transparent background.
  [#1065](https://github.com/griptape-ai/griptape-nodes-library-standard/issues/1065)
