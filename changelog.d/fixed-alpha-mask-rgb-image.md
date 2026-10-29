- Display Mask, Display Channel, Apply Mask, and Load Image return a fully opaque (all-white) mask
  when `channel` is `alpha` and the image is RGB, instead of the red channel.
  [#887](https://github.com/griptape-ai/griptape-nodes-library-standard/issues/887)
