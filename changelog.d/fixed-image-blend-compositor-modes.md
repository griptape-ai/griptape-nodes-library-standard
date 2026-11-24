- `Image Blend Compositor`'s `logical_and`, `logical_or` and `logical_xor` modes threshold both
  images to black and white at mid-gray and combine them, instead of failing or returning the blend
  image. `exclusion` is a true exclusion blend instead of failing.
  [#1060](https://github.com/griptape-ai/griptape-nodes-library-standard/issues/1060)
