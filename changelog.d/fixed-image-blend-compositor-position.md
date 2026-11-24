- `Image Blend Compositor` modes other than `normal` cover the whole base image when the blend
  image is centered. Before, the resized result was offset by the original blend image's centering,
  so part of the base showed through unblended and the rest was cropped.
  [#1060](https://github.com/griptape-ai/griptape-nodes-library-standard/issues/1060)
