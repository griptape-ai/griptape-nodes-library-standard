- `Add Overlay` accepts a still image as the base `video` input, so a video overlay can be
  composited onto a picture. The result takes the overlay video's length and frame rate. An image
  overlay on a video base already worked and still runs for the length of the video. Connecting
  two images is an error, because the result would have no length.
  [#985](https://github.com/griptape-ai/griptape-nodes-library-standard/issues/985)
