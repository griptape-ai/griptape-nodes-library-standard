- `OpenAI Image Generation` offers `GPT Image 2.6 Sunburst` and `GPT Image 2.6 Flare`. They take the
  same sizes, quality, background and reference images as the other GPT Image 2 models, and add a
  `complexity` setting (`low`, `medium` or `high`, default `high`) for how much the model plans before
  it renders. Both can also output an animated GIF: choose `gif` as the output format, then set
  `frame_count` (2 to 16, default 9) and `frame_duration_ms` (20 to 2000, default 200). `size` is the
  size of one frame, and the frames, their size and the number of images have to fit the model's
  pixel budget, so a large size can limit how many frames fit.
  [#912](https://github.com/griptape-ai/griptape-nodes-library-standard/issues/912)
