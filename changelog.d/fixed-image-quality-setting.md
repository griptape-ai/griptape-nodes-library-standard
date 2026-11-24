- `quality` sets the JPEG and WEBP compression in `Resize Image`, `Flip Image`, `Rotate Image`,
  `Adjust Image EQ`, `Adjust Image Levels`, `Bloom Effect`, `Gaussian Blur Image`,
  `Grayscale Image`, and `Add Bounding Boxes`. Its default of 95 now applies, so JPEG and WEBP
  outputs left at the default come out larger than before. `Crop Image` writes WEBP lossy at its
  `output_quality`; pick PNG for lossless output.
  [#1064](https://github.com/griptape-ai/griptape-nodes-library-standard/issues/1064)
