- `Resize Image` and `Resize Video` have a `divisible by` resize mode that snaps the output width and
  height to multiples of a `divisor`, shown in the live preview, for models that only accept such
  sizes. Set `rounding`, `preserve_aspect_ratio`, and `anchor_dimension` to control how; `Resize
  Video` takes even divisors only.
  https://github.com/griptape-ai/griptape-nodes-library-standard/issues/987
