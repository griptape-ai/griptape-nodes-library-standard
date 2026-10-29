- **Breaking:** Apply Mask outputs straight alpha: it keeps the input colors and only sets the
  mask as alpha, so soft edges no longer go dark in Merge Images. Workflows that worked around the
  old premultiplied output (Unpremultiply after Apply Mask, or Merge Images with `input_alpha` set
  to `premultiplied`) give a different result now; remove the workaround.
  [#888](https://github.com/griptape-ai/griptape-nodes-library-standard/issues/888)
