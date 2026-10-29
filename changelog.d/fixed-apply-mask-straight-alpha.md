- **Breaking:** Apply Mask outputs straight alpha: it keeps the input colors and only sets the
  mask as alpha, so soft edges no longer go dark in Merge Images. Workflows that worked around the
  old premultiplied output (Unpremultiply after Apply Mask, or Merge Images with `input_alpha` set
  to `premultiplied`) give a different result now; remove the workaround. Nodes that drop alpha
  without compositing, such as Bloom Effect and Color Match, now show the original colors where
  the mask was black instead of black. Save Image to `.jpg` composites over black, so it keeps the
  old look.
  [#888](https://github.com/griptape-ai/griptape-nodes-library-standard/issues/888)
