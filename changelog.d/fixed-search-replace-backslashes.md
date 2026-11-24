- **Breaking:** `Search Replace Text` inserts `replacement_text` exactly as typed when `use_regex`
  is off, so Windows paths and other backslashes come through unchanged. A workflow that relied on
  `\n` becoming a newline in plain mode now gets a literal `\n`; turn on `use_regex` for escapes.
  With `use_regex` on, an invalid pattern or group reference fails the node with an error instead of
  returning the input unchanged.
  [#1072](https://github.com/griptape-ai/griptape-nodes-library-standard/issues/1072)
