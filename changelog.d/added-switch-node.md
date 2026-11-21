- `Switch` sends the flow down one of many exits. Add a case per row, and the first case that matches
  the value, from the top, sets the exit, with `Default` when none match. With `Pattern matching` on,
  a row with `*` or `?` is a pattern, so `*fx` matches `fx` and `cfx`. Case is ignored by default. For
  a dictionary, JSON or YAML value, set `Path` to the field to switch on, such as `department`.
  [#1023](https://github.com/griptape-ai/griptape-nodes-library-standard/issues/1023)
