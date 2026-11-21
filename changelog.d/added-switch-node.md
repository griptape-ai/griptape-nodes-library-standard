- `Switch` sends the flow down one of many exits: add a case per row, and the first case that
  matches the value sets the exit, with `Default` when none match. `*` and `?` in a case are patterns
  (`*fx` matches `fx` and `cfx`), a bool adds `true` and `false` cases, and `Path` picks a field out
  of a dictionary, JSON or YAML value. Each case has a row in `Values` for the data to pass on, and
  `Result` carries the matching case's data.
  [#1023](https://github.com/griptape-ai/griptape-nodes-library-standard/issues/1023)
