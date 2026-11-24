- `Json Replace` and `Yaml Replace` write the value when the path ends in a list index, such as
  `items[0]`, and when a list index past the end is followed by a key, such as `items[2].title`.
  The list is padded with nulls up to the index.
  [#1033](https://github.com/griptape-ai/griptape-nodes-library-standard/issues/1033)
  [#1081](https://github.com/griptape-ai/griptape-nodes-library-standard/issues/1081)
