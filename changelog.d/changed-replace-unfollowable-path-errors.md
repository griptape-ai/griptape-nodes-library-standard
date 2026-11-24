- **Breaking:** `Json Replace` and `Yaml Replace` fail with an error naming the path when it runs
  into a value that is not a mapping, list or null, such as `a.b` on `a: 1`, or when a part has
  more than one list index, such as `a[0][1]`. They used to pass the input through unchanged, or
  write to a key named `a[0]`. Fix the path, or the input, to match the data.
  [#1033](https://github.com/griptape-ai/griptape-nodes-library-standard/issues/1033)
