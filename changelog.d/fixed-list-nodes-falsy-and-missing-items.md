- `Get Index of Item`, `Get List Contains Item` and `Replace In List` treat `0`, empty text and
  `False` as real values, so searching a list for `0` finds it and replacing with `0` works.
  [#1070](https://github.com/griptape-ai/griptape-nodes-library-standard/issues/1070)
