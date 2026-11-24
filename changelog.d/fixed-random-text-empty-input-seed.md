- **Breaking:** `Random Text` with empty `input_text` and `selection_type` set to `character` or
  `word` follows `seed`, so the same seed gives the same result. Workflows that relied on a new value
  each run should turn on `randomize_seed`.
  [#1074](https://github.com/griptape-ai/griptape-nodes-library-standard/issues/1074)
