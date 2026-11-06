- `Summarize Agent Memory`, `Split Video` timecode parsing, and agents connected through
  `Agent To Tool` now check the model against your organization's model policy before calling it,
  and stop when it is not permitted. Before, they called the model without the check.
  [#431](https://github.com/griptape-ai/griptape-nodes-library-standard/issues/431)
