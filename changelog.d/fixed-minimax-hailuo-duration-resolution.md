- `MiniMax Hailuo Video Generation`: choosing a model now selects its default **resolution** when
  the current one isn't offered, so picking Hailuo 2.3 Fast no longer fails on the first run. A
  duration the model doesn't offer, such as 10 s on Hailuo 2.3 Fast, is no longer sent to MiniMax, and
  unsupported durations and resolutions are reported before the run starts.
  [#1042](https://github.com/griptape-ai/griptape-nodes-library-standard/issues/1042)
