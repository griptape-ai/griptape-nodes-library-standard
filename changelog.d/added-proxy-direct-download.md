- Nodes that support it gain an advanced `Direct Download` option. When on, outputs are downloaded
  straight from the provider to this machine instead of being hosted by Griptape Cloud, which suits
  very large outputs; Griptape Cloud does not keep them and the provider's links expire. These nodes
  also stop a run before submitting when the estimated output is over 2 GB and the option is off,
  and fetch an output straight from the provider if Griptape Cloud could not host it.
  [#915](https://github.com/griptape-ai/griptape-nodes-library-standard/issues/915)
