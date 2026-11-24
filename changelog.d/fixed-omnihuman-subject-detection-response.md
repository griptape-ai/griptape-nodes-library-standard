- OmniHuman Subject Detection reads the detection result whether or not the provider response is
  wrapped, no longer fails with a `TypeError` when the response carries no `resp_data`, and fails
  the node, naming the provider status, when detection did not finish or reported no status. A
  `provider_response` that can't be parsed fails the node with the parse error.
  [#1035](https://github.com/griptape-ai/griptape-nodes-library-standard/issues/1035)
