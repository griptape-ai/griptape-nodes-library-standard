- **Breaking:** Open Ai Prompt offers a fixed `model` list, so it can be added and saved workflows
  load without a working OpenAI key or network. A saved workflow whose model isn't on the list
  keeps it, and the run fails before it starts with an error naming that model; pick a model from
  the list to fix it. Models your license does not permit are flagged, and the run fails if one is
  selected. Requires engine 0.114.0.
  [#529](https://github.com/griptape-ai/griptape-nodes-library-standard/issues/529)
