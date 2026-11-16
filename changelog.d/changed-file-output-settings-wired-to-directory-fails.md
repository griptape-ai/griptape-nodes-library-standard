- **Breaking:** `Extract Frames` and `Extract PDF Images` now fail when a `File Output Settings`
  node is wired to **Directory** instead of a `Directory Output Settings` node, since it names a
  file, not a folder. Disconnect it and use the gear button to connect a
  `Directory Output Settings` node.
  [#945](https://github.com/griptape-ai/griptape-nodes-library-standard/issues/945)
