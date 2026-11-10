- The gear button on `Extract Frames` and `Extract PDF Images`' **Directory** parameter now
  connects a `Directory Output Settings` node, which writes into the folder it names. Before, it
  connected a `File Output Settings` node, which names a file, so the run wrote into that file's
  parent folder instead of the folder requested. `Extract Frames` and `Extract PDF Images` running
  in a worker now honor the connected `Directory Output Settings` node too, instead of only the
  **Directory** value.
  [#945](https://github.com/griptape-ai/griptape-nodes-library-standard/issues/945)
  [griptape-nodes-engine#5971](https://github.com/griptape-ai/griptape-nodes-engine/issues/5971)
