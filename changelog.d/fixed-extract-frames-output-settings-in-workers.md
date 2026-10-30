- `Extract Frames` running in a worker or inside an Isolated Execution group now saves into the
  folder from a `File Output Settings` node wired to **Directory**. A connected `File Output
  Settings` node with no filename set now fails the node instead of silently using **Directory**.
  [griptape-nodes-engine#5971](https://github.com/griptape-ai/griptape-nodes-engine/issues/5971)
