- **Breaking:** `Extract Frames` now fails when the `File Output Settings` node wired to
  **Directory** has no filename set. Before, it ignored that node and saved to the **Directory**
  value. Set a filename on the `File Output Settings` node, or disconnect it to use **Directory**.
  [griptape-nodes-engine#5971](https://github.com/griptape-ai/griptape-nodes-engine/issues/5971)
