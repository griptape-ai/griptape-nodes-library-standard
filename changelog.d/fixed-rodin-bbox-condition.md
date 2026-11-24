- **Breaking:** `Rodin 3D Generation` fails before the run when **Bounding Box** isn't three whole
  numbers from 1 to 2048. Before, decimals were rounded down and other input was dropped without a
  message. A saved workflow with a decimal or malformed box now stops with an error: enter three
  whole numbers such as `100,200,300`.
  [#1049](https://github.com/griptape-ai/griptape-nodes-library-standard/issues/1049)
