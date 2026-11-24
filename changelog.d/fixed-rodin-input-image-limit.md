- **Breaking:** `Rodin 3D Generation` fails before the run when more than 5 **Input Images** are
  connected. Before, it sent the first 5 and dropped the rest without a message. A saved workflow
  with 6 or more images now stops with an error: remove images until 5 remain.
  [#966](https://github.com/griptape-ai/griptape-nodes-library-standard/issues/966)
