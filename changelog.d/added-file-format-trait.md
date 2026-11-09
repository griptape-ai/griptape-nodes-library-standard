- Output format dropdowns on the image, video, audio, and 3D generation and processing nodes now carry
  the engine's `FileFormat` trait, so a host running the workflow (such as Nuke) can hide formats it
  cannot open. Needs engine 0.107.0 or later.
  [griptape-nodes-engine#6193](https://github.com/griptape-ai/griptape-nodes-engine/issues/6193)
