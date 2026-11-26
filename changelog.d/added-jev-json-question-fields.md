- The JEV question nodes accept JSON in their question text, `Yes means` and `No means`, Pick One
  option descriptions and Rate levels. A JSON object or array reaches JEV as a structured value, so a
  Pick One option can carry a rubric such as `Lighting: {"what": ..., "not_for": ...}` that sharpens
  the line between options. Plain text and invalid JSON are sent as text, as before.
  [#1026](https://github.com/griptape-ai/griptape-nodes-library-standard/issues/1026)
