- **Breaking:** To JSON and JSON Input output parsed JSON objects, lists, and numbers in place of
  JSON text, repairing malformed JSON such as `[1,2`. Plain text that isn't JSON fails the run with
  an error in place of outputting an empty string, as does JSON after a lead-in sentence unless it
  is in a fenced `json` code block; fix the upstream text or remove the node.
  Display JSON shows repaired JSON as an object and plain text as text or YAML in place of an
  empty string. Sort List sorts a malformed JSON list such as `[3,1,2,` in place of returning an
  empty list.
  [#1071](https://github.com/griptape-ai/griptape-nodes-library-standard/issues/1071)
  [#1045](https://github.com/griptape-ai/griptape-nodes-library-standard/issues/1045)
