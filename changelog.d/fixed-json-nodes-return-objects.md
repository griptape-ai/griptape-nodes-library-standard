- **Breaking:** To JSON and JSON Input output the parsed object, list or value instead of JSON text,
  and fail with an error when the input text isn't JSON instead of outputting an empty string. Blank
  text gives an empty value. Display JSON outputs an object for malformed JSON it repairs, shows
  plain text as YAML, and fails with an error when the text is neither. A workflow that passed the
  text output on as text needs a To Text node after it.
  [#1071](https://github.com/griptape-ai/griptape-nodes-library-standard/issues/1071)
