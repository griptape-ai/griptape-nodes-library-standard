- **Breaking:** `Extract Frames` and `Extract PDF Images`'s **Directory** is now a folder name
  placed by the project's `save_output_directory` situation, or an absolute path used as given.
  Saved workflows using the old default are updated automatically; any other `{...}` template
  value must be re-entered as a folder name. On `Extract Frames`, **Directory** now sits above
  **Output Settings** instead of inside it.
  [#945](https://github.com/griptape-ai/griptape-nodes-library-standard/issues/945)
