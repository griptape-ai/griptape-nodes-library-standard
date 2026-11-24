- Resolve Macro Path and Resolve Macro Paths return URLs and filenames with braces unchanged, so a
  URL is no longer turned into a bogus local path and a filename like `notes {draft}.txt` no longer
  fails the run.
  [#1043](https://github.com/griptape-ai/griptape-nodes-library-standard/issues/1043)
