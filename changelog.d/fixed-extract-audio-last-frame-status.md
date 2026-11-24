- `Extract Audio` and `Extract Last Frame` fill `was_successful` and `result_details` after a run,
  and a failure such as a missing file or a video with no audio goes to a connected failure output
  instead of stopping the workflow.
  [#1068](https://github.com/griptape-ai/griptape-nodes-library-standard/issues/1068)
