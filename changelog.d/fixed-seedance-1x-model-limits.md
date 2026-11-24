- `Seedance Video Generation` stops before the run when the model doesn't support the chosen
  `resolution` or `duration`, and names the values it does: the 1.0 models take 5 or 10 seconds,
  and 1.5 Pro takes 480p or 720p. Resolution, ratio, duration, and camera-fixed are sent only as
  request settings, not as flags appended to the prompt text.
  [#1052](https://github.com/griptape-ai/griptape-nodes-library-standard/issues/1052)
