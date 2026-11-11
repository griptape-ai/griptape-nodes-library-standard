- `Kling Image to Video Generation` and `Kling Omni Video Generation` no longer drop Kling 4.0-only
  inputs without saying so after switching back to an earlier model. Populated `middle_frames`,
  `voice_ids` and an `audio` of `original` stay visible, and the run is blocked with a message
  saying to clear them or select Kling 4.0.
