- Trim Video and Split Video cut at the requested frame on 30 and 60 fps videos. They were treated
  as drop-frame NTSC, so timecode cuts landed up to two frames off. Frame-range cuts no longer drop
  or add a frame at either end, and a time whose seconds end in .9995 or more no longer cuts almost
  a second early.
  [#1028](https://github.com/griptape-ai/griptape-nodes-library-standard/issues/1028)
  [#1021](https://github.com/griptape-ai/griptape-nodes-library-standard/issues/1021)
