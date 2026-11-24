---
description: Records video and audio from your webcam in the editor and outputs the clip as a video.
when_to_use: Use it when you want to record a clip on the spot instead of loading a file.
---

## When to use it

You want a video from your webcam in the workflow, for example a reference performance or a quick test clip. Record in the node's widget, preview the take, then accept it.

## Parameters

### recording

The capture widget. Record, review the playback, and click Accept to keep the take. Nothing reaches the output until you accept. The node can be run only after a take has been accepted. Otherwise it stops with "No video recorded".

### output_video

The accepted take, saved as a video file. It is updated as soon as you accept, without running the node.

### output_file

The file name used for the saved clip. The default is `webcam_capture.mp4`. The browser decides the recording format. If it records WebM, as Firefox does, the file extension is changed to `.webm` to match.
