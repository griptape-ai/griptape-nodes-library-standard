---
description: Shows a video in the node and passes it through unchanged.
when_to_use: Use it when you want to preview a video at a point in a workflow, such as a render between two processing steps.
---

## When to use it

You want to see a video on the canvas without changing it. Connect a video to `video` and the player on the node shows it. The same video comes out of the `video` output, so the node can sit in the middle of a chain. For example, drop one after a [Resize Video](ResizeVideo.md) node to check a shot before it goes on to a save step.

The node does not process the video. It also accepts a video you set directly on the node.
