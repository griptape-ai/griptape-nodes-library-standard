---
description: Matches the colors of a video to a reference image using the color-matcher library.
when_to_use: Use it when you want a clip to take on the color palette of a still, such as a grade reference frame or a hero shot from another sequence.
---

## When to use it

You have a reference still and want a clip to look like it. For example, take one graded frame from a hero shot as `reference_image` and match the rest of the sequence to it. For one image instead of a video, use [Color Match](../image/ColorMatch.md). For manual corrections, see [Adjust Video EQ](AdjustVideoEQ.md) and [Add Color Curves](AddColorCurves.md).

Both `reference_image` and `target_video` are needed. If either is empty, validation fails with "Connect a video to 'target_video'." or "Connect an image to 'reference_image'." and the run does not start.

The output is re-encoded as H.264. Each ffmpeg step times out after 600 seconds.

## Parameters

### reference_image

The image whose colors are copied.

### target_video

The video to change.

### transfer_method

`ffmpeg-haldclut` (default) runs the color match once on a lookup table image and applies it to the whole video in a single ffmpeg pass. This is the faster method. The audio is copied unchanged.

`frame-by-frame` extracts every frame, color-matches each one, and reassembles the video. It is slower and uses more disk and memory. The audio is re-encoded to AAC at 192 kbps. The progress bar shows only for this method.

### transfer_algorithm

`mkl` (default), `hm`, `reinhard`, `mvgd`, `hm-mvgd-hm` or `hm-mkl-hm`. These are the algorithms of the color-matcher library: Monge-Kantorovich Linearization, Histogram Matching, Reinhard, Multi-Variate Gaussian Distribution, and two compound methods that run histogram matching before and after another step.

### strength

From 0.0 to 10.0, default 1.0. At 0 nothing changes. At 1 the full transfer is applied. Values above 1 exaggerate the effect.
