---
description: Saves chosen frames from a video as image files, either specific frames or every Nth frame.
when_to_use: Use it when you need still images from a clip, such as a reference frame, a contact sheet source or a frame sequence.
---

## When to use it

You want frames out of a video as PNG, JPG or WebP files. Connect a video to `input_video`, choose which frames you want, and run the node. The frames are written to a folder and returned as a list of paths and as a sequence.

Frame numbers start at 1.

## Parameters

### frame_selection_mode

`list` (shown as **Specific Frames**) extracts the frames named in `input_frame_numbers`. `every_Nth` (shown as **Every N Frames**) extracts frames 1, 1+N, 1+2N and so on across the whole clip. Only the field for the chosen mode is shown.

### input_frame_numbers

Comma-separated frame numbers and inclusive ranges, such as `1,4,5-9,11`. Duplicates are removed and the list is sorted. Entries that are not numbers, or are below 1, are ignored. You can also place markers on the video player instead of typing.

Frames past the end of the clip are skipped with a warning in the log. The node still succeeds.

### every_n

The step for `every_Nth`. It must be 1 or more. The frame count comes from the video's duration and frame rate.

### file_prefix, frame_padding

Files are named `<file_prefix>.<number>.<format>`, with the number zero-padded to `frame_padding` digits. For example `frames.0004.png`. The number is the frame's position in the video, not a counter.

### directory

The folder to save into. A plain name such as `frames` is placed by the project's output directory setting and versioned. An absolute path is used as it is. The folder is created if it does not exist.

### output_frames

Absolute paths of the files that were written, in order.

### sequence

The same frames as one sequence object, with frame numbers, paths and the filename pattern. Connect it to nodes that take a sequence. It can be empty if the sequence could not be built. The status then says so.
