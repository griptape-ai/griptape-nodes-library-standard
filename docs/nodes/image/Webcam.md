---
description: Captures still images from your camera into a gallery and outputs the selected one, or all of them.
when_to_use: Use it when you want to photograph something with a webcam and feed the frame into a workflow, such as a reference shot of a prop or a set.
---

## When to use it

You want an image from a camera instead of a file. For example, you can photograph a set piece or a lighting reference during a shoot, then send it to [Describe Image](DescribeImage.md) or [Color Match](ColorMatch.md).

## Example

1. Add a **Webcam** node.
1. Use the capture widget on the node to take one or more snapshots. Each snapshot is added to the gallery.
1. Click a thumbnail in the gallery to choose which capture goes to `image`. The latest capture is selected by default.
1. Connect `image` to the next node, or `images` to a node that takes a list.
1. Run the workflow.

## Parameters

### snapshot

The capture widget. It captures frames and holds the gallery. The browser asks for camera permission when you start the camera. The node has no camera, resolution or auto-capture settings. Those come from the browser and the widget.

### image

The selected image from the gallery. This is the latest capture unless you pick another thumbnail.

### images

All captured images, as a list.

### output_file

Where each capture is saved. The default is `webcam/snapshot.jpg` in the project.

## Notes

- The gallery is stored with the node, so captures stay after you reload the workflow.
- The widget has a control to clear the gallery. This empties both `image` and `images`.
- Running the node with an empty gallery fails with "No images captured". Take a snapshot first.
- If a capture cannot be saved, the node logs a warning and drops that capture.
