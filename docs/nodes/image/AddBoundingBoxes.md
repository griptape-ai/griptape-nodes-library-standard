---
description: Draws labeled rectangles on an image from bounding box data.
when_to_use: Use it to check detection or region data against a frame, for example to review tracked regions or detected subjects on a plate.
---

## When to use it

You have coordinates for regions in an image and want to see them drawn on it. For example, a detection node returns the position of each character in a plate and you want to check the boxes before using them to drive a mask.

## Example

1. Add a **Load Image** node and pick a frame.
1. Produce box data, for example from a detection node, or write it by hand:

    ```
    [
      {"x": 100, "y": 50, "width": 200, "height": 150, "label": "hero", "confidence": 0.95},
      {"x": 350, "y": 120, "width": 180, "height": 200, "label": "extra", "confidence": 0.87}
    ]
    ```

1. Add **Add Bounding Boxes**. Connect the frame to `input_image` and the box data to `bounding_boxes`.
1. Set `label_key` to `{label}: {confidence}`.
1. Run the workflow. The annotated frame is in `output`.

## Parameters

### input_image

The image to draw on. It is required when the node runs.

### bounding_boxes

One dict or a list of dicts. Each dict needs `x` and `y` (the top-left corner, both 0 or more) and `width` and `height` (both more than 0). Values can be integers or strings that convert to integers. Floats are rejected. Extra keys are allowed and can be used in labels. The node fails validation if the input is missing or any box is invalid, and the message gives the box index.

### box_color

Default `#FF0000`. The outline color.

### line_thickness

1 to 10 pixels, default 2.

### show_labels and label_key

`show_labels` is on by default. `label_key` is a template. Each `{key}` is replaced with that key's value from the box dict. The default is `{x}, {y}, width: {width}, height: {height}`.

A key that is not in the box dict is left as written. If none of the keys in the template can be filled, no label is drawn for that box. An empty template draws no labels.

The label is white text on a black background, placed above the box's top-left corner. If it would run off the top of the image, it is placed just inside the box. The text height is 4% of the image height.

### output_format and output_file

`output_format` defaults to `auto`, which keeps the input image's format. The result is saved to the project location set in `output_file`.

### output

The image with the boxes drawn. The input image is not changed.
