---
description: Draws text onto an existing image, with colors, alignment, margin and `{key}` placeholders filled from a dictionary.
when_to_use: Use it when you need a burn-in, caption or label on a frame, such as shot name, version and status on a dailies frame.
---

## When to use it

You have an image and want text on it. Typical uses are titles, captions, and stamps such as filenames, timestamps or IDs. The text can contain `{key}` placeholders that are filled from a separate dictionary, so one template can label many images.

## Example

A burn-in for a dailies frame:

1. Load a rendered frame with **Load Image**.
1. Add **Add Text To Existing Image** and connect the frame to `input_image`.
1. Set `text` to `{shot}  |  v{version}  |  {status}`.
1. Set `template_values` to `{"shot": "SH010", "version": "003", "status": "Pending review"}`.
1. Set `text_vertical_alignment` to `bottom`.
1. Set `text_background` to a semi-transparent black such as `#00000080` and `text_color` to white `#ffffffff`.
1. Run the node and connect `output` to [Display Image](DisplayImage.md).

## Parameters

### text

The text template. Placeholders are expanded only in the rendered image. The `text` output stays the original template.

A placeholder is `{` plus letters, digits or underscores plus `}`. Keys with other characters, such as spaces or dashes, are not expanded. If `text` is empty after expansion, the output is the input image saved as a PNG.

### template_values

Optional dictionary used to expand placeholders in `text`. If a key is missing, the placeholder stays in the rendered text and a message like `key: key not found in dictionary input` is added to `result_details`.

### text_color, text_background

Hexa colors in `#RRGGBBAA` form, so both support transparency. Defaults are `#ffffffff` for the text and `#000000ff` for the rectangle behind it.

### margin

Inset from the image edges, default 10. Must be 0 or more or the node fails.

### font_size

Default 36. Must be greater than 0 or the node fails. The node uses Pillow's built-in font, so size is the only font setting.

### output

The updated image. It is always a PNG, written to the location set by `output_file`.

### result_details

Success or failure details, including missing placeholder keys.

## Notes

The node shows a preview of the text as you change the settings. The output is written when the node runs.
