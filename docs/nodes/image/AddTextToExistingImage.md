---
description: Draws text onto an existing image, with colors, alignment, margin and `{key}` placeholders filled from a dictionary.
when_to_use: Use it when you need a burn-in, caption or label on a frame, such as shot name, version and status on a dailies frame.
---

## When to use it

You have an image and want text on it, such as a title, a caption or a stamp with a filename, timestamp or ID. The text can contain `{key}` placeholders filled from a dictionary, so one template can label many images.

## Example

A burn-in for a dailies frame:

1. Connect a frame from **Load Image** to `input_image` on **Add Text To Existing Image**.
1. Set `text` to `{shot}  |  v{version}  |  {status}`.
1. Set `template_values` to `{"shot": "SH010", "version": "003", "status": "Pending review"}`.
1. Set `text_vertical_alignment` to `bottom`, `text_background` to `#00000080` and `text_color` to `#ffffffff`.
1. Run the node and connect `output` to [Display Image](DisplayImage.md).

## Parameters

### text

The text template. Placeholders are expanded only in the rendered image. The `text` output stays the original template.

A placeholder is `{` plus letters, digits or underscores plus `}`. Anything else, such as a key with a space or dash, is left as typed. If the expanded text is empty or only whitespace, nothing is drawn and the output is the input image saved as a PNG.

### template_values

Optional dictionary used to fill the placeholders in `text`. If a key is missing, the placeholder stays in the text and a line such as `key: status not found in dictionary input` is added to `result_details`. The run still succeeds. If `template_values` is not a dictionary it is ignored, so every placeholder counts as missing.

### text_color, text_background

Colors in `#RRGGBBAA` form, so both support transparency. Values without the alpha part, such as `#ff0000`, and color names are rejected. Defaults are `#ffffffff` for the text and `#000000ff` for the rectangle behind it.

### text_vertical_alignment, text_horizontal_alignment

Vertical is `top` (default), `center` or `bottom`. Horizontal is `left` (default), `center` or `right`.

### margin

Inset from the image edges to the text, default 10. Must be 0 or more or the node fails. The background rectangle extends 4 pixels past the text on each side.

### font_size

Default 36. Must be greater than 0 or the node fails. Size is the only font setting, because the node uses Pillow's built-in font.

### output

The updated image. It is always an RGBA PNG, whatever the input format, written to the location set by `output_file`.

### result_details

Success or failure details, including missing keys.
