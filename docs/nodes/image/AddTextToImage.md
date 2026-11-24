---
description: Creates a new image of a chosen size and background color with text drawn in the center.
when_to_use: Use it when you need a quick title card, label or placeholder image made from text.
---

## When to use it

You need an image that is just text on a flat color, such as a title card or a labeled placeholder. Set the size, colors and text and the node produces a PNG. To draw text on an image you already have, use [Add Text to Existing Image](AddTextToExistingImage.md).

## Parameters

### text

Drawn once, centered in the image. The text is not wrapped, so long lines run past the edges. Use line breaks to split it yourself. If the text is empty or only spaces, you get the plain background.

### font_size

In points, and must be above 0. The node draws with a built-in default font. You can't choose another.

### background_color

A hex color. The result is a solid, opaque image, so any alpha in the color is ignored. The same applies to `text_color`.

### width and height

In pixels, and both must be above 0. The default is 512 by 512.
