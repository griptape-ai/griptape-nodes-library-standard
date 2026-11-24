---
description: Works out a width, height and aspect ratio from a preset or your own values, with optional swap and scale.
when_to_use: Use it when you need matching width, height and ratio values to feed into a generation or resize node.
---

## When to use it

Pick a preset or type a width, height or ratio, and the node keeps the other values in step. Connect `final_width` and `final_height` to a model node or a resize node. Use `final_ratio_str` for models that take a ratio such as `16:9`.

## Parameters

### preset

Pixel presets such as `1152x896 (9:7)` set the exact width and height. Ratio-only presets such as `16:9 landscape` set the ratio and work out the pixels from the size you already have. Typing a width, height or ratio switches the preset to `Custom`.

### width

Changing it updates `ratio_str` and sets `preset` to `Custom`.

### height

Changing it updates `ratio_str` and sets `preset` to `Custom`.

### ratio_str

Written as `width:height` with whole numbers, such as `16:9`. Setting it keeps the larger of the current width and height and recalculates the other from the ratio. It then sets `preset` to `Custom`. Ratio-only presets work the same way.

### upscale_value

Multiplies the width and height. The results are rounded down to whole pixels.

### swap_dimensions

Swaps width and height before `upscale_value` is applied.

### final_ratio_str

The ratio of the final size, reduced to lowest terms. Common ratios such as `16:9` are recognized even when rounding made the pixel values slightly off.

### final_ratio_decimal

Final width divided by final height.
