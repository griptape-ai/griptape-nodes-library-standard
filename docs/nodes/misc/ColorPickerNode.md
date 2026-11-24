---
description: Picks a color with a visual picker and outputs it as text in several formats.
when_to_use: Use it when you need a color value, such as a hex code or an RGB string, for another node.
---

## When to use it

Pick a color by eye and pass the result on as text. Each parameter holds the same kind of color in a different notation, so you can connect the one the next node expects.

## Parameters

### hex and hexa

Hex codes such as `#ffffff`. `hexa` adds an alpha channel, such as `#ffffffff`.

### rgb and rgba

Strings such as `rgb(255, 255, 255)` and `rgba(255, 255, 255, 1.0)`.

### hsl, hsla, hsv and hsva

Strings such as `hsl(0, 0%, 100%)` and `hsva(0, 0%, 100%, 1.0)`. The `a` forms add alpha.

> [!NOTE]
> All eight values default to white. The node source does not keep them in sync with each other. Use the parameter whose format you need and set the color there.
