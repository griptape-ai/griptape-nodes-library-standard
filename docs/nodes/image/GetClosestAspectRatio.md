---
description: Picks the aspect ratio from a list that is numerically closest to a target ratio.
when_to_use: Use it when a model only accepts certain aspect ratios and you want the nearest supported one for an arbitrary size.
---

## When to use it

A model may accept only a fixed list of ratios, while your image has an arbitrary shape. Give this node the target ratio and the allowed list and it returns the closest entry. Pair it with [Image Details](ImageDetails.md) to start from an existing image's `ratio_str`.

## Parameters

### aspect_ratio

The target, written as `width:height`. Decimals are allowed.

### aspect_ratios

The candidates, separated by commas, spaces or new lines. The default list covers common ratios from 1:1 to 21:9. Any other text in the field is an error. Decimals are allowed.

### closest_aspect_ratio

The candidate whose width divided by height is nearest to the target. It is returned as you typed it. If two candidates are equally close, the earlier one in the list wins. The output is empty and the node fails if an entry is invalid or its height is zero.
