---
description: Collects images into a list, with options to flatten, remove duplicates and remove blank items.
when_to_use: Use it when a node needs a list of images and you have them as separate connections.
---

## When to use it

Connect several images to `items` to build one list of images. Pass the `output` list to nodes that take a list of images.

It works like [CreateList](CreateList.md) but accepts images only.

## Parameters

### flatten_list

If an item is itself a list, its contents are added in its place. This opens up one level only.

### remove_duplicates

Keeps one copy of each repeated item. The original order is not kept.
