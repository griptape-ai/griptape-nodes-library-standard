---
description: Shows the contents of a list as one typed entry per item, so each item can be viewed or connected on its own.
when_to_use: Use it to inspect a list or to pull individual items out of it, such as the plates in a list of images.
---

## When to use it

You want to see what is in a list, or send one item from it to another node. The node builds a `display_list` entry for each item and sets its type from the item.

## Example

1. Connect a list of rendered frames to `items`.
1. Run the workflow. `display_list` shows one image per list item.
1. Connect any single entry to a node that needs one image.

## Parameters

### items

The list to show. It is also passed through as an output, so the node can sit in the middle of a chain.

### display_list

One entry per item. Each entry is typed from its item: text, integer, float, boolean, dictionary, image, video, audio or 3D. Other values use the `any` type. If the connected list declares an element type (for example `list[Sequence]`), that type is used for every entry.

While `items` is connected, the entries cannot be edited by hand. Entries are removed when the list gets shorter and cleared when it is empty. Connections to an entry are removed if its type changes and the new type no longer fits the target.
