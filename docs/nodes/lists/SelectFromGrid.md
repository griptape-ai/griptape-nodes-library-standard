---
description: Shows a list as a grid of thumbnails or text cells and outputs the items you click.
when_to_use: Use it when you want to pick by eye which images, videos, clips or text items continue through the workflow.
---

## When to use it

You have a list of results, such as a batch of generated images, and only some should go on. Connect the list, run the node, then click the items you want in the grid.

The selection comes out in the same form as the input items, so images stay images.

## What the grid shows

| Input item | Grid cell |
| ---------- | --------- |
| Image artifact, or text ending in `.png`, `.jpg`, `.jpeg`, `.gif`, `.webp`, `.bmp`, `.svg` | Image |
| Video artifact, or text ending in `.mp4`, `.webm`, `.mov`, `.avi`, `.mkv` | Video |
| Audio artifact, or text ending in `.mp3`, `.wav`, `.ogg`, `.flac`, `.aac`, `.m4a` | Audio |
| Other text | Text |
| Dictionary | The dictionary as JSON |

A dictionary with a `value` key is shown as that value. Add a `label` key to set the cell's label.

Images show right away and are upgraded to engine previews when the node runs. Run the node once so the grid fills in.

## Parameters

### multi_select

On by default. When on, `selected_items` is the output and returns a list. When off, only one item can be selected, `selected_item` replaces it and returns that single item, or nothing if none is selected.

Switching this setting clears the current selection.

### selected_items

The chosen items as a list. Visible only when `multi_select` is on.

### selected_item

The single chosen item. Visible only when `multi_select` is off.

> [!NOTE]
> The selection is kept when the input list changes only if the new list has the same length. Otherwise it is cleared.
