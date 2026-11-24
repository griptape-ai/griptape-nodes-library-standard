---
description: Shows Gaussian splats in an inline 3D viewer and passes the best-quality one on as an output.
when_to_use: Use it when you want to look at a splat in the workflow, or to pick between resolutions of the same splat.
---

## When to use it

You have splats from [World Labs World Generation](WorldLabsWorldGeneration.md) or a file on disk. Connect them, or pick a local file for each input, and view the result on the node. The viewer has its own resolution dropdown and load button, so you can switch resolutions without running the node again.

You do not need to fill all three inputs. Wire only the ones you have.

## Parameters

### splat_100k, splat_500k, splat_full_res

One splat for each resolution. They can be an artifact from another node or a file you pick. A text value is read as a URL or path. Disconnecting an input clears it.

### splat_out

A single splat for the next node. It is the full-resolution input if set, otherwise the 500k input, otherwise the 100k input. It is empty when no input is set.
