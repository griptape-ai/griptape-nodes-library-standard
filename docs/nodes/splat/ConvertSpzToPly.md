---
description: Converts an SPZ Gaussian splat to a PLY file that opens in Blender, Unreal and other 3D tools.
when_to_use: Use it when a splat from World Labs is in SPZ format and the tool you want to use needs PLY.
---

## When to use it

SPZ is a compressed splat format. Many tools cannot open it. This node reads an SPZ file or URL and saves a standard Gaussian splat PLY file.

The conversion happens on your machine. It handles SPZ versions 1 to 4. The axes are converted for PLY, so the result is not the same data layout as the SPZ.

## Parameters

### splat_in

An SPZ splat, such as an output of [World Labs World Generation](WorldLabsWorldGeneration.md), or a local `.spz` file. If the input is already marked as PLY, it is passed through without conversion. A PLY file that is not marked this way fails because the node tries to read it as SPZ.

### output_file

Where the PLY file is saved. The default is `splat_converted.ply`.

### splat_out

The PLY file. Connect it to [Load Splat](LoadSplat.md) or open the saved file in another tool.
