---
description: Reads the metadata stored in an image and outputs it as a dictionary and as individual outputs.
when_to_use: Use it when you want to recover information stored in an image, such as a prompt, a seed, camera EXIF or GPS data.
---

## When to use it

Connect an image to see what metadata it carries. PNG images give their text chunks. JPEG, TIFF and MPO images give their EXIF data. An image in any other format returns an empty result.

The node reads as soon as an image is set, without running the workflow.

## Parameters

### metadata

All entries as one dictionary, for passing to other nodes. It is empty if the image has no metadata.

### Per-key outputs

The node also adds one text output for every entry it finds, grouped by the part of the key before the first underscore. For example, `GPS_Latitude` goes in a group called `GPS`. Keys that start with `gtn_` go in `Griptape Nodes`, and keys with no underscore go in `Other`. The outputs change to match each new image.

For JPEG and TIFF, GPS coordinates are shown as decimal degrees. Entries written by [Write Image Metadata](WriteImageMetadataNode.md) are included.
