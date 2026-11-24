---
description: Writes custom key-value metadata into an image file.
when_to_use: Use it when you want to embed information such as a prompt, seed or credit inside an image so it travels with the file.
---

## When to use it

Connect an image and a dictionary and get back a copy of the image with those entries embedded. Read them later with [Read Image Metadata](ReadImageMetadataNode.md).

PNG images store the entries as text chunks. JPEG, TIFF and MPO images store them as JSON in the EXIF UserComment field. The node picks the method from the image's format. Other formats fail.

## Parameters

### metadata

A dictionary. It can't be empty. Keys that start with `gtn_` are reserved for metadata the engine adds to its own outputs, and using one fails the run.

For PNG, entries are merged with the text chunks already in the file, and a matching key is replaced.

### output_image

A copy of the input in the same format as the input. The `output_file` default name ends in `.png`, so change it when you work with other formats.
