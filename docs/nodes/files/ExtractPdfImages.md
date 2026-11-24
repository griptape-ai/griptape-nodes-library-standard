---
description: Saves the images embedded in a PDF, such as scanned pages, to a folder, with optional rotation and format conversion.
when_to_use: Use it when you need the pictures inside a PDF as image files, for example the pages of a scanned document.
---

## When to use it

A scanned PDF stores each page as an image. This node saves those images to a folder and returns them as a list you can send to image nodes. Images keep the resolution they have in the PDF. Nothing is resized.

A PDF with only text and vector graphics has no embedded images, and the node fails with that message. To read its text, use [LoadPdf](LoadPdf.md).

## Parameters

### path

The PDF file. Use the file picker or connect a path.

### password

Only needed for a password-protected PDF. If the PDF is protected and the password is missing or wrong, the node fails.

### rotation_preview

A viewer that shows the pages turned by `rotation`, so you can check the result before extracting.

### rotation

Turns every extracted image: 90 degrees clockwise, 180 degrees or 90 degrees counter-clockwise. Use it when scans are stored sideways. If a page is already rotated inside the PDF, that rotation is added on top.

### file_format

`original` keeps the format stored in the PDF. JPEG and PNG images are saved unchanged, so a JPEG scan is not re-encoded unless you rotate it. Images stored in another format, such as JPEG 2000, are saved as PNG. Choose `png`, `jpg` or `webp` to convert everything. JPG has no transparency, so transparent areas are filled with white.

### directory

The folder to save into. A plain name such as `extracted` is placed and versioned by the project's output folder setting. An absolute path is used as given. Click the gear to connect a [DirectoryOutputSettings](../filesystem/DirectoryOutputSettings.md) node.

> [!WARNING]
> Do not connect a File Output Settings node to `directory`. It names a file, not a folder, and the node fails.

### file_prefix

Start of each file name. Files are numbered in order, so `page` gives `page_001.png`, `page_002.png`.

### number_padding

How many digits the number has, from 1 to 8. A page with several images produces several files, so the numbers count images, not pages.

### images

The saved images in page order, as a list.

### output_directory

The folder the images were saved in.
