---
description: Reads a PDF from disk and returns its text, the text of each page and the page count.
when_to_use: Use it when you want to feed the words of a PDF to an Agent or another text node.
---

## When to use it

You have a PDF report, contract or article and need its text in your workflow. The node shows a page viewer so you can check you picked the right file, then outputs the text. To pull out the images inside a PDF instead, use [ExtractPdfImages](ExtractPdfImages.md).

## Parameters

### path

The PDF file. Use the file picker or connect a path. Only local files are supported.

### password

Only needed for a password-protected PDF. Leave it empty otherwise.

### text

All pages joined into one string, with a blank line between pages.

### pages

A list with the text of each page, in order. Use it when you want to handle pages one at a time.

### page_count

The number of pages.
