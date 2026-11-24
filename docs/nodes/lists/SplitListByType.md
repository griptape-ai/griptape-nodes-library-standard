---
description: Sorts a mixed list into separate lists of images, audio, video, text and everything else.
when_to_use: Use it when a list holds different media types and each type must go to its own input.
---

## When to use it

Generation nodes take images on one input and audio on another. A list built at runtime can hold both. Connect the mixed list here and wire each output to the input it belongs on.

Nothing is dropped. Items that match no type go to `other`.

## Parameters

### images

Image artifacts, including image URL artifacts.

### audio

Audio artifacts and audio URL artifacts.

### video

Video URL artifacts.

### text

Items that are plain text. A text item holding a file path or URL stays in `text`. It is not sorted by its file extension.

### other

Everything else, such as numbers, dictionaries and unrecognized artifacts.
