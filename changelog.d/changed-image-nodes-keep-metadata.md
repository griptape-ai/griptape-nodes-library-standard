- Image nodes keep the source image's metadata (EXIF, XMP, text, color profile) in their outputs,
  apply the chosen JPEG/WEBP quality, and read EXR and 16-bit images. Image work now goes through
  the engine's image API, and Pillow is no longer a dependency.
  [#927](https://github.com/griptape-ai/griptape-nodes-library-standard/issues/927)
