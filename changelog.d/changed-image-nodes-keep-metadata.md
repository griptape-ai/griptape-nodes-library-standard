- Image nodes keep the source image's EXIF, color profile and text metadata in their outputs (XMP
  too when the output is PNG), apply the chosen JPEG/WEBP quality, and read EXR, TGA and 16-bit
  images. Image work now goes through the engine's image API, and Pillow is no longer a dependency.
  [#927](https://github.com/griptape-ai/griptape-nodes-library-standard/issues/927)
