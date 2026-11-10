# Changelog

All notable changes to the Griptape Nodes Standard Library will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/2.0.0/), and each version is
the `library_version` in `griptape_nodes_library.json`.
**Breaking** marks a change that can stop a saved workflow from working without edits.

## [Unreleased]

<!-- Entries go in changelog.d/, one file each. See changelog.d/README.md. -->

## [0.90.2] - 2026-11-09

### Added

- The limits on unpacking an EXR ZIP (frames per archive, total unpacked size and size of one
  frame) are now library settings under `exr_sequence`, with the previous values as the defaults,
  so they can be raised without a code change.
  [#915](https://github.com/griptape-ai/griptape-nodes-library-standard/issues/915)
- New `LTX Video Alpha Generation` node creates a greyscale alpha matte for a video, where white is foreground, black is transparent and greys are partly transparent. Turn on `output RGBA ProRes` to also save the input as a ProRes 4444 `.mov` with the matte as its alpha channel. These files are very large, so the option is off by default. The node also checks the video against LTX's limits before submitting, and waits longer for large linked videos.
  [#915](https://github.com/griptape-ai/griptape-nodes-library-standard/issues/915)
- `LTX Video to Video HDR` gains an `LTX HDR (ACES)` model, now the default for new nodes, with an
  `output colour space` choice (ACEScg, ACEScct, linear sRGB or ACES2065-1) shown only for it. The node
  also outputs a `preview_video` (10-bit HLG MP4), an `EXR Sequence` and its `Colour Space` alongside
  the EXR ZIP. `LTX 2.3 Pro` is deprecated but still works in saved workflows. The node has a
  `Direct Download` option for outputs too large to host, and stops a run before submitting when the
  estimated output is over 4 GiB (about 4.3 GB) and that option is off.
  [#915](https://github.com/griptape-ai/griptape-nodes-library-standard/issues/915)
- The `LTX Video Upscale` node upscales a video to 1080p, 4K or 8K with LTX AI through Griptape
  Cloud. Pick the size in `resolution`. Before it runs, the node checks the input's codec, size,
  frame count and frame rate, and names the limit a video breaks. Its advanced `Direct Download`
  option fetches the output straight from LTX instead of having Griptape Cloud host it.
  [#917](https://github.com/griptape-ai/griptape-nodes-library-standard/issues/917)
- Nodes that support it gain an advanced `Direct Download` option. When on, outputs are downloaded
  straight from the provider to this machine instead of being hosted by Griptape Cloud, which suits
  very large outputs; Griptape Cloud does not keep them. These nodes also stop a run before
  submitting when the estimated output is over about 4.3 GB and the option is off,
  and fetch an output straight from the provider if Griptape Cloud could not host it.
  [#915](https://github.com/griptape-ai/griptape-nodes-library-standard/issues/915)

### Changed

- LTX Video to Video HDR uploads the input video and sends LTX a public URL, falling back to a
  size-checked base64 data URI, and its input frame limits now match the LTX docs (273, 193 and 65
  frames at up to 1080p, 1440p and 2160p).
  [#915](https://github.com/griptape-ai/griptape-nodes-library-standard/issues/915)

## [0.90.1] - 2026-10-29

### Fixed

- **Breaking:** Display Mask, Display Channel, Apply Mask, and Load Image return a fully opaque
  (all-white) mask when `channel` is `alpha` and the image is RGB, instead of the red channel. A
  saved workflow that relied on the red-channel mask should set `channel` to `red`.
  [#887](https://github.com/griptape-ai/griptape-nodes-library-standard/issues/887)
- **Breaking:** Apply Mask outputs straight alpha: it keeps the input colors and only sets the
  mask as alpha, so soft edges no longer go dark in Merge Images. Workflows that worked around the
  old premultiplied output (Unpremultiply after Apply Mask, or Merge Images with `input_alpha` set
  to `premultiplied`) give a different result now; remove the workaround. Nodes that drop alpha
  without compositing, such as Bloom Effect and Color Match, now show the original colors where
  the mask was black instead of black. Save Image to `.jpg` composites over black, so it keeps the
  old look.
  [#888](https://github.com/griptape-ai/griptape-nodes-library-standard/issues/888)

[Unreleased]: https://github.com/griptape-ai/griptape-nodes-library-standard/compare/v0.90.2...HEAD
[0.90.2]: https://github.com/griptape-ai/griptape-nodes-library-standard/compare/v0.90.1...v0.90.2
[0.90.1]: https://github.com/griptape-ai/griptape-nodes-library-standard/compare/v0.90.0...v0.90.1
