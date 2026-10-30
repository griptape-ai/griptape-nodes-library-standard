# Changelog

All notable changes to the Griptape Nodes Standard Library will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/2.0.0/), and each version is
the `library_version` in `griptape_nodes_library.json`.
**Breaking** marks a change that can stop a saved workflow from working without edits.

## [Unreleased]

<!-- Entries go in changelog.d/, one file each. See changelog.d/README.md. -->

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

[Unreleased]: https://github.com/griptape-ai/griptape-nodes-library-standard/compare/v0.90.1...HEAD
[0.90.1]: https://github.com/griptape-ai/griptape-nodes-library-standard/compare/v0.90.0...v0.90.1
