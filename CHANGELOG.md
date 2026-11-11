# Changelog

All notable changes to the Griptape Nodes Standard Library will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/2.0.0/), and each version is
the `library_version` in `griptape_nodes_library.json`.
**Breaking** marks a change that can stop a saved workflow from working without edits.

## [Unreleased]

<!-- Entries go in changelog.d/, one file each. See changelog.d/README.md. -->

## [0.91.0] - 2026-11-11

### Added

- `Agent`, `Griptape Cloud Prompt`, and `Describe Image` offer Claude Haiku 5.5 through
  Griptape Cloud.
- Extract PDF Images node pulls the images out of a PDF, such as the pages of a scanned document,
  at the resolution they are stored at. It saves them into a versioned folder like
  `extracted_v001/page_001.png`, lets you pick `png`, `jpg`, `webp`, or the stored format, and its
  `rotation` parameter turns every image by 90° or 180° with a live preview for scans stored sideways.
  [#934](https://github.com/griptape-ai/griptape-nodes-library-standard/issues/934)
- Output format dropdowns on the image, video, audio, and 3D generation and processing nodes now carry
  the engine's `FileFormat` trait, so a host running the workflow (such as Nuke) can hide formats it
  cannot open. Needs engine 0.107.0 or later.
  [griptape-nodes-engine#6193](https://github.com/griptape-ai/griptape-nodes-engine/issues/6193)
- Flatten List node turns a list of nested lists into one flat list, keeping item order. `depth`
  limits how many levels it flattens; 0 flattens every level.
  [#936](https://github.com/griptape-ai/griptape-nodes-library-standard/issues/936)
- FLUX 3 Image Generation, FLUX 3 Video Generation, FLUX 3 Video Edit, and FLUX 3 Video Upscale
  nodes run Black Forest Labs' FLUX 3 models (`flux-3-image`, `flux-3-image-klein`,
  `flux-3-video`, `flux-video-edit-v1`, and `flux-video-upscale-v1`).
  [#897](https://github.com/griptape-ai/griptape-nodes-library-standard/issues/897)
- Added interactive previews for some image and video nodes.
- Ask Yes/No, Pick One, and Rate accept an optional Images input, so questions can classify or
  compare images. Context is optional when images are connected.
  [#778](https://github.com/griptape-ai/griptape-nodes-library-standard/issues/778)
- `Kling Text to Video Generation`, `Kling Image to Video Generation` and `Kling Omni Video
  Generation` offer Kling 4.0 and Kling 4.0 Flash: 3-30s (Flash 3-20s) at 720p or 1080p (Flash
  720p), with a 21:9 aspect ratio. Image to Video takes middle keyframes with Kling 4.0, and Omni
  takes voice IDs and an `audio` setting that keeps the reference video's sound.
  https://github.com/griptape-ai/griptape-nodes-library-standard/issues/849
- Rotate Image node turns an image by 90°, 180°, or a custom angle.
  [#906](https://github.com/griptape-ai/griptape-nodes-library-standard/issues/906)
- Eleven Labs Sound Effect Generation has a model dropdown offering Eleven Text to Sound v3
  alongside v2, which stays the default.
  [#894](https://github.com/griptape-ai/griptape-nodes-library-standard/issues/894)

### Changed

- **Breaking:** `Extract Frames` now fails when the `File Output Settings` node wired to
  **Directory** has no filename set. Before, it ignored that node and saved to the **Directory**
  value. Set a filename on the `File Output Settings` node, or disconnect it to use **Directory**.
  [griptape-nodes-engine#5971](https://github.com/griptape-ai/griptape-nodes-engine/issues/5971)
- `Adjust Mask Size`'s adjustment is now a parameter you can set on the node or connect a value to.
- `Combine Masks` shows only the connections for its incoming masks, not each mask's image, since the
  interactive preview shows the combined result.
- `Image Grid Splitter`'s `preview` output no longer shows its image on the node, since the interactive
  preview shows the grid. It still outputs the image.
- Image nodes with a preview no longer update their output until you run them.
- List inputs that take images, videos, or audio, such as `input_images` on Google Image
  Generation and `additional_images` on Grok Image Edit, convert a file path to media in its own
  row, without rewriting the rest of the list. The library requires engine 0.105.0.
- Searching the node picker for "reformat", Nuke's name for resizing, finds Resize Image and
  Resize Video.
  [#902](https://github.com/griptape-ai/griptape-nodes-library-standard/issues/902)
- `Crop Image` and `Crop Video`'s crop editors and their settings now stay in sync while you drag either one.

### Fixed

- Extract Frames saves into your home folder when `directory` starts with `~/`. It used to create a
  folder named `~` inside the project outputs folder.
  [#933](https://github.com/griptape-ai/griptape-nodes-library-standard/issues/933)
- `Extract Frames` running in a worker now saves into the folder from a `File Output Settings`
  node wired to **Directory**, instead of the **Directory** value.
  [griptape-nodes-engine#5971](https://github.com/griptape-ai/griptape-nodes-engine/issues/5971)
- LTX Video to Video HDR's `project_path` shows placeholder text before a run saying the saved EXR
  ZIP's path appears there, and its tooltip points to `output_file` for where the ZIP is saved.
  [#946](https://github.com/griptape-ai/griptape-nodes-library-standard/issues/946)
- `Set Color to Transparent`'s `grow_shrink` tooltip now says positive values grow the transparent area
  and negative values shrink it, which is what the setting has always done.
- `Workflow Node` inside an Isolated Execution group now reuses the workflow it imported on
  earlier runs instead of importing another copy on every run, and runs instead of failing to read
  the workflow's inputs and outputs. Requires engine 0.106.0.
  [#883](https://github.com/griptape-ai/griptape-nodes-library-standard/issues/883)

## [0.90.3] - 2026-11-09

### Added

- `LTX Video Reframe` reframes a video to a new resolution or aspect ratio (1:1, 4:5, 5:4, 9:16 or
  16:9), with the model generating the extra area to match the original. Choose where the video sits
  with `placement_mode`: `center`, `anchor` (an edge or corner) or `rectangle` (exact `x`, `y`,
  `width` and `height`). Input videos can be up to 60 seconds and 1,800 frames.
  [#916](https://github.com/griptape-ai/griptape-nodes-library-standard/issues/916)

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

[Unreleased]: https://github.com/griptape-ai/griptape-nodes-library-standard/compare/v0.91.0...HEAD
[0.91.0]: https://github.com/griptape-ai/griptape-nodes-library-standard/compare/v0.90.3...v0.91.0
[0.90.3]: https://github.com/griptape-ai/griptape-nodes-library-standard/compare/v0.90.2...v0.90.3
[0.90.2]: https://github.com/griptape-ai/griptape-nodes-library-standard/compare/v0.90.1...v0.90.2
[0.90.1]: https://github.com/griptape-ai/griptape-nodes-library-standard/compare/v0.90.0...v0.90.1
