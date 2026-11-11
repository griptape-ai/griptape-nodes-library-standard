---
name: add-node-preview
description: Add an interactive preview to an image or video node — the ImageProcessorPreview widget drawing the node's result in the browser as its settings change, via ImagePreviewMixin. Covers the Python side, writing the JS operation, checking it against Pillow or FFmpeg, and testing.
argument-hint: <node-class> [operation-name]
allowed-tools: Read Edit Write Grep Glob Bash
---

# Add an interactive preview to a node

An interactive preview shows a node's result above its settings and redraws in the browser while the user drags them. The node still processes the real media when it runs; the preview never writes outputs. One shared widget, `widgets/ImageProcessorPreview/`, draws every preview. Each node names an operation, and each operation is a small JS entry that mirrors the node's processing.

The settings reach the widget as `synced_parameters` (see the `griptape-nodes-widget-dev` skill), so the node only sends what isn't a parameter: media URLs and anything it works out, such as an image's mode. Requires engine 0.107.0+ and the matching editor.

## 1. Python: opt the node in

Add `ImagePreviewMixin` (`image/image_preview.py`) first in the bases, name the operation and the settings, and add the preview after the media input. `BaseImageProcessor` and `BaseVideoProcessor` already include the mixin and call `_add_preview_parameter()`; their subclasses only set the two attributes.

```python
class FlipImage(BaseImageProcessor):
    """Flip image horizontally or vertically."""          # docstring FIRST, then the attributes

    PREVIEW_OPERATION = "flip"                           # key in _operations.js
    PREVIEW_PARAMETERS: ClassVar[list[str]] = ["direction"]
```

For any other base class:

```python
class InvertMask(ImagePreviewMixin, DataNode):
    PREVIEW_OPERATION = "invert_mask"
    PREVIEW_IMAGES: ClassVar[dict[str, str]] = {"input_mask": "image_url"}   # media param -> key the widget reads

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.add_parameter(ParameterImage(name="input_mask", ...))
        self._add_preview_parameter()                    # right after the media input
```

Class attributes and hooks:

| Name | Use |
|---|---|
| `PREVIEW_OPERATION` | The operation key in `_operations.js`. `None` means no preview. |
| `PREVIEW_PARAMETERS` | Settings the operation reads. They become `synced_parameters`, so they arrive in the widget's `value` live. |
| `PREVIEW_IMAGES` | Media parameters, mapped to `image_url` or `video_url`. Default `{"input_image": "image_url"}`; `BaseVideoProcessor` uses `{"video": "video_url"}`; `{}` for generated output (Color Bars, Create Text Image). |
| `PREVIEW_PARAMETER_NAME` | Defaults to `"preview"`. Override when the node already has a parameter by that name (Image Grid Splitter uses `"grid_preview"`). |
| `_preview_extras()` | Data the node works out that isn't a parameter: `{"image_mode": self._image_mode("input_image")}`, a detected grid, a list of masks. Stored with the preview. |
| `_preview_session_extras()` | Values only valid in this engine session, such as `self._pillow_font_url()`. Recomputed whenever the preview's value is set, including on load. |
| `_is_preview_source(name)` | Which parameters trigger a recompute. Override for sources `PREVIEW_IMAGES` can't list, such as the items of a `ParameterList` (Combine Masks). |

Also:

- **Remove any engine-side "live preview".** Many nodes processed the image in `after_value_set` (`_process_image_immediately`) and wrote `output` on every change. With a preview, outputs change only on a run. Keep `after_value_set` overrides that do other things (show/hide) and make sure they end with `return super().after_value_set(...)`, or the mixin's hook never runs.
- **The media input row is hidden** (`hide_property`), which also hides its file browser. Media then arrives by connection. That's the agreed behaviour; a value picked before is kept and still previews.
- **Changelog:** the library has "Added interactive previews for some image and video nodes." Add to it only for a behaviour change users will notice, such as an output no longer updating before a run.

## 2. JS: write the operation

Add an entry to `OPERATIONS` in `widgets/ImageProcessorPreview/_operations.js` (list the key explicitly; a test parses that object). Larger groups live in their own module (`_video_operations.js`, `_pixel_nodes.js`, `_color_bars.js`, `_text.js`) and are imported. One of these shapes:

```js
{
  size(v, w, h) { return { width, height } | { error } },   // output size in source pixels; default: input size
  label(v) { return "text before the size in the status bar" },
  pixels(data, v, view) { /* edit RGBA in place, at preview size */ },   // colour ops
  draw(ctx, img, v, view) { /* draw the result yourself */ },           // geometry, text
  generate: true, pixels(v, width, height) | render(ctx, v, size),      // no source (full-size, then scaled)
  multi: true, combine(datas, modes, view),                             // several images (Combine Masks)
}
```

`v` holds the synced settings and extras; read every setting with the node's default as fallback (`num(v.radius, 5)`). `view` is `{ width, height, scale, sourceWidth, sourceHeight, outputWidth, outputHeight }`: preview pixels, preview pixels per source pixel, and sizes in source pixels. Scale distances that are in source pixels (blur radius, shifts, margins) by `view.scale`. Video operations run on the frame picked with the scrubber; the shell handles that.

Port the node's processing line by line, including its quirks; the preview should match the output, not an idealised version. Shared helpers: `_pixels.js` has Pillow's luma, `Image.blend` byte maths, LUT application and its three-pass box blur; `_video_operations.js` has BT.601 limited-range YUV conversion.

## 3. Check it against the node

Run the node's own code (or the FFmpeg filter it builds) and the JS on the same pixels, and compare. This has caught real bugs every time. Pattern:

1. A Python script builds a small test image (random pixels for colour ops, smooth gradients for YUV filters), runs the node's real helper or filter, and writes inputs and expected outputs to JSON. Run it with the engine under test: `UV_PROJECT_ENVIRONMENT=.venv-dev uv run --with-editable ../../griptape-nodes-engine python -I script.py`.
2. A Node script imports the operation (copy the JS files to `.mjs` and rewrite the import paths) and compares byte by byte: count of differing bytes and the largest difference.
3. Put scratch files in `/tmp`, not the repo, and delete them.

What "matching" has meant so far: exact for LUT and geometry operations (levels, grayscale, curves, flip, crop, extend canvas, color bars); within one or two levels for anything that blurs or converts colour (Pillow and FFmpeg round per pass).

Traps found this way:

- Python's float `%` is `fmod` plus the divisor when the signs differ, so a tiny negative remainder becomes exactly the divisor. `((a % m) + m) % m` gets it wrong (Zone Plate).
- Python's `round()` rounds halves to even. `int()` truncates toward zero (`Math.trunc`), `//` floors (`Math.floor`).
- PIL rectangles include both corners: `fillRect(x0, y0, x1 - x0 + 1, y1 - y0 + 1)`.
- FFmpeg keeps some scales as 32-bit floats (vignette's aspect): use `Math.fround`.
- FFmpeg filters mostly run on limited-range YUV, so convert, apply, convert back. In 4:2:0 a padded picture with an odd size loses its last row.
- FFmpeg's `color` test source rounds odd sizes down to even; test with even sizes, and keep the chain in `yuv420p` (an `rgb24` format filter changes how `pad` places offsets).
- The browser decodes every image to RGBA, with fully transparent pixels' colour zeroed. Nodes that branch on the Pillow mode (RGB vs RGBA vs L) must send `image_mode` in the extras.

## 4. Test

- Add the node to `PREVIEWED` in `tests/unit/image/test_image_processor_preview.py`. That checks the synced settings exist, the operation key exists in the JS, a source reaches the preview, and a source loaded with `initial_setup=True` reaches it too.
- Keep tests off the network: the file's `no_downloads` fixture stubs image loading; extend it for new loaders.
- Run against the engine under test: `UV_PROJECT_ENVIRONMENT=.venv-dev uv run --with-editable ../../griptape-nodes-engine pytest tests/unit/image tests/unit/video`. Against the released engine, the `synced_parameters` tests fail until the engine ships.
- A new test must fail without the change. Swap in the previous file (`git show HEAD:path > path`), run it, put it back. A test that passes on the old code (for example because a network fetch fails silently and leaves the same empty value) proves nothing.
- Syntax-check the JS: copy to `.mjs`, fix the import paths, and run `node --check`.

## Gotchas

- `initial_setup=True` (loading a workflow) skips `before_value_set` and `after_value_set`. The mixin handles the preview by overriding `set_parameter_value`; don't rely on `after_value_set` alone for anything that must be right after a load.
- Put `PREVIEW_*` attributes **below** the class docstring, or it stops being `__doc__`. When inserting them with a script, add any import first and find the insertion point after, or the offsets shift.
- A `ParameterList` can't take a single image connection; connect to an item. With the MCP, `AddParameterToNodeRequest(parent_container_name="masks")` adds one.
- Don't redistribute fonts. Text previews get Pillow's built-in font from the node at runtime (`_pillow_font_url`, a session extra).
- Image Blend Compositor has no preview yet: its `exclusion` and `logical_*` modes raise on RGB images and `logical_and` falls back to normal. Fix those first.
