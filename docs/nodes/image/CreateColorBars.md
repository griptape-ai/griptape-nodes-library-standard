---
description: Generates a color bar or test pattern image at a size you choose.
when_to_use: Use it when you need a reference pattern, such as slate or leader bars for a review reel, or a known image to test a display or a pipeline.
---

## When to use it

You need a test pattern as an image. For example, generate bars at the delivery resolution to put at the head of a dailies reel, or a ramp to check that a display or a color transform is not clipping.

## Example

1. Add **Create Color Bars**.
1. Leave `bar_type` on `SMPTE 219-100 Bars` and set `width` and `height` to 1920 and 1080.
1. Run the workflow. The pattern is in `image`.
1. Connect `image` to **Save Image** to write it to disk, or to a video node to build a leader.

## Parameters

### bar_type

Default `SMPTE 219-100 Bars`. The choices are:

- SMPTE and related bars: `SMPTE 219-100 Bars`, `SMPTE 75% Bars`, `SMPTE Bars`, `SMPTE 219+i Bars`, `100% Full Field Bars`, `75% Full Field Bars`, `75% Bars Over Red`, `EBU Bars`, `ARIB 28-100`, `ARIB 28-75`, `ARIB 28+i`, `HD Color Bars`, `Tartan Bars`.
- Solid fields: `Full Field White`, `Blue`, `Cyan`, `Green`, `Magenta`, `Red` and `Yellow`.
- Ramps and steps: `Stair 5 Step`, `Stair 5 Step Vert`, `Stair 10 Step`, `Stair 10 Step Vert`, `Y Ramp Up`, `Y Ramp Down`, `Vertical Ramp`, `Legal Chroma Ramp`, `Full Chroma Ramp`, `Chroma Ramp`.
- Other patterns: `Zone Plate`, `Multi Burst`, `Pluge`, `Bowtie`, `Pathological EG`, `Pathological PLL`, `Pathological EG/PLL`, `AV Delay Pattern 1`, `AV Delay Pattern 2`, `Bouncing Box`.

The node draws every pattern directly in 8-bit RGB. `Bouncing Box` is a single still frame, not an animation.

### width and height

Pixels, default 1920 and 1080. The node fails if either is 0 or less.

### pluge_ire_setup, pluge_bar_count, pluge_orientation

Shown only when `bar_type` is `Pluge`. The pattern is gray bars near black on a black background.

- `pluge_ire_setup`: `NTSC 7.5 IRE` (default), `PAL 0 IRE` or `RGB Full Range`. Sets the black level and the steps around it. IRE values are converted with `IRE / 100 * 255`, and negative values become 0.
- `pluge_bar_count`: default 3. Values are limited to 2 through 5. Two bars are black and just above black. Three bars add a super black bar first. Four and five add steps between black and just above black.
- `pluge_orientation`: `vertical` (default) or `horizontal`.

### image

The generated pattern as a PNG, saved to the project location set in `output_file`. The node has a live preview above its settings. The saved image is produced when the node runs.
