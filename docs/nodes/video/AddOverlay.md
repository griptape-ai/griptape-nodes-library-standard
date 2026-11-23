---
description: Composites an overlay video or image on top of a base video, with blend modes, an optional matte and alpha edge cleanup.
when_to_use: Use it when you want to lay one layer over another, such as film grain, dust, a logo or a rendered element with alpha over a plate.
---

## When to use it

You have a base and a layer to put on it. For example, put a grain or dust pass over a plate with `grainmerge`, or lay a CG render that carries its own alpha over a background plate.

The base (`video`) can be a video or a still image. The overlay can be a video or an image, including a PNG with transparency. At least one of the two must be a video, because that sets the length of the result. If the base is a still image, the result takes the overlay's length and frame rate. If the base is a video, the overlay is looped and trimmed to the base's length.

The output has no audio. The node maps only the composited picture. The ffmpeg run times out after 300 seconds.

If `overlay_video` is empty, the base is re-encoded and nothing is added.

## Parameters

### overlay_video

Shown as "Overlay (Video or Image)". The layer that goes on top. Connect it, it has no field on the node.

### matte_video

Shown as "Matte (Video or Image)". Optional. Its brightness replaces the overlay's alpha: white is visible, black is transparent. It is stretched to the overlay's size.

### alpha_convention

`straight` (default) or `premultiplied`. How the overlay's own alpha is stored. Renderers and keyers often write premultiplied. Choosing wrong fringes soft edges, dark if premultiplied is read as straight and bright the other way. It does not matter when a matte supplies the alpha.

### blend_mode

`overlay` (default), `screen`, `lighten`, `softlight`, `grainmerge`, `grainextract`, `glow` or `hardlight`.

`overlay` is a normal alpha composite of the layer over the base, centered, at the layer's own size. Every other mode blends the colors with the matching ffmpeg blend filter and then mixes the result back using the layer's alpha, so transparency is respected in all modes. In these modes the layer is first scaled to the base's size using `sizing`.

### amount

From 0.0 to 1.0, default 1.0. Scales the layer's alpha. At 0 the layer has no effect.

### choke

Whole pixels, 0 to 10 on the slider, default 0. Shrinks the layer's alpha inward. Use it to remove a colored halo from a soft edge. You can type a value above 10.

### edge_softness

Pixels, 0.0 to 10.0 on the slider, default 0. Blurs the layer's alpha after the choke so a choked edge does not look hard. You can type a value above 10.

### sizing

`Scale to cover` (default) keeps the layer's aspect ratio and crops the excess. `Scale to fit` stretches it to the base's size. It applies to the blend modes only, because `overlay` mode does not scale the layer.

### output_frame_rate

`auto` keeps the frame rate of the base video. A specific rate converts the result to it.

### processing_speed

Trades encode time against quality: `fast`, `balanced` or `quality`.
