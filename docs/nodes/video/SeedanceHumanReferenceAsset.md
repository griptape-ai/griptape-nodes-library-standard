---
description: Packages an image, video or audio file as a private-asset reference for Seedance reference inputs.
when_to_use: Use it when you want to use a human likeness or voice as a Seedance reference and the media must be registered as a provider private asset.
---

## When to use it

Seedance treats human references, such as AIGC virtual-portrait images, as private assets. Plugging the file straight into a reference input does not register it that way. This node wraps the media and marks its kind. Connect the `asset_reference` output to the matching reference input on [Seedance 2.0 Video Generation](Seedance20VideoGeneration.md) or [Seedance 2.5 Video Generation](Seedance25VideoGeneration.md). Nothing is registered until the generation node runs. It registers the asset with the provider, waits up to about 3 minutes for it to become active, and then uses it.

> [!NOTE]
> Private assets need Griptape authentication. The Seedance node refuses them when its API Key Provider option is set to use your own key.

## Parameters

### asset_kind

Choose Image, Video or Audio. Only the matching input is shown. The output must go to the reference input of the same kind. A mismatch, such as an Image reference in a video input, is reported before the run.

### image / video / audio

The media for the selected kind. If it is empty, `asset_reference` is empty.

### asset_reference

The wrapped media and its kind. It is updated as soon as you change the kind or the media, so downstream nodes can preview it without running this node.

## Access

The provider-asset feature is gated by organization. The node checks access the first time you edit it and again before each run. If access is denied, a badge appears on `asset_kind` and the run is blocked. If the check cannot be completed, you get a warning badge and the run goes ahead.
