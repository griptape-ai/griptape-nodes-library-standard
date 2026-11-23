---
description: Generates an explorable 3D world as a Gaussian splat from text, an image, several images or a video, using World Labs Marble.
when_to_use: Use it when you want a navigable 3D scene, not a single object, built from a description or reference footage.
---

## When to use it

You want a whole environment you can look around in. Pick an `input_type`, supply the matching input and run. The node runs through the Griptape model proxy and needs a Griptape Cloud API key (`GT_CLOUD_API_KEY`). It polls until the job finishes. The default timeout is 10 minutes. If a run times out, the Refresh button in the Status group retrieves the result.

The splats come out at three sizes. Connect them to [Load Splat](LoadSplat.md) to view them. To open a splat in Blender or Unreal, convert it with [Convert SPZ to PLY](ConvertSpzToPly.md).

## Parameters

### model

Marble 1.1 Plus has variable pricing and makes larger worlds. 1.1 is the default. 1.0 is the previous version. 1.0 Draft is faster and cheaper. The dropdown marks models your license does not allow, and the node refuses to run with one selected.

### input_type

Chooses which inputs show and are used.

| Input type | Needs |
| --- | --- |
| Text | `text_prompt` |
| Image | `image`. `text_prompt` is optional guidance. |
| Multi-Image | `images`, 2 to 4 images, or up to 8 with `enable_reconstruction`. `text_prompt` is optional. |
| Video | `video`. `text_prompt` is optional. |

### is_panorama

Image input only. Check it if your image is already a panorama. This skips panorama generation and saves 80 credits.

### azimuth_angles

Multi-Image only. Optional comma-separated angles in degrees, one per image in order: 0 is front, 90 is right, 180 is back, 270 is left. For example `0,90,180,270`. Each angle must be between 0 and 360.

### enable_reconstruction

Multi-Image only. Raises the image limit from 4 to 8.

### disable_recaption

Hidden by default, and hidden again for Text input. Sends your `text_prompt` as written instead of letting the provider rewrite it. A value set earlier still applies to Text input even though the option is hidden.

### display_name and tags

Hidden by default. `display_name` is cut to 64 characters. `tags` is a comma-separated list.

### seed

Use the same seed for repeatable results. Turn on `randomize_seed` to pick a new one each run.

### output_file

Where the full-resolution splat is saved. The other files are saved in the same folder with fixed names: `splat_100k.spz`, `splat_500k.spz`, `collider_mesh.glb` and `panorama.jpg`.

### Outputs

- `splat_100k`, `splat_500k` and `splat_full_res` are the SPZ splat files.
- `mesh` is a collider mesh in GLB format.
- `panorama` and `thumbnail` are images. The thumbnail is the provider's own link and is not saved to the project.
- `caption` is a text description of the world.
- `world_id` identifies the world at World Labs.

If some files cannot be retrieved, the node still succeeds and the status lists the missing ones. It fails only when none can be retrieved.
