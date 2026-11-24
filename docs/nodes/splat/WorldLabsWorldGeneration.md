---
description: Generates an explorable 3D world as a Gaussian splat from text, an image, several images or a video, using World Labs Marble.
when_to_use: Use it when you want a navigable 3D scene, not a single object, built from a description or reference footage.
---

## When to use it

You want a whole environment you can look around in, not a single object. Pick an `input_type`, supply the matching input and run.

View the splats with [Load Splat](LoadSplat.md). To open one in Blender or Unreal, convert it with [Convert SPZ to PLY](ConvertSpzToPly.md).

## What it needs

- A Griptape Nodes license or a Griptape Cloud API key (`GT_CLOUD_API_KEY`). To use your own World Labs key (`WORLD_LABS_API_KEY`) instead, set **API Key Provider** to **Customer**.
- The node waits for the result. `timeout` sets how long, 600 seconds by default. If a run times out, the **Refresh** button on the node fetches the result later.
- The input that matches `input_type`: text, one image, 2 to 4 images (up to 8 with `enable_reconstruction`), or a video. Each image or video must be 10 MB or smaller. `text_prompt` is optional guidance for every type except Text, where it is the input.

## Parameters

### model

Marble 1.1 Plus makes larger worlds and has variable pricing. 1.1 is the default. 1.0 Draft is faster and cheaper. The dropdown marks models your license does not allow, and the node refuses to run with one selected.

### input_type

Text, Image, Multi-Image or Video. It decides which inputs show.

### is_panorama

Image input only. Check it if your image is already a panorama. This skips panorama generation and saves 80 credits.

### azimuth_angles

Multi-Image only. Optional angles in degrees, one per image in order, such as `0,90,180,270`. 0 is front, 90 right, 180 back and 270 left. Each angle must be between 0 and 360.

### enable_reconstruction

Multi-Image only. Raises the image limit from 4 to 8.

### disable_recaption

Hidden by default, and hidden for Text input. Sends your `text_prompt` as written instead of letting the provider rewrite it.

### display_name and tags

Hidden by default. `display_name` is cut to 64 characters. `tags` is a comma-separated list.

### seed

Reuse a seed for repeatable results, or turn on `randomize_seed`.

### output_file

Where the full-resolution splat is saved. The other files go in the same folder under fixed names: `splat_100k.spz`, `splat_500k.spz`, `collider_mesh.glb` and `panorama.jpg`.

### Outputs

`splat_100k`, `splat_500k` and `splat_full_res` are SPZ splat files. `mesh` is a GLB collider mesh. `panorama` and `thumbnail` are images (the thumbnail is the provider's link and is not saved to the project), `caption` describes the world and `world_id` identifies it at World Labs.

## Edge cases

- If some files cannot be retrieved, the node still succeeds and the status lists the missing ones. It fails only when none can be retrieved.
