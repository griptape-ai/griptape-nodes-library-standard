---
description: Generates a 3D model with Rodin Gen-2 from a text prompt, from up to five images, or from both.
when_to_use: Use it when you want a textured 3D model from a description or reference photos, and you need a format other than GLB such as FBX, OBJ, USDZ or STL.
---

## When to use it

You want a 3D asset without modelling it. Write a prompt, connect reference images, or use both. Use it over the Tripo nodes when you need USDZ, OBJ or STL. Rodin also returns a preview render and may return companion files such as textures. All are saved to the project.

## What it needs

- A Griptape Nodes license or a Griptape Cloud API key (`GT_CLOUD_API_KEY`). To use your own Hyper3D Rodin key (`RODIN_API_KEY`) instead, set **API Key Provider** to **Customer**.
- The node waits for the result. `timeout` sets how long, 1200 seconds by default, or 0 for no limit. If a run times out, the **Refresh** button on the node fetches the result later.
- A prompt, up to five images, or both. One of the two is required.

## Parameters

### input_images

Up to five. Extras are ignored.

### condition_mode

Used only with more than one image. `concat` treats them as views of the same object. `fuse` combines the objects into one model.

### geometry_file_format

glb, usdz, fbx, obj or stl. The node fails if Rodin returns no file in that format.

### quality

Sets the target face count: high is 50k, medium is 18k, low is 8k and extra-low is 4k. `quality_override` replaces it.

### mesh_mode

`Quad` makes four-sided faces. `Raw` makes triangles.

### Advanced options

Hidden until you show them.

- `use_original_alpha` keeps the transparency of the input images.
- `quality_override` sets an exact poly count and overrides `quality`. Use 1,000 to 200,000 for Quad and 500 to 1,000,000 for Raw.
- `ta_pose` forces a T or A pose for human-like models.
- `bbox_condition` limits the model's size as `Width(Y), Height(Z), Length(X)`, for example `1,2,3`. Values are rounded down to whole numbers. Anything other than three numbers is ignored without an error.
- `high_pack` gives 4K textures instead of 2K and 16 times the face count in Quad mode.
- `preview_render` adds an extra high-quality render to the downloaded files.

### seed

Reuse a seed for repeatable results, or turn on `randomize_seed`.

### output_file

Where the preview image is saved, always as `.webp`. The model and other files go in the same folder.

### model_url

The generated model. If Rodin returns several files in that format, the largest is used.

### preview_image

The preview render. Hidden on the node, available as an output.

### all_files

The locations of every saved file.
