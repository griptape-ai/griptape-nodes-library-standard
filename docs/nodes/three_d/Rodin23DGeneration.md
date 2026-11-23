---
description: Generates a 3D model with Rodin Gen-2 from a text prompt, from up to five images, or from both.
when_to_use: Use it when you want a textured 3D model from a description or reference photos, and you need a format other than GLB such as FBX, OBJ, USDZ or STL.
---

## When to use it

You want a 3D asset without modelling it. Write a prompt, connect reference images, or use both. You must give at least a prompt or one image.

The node runs through the Griptape model proxy and needs a Griptape Cloud API key (`GT_CLOUD_API_KEY`). Generations can be slow, especially at high quality or with HighPack. The default timeout is 20 minutes. If a run times out, the Refresh button in the Status group retrieves the result.

Rodin also returns a preview render and may return companion files such as textures. All of them are saved to the project.

## Parameters

### input_images

Up to five images. Extra images are ignored.

### condition_mode

Only used when you connect more than one image.

- `concat` treats the images as several views of the same object.
- `fuse` combines the objects in the images into one model.

### geometry_file_format

The format to generate: glb, usdz, fbx, obj or stl. If Rodin does not return a file in that format, the node fails.

### quality

Sets the target face count: high is 50k faces, medium is 18k, low is 8k and extra-low is 4k. `quality_override` replaces it.

### mesh_mode

`Quad` makes four-sided faces and `Raw` makes triangles.

### Advanced options

These are hidden until you show them.

- `use_original_alpha` keeps the transparency of the input images.
- `quality_override` sets an exact poly count. The tooltip gives 1,000 to 200,000 for Quad and 500 to 1,000,000 for Raw, and recommends 150,000 or more for Gen-2. It overrides `quality`.
- `ta_pose` forces a T or A pose for human-like models.
- `bbox_condition` limits the model's size as `Width(Y), Height(Z), Length(X)`, for example `1,2,3`. Values are rounded down to whole numbers. Anything other than three numbers is ignored without an error.
- `high_pack` gives 4K textures instead of 2K and 16 times the face count in Quad mode.
- `preview_render` adds an extra high-quality render to the downloaded files.

### seed

Use the same seed for repeatable results. Turn on `randomize_seed` to pick a new one each run.

### output_file

Where the preview image is saved. The extension is always `.webp`. The model and any other files Rodin returns are saved in the same folder under the names Rodin gives them.

### model_url

The generated model. If Rodin returns several files in the requested format, the largest one is used.

### preview_image

The preview render. It is hidden on the node and available as an output.

### all_files

The locations of every saved file, including the model, preview and companion files.
