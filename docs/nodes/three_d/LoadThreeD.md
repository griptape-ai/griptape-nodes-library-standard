---
description: Loads a 3D model file with an inline viewer, and can output a snapshot image of it.
when_to_use: Use it when you want to bring a 3D model into a workflow, such as a prop or set piece, or take a still image of it.
---

## When to use it

You have a 3D model and want to pass it to other nodes or preview it. Pick a file with the file browser in the `3d` field, type or paste a path or URL in `path`, or connect a 3D model from another node. The model shows in the viewer on the node.

The `3d` and `path` fields stay in sync. When a connection feeds `3d`, `path` is not editable.

Supported extensions for the file picker are `.glb`, `.gltf`, `.stl`, `.obj`, `.fbx`, `.ply` and `.dae`. For a GLTF-only version, see [Load GLTF](LoadGLTF.md).

## Parameters

### 3d

Shown in the editor as "3D File or path to 3D file". The viewer has a snapshot button.

### path

Shown as "File Path or URL". A local file or a URL.

### image

A still image of the model. It is filled when the loaded model carries an image URL in its metadata, which is what the viewer's snapshot button saves. If no snapshot has been saved, `image` stays empty.
