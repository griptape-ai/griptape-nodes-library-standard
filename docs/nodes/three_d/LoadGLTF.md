---
description: Loads a GLTF 3D model with an inline viewer, and can output a snapshot image of it.
when_to_use: Use it when you want to bring a GLTF model into a workflow, or take a still image of one.
---

## When to use it

You have a GLTF file and want to pass it to other nodes or preview it. Pick a file with the file browser in the `gltf` field, or connect a GLTF from another node. The model shows in the viewer on the node.

## Parameters

### image

A still image of the model. It is filled in when the loaded model carries an image URL in its metadata, which is what the viewer's snapshot button saves. If no snapshot has been saved, `image` stays empty.
