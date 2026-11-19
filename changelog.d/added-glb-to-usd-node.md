- `GLB to USD` (beta) converts a GLB/glTF model, such as Tripo or Rodin output, to `usdz`, `usdc`
  or `usda` with UsdPreviewSurface, MaterialX or both, so 3D results load straight into USD tools
  like Nuke. Models must be self-contained: a `.glb`, or a `.gltf` with its buffers and images
  embedded. Unlit materials are converted as lit PBR using their glTF fallback values. Unsupported
  glTF features are listed in the node's result details.
