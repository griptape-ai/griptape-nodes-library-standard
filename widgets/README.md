# Widgets

Custom parameter widgets for the standard library. Each widget is authored in TypeScript in this
package and built into one self-contained browser module.

```
widgets/
  src/<Name>/            widget sources (entry: <Name>.ts, value shapes in _types.ts)
  scripts/build.mjs      the build: esbuild, staged, verified, then promoted; ENTRIES lists the widgets
  test/                  vitest tests (happy-dom), one file per widget plus the build's own
  dist/                  built bundles, committed
```

| Widget | Node |
| --- | --- |
| CropImageEditor | Crop Image |
| CropVideoEditor | Crop Video |
| ImageProcessorPreview | the image and video processor nodes' live preview |
| KlingMultiShotEditor | Kling video generation, multi-shot mode |
| MaskAdjustmentPreview | Adjust Mask Size |
| SelectFromGrid | Select From Grid |
| SplatViewer | the Gaussian splat nodes' viewer |
| VideoCapture | Video Capture |
| WebcamCapture | Webcam |

## The editor's widget contract

The props a widget is mounted with, the handle it returns, and the module shape the editor loads
are typed by `@griptape-ai/nodes-widget-api`, which the Griptape Nodes editor publishes from its
widget loader. Widgets import it with `import type`, so nothing from it reaches a bundle.

For now the package is published only to a local registry, which `.npmrc` points the
`@griptape-ai` scope at (`http://localhost:4873/`). Start that registry and publish the package
from a checkout of the editor (`npm run registry:local`, then `npm run publish:widget-api:local`)
before `make widgets/install`. Without it, `npm ci` fails to fetch the package.

## Authoring

Needs Node.js 22.12 or later and npm. From the repository root:

```sh
make widgets/install     # npm ci from package-lock.json, lifecycle scripts disabled
make widgets/watch       # rebuild dist/ on every save
make widgets/typecheck
make widgets/test
make widgets/build       # one-off build into dist/
```

The editor doesn't hot-reload widgets. After a rebuild, refresh the editor (reload the browser
tab) to load the new bundle.

To add, remove, or upgrade a build dependency, run `npm install <pkg>@<version> --save-exact
--ignore-scripts` (or edit `package.json` and run `npm install --ignore-scripts`) in `widgets/`,
and commit `package.json` and `package-lock.json` together. Keep versions exact. Dependencies
must work without install scripts: platform binaries, as esbuild and TypeScript ship them, come
from optional dependencies.

Commit `dist/` with the sources that produced it. CI rebuilds and runs `make widgets/check-dist`,
which fails if a bundle in `dist/` is changed, added, or removed by the build, and the release
workflow runs the same check before tagging.

Third-party packages a widget imports at runtime are bundled into it at the exact versions in
`package-lock.json`. `SplatViewer` bundles three.js and Spark this way, which makes it by far the
largest bundle.

### What the build guarantees

- One ES module per entry, `dist/<Name>.js`, whose only export is the default mount function.
- Helpers, styles, and icons are bundled into the entry: no imports of any kind, no shared
  chunks, no CSS or asset sidecars. The build fails rather than emit them.
- The same sources produce the same bytes: no timestamps, absolute paths, or source maps.
- The bundle is built into `.build-tmp/`, checked, and only then moved into `dist/`; a failed
  build leaves `dist/` as it was. Files in `dist/` the build didn't produce are removed.

`watch` writes straight to `dist/` without the staging step, so run `make widgets/build` before
committing.

## Distribution

`griptape_nodes_library.json` declares

```json
"widget_build": { "mode": "prebuilt", "root": "widgets" }
```

and points each widget at its bundle in `widgets/dist/`. In prebuilt mode
the engine serves the committed bundle as is. Users need no Node.js, npm, or beta settings, and
the engine runs nothing from this package.

## Building from source in the engine

A library can instead declare `"mode": "source"`, and the engine builds its widgets itself by
running, in the widget root:

```sh
npm ci --include=dev --include=optional --ignore-scripts
npm run build --ignore-scripts
```

The standard library ships prebuilt. Source mode is for library authors and for testing a
library checkout whose bundles aren't built. To use it, on the machine running the engine:

1. Install Node.js and npm, and set the engine settings `widget_tooling.node_executable` and
   `widget_tooling.npm_executable` to their paths.
2. Enable the `widget_source_builds` beta.
3. Run **Prepare Widgets** for the library, or approve the build when installing or updating
   it. The engine only builds after this explicit authorisation.

**Building a library runs its code.** `npm run build` executes the library's build script and the
code of every build dependency it installs, as your user, with your user's access to files and
the network. Only prepare libraries you trust. Lifecycle scripts are disabled, which narrows what
runs during install but doesn't make the build itself sandboxed.

Until a source-mode library is prepared, its nodes still load, and its custom widgets fall back
to the ordinary parameter controls with a diagnostic explaining what's missing. If a build fails
or the tooling changes, fix the cause and run **Prepare Widgets** again to repair it. A prepared
bundle keeps working if the beta is turned off afterwards.

Offline, `npm ci` can only install from the npm cache, so building without network access needs
a cache already populated with every package in `package-lock.json` (for example by running
`make widgets/install` once while online).
