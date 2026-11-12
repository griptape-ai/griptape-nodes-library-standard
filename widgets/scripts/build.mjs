// Builds each widget entry into one self-contained browser ES module in dist/.
//
//   node scripts/build.mjs           build once: stage, verify, then promote into dist/
//   node scripts/build.mjs --watch   rebuild into dist/ on every source change
//
// The engine runs `npm run build --ignore-scripts` in source mode, so this script is the whole
// build: no pre/post hooks, no install-time steps.

import { realpathSync } from "node:fs";
import { mkdir, mkdtemp, readFile, readdir, rename, rm } from "node:fs/promises";
import { dirname, join, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import * as esbuild from "esbuild";

export const WIDGETS_ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");

/** Output name (without `.js`) to entry module, relative to the package root. */
export const ENTRIES = {
  CropImageEditor: "src/CropImageEditor/CropImageEditor.tsx",
  CropVideoEditor: "src/CropVideoEditor/CropVideoEditor.tsx",
  ImageProcessorPreview: "src/ImageProcessorPreview/ImageProcessorPreview.tsx",
  KlingMultiShotEditor: "src/KlingMultiShotEditor/KlingMultiShotEditor.tsx",
  MaskAdjustmentPreview: "src/MaskAdjustmentPreview/MaskAdjustmentPreview.tsx",
  SelectFromGrid: "src/SelectFromGrid/SelectFromGrid.tsx",
  SplatViewer: "src/SplatViewer/SplatViewer.tsx",
  VideoCapture: "src/VideoCapture/VideoCapture.tsx",
  WebcamCapture: "src/WebcamCapture/WebcamCapture.tsx",
};

const STAGING_DIR = ".build-tmp";

// Module syntax that would make the browser fetch something besides the bundle itself. Bundles are
// minified, so a statement starts a line or follows a `;` or `}`.
const FORBIDDEN_SYNTAX = [
  { pattern: /(?:^|[;}])\s*import\s*(?:[\w*{}\s,$]+from\s*)?["']/m, what: "an import statement" },
  { pattern: /(?:^|[;}])\s*export\s*(?:\*|\{[^}]*\})\s*from\s*["']/m, what: "a re-export" },
  { pattern: /\bimport\s*\(/, what: "a dynamic import" },
];
const EXPORT_STATEMENT = /(?:^|[;}])\s*export\b/gm;
const DEFAULT_EXPORT = /(?:^|[;}])\s*export\s*\{\s*[\w$]+ as default\s*\};?\s*$/;

/**
 * @param {string} root
 * @param {Record<string, string>} entries
 * @param {string} outdir
 * @returns {esbuild.BuildOptions}
 */
export function buildOptions(root, entries, outdir) {
  return {
    absWorkingDir: root,
    entryPoints: Object.fromEntries(Object.entries(entries).map(([name, entry]) => [name, join(root, entry)])),
    outdir,
    bundle: true,
    format: "esm",
    splitting: false,
    platform: "browser",
    target: "es2020",
    jsx: "automatic",
    jsxImportSource: "@griptape-ai/nodes-widget-api",
    minify: true,
    sourcemap: false,
    legalComments: "inline",
    charset: "utf8",
    logLevel: "warning",
    metafile: true,
    write: true,
  };
}

/**
 * Checks one emitted bundle. Throws with every problem found.
 *
 * @param {string} name output file name
 * @param {string} text bundle contents
 * @param {esbuild.Metafile["outputs"][string]} meta esbuild's record of the output
 */
export function verifyBundle(name, text, meta) {
  const problems = [];
  if (meta.imports.length > 0) {
    problems.push(`imports ${meta.imports.map((i) => i.path).join(", ")}`);
  }
  if (meta.exports.length !== 1 || meta.exports[0] !== "default") {
    problems.push(`exports [${meta.exports.join(", ")}] instead of a single default export`);
  }
  for (const { pattern, what } of FORBIDDEN_SYNTAX) {
    if (pattern.test(text)) problems.push(`contains ${what}`);
  }
  const exportStatements = text.match(EXPORT_STATEMENT) ?? [];
  if (exportStatements.length !== 1 || !DEFAULT_EXPORT.test(text)) {
    problems.push("does not end in a single `export { … as default }`");
  }
  if (problems.length > 0) {
    throw new Error(`${name}: ${problems.join("; ")}`);
  }
}

/**
 * Builds every entry into a staging directory, verifies the result, and moves the bundles into
 * `outDir`. Files in `outDir` that the build did not produce are removed. If the build or its
 * checks fail, `outDir` is left as it was.
 *
 * @param {{ root?: string, outDir?: string, entries?: Record<string, string> }} [options]
 * @returns {Promise<string[]>} paths of the promoted bundles
 */
export async function buildWidgets({ root = WIDGETS_ROOT, outDir, entries = ENTRIES } = {}) {
  const target = outDir ?? join(root, "dist");
  const stagingRoot = join(root, STAGING_DIR);
  await mkdir(stagingRoot, { recursive: true });
  const staging = await mkdtemp(join(stagingRoot, "build-"));

  try {
    const result = await esbuild.build(buildOptions(root, entries, staging));
    const metafile = result.metafile;
    if (!metafile) throw new Error("esbuild returned no metafile");

    const expected = Object.keys(entries).map((name) => `${name}.js`).sort();
    const emitted = (await readdir(staging)).sort();
    if (JSON.stringify(emitted) !== JSON.stringify(expected)) {
      throw new Error(`expected exactly [${expected.join(", ")}] but the build emitted [${emitted.join(", ")}]`);
    }

    for (const file of emitted) {
      const outPath = join(staging, file);
      const meta = metafile.outputs[relative(root, outPath).split("\\").join("/")];
      if (!meta) throw new Error(`${file}: missing from esbuild's metafile`);
      verifyBundle(file, await readFile(outPath, "utf8"), meta);
    }

    await mkdir(target, { recursive: true });
    // Each rename is atomic, the set is not: with several entries a reader can briefly see a mix.
    for (const file of emitted) {
      await rename(join(staging, file), join(target, file));
    }
    for (const file of await readdir(target)) {
      if (!emitted.includes(file)) await rm(join(target, file), { recursive: true, force: true });
    }
    return emitted.map((file) => join(target, file));
  } finally {
    await rm(staging, { recursive: true, force: true });
  }
}

async function watch() {
  const ctx = await esbuild.context({
    ...buildOptions(WIDGETS_ROOT, ENTRIES, join(WIDGETS_ROOT, "dist")),
    metafile: false,
    logLevel: "info",
  });
  await ctx.watch();
  console.log("Watching src/ — refresh the editor to load a rebuilt widget.");
}

async function main() {
  if (process.argv.includes("--watch")) {
    await watch();
    return;
  }
  const outputs = await buildWidgets();
  for (const file of outputs) console.log(`built ${relative(WIDGETS_ROOT, file)}`);
}

// Run as a script, however it was invoked (relative path, symlink), but not when imported.
function invokedDirectly() {
  const script = process.argv[1];
  if (!script) return false;
  try {
    return realpathSync(script) === realpathSync(fileURLToPath(import.meta.url));
  } catch {
    return false;
  }
}

if (invokedDirectly()) {
  main().catch((err) => {
    console.error(err instanceof Error ? err.message : err);
    process.exitCode = 1;
  });
}
