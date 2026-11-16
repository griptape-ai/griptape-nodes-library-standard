import { cp, readFile, readdir, writeFile } from "node:fs/promises";
import { join } from "node:path";

import { afterAll, beforeAll, describe, expect, it } from "vitest";

import { WIDGET_HOST_KEY } from "@griptape-ai/nodes-widget-api/host";

import { ENTRIES, WIDGETS_ROOT, buildWidgets, verifyBundle } from "../scripts/build.mjs";
import { buildIntoScratch, importMount, removeDir, scratchDir } from "./support/bundle";

// Module syntax that would make the browser fetch something besides the bundle itself.
// Bundles are minified, so a statement starts a line or follows a `;` or `}`.
const STATIC_IMPORT = /(?:^|[;}])\s*import\s*(?:[\w*{}\s,$]+from\s*)?["']/m;
const RE_EXPORT = /(?:^|[;}])\s*export\s*(?:\*|\{[^}]*\})\s*from\s*["']/m;
const DYNAMIC_IMPORT = /\bimport\s*\(/;

// These build into a scratch directory, so they pass with or without a built widgets/dist.
// `make widgets/check-dist` checks the committed bundles against a fresh build.
describe("built bundle", () => {
  let outDir: string;

  beforeAll(async () => {
    ({ outDir } = await buildIntoScratch());
  });

  afterAll(async () => {
    await removeDir(outDir);
  });

  it("emits exactly one JavaScript file per entry and nothing else", async () => {
    const files = (await readdir(outDir)).sort();
    expect(files).toEqual(Object.keys(ENTRIES).map((name) => `${name}.js`).sort());
  });

  it("imports as an ES module whose default export is the mount function", async () => {
    const mount = await importMount(join(outDir, "CropImageEditor.js"));
    expect(typeof mount).toBe("function");
  });

  it("has no imports, re-exports, or dynamic imports, and only a default export", async () => {
    const text = await readFile(join(outDir, "CropImageEditor.js"), "utf8");
    expect(text).not.toMatch(STATIC_IMPORT);
    expect(text).not.toMatch(RE_EXPORT);
    expect(text).not.toMatch(DYNAMIC_IMPORT);
    expect(text.match(/(?:^|[;}])\s*export\b/gm)).toHaveLength(1);
    expect(text).toMatch(/[;}]export\s*\{\s*[\w$]+ as default\s*\};?\s*$/);
  });

  it("is minified", async () => {
    const text = await readFile(join(outDir, "CropImageEditor.js"), "utf8");
    expect(text).not.toMatch(/\bfunction CropImageEditor\b/);
  });

  it("carries no machine paths or timestamps", async () => {
    const text = await readFile(join(outDir, "CropImageEditor.js"), "utf8");
    expect(text).not.toContain(WIDGETS_ROOT);
    expect(text).not.toMatch(/\/(?:home|Users|tmp)\//);
    expect(text).not.toMatch(/sourceMappingURL/);
  });

});

describe("rebuilding after a helper change", () => {
  let copyRoot: string;

  beforeAll(async () => {
    copyRoot = await scratchDir("copy");
    await cp(join(WIDGETS_ROOT, "src"), join(copyRoot, "src"), { recursive: true });
  });

  afterAll(async () => {
    await removeDir(copyRoot);
  });

  it("is deterministic, and changing a helper module changes the entry bundle bytes", async () => {
    // Given a copy of the sources that builds to the same bytes as the original
    const original = await buildIntoScratch(WIDGETS_ROOT, ["CropImageEditor"]);
    const unchanged = await buildIntoScratch(copyRoot, ["CropImageEditor"]);
    const originalBytes = await readFile(join(original.outDir, "CropImageEditor.js"));
    const unchangedBytes = await readFile(join(unchanged.outDir, "CropImageEditor.js"));
    expect(unchangedBytes.equals(originalBytes)).toBe(true);

    // When only a helper module (not the entry) is edited
    const stylesPath = join(copyRoot, "src", "CropImageEditor", "_styles.ts");
    const styles = await readFile(stylesPath, "utf8");
    expect(styles).toContain('"#1bb0f4"');
    await writeFile(stylesPath, styles.replace('"#1bb0f4"', '"#ff00ff"'));
    const changed = await buildIntoScratch(copyRoot, ["CropImageEditor"]);
    const changedBytes = await readFile(join(changed.outDir, "CropImageEditor.js"));

    // Then the single entry bundle carries the change
    expect(changedBytes.equals(originalBytes)).toBe(false);
    expect(changedBytes.toString("utf8")).toContain('"#ff00ff"');
    expect(await readdir(changed.outDir)).toEqual(["CropImageEditor.js"]);

    await Promise.all([original, unchanged, changed].map((b) => removeDir(b.outDir)));
  });
});

describe("the editor's React", () => {
  const hostGlobal = globalThis as Record<string, unknown>;
  const dirs: string[] = [];

  afterAll(async () => {
    await Promise.all(dirs.map(removeDir));
  });

  it("is read from the editor when a bundle loads, which fails without it", async () => {
    // Given a bundle that hasn't been imported yet, and no editor host
    const { outDir } = await buildIntoScratch(undefined, ["KlingMultiShotEditor"]);
    dirs.push(outDir);
    const host = hostGlobal[WIDGET_HOST_KEY];
    delete hostGlobal[WIDGET_HOST_KEY];

    // Then importing it fails with a message the editor shows
    try {
      await expect(importMount(join(outDir, "KlingMultiShotEditor.js"))).rejects.toThrow(/doesn't provide it/);
    } finally {
      hostGlobal[WIDGET_HOST_KEY] = host;
    }
  });

  it("fails the build for a React module the editor doesn't provide", async () => {
    const root = await scratchDir("server");
    dirs.push(root);
    await writeFile(join(root, "Widget.tsx"), 'import { renderToString } from "react-dom/server";\nexport default renderToString;\n');

    await expect(buildWidgets({ root, outDir: join(root, "out"), entries: { Widget: "Widget.tsx" } })).rejects.toThrow(
      /doesn't provide "react-dom\/server"/,
    );
  });

  it("fails verification for a bundle that includes React", () => {
    const meta = { imports: [], exports: ["default"], inputs: { "node_modules/react/cjs/react.production.js": {} } };

    expect(() => verifyBundle("Widget.js", "export{a as default};", meta as never)).toThrow(/bundles React/);
  });
});
