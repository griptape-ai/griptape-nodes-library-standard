import { cp, readFile, readdir, writeFile } from "node:fs/promises";
import { join } from "node:path";

import { afterAll, beforeAll, describe, expect, it } from "vitest";

import { ENTRIES, WIDGETS_ROOT } from "../scripts/build.mjs";
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
