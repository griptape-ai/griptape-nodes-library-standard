import { mkdtemp, mkdir, rm } from "node:fs/promises";
import { join } from "node:path";
import { pathToFileURL } from "node:url";

import { WIDGETS_ROOT, buildWidgets } from "../../scripts/build.mjs";
import type { WidgetMount } from "../../src/types/widget";

export const SCRATCH_ROOT = join(WIDGETS_ROOT, ".build-tmp");

// A scratch directory inside the package, so it is ignored by git and cleaned by the caller.
export async function scratchDir(prefix: string): Promise<string> {
  await mkdir(SCRATCH_ROOT, { recursive: true });
  return mkdtemp(join(SCRATCH_ROOT, `${prefix}-`));
}

export async function removeDir(dir: string): Promise<void> {
  await rm(dir, { recursive: true, force: true });
}

// Builds the widgets from `root` into a fresh directory and returns where they landed.
export async function buildIntoScratch(root: string = WIDGETS_ROOT): Promise<{ outDir: string }> {
  const outDir = await scratchDir("out");
  await buildWidgets({ root, outDir });
  return { outDir };
}

export async function importMount<TProps>(file: string): Promise<WidgetMount<TProps>> {
  const mod = (await import(/* @vite-ignore */ pathToFileURL(file).href)) as { default: WidgetMount<TProps> };
  return mod.default;
}
