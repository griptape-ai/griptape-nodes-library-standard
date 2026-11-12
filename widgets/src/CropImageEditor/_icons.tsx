// Lucide SVG icon paths (MIT licensed)

export const ICON_PATHS = {
  "rotate-ccw": `<path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/><path d="M3 3v5h5"/>`,
} as const satisfies Record<string, string>;

export type IconName = keyof typeof ICON_PATHS;

export function Icon({ name, size = 14 }: { name: IconName; size?: number }) {
  return (
    <svg
      viewBox="0 0 24 24" width={size} height={size}
      fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"
      style="display:block;flex-shrink:0;pointer-events:none;"
      innerHTML={ICON_PATHS[name] || ""}
    />
  );
}
