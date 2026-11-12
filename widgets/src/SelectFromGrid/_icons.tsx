// Lucide SVG icon paths (MIT licensed)

export const ICON_PATHS = {
  "music":         `<path d="M9 18V5l12-2v13"/><circle cx="6" cy="18" r="3"/><circle cx="18" cy="16" r="3"/>`,
  "loader-circle": `<path d="M21 12a9 9 0 1 1-6.219-8.56"/>`,
  "alert-circle":  `<circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/>`,
  "check":         `<polyline points="20 6 9 17 4 12"/>`,
  "maximize-2":    `<polyline points="15 3 21 3 21 9"/><polyline points="9 21 3 21 3 15"/><line x1="21" x2="14" y1="3" y2="10"/><line x1="3" x2="10" y1="21" y2="14"/>`,
  "x":             `<path d="M18 6 6 18"/><path d="m6 6 12 12"/>`,
  "chevron-left":  `<path d="m15 18-6-6 6-6"/>`,
  "chevron-right": `<path d="m9 18 6-6-6-6"/>`,
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
