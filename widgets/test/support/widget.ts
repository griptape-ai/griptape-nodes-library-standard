import { vi } from "vitest";

import type { WidgetHandle, WidgetMount, WidgetMountResult, WidgetProps } from "@griptape-ai/nodes-widget-api";

/**
 * Props as the editor passes them: the given ones over an empty label, no value, and mock
 * `onChange` / `onLiveChange` callbacks.
 */
export function widgetProps<TValue, TLiveValues extends object = Record<string, unknown>>(
  props: Partial<WidgetProps<TValue, TLiveValues>> = {},
): WidgetProps<TValue, TLiveValues> {
  return {
    label: "",
    value: undefined,
    onChange: vi.fn(),
    onLiveChange: vi.fn(),
    ...props,
  };
}

/** A mounted widget's handle with both methods present, the way the editor treats what mount returns. */
export interface MountedWidget<TProps> {
  cleanup: () => void;
  update: (props: TProps) => void;
  /** What the mount function returned. */
  result: WidgetMountResult<TProps>;
}

/** Mounts a widget and normalises its result: a bare cleanup function, a partial handle, or nothing. */
export function mountWidget<TProps>(mount: WidgetMount<TProps>, container: HTMLElement, props: TProps): MountedWidget<TProps> {
  const result = mount(container, props);
  const handle: WidgetHandle<TProps> = typeof result === "function" ? { cleanup: result } : (result ?? {});
  return {
    cleanup: () => handle.cleanup?.(),
    update: (next) => handle.update?.(next),
    result,
  };
}
