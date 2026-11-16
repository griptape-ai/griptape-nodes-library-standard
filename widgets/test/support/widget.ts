import { act } from "react";
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
  /** Cleans up, then lets the widget's deferred React unmount and its effect cleanups run. */
  cleanup: () => Promise<void>;
  /** Updates and lets React render the new props. */
  update: (props: TProps) => void;
  /** What the mount function returned. */
  result: WidgetMountResult<TProps>;
}

/**
 * Mounts a widget, lets React render it, and normalises its result: a bare cleanup function, a
 * partial handle, or nothing.
 */
export function mountWidget<TProps>(mount: WidgetMount<TProps>, container: HTMLElement, props: TProps): MountedWidget<TProps> {
  let result: WidgetMountResult<TProps> = undefined;
  act(() => {
    result = mount(container, props);
  });
  const handle: WidgetHandle<TProps> = typeof result === "function" ? { cleanup: result } : (result ?? {});
  return {
    cleanup: () => act(async () => handle.cleanup?.()),
    update: (next) => act(() => handle.update?.(next)),
    result,
  };
}

/** Runs `fn` (an event, a timer, a resolved load) and lets React render what it changed. */
export function flush(fn: () => void): void {
  act(fn);
}

/** Lets pending promises settle and React render what they changed. */
export async function settle(): Promise<void> {
  await act(async () => {
    await new Promise((resolve) => setTimeout(resolve, 0));
  });
}
