// The contract between a library widget and the Griptape Nodes editor's widget loader.
//
// This mirrors the editor's widget loader (WidgetLoader and the param-display prop types in the
// GUI), which imports a widget module, calls its default export with a container element and the
// props below, and calls the returned handle's `update` on each new set of props and `cleanup`
// when the widget unmounts. It is kept here, not imported, because the editor publishes no types
// package; when the loader's props change, this file changes with them.

/** Names of the node parameters a widget reads and writes. */
export type ParameterName = string;

export interface WidgetProps<TValue, TLiveValue = Partial<TValue>> {
  /** The parameter's current value from the engine. Absent until the engine sends one. */
  value?: TValue | null;
  /** Commits a new value; the editor sends it to the engine. */
  onChange?: (value: TValue) => void;
  /**
   * Shows a value on the node's parameters while an interaction is under way, without sending it
   * to the engine. A later `onChange` commits.
   */
  onLiveChange?: (value: TLiveValue) => void;
  /** The node is read-only: the widget must not emit changes. */
  disabled?: boolean;
  /**
   * Parameters whose values come from an incoming connection, worked out by the editor from the
   * widget's `synced_parameters`. The widget must not change them.
   */
  locked?: readonly ParameterName[];
}

export interface WidgetHandle<TProps> {
  /** Removes everything the widget added and stops its observers. */
  cleanup: () => void;
  /** Applies a new set of props to the mounted widget. */
  update: (props: TProps) => void;
}

/** A widget module's default export. */
export type WidgetMount<TProps> = (container: HTMLElement, props: TProps) => WidgetHandle<TProps>;
