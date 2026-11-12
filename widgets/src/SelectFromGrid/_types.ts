// Value shapes the grid exchanges with the editor and the SelectFromGrid node.

export type GridLayout = "grid" | "masonry";

/**
 * One list item as the node serialises it. Media items carry a browser URL (empty while the
 * node is still resolving it), image items may add a full-resolution `full_url`, and video items
 * a `thumbnail`. `source` is the item's path for the saved workflow; the widget doesn't read it.
 */
export interface GridItem {
  type: "image" | "video" | "audio" | "dict" | "text" | (string & {});
  url?: string;
  full_url?: string;
  thumbnail?: string;
  value?: unknown;
  label?: string;
  source?: string;
  [key: string]: unknown;
}

/** Node-author configuration, preserved across list updates. */
export interface GridSettings {
  multi_select?: boolean;
  [key: string]: unknown;
}

/**
 * The widget's parameter value (the node's `grid` ParameterDict). Fields the widget doesn't
 * know are passed back unchanged.
 */
export interface GridValue {
  items?: GridItem[];
  selected_indices?: number[];
  columns?: number;
  layout?: GridLayout;
  settings?: GridSettings;
  show_labels?: boolean;
  label_size?: number;
  [key: string]: unknown;
}

/** What the widget commits: the incoming value with the grid's own fields filled in. */
export type CommittedGridValue = GridValue & {
  items: GridItem[];
  selected_indices: number[];
  columns: number;
  layout: GridLayout;
  show_labels: boolean;
  label_size: number;
};
