// Value shapes the multi-shot editor exchanges with the editor and the Kling video nodes.

/**
 * One shot in the sequence. The node's default is `{ name, duration, description }`; the widget
 * adds an `id` to each shot and sends it back. Fields the widget doesn't know are passed back
 * unchanged.
 */
export interface KlingShot {
  id?: string;
  name: string;
  /** Seconds. */
  duration: number;
  description: string;
  [key: string]: unknown;
}

/** The widget's parameter value: the `shots` list. */
export type KlingShotList = KlingShot[];
