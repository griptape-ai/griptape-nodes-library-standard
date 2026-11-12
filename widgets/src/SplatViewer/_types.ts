// Value shapes the splat viewer receives from the LoadSplat node.

/** A resolution the node aggregates, keyed as in `viewer_state.splats`. */
export type SplatResolution = "100k" | "500k" | "full_res";

/** One wired splat: its URL and the artifact's metadata. */
export interface SplatEntry {
  url: string;
  /** `created_at` or `content_hash`, when present, tells a re-run apart from the same URL. */
  meta?: Record<string, unknown>;
}

/** The splats the node has wired, by resolution. Unwired resolutions are `null`. */
export type SplatMap = Partial<Record<SplatResolution, SplatEntry | null>>;

/** What the node aggregates: `{ "splats": { "100k": …, "500k": …, "full_res": … } }`. */
export interface SplatViewerState {
  splats: SplatMap;
}

/**
 * The widget's parameter value. The node's `viewer_state` parameter is a `str` holding the state
 * as JSON; the widget also accepts the parsed object.
 */
export type SplatViewerValue = string | SplatViewerState;
