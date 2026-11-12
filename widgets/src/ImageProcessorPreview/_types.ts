// Value shapes the preview receives from ImagePreviewMixin, and the shape of each operation.

/** One image for an operation that combines several (Combine Masks), with its Pillow mode. */
export interface PreviewImage {
  url: string;
  mode?: string;
}

/**
 * The widget's parameter value. The node sends its operation, which kind of media it previews, the
 * media's URL and any extras; the node's settings arrive under their own names as synced parameters.
 */
export interface PreviewValue {
  /** The node's PREVIEW_OPERATION, a key in OPERATIONS. */
  operation?: string;
  media?: "image" | "video";
  image_url?: string;
  video_url?: string;
  /** The font Pillow draws with, for the text previews. */
  font_url?: string;
  /** The source image's Pillow mode, for operations that branch on it. */
  image_mode?: string;
  /** Combine Masks' masks; `skipped` counts the ones that can't be previewed. */
  images?: PreviewImage[];
  skipped?: number;
  /** The grid Image Grid Splitter's auto detection found. */
  detected_rows?: number;
  detected_cols?: number;
  /** The node's settings, as its parameters hold them. */
  [setting: string]: unknown;
}

/** The preview never shows values on the node's parameters. */
export type PreviewLiveValues = Record<string, never>;

/** What a multi-image operation is drawn at: preview pixels, and the sources' size in source pixels. */
export interface MultiView {
  width: number;
  height: number;
  /** Preview pixels per source pixel. */
  scale: number;
  sourceWidth: number;
  sourceHeight: number;
}

/** What an operation on one source is drawn at, with its output size in source pixels. */
export interface PreviewView extends MultiView {
  outputWidth: number;
  outputHeight: number;
}

/** A size the operation can't produce, shown instead of the preview. */
export interface SizeError {
  error: string;
  width?: undefined;
  height?: undefined;
}

export type SizeResult = { width: number; height: number; error?: undefined } | SizeError;

interface Labelled {
  /** Text shown before the size in the status bar. */
  label?: (v: PreviewValue) => string;
  /** Draws text with the font the node sends in `font_url`. */
  usesFont?: boolean;
}

interface OnSource extends Labelled {
  generate?: undefined;
  multi?: undefined;
  /** Output size in source pixels for a w × h input. Defaults to the input size. */
  size?: (v: PreviewValue, w: number, h: number) => SizeResult;
}

/** Edits the scaled input's RGBA pixels in place, for colour operations. */
export interface PixelsOperation extends OnSource {
  pixels: (data: Uint8ClampedArray, v: PreviewValue, view: PreviewView) => void;
  draw?: undefined;
}

/** Draws the result into ctx, which is view.width × view.height. */
export interface DrawOperation extends OnSource {
  pixels?: undefined;
  draw: (ctx: CanvasRenderingContext2D, img: CanvasImageSource, v: PreviewValue, view: PreviewView) => void;
}

interface Generated extends Labelled {
  generate: true;
  multi?: undefined;
  size: (v: PreviewValue) => SizeResult;
}

/** Generates the image's RGBA pixels at full size, with no source. */
export interface GeneratedPixelsOperation extends Generated {
  pixels: (v: PreviewValue, width: number, height: number) => Uint8ClampedArray<ArrayBuffer>;
  render?: undefined;
}

/** Draws the image at full size, with no source. */
export interface GeneratedRenderOperation extends Generated {
  pixels?: undefined;
  render: (ctx: CanvasRenderingContext2D, v: PreviewValue, size: { width: number; height: number }) => void;
}

/** Combines several images (Combine Masks). modes[k] is the mode of datas[k]. */
export interface MultiOperation extends Labelled {
  generate?: undefined;
  multi: true;
  combine: (datas: Uint8ClampedArray[], modes: string[], view: MultiView) => Uint8ClampedArray<ArrayBuffer>;
}

export type PreviewOperation =
  | PixelsOperation
  | DrawOperation
  | GeneratedPixelsOperation
  | GeneratedRenderOperation
  | MultiOperation;
