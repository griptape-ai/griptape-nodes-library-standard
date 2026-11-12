// Value shapes the mask preview exchanges with the editor and the AdjustMaskSize node.

/**
 * The widget's parameter value. The node sends the video URLs and frame position; `adjustment`
 * comes from the node's synced adjustment parameter.
 */
export interface MaskPreviewValue {
  original_video_url?: string;
  mask_video_url?: string;
  /** Pixels to grow the mask by, -25 to 25. Negative values shrink it. */
  adjustment?: number | null;
  current_frame?: number | null;
  total_frames?: number | null;
  [key: string]: unknown;
}

/** What the widget commits. Only these fields are sent; anything else in the value is dropped. */
export interface CommittedMaskPreviewValue extends MaskPreviewValue {
  original_video_url: string;
  mask_video_url: string;
  adjustment: number;
  current_frame: number;
  total_frames: number;
}

/** What the widget shows on the node's adjustment parameter while the slider is dragged. */
export interface LiveMaskPreviewValue {
  adjustment: number;
}
