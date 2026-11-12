// Value shapes the crop editor exchanges with the editor and the CropVideo node.

/** A crop rectangle in video pixels, as the sidebar hands it to `onApply`. */
export interface CropRect {
  left: number;
  top: number;
  width: number;
  height: number;
}

/** The crop box the node's custom fields describe, in video pixels. */
export interface CustomCrop {
  custom_left: number;
  custom_top: number;
  custom_width: number;
  custom_height: number;
}

/**
 * The widget's parameter value. The node sends the video and its size; the crop settings come
 * from the node's synced crop parameters. Fields the widget doesn't know are passed back unchanged.
 */
export interface CropVideoValue extends Partial<CustomCrop> {
  video_url?: string;
  video_path?: string;
  video_width?: number;
  video_height?: number;
  total_frames?: number;
  /** A RESOLUTION_PRESETS name, a `WxH` string, or `"Custom"`. */
  crop_size?: string;
  /** A named position or `"Custom"`. */
  crop_position?: string;
  [key: string]: unknown;
}

/** What the widget commits: the incoming value with the box set and both presets on Custom. */
export type CommittedCropVideoValue = CropVideoValue & CustomCrop & { crop_size: "Custom"; crop_position: "Custom" };

/** What the widget shows on the node's parameters during a drag. */
export type LiveCropVideoValue = CustomCrop;
