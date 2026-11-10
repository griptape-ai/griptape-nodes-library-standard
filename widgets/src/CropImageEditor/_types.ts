// Value shapes the crop editor exchanges with the editor and the CropImage node.

/** A crop rectangle in image pixels. */
export interface CropRect {
  left: number;
  top: number;
  width: number;
  height: number;
}

/**
 * The widget's parameter value. The node sends the image and its size; the crop fields come from
 * the node's synced crop parameters. Fields the widget doesn't know are passed back unchanged.
 */
export interface CropValue extends Partial<CropRect> {
  image_url?: string;
  img_width?: number;
  img_height?: number;
  zoom?: number | null;
  rotate?: number | null;
  [key: string]: unknown;
}

/** What the widget commits: the incoming value with the crop, zoom, and rotation applied. */
export type CommittedCropValue = CropValue & CropRect & { zoom: number; rotate: number };

/** What the widget shows on the node's parameters during a drag. */
export type LiveCropValue = CropRect & { zoom: number; rotate: number };
