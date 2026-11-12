// Value shapes the webcam widget exchanges with the Webcam node over its `snapshot` parameter.
// Each capture runs one upload handshake; selection and clearing are single messages.

/** One saved photo in the gallery. */
export interface GalleryItem {
  /** Where the thumbnail loads from. */
  url: string;
  /** The saved file's location, which the node turns back into an artifact. */
  _path?: string;
}

/** The gallery as the node reports it. */
export interface WebcamGallery {
  gallery_items: GalleryItem[];
  selected_index: number;
  gallery_count: number;
}

/** The node's resting value, and its reply to `clear_gallery`. */
export interface WebcamIdle extends WebcamGallery {
  state: "idle";
  _emitSeq?: number;
}

/** Sent by the widget for each capture: asks the node where to PUT the photo. */
export interface WebcamRequestingUploadUrl {
  state: "requesting_upload_url";
  _emitSeq: number;
}

/** Sent by the node in reply: the URL the widget uploads the photo to. */
export interface WebcamUploadReady {
  state: "upload_ready";
  _uploadUrl: string;
  _emitSeq: number;
}

/** Sent by the widget once the upload finished, whether or not it succeeded. */
export interface WebcamAccepted {
  state: "accepted";
  _emitSeq: number;
}

/** Sent by the node once it handled a capture: the gallery with the new photo, if it saved. */
export interface WebcamProcessed extends WebcamGallery {
  state: "processed";
  _emitSeq: number;
}

/** Sent by the widget when the user picks a thumbnail. */
export interface WebcamSelected {
  state: "selected";
  selected_index: number;
  gallery_items: GalleryItem[];
  _emitSeq: number;
}

/** Stored by the node after a selection. The widget ignores it. */
export interface WebcamSelectionConfirmed extends WebcamGallery {
  state: "selection_confirmed";
  _emitSeq: number;
}

/** Sent by the widget when the user clears the gallery. */
export interface WebcamClearGallery {
  state: "clear_gallery";
  _emitSeq: number;
}

/** The `snapshot` parameter's value, in either direction. */
export type WebcamCaptureValue =
  | WebcamIdle
  | WebcamRequestingUploadUrl
  | WebcamUploadReady
  | WebcamAccepted
  | WebcamProcessed
  | WebcamSelected
  | WebcamSelectionConfirmed
  | WebcamClearGallery;

/** The widget has no synced parameters, so it previews nothing. */
export type WebcamCaptureLiveValues = Record<string, never>;
