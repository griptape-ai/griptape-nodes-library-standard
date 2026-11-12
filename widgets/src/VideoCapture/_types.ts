// Value shapes the video capture widget exchanges with the VideoCapture node over its
// `recording` parameter. Each step of the upload handshake is one state.

/** The node's resting value: nothing recorded or accepted yet. */
export interface VideoCaptureIdle {
  state: "idle";
}

/** Sent by the widget on Accept: asks the node where to PUT the recording. */
export interface VideoCaptureRequestingUploadUrl {
  state: "requesting_upload_url";
  /** MIME type of the recorded blob, so the node picks the file extension. */
  _mime: string;
  _emitSeq: number;
}

/** Sent by the node in reply: the URL the widget uploads the recording to. */
export interface VideoCaptureUploadReady {
  state: "upload_ready";
  _uploadUrl: string;
  _emitSeq: number;
}

/** Sent by the widget once the upload succeeded. */
export interface VideoCaptureAccepted {
  state: "accepted";
  _emitSeq: number;
}

/** Sent by the node once it saved the upload as the output video. */
export interface VideoCaptureProcessed {
  state: "processed";
  url: string;
  _emitSeq: number;
}

/** Sent by the node when the upload can't be found. */
export interface VideoCaptureError {
  state: "error";
  message: string;
  _emitSeq: number;
}

/** The `recording` parameter's value, in either direction. */
export type VideoCaptureValue =
  | VideoCaptureIdle
  | VideoCaptureRequestingUploadUrl
  | VideoCaptureUploadReady
  | VideoCaptureAccepted
  | VideoCaptureProcessed
  | VideoCaptureError;

/** The widget has no synced parameters, so it previews nothing. */
export type VideoCaptureLiveValues = Record<string, never>;
