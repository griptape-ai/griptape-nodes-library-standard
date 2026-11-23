---
description: Adds 2 to 20 seconds of new footage to the start or end of an existing video using LTX.
when_to_use: Use it when a clip ends too soon, or you need a lead-in before it starts.
---

## When to use it

You have a clip and want more of it. Connect the video to `video`, choose whether to extend the `start` or the `end`, set how many seconds to add, and run the node. An optional `prompt` says what should happen in the new part.

The tooltip on `video` lists the input limits: 16:9 or 9:16, at least 73 frames and up to 4K. LTX enforces them. The node only checks the size of the upload.

To send the video, the node uploads it to Griptape Cloud as a short-lived public link and deletes it after the run. If that is not possible it embeds the video in the request instead, which only works for files up to about 11 MB.

The node runs through the Griptape model proxy and needs a Griptape Cloud API key. Switch `api_key_provider` to **Customer** to use your own LTX Studio key instead, stored in the `LTX_API_KEY` secret. `timeout` sets how long the node waits, in seconds, and defaults to 600. 0 means no limit. If the wait ends first, the Refresh button in the Status group retrieves a finished result.

## Parameters

### mode

`end` (the default) continues after the last frame. `start` adds footage before the first frame.

### duration

Seconds of new footage, as a whole number from 2 to 20.

### context

Seconds of the source video that LTX looks at to match the new footage. The default is 1. Values run from 0 to 20.

> [!WARNING]
> At 0 the value is left out of the request. LTX then uses the most context it can, and you are billed for the full per-request frame cap. A badge on the parameter shows this.

The cost is the context plus the duration, rounded up to whole seconds, up to a cap that depends on the input frame rate. Requests above the cap are rejected.

### model

`ltx-2-pro` or `ltx-2-3-pro`. The default is `ltx-2-3-pro`.
