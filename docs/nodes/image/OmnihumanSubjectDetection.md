---
description: Finds the subjects in an image and returns a mask image for each, using OmniHuman 1.5.
when_to_use: Use it when you need masks of the people or characters in an image, for example to choose which subject speaks in an OmniHuman video.
---

## When to use it

You have an image with several people or characters and want to pick which one speaks in an OmniHuman video. This node returns a mask image for each detected subject. It is the optional second step of the OmniHuman workflow. If the image has one obvious subject, skip it.

To only check whether the image is usable, use [OmniHuman Subject Recognition](OmnihumanSubjectRecognition.md).

## What it needs

- A Griptape Nodes license or a Griptape Cloud API key (`GT_CLOUD_API_KEY`). To use your own ByteDance Seed key (`SEED_API_KEY`) instead, set **API Key Provider** to **Customer**.
- The node waits for the result. `timeout` sets how long, 600 seconds by default, or 0 for no limit. If a run times out, the **Refresh** button on the node fetches the result later.
- The service downloads the image itself. A local image is uploaded to a short-lived public link, which is deleted after the run.

## Parameters

### image_url

The image must be reachable from the internet. A connected local image is uploaded for you, see What it needs.

### mask_image_urls

A list of URLs, one mask image per detected subject, as returned by the provider. The node outputs masks only. Bounding boxes and profile images are not output, although the full provider response is in `result_details`.

### contains_subject

True when the service reports a human subject.
