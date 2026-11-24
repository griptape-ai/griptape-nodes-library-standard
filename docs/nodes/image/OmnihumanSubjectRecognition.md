---
description: Checks whether an image contains a human, human-like or anthropomorphic subject, using OmniHuman 1.5.
when_to_use: Use it when you want to confirm an image will work with OmniHuman before spending time on a video generation.
---

## When to use it

OmniHuman animates people and similar characters. This node asks the service whether your image has a suitable subject and returns a yes or no in `contains_subject`. Use it as a quick gate before generation, for example to branch on the result. Skip it if you already know the image is fine.

## What it needs

- A Griptape Nodes license or a Griptape Cloud API key (`GT_CLOUD_API_KEY`). To use your own ByteDance Seed key (`SEED_API_KEY`) instead, set **API Key Provider** to **Customer**.
- The node waits for the result. `timeout` sets how long, 600 seconds by default, or 0 for no limit. If a run times out, the **Refresh** button on the node fetches the result later.
- The service downloads the image itself. A local image is uploaded to a short-lived public link, which is deleted after the run.

## Parameters

### image_url

The image must be reachable from the internet. A connected local image is uploaded for you, see What it needs.

### contains_subject

True when the service reports a suitable subject. It is false if the run fails or the provider doesn't finish.

### result_details

The raw provider response on success, or the provider status if it didn't finish.
