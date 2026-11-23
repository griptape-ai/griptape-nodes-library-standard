---
description: Finds the subjects in an image and returns a mask image for each, using OmniHuman 1.5.
when_to_use: Use it when you need masks of the people or characters in an image, for example to choose which subject speaks in an OmniHuman video.
---

## When to use it

This is the optional second step of the OmniHuman workflow. It detects subjects in an image and gives you a mask image for each one. You only need it if you want to pick a specific subject during video generation. If the image has one obvious subject, you can skip it.

To only check whether the image is usable, use [OmniHuman Subject Recognition](OmnihumanSubjectRecognition.md).

The node runs through the Griptape model proxy. You need a Griptape Cloud credential, or you can switch `api_key_provider` to use your own ByteDance Seed key (`SEED_API_KEY` in Settings, Secrets).

## Parameters

### image_url

The service downloads the image itself, so it must be reachable from the internet. When you connect a local image, the node uploads it to a temporary public location and deletes it after the run.

### mask_image_urls

A list of URLs, one mask image per detected subject, as returned by the provider. The node outputs masks only. Bounding boxes and profile images are not output, although the full provider response is in `result_details`.

### contains_subject

True when the service reports a human subject.
