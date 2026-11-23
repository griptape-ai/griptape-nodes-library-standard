---
description: Checks whether an image contains a human, human-like or anthropomorphic subject, using OmniHuman 1.5.
when_to_use: Use it when you want to confirm an image will work with OmniHuman before spending time on a video generation.
---

## When to use it

OmniHuman animates people and similar characters. This node asks the service whether your image has a suitable subject and returns a yes or no in `contains_subject`. Use it as a quick gate before generation, for example to branch on the result. Skip it if you already know the image is fine.

It runs through the Griptape model proxy. You need a Griptape Cloud credential, or you can switch `api_key_provider` to use your own ByteDance Seed key (`SEED_API_KEY` in Settings, Secrets).

## Parameters

### image_url

The service downloads the image itself, so it must be reachable from the internet. When you connect a local image, the node uploads it to a temporary public location and deletes it after the run.

### contains_subject

True when the service reports a suitable subject. It is false if the run fails or the provider doesn't finish.

### result_details

The raw provider response on success, or the provider status if it didn't finish.
