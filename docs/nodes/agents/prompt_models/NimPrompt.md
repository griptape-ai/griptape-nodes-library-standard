---
description: Sets up an NVIDIA NIM model for an Agent to use.
when_to_use: Use it when you want an Agent to run on a model served through NVIDIA's API instead of the default.
---

## When to use it

Choose a model from NVIDIA and connect `prompt_model_config` to the `model` input of an [Agent](../Agent.md). The node does not call the model itself. It only passes the settings along.

You need an NVIDIA API key saved as `NVIDIA_API_KEY`. The workflow will not run without it.

## Parameters

### model

The default is `deepseek-ai/deepseek-v3.1`. Models your license denies are marked in the list, and running the node with one selected fails.

### top_p

Default 0.9. This node has no `seed` or `top_k` setting.

### use_native_tools

Leave it on if the agent uses tools. Tool use through prompting is not supported, so an agent with tools fails when this is off.

### max_tokens

Zero or less leaves the limit to the provider default.
