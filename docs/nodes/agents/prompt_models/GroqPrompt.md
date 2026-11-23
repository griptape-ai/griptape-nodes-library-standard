---
description: Sets up a Groq model for an Agent to use.
when_to_use: Use it when you want an Agent to run on a model hosted by Groq instead of the default.
---

## When to use it

Choose a Groq model and connect `prompt_model_config` to the `model` input of an [Agent](../Agent.md). The node does not call the model itself. It only passes the settings along.

You need a Groq API key saved as `GROQ_API_KEY`. The workflow will not run without it.

## Parameters

### model

The default is `gemma2-9b-it`. Models your license denies are marked in the list, and running the node with one selected fails.

### top_p

Default 0.9. This node has no `seed` or `top_k` setting.

### use_native_tools

Leave it on if the agent uses tools. Tool use through prompting is not supported, so an agent with tools fails when this is off.

### max_tokens

Zero or less leaves the limit to the provider default.
