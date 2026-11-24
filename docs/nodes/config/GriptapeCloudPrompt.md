---
description: Configures a model from the Griptape Cloud catalog, using a Griptape Nodes license or a Griptape Cloud API key, for use by an Agent.
when_to_use: Use it when an Agent should use a specific Griptape Cloud model, or when you need control over temperature, top_p or retries.
---

## When to use it

This node builds a model configuration. It does not call the model itself. Connect `prompt_model_config` to an [Agent](../agents/Agent.md)'s `prompt model` input and the Agent uses that model.

Griptape Cloud serves models from several providers, so you don't need a separate key for each. An Agent already uses Griptape Cloud models by default through its own `prompt model` dropdown. Use this node when you want the extra settings.

## What it needs

- A Griptape Nodes license or a Griptape Cloud API key (`GT_CLOUD_API_KEY`). The workflow fails validation if neither is set.

## Example

An Agent summarizes dailies notes and should give the same answer each time.

1. Add **Griptape Cloud Prompt** and pick a `model`.
1. Set `temperature` to 0.
1. Connect `prompt_model_config` to the Agent's `prompt model` input.

## Parameters

### model

The default is `gpt-4.1-mini`. The list is the Griptape Cloud catalog and includes Claude, Gemini, GPT, DeepSeek and Llama models. Models your license does not permit are flagged, and the run fails if one is selected. Workflows saved with older model names load as the current equivalent.

### temperature

From 0 to 1, default 0.1.

### top_p

From 0 to 1, default 0.9. It is hidden for DeepSeek models and is not sent for the OpenAI o-series models.

### max_attempts_on_fail

From 1 to 100, default 2. The total number of attempts when a request fails.

### use_native_tools

On by default. Agents that use tools fail when this is off, because the node does not support prompted tool use.

### max_tokens

The maximum tokens to generate. A value of 0 or less, the default of -1, leaves the limit to the provider. Some models have their own limit that replaces your value. Claude models use 64000.

### prompt_model_config

The model configuration. Connect it to an Agent.
