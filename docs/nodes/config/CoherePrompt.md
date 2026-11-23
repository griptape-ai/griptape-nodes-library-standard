---
description: Configures a Cohere model with your own Cohere API key, for use by an Agent.
when_to_use: Use it when an Agent should call Cohere directly with your key, or when you need control over temperature, p, k or retries.
---

## When to use it

This node builds a model configuration. It does not call the model itself. Connect `prompt_model_config` to an [Agent](../agents/Agent.md)'s `prompt model` input and the Agent uses that model.

It talks to Cohere directly and needs `COHERE_API_KEY` in Settings under API Keys & Secrets. A warning appears on the node when the key is missing, and the workflow fails validation.

## Example

1. Add **Cohere Prompt**.
1. Set `max_tokens` to 1000 to keep answers short.
1. Connect `prompt_model_config` to the Agent's `prompt model` input.

## Parameters

### model

`command-r-plus` is the only choice. Workflows saved with the older display name load as `command-r-plus`. If your license does not permit the model, the run fails.

### temperature

From 0 to 1, default 0.1.

### p

From 0 to 1, default 0.9. The nucleus sampling cutoff, sent to the model as `top_p`.

### k

Limits how many tokens are considered at each step. Default 50. Sent to the model as `top_k`.

### max_attempts_on_fail

From 1 to 100, default 2. The total number of attempts when a request fails.

### use_native_tools

On by default. Agents that use tools fail when this is off, because the node does not support prompted tool use.

### max_tokens

The maximum tokens to generate. A value of 0 or less, the default of -1, leaves the limit to the provider.

### prompt_model_config

The model configuration. Connect it to an Agent.
