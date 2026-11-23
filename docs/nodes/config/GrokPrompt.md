---
description: Configures an xAI Grok model with your own xAI API key, for use by an Agent.
when_to_use: Use it when an Agent should call Grok directly with your xAI key, or when you need control over temperature, top_p or retries.
---

> [!WARNING]
> **Billing Required for xAI API Usage**
>
> The Grok Prompt node requires an xAI account and billing information to be set up before xAI API keys will work. Without completing the billing setup, any nodes using xAI will fail, even with a valid API key. See [this guide](https://docs.griptapenodes.com/en/stable/guides/integrations/grok/) for instructions on setting up an xAI account with billing.

## When to use it

This node builds a model configuration. It does not call the model itself. Connect `prompt_model_config` to an [Agent](../agents/Agent.md)'s `prompt model` input and the Agent uses that model.

It talks to xAI directly and needs `GROK_API_KEY` in Settings under API Keys & Secrets. A warning appears on the node when the key is missing, and the workflow fails validation.

## Example

1. Add **Grok Prompt** and pick a `model`.
1. Set `top_p` to 0.7 for more focused answers.
1. Connect `prompt_model_config` to the Agent's `prompt model` input.

## Parameters

### model

`grok-3-beta` is the default. The other choices are `grok-3-fast-beta`, `grok-3-mini-beta`, `grok-3-mini-fast-beta` and `grok-2-vision-1212`. Workflows saved with older display names load as the matching model id. Models your license does not permit are flagged, and the run fails if one is selected.

### temperature

From 0 to 1, default 0.1.

### top_p

From 0 to 1, default 0.9.

### max_attempts_on_fail

From 1 to 100, default 2. The total number of attempts when a request fails.

### use_native_tools

On by default. Agents that use tools fail when this is off, because the node does not support prompted tool use.

### max_tokens

The maximum tokens to generate. A value of 0 or less, the default of -1, leaves the limit to the provider.

### prompt_model_config

The model configuration. Connect it to an Agent.
