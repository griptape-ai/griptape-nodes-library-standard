---
description: Configures an Anthropic Claude model with your own Anthropic API key, for use by an Agent.
when_to_use: Use it when an Agent should call Anthropic directly with your key, or when you need control over temperature, top_p, top_k or retries.
---

## When to use it

This node builds a model configuration. It does not call the model itself. Connect `prompt_model_config` to an [Agent](../agents/Agent.md)'s `prompt model` input and the Agent uses that model.

Use it when the Agent should call Anthropic directly. If you don't have an Anthropic key, use [Griptape Cloud Prompt](GriptapeCloudPrompt.md), which offers Claude models through Griptape Cloud.

## What it needs

- An Anthropic API key, stored as `ANTHROPIC_API_KEY` in Settings under API Keys & Secrets. This node does not use a Griptape Nodes license or a Griptape Cloud API key.
- If the key is missing, a warning appears on the node and the workflow fails validation.

## Example

An Agent reviews a long shot list and needs room for a detailed answer.

1. Add **Anthropic Prompt** and pick a `model`.
1. Set `max_tokens` to 4000.
1. Connect `prompt_model_config` to the Agent's `prompt model` input.

## Parameters

### model

`claude-sonnet-4-6` is the default. The other choices are `claude-opus-4-7` and `claude-haiku-4-5`. Workflows saved with older Claude model names load as the nearest current model. Models your license does not permit are flagged, and the run fails if one is selected.

### temperature

From 0 to 1, default 0.1.

### top_p

From 0 to 1, default 0.9.

### top_k

Limits how many tokens are considered at each step. Default 50.

### max_attempts_on_fail

From 1 to 100, default 2. The total number of attempts when a request fails.

### use_native_tools

On by default. Agents that use tools fail when this is off, because the node does not support prompted tool use.

### max_tokens

The maximum tokens to generate. A value of 0 or less, the default of -1, leaves the limit to the provider.

### prompt_model_config

The model configuration. Connect it to an Agent.
