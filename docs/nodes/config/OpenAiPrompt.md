---
description: Configures an OpenAI chat model with your own OpenAI API key, for use by an Agent.
when_to_use: Use it when an Agent should call OpenAI directly with your key, or when you need control over temperature, top_p or retries.
---

## When to use it

This node builds a model configuration. It does not call the model itself. Connect `prompt_model_config` to an [Agent](../agents/Agent.md)'s `prompt model` input and the Agent uses that model.

Use it when the Agent should call OpenAI directly. If you don't have an OpenAI key, use [Griptape Cloud Prompt](GriptapeCloudPrompt.md), or leave the Agent on its default model.

## What it needs

- An OpenAI API key, stored as `OPENAI_API_KEY` in Settings under API Keys & Secrets. This node does not use a Griptape Nodes license or a Griptape Cloud API key.

## Example

An Agent that writes dailies notes should be consistent from run to run.

1. Add **Open Ai Prompt** and pick a `model`.
1. Set `temperature` to 0.
1. Connect `prompt_model_config` to the Agent's `prompt model` input.

## Parameters

### model

The list is fixed, so the node can be added without a working key. The default is `gpt-5.6-sol`. A saved workflow whose model isn't on the list loads with `gpt-5.6-sol`. Models your license does not permit are flagged, and the run fails if one is selected.

### temperature

From 0 to 1, default 0.1. Lower values give more repeatable answers.

### top_p

From 0 to 1, default 0.9. This is the nucleus sampling cutoff.

### max_attempts_on_fail

From 1 to 100, default 2. The total number of attempts when a request fails.

### use_native_tools

On by default. Agents that use tools fail when this is off, because the node does not support prompted tool use.

### max_tokens

The maximum tokens to generate. A value of 0 or less, the default of -1, uses the model's own limit.

### prompt_model_config

The model configuration. Connect it to an Agent.
