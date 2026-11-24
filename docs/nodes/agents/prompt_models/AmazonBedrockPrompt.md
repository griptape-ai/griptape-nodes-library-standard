---
description: Sets up an Amazon Bedrock model for an Agent to use.
when_to_use: Use it when you want an Agent to run on a model hosted in your AWS account through Bedrock.
---

## When to use it

Choose a Bedrock model and connect `prompt_model_config` to the `model` input of an [Agent](../Agent.md). When the node runs, it opens an AWS session so bad credentials fail here with a clear message.

You need three secrets saved in Settings: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` and `AWS_DEFAULT_REGION`. The workflow will not run without them.
## Parameters

### model

A text field. It starts at `us.anthropic.claude-opus-4-7`, and you can type any Bedrock model ID. The Claude 4.x and DeepSeek R1 models need the `us.` prefix because they are only available through cross-region inference profiles. Older Claude 3 IDs, and Claude 4.x IDs without the prefix, are replaced with the current equivalent and a notice tells you to save the workflow.

### max_tokens

This node starts with 100, which is short for most answers. Raise it if replies are cut off. Bedrock tends to fail when no limit is set, so avoid zero or less.

### use_native_tools

Leave it on if the agent uses tools. Tool use through prompting is not supported, so an agent with tools fails when this is off.

This node has no `seed`, `min_p` or `top_k` setting.
