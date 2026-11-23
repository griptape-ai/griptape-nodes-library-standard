---
description: Sets up a model running on your own Ollama server for an Agent to use.
when_to_use: Use it when you want an Agent to run on a local model through Ollama, with no API key.
---

## When to use it

Run the Ollama server on your machine (or another you can reach), pull a model, then connect `prompt_model_config` to the `model` input of an [Agent](../Agent.md). No API key is needed.

If the node can't reach the server, it shows an Install Ollama message with a download link. If the server is running but has no models, it tells you to run `ollama pull <model>`.

## Parameters

### model

Lists the models installed on the server, sorted by name. The list is read when the node is created, so if Ollama wasn't running then, start it and press the refresh button next to the dropdown. Changing `base_url` or `port` also reloads the list. The workflow fails if the selected model isn't installed.

### connection_settings

`base_url` defaults to `http://127.0.0.1` and `port` to `11434`. Change them to use a server on another machine.

### use_native_tools

Leave it on if the agent uses tools. Tool use through prompting is not supported, so an agent with tools fails when this is off.
### max_tokens

Zero or less leaves the limit to the provider default.
