---
description: Runs a prompt through an agent that can use the tools of one MCP server.
when_to_use: Use it when you want a one-off task done with the tools from an MCP server, without building a full Agent setup.
---

## When to use it

You have an MCP server enabled in your settings and want a model to use its tools to answer a prompt. Pick the server, write the prompt and run. The result is streamed into `output`, with a `[Using tool ...]` line each time the model calls a tool.

To keep talking to the same agent afterwards, connect its `agent` output to another node. The returned agent keeps the new conversation and the tools and rules it came in with. The MCP server's tool is not added to it.

## Parameters

### mcp_server_name

Lists the MCP servers you have enabled. Use the refresh button after enabling a server. The run fails if the chosen server is not enabled. If the server has rules configured, they are added to the agent as a ruleset.

### agent

Optional. Connect an existing agent to continue its conversation and reuse its model, tools and rules. The `model_provider` and `model` parameters hide while an agent is connected. If the agent already has a tool for the same MCP server, that tool is skipped for this run.

### model_provider / model

Used only when no agent is connected. The default provider is Griptape Cloud, with a shorter list of models than the Agent node offers because not every model handles tool calls reliably. You can connect a Prompt Model Config to `model` instead.

### max_subtasks

Hidden by default. Limits how many tool-calling steps the model can take, from 1 to 100 (default 20). If the limit is hit, `output` is `Exceeded tool limit of N subtasks per task` instead of an answer.

### prompt

Required.

### context

Optional inputs of any type. Their values are appended to the end of the prompt.
