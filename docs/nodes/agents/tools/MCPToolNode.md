---
description: Gives an agent access to the tools of one MCP server.
when_to_use: Use it when you want an Agent to call the tools of an MCP server you have enabled.
---

## When to use it

Pick an MCP server and connect the `tool` output to an [Agent](../Agent.md)'s tools input. The agent can then call that server's tools.

## Parameters

### mcp_server_name

Lists the MCP servers you have enabled in settings. Use the refresh button after enabling a server. The run fails with an error if the chosen server is not enabled.

### tool

The tool definition for the agent. It includes the server's configuration and any rules set for that server.
