---
description: Shows an agent's conversation history as JSON.
when_to_use: Use it when you want to see what an agent remembers, or pass its past exchanges to other nodes.
---

## When to use it

You want to check what an [Agent](../Agent.md) has been told and what it answered. Connect the agent and run the node. The `agent` output passes the agent along unchanged, so you can keep it in the chain.

## Parameters

### memory

A JSON object with one key, `runs`. Each run is one exchange, with an `input` (what the agent was given) and an `output` (what it answered). With no agent connected, `runs` is an empty list.
