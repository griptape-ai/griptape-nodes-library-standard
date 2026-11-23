---
description: Wipes an agent's conversation history and returns the agent.
when_to_use: Use it when you want an agent to start fresh without losing its model, tools or rules.
---

## When to use it

An [Agent](../Agent.md) keeps its past exchanges and uses them as context on the next run. Put this node after the agent to forget all of that. The returned agent keeps its model, tools and rulesets. Only the conversation is removed.

To keep a short version of the history instead, use [Summarize Agent Memory](SummarizeAgentMemory.md).

## Parameters

### agent

Connect an agent to clear. The same parameter returns the agent with an empty history. With no agent connected, the node does nothing.
