---
description: Gives an agent a calculator for exact arithmetic and simple algebra.
when_to_use: Use it when an agent has to add up, convert or compare numbers and you do not want it guessing.
---

## When to use it

Language models are unreliable at arithmetic. Connect this tool to an [Agent](../agents/Agent.md) and it can evaluate an expression instead of estimating the answer. For example, an agent that totals the frames in a shot can compute `14 * 24` rather than guess.

The agent sends one single-line Python-style expression at a time. If the expression is invalid, the agent receives the error text.

## Example

1. Add a **Calculator** node.
1. Connect its `tool` output to an Agent's `tools` input.
1. Ask the Agent a question that needs math, such as "How many frames are in a 14 second shot at 24 fps?"

## Parameters

### tool

Connect this to an Agent's `tools` input, or to a [Tool List](ToolList.md).
