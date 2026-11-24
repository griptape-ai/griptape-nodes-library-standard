---
description: Combines up to four tools into one list that you can connect to an Agent.
when_to_use: Use it when an agent needs several tools and you would rather connect one wire than several.
---

## When to use it

An [Agent](../agents/Agent.md) accepts tools one at a time, so you can connect each tool node directly. Tool List gathers up to four of them into a single `tool_list` output. This is handy when the same set of tools feeds more than one Agent.

## Example

A shot-tracking agent needs to read the production notes in your workspace and know today's date.

1. Add a [File Manager](FileManager.md) and a [Date Time](DateTime.md) node.
1. Add **Tool List** and connect their `tool` outputs to `tool_1` and `tool_2`.
1. Connect `tool_list` to the Agent's `tools` input.

## Parameters

Empty inputs are skipped, so you can leave `tool_3` and `tool_4` unconnected. The node accepts only single tools on `tool_1` to `tool_4`. To combine more than four, chain a second Tool List into the Agent alongside the first.
