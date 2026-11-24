---
description: Wraps an Agent as a tool that other Agents can call.
when_to_use: Use it when you want one Agent to hand part of its work to a specialist Agent.
---

## When to use it

Set up an [Agent](../agents/Agent.md) for a specific job, such as research or copy editing. Connect it here, name it, and describe what it does. Connect the `tool` output to the `tools` input of another Agent. That Agent decides when to call it.

Without a connected agent, `tool` is empty.

## Parameters

### name

The name the calling Agent sees for the tool.

### description

Required. The calling Agent reads this to decide when to use the tool, so say what it does and when. An empty description raises an error.

### off_prompt

Not used. It stays so older saved workflows still load.
