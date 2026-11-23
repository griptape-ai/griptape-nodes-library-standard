---
description: Asks the agent's model to summarize the conversation, then replaces the history with that summary.
when_to_use: Use it when a long agent conversation is getting heavy and you want to keep the gist and drop the detail.
---

## When to use it

A long conversation costs more to send on every run. This node asks the agent's own model for a summary and replaces the whole history with it. The history becomes a single exchange with the input `conversation summary` and the summary as the output.

The summary run uses the agent's model and rulesets. If the agent has no model set, the default Griptape Cloud model is used. If the history is empty, the agent is returned unchanged and no model call is made.

## Parameters

### prompt

The instruction sent to the model. The default asks for a summary with specific and useful details and no other text. The parameter is hidden until you expose it. Edit it to change what gets kept, for example "Summarize the decisions we made".

### summary

The summary text. The agent's new history holds the same text.
