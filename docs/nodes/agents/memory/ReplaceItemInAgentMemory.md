---
description: Edits one past exchange in an agent's conversation history.
when_to_use: Use it when you want to correct or rewrite something the agent said or was asked earlier, without clearing the whole history.
---

## When to use it

You want to fix one exchange in an [Agent](../Agent.md)'s history, such as a wrong answer that would keep misleading later runs. Connect the agent, pick the exchange, and type the new input, the new output, or both. The node returns the agent with that one exchange changed.

Use [Display Agent Memory](DisplayAgentMemory.md) to see the whole history first.

## Parameters

The memory parameters stay hidden until an agent is connected.

### memory_to_replace

Lists each exchange as its position in the history followed by the first 60 characters of its input, such as `0: Hey, how's it going?`. Positions start at 0. The list reads the agent's history when it is connected or changes, and shows `No memories available` when the history is empty. The node does nothing if no exchange is selected.

### orig_output

Read-only. Shows the output of the selected exchange.

### new_input / new_output

Leave a field blank to keep the original. The exchange is replaced with the new values only for the fields you fill in.
