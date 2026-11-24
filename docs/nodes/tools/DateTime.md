---
description: Gives an agent tools to read the current date and time and to calculate with dates.
when_to_use: Use it when an agent needs to know today's date or work out a deadline, such as the delivery date for a shot.
---

## When to use it

A language model does not know what day it is. Connect this tool to an [Agent](../agents/Agent.md) and it can:

- Get the current date and time.
- Parse a relative date such as "in 2 days" or "20 minutes ago".
- Add or subtract a time span from a date.
- Work out the difference between two dates.

The current time is read from the machine running the workflow, in its local time zone.

## Example

1. Add a **Date Time** node.
1. Connect its `tool` output to an Agent's `tools` input.
1. Prompt the Agent with "A shot is due 10 days from today. What is the date, and how many days are left until the 1st of next month?"

## Parameters

### tool

Connect this to an Agent's `tools` input, or to a [Tool List](ToolList.md).
