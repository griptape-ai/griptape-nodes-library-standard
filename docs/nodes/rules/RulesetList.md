---
description: Combines up to four rulesets into one list.
when_to_use: Use it when an Agent should follow several rulesets and each one is kept in its own Ruleset node.
---

## When to use it

You have more than one [Ruleset](Ruleset.md) and want to pass them to an Agent through a single connection. Empty inputs are skipped, so you can connect one to four.

## Example

1. Add a **Ruleset** named "Naming" with the rules for shot and asset names.
1. Add a second **Ruleset** named "Tone" with rules for how notes are phrased.
1. Add **Ruleset List** and connect them to `ruleset_1` and `ruleset_2`.
1. Connect `rulesets` to the **Behaviors (Rulesets)** input of an [Agent](../agents/Agent.md).

## Parameters

### ruleset_1 to ruleset_4

Optional Ruleset inputs.

### rulesets

The connected rulesets as a list, in input order.
