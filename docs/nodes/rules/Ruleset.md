---
description: Creates a named set of rules that guide how an Agent responds.
when_to_use: Use it to constrain or direct the output of an Agent, for example to enforce naming conventions or a tone.
---

## When to use it

You want to give an [Agent](../agents/Agent.md) standing instructions. Connect `ruleset` to the Agent's **Behaviors (Rulesets)** input. To use several, combine them with [Ruleset List](RulesetList.md).

## Example

1. Add a **Ruleset** and set `name` to `ShotNotes`.
1. Set `rules` to the following. A blank line separates one rule from the next.

    ```
    Refer to shots by their code, such as SH010.

    Keep each note under two sentences.

    Never suggest a fix that changes the approved plate.
    ```

1. Connect `ruleset` to an Agent's **Behaviors (Rulesets)** input.

## Parameters

### name

The ruleset name. Defaults to `Behavior`.

### rules

The rules as text. Each blank-line-separated block becomes one rule. Lines directly under each other without a blank line stay in the same rule. The field's placeholder says "one per line", but the node splits on blank lines.

### ruleset

The output, a name and a list of rules.
