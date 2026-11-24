---
description: Saves text from your workflow to a UTF-8 file in your project.
when_to_use: Use it when you want to keep generated or processed text, such as shot notes or a log, as a file.
---

## When to use it

Connect text from an [Agent](../agents/Agent.md), a template or any other text source to write it to disk. The node has no outputs, so it is usually the last step of a chain. The saved location is written to the log.

## Example

1. Add an **Agent** that turns a list of shot statuses into a daily report.
1. Add **Save Text** and connect the agent's `output` to `text`.
1. Set `output_file` to `dailies_report.txt` and pick a `situation`.
1. Run. The file is written to the location the situation defines.

## Parameters

### text

The text to save. It is encoded as UTF-8.

### situation

Picks where the file is saved, using the situations defined in your project. Use the refresh button to reload the list. The dropdown hides when a file-destination node is connected to `output_file`, because that node decides the location.

### output_file

The file name to save as. The default is `griptape_output.txt`. Whether an existing file is overwritten or renamed is decided by the project situation, not by this node.
