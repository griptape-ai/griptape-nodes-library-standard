---
description: Gives an agent a tool that lists, reads and writes text files in your workspace directory.
when_to_use: Use it when an agent should read notes or save its output as files without you wiring up file nodes.
---

## When to use it

Connect this tool to an [Agent](../agents/Agent.md) and it can work with files in your workspace directory. It offers three actions:

- List the files in a folder.
- Read one or more text files.
- Save text to a file, creating folders as needed.

Paths are relative to the workspace directory. A path that resolves outside the workspace is refused. The tool reads and writes text only, and it cannot copy, move or delete files.

## Example

1. Add a **File Manager** node.
1. Connect its `tool` output to an Agent's `tools` input.
1. Prompt the Agent with "Read notes/SH010_dailies.txt and save a three-line summary to notes/SH010_summary.txt."

## Parameters

### tool

Connect this to an Agent's `tools` input, or to a [Tool List](ToolList.md). The node's badge shows the workspace directory in use.
