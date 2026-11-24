---
description: Gives an agent a tool that fetches a web page and reads its main text.
when_to_use: Use it when an agent needs to read the content of a URL, such as a vendor's release notes or a reference page.
---

## When to use it

Connect this tool to an [Agent](../agents/Agent.md) and it can open a URL you give it, or one it found with [Web Search](WebSearch.md), and read the page. It needs no API key.

The tool returns the page's main text with links included. If the page can't be fetched or has no extractable content, the agent receives an error message and can try something else.

## Example

1. Add a **Web Scraper** node.
1. Connect its `tool` output to an Agent's `tools` input.
1. Prompt the Agent with "Summarize the changes on this page: https://example.com/release-notes".

## Parameters

### tool

Connect this to an Agent's `tools` input, or to a [Tool List](ToolList.md).
