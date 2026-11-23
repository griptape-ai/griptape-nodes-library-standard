---
description: Gives an agent a tool that searches the web with Exa, DuckDuckGo or Google.
when_to_use: Use it when an agent needs current information that is not in its training data.
---

## When to use it

Connect this tool to an [Agent](../agents/Agent.md) and it can run web searches. Each search returns up to five results with a title, URL and description. Pair it with [Web Scraper](WebScraper.md) if the agent should then read the pages.

## Example

1. Add a **Web Search** node and choose a `search_engine`.
1. Connect its `tool` output to an Agent's `tools` input.
1. Prompt the Agent with "Find recent articles about OpenEXR deep compositing workflows."

## Parameters

### search_engine

Exa is the default. Each engine reads its keys from your Settings under API Keys & Secrets.

| Engine | Keys needed |
| --- | --- |
| Exa | `EXA_API_KEY` |
| DuckDuckGo | None |
| Google | `GOOGLE_API_KEY` and `GOOGLE_API_SEARCH_ID` |

If a key is missing, the node shows a warning and the workflow fails validation before it runs. Pick DuckDuckGo to search without a key.

### tool

Connect this to an Agent's `tools` input, or to a [Tool List](ToolList.md).
