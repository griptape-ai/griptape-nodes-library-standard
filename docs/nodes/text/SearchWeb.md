---
description: Runs a web search and returns the results as text, optionally summarized by a model.
when_to_use: Use it when a workflow needs current information from the web and you do not need an Agent in the loop.
---

## When to use it

You have a query and want the search results as text. For example, look up a codec's recommended settings and pass the result to an [Agent](../agents/Agent.md) or a [Display Text](DisplayText.md) node.

To let an Agent decide for itself when to search, use the [Web Search](../tools/WebSearch.md) tool instead.

Without `summarize`, `output` holds up to five results, one per line, each as JSON with `title`, `url` and `description`. With `summarize` on, a model reads the results and writes an answer. The model call runs through Griptape Cloud, so a Griptape key is needed for that. The node's own validation only checks the search engine keys.

## Example

1. Add **Search Web** and set `prompt` to "OpenEXR multipart deep file format specification".
1. Leave `search_engine` as DuckDuckGo, which needs no key.
1. Turn on `summarize` to get a short written answer instead of the raw result list.
1. Run. The text is in `output`.

## Parameters

### prompt

The search query. An empty or whitespace-only query does nothing.

### summarize

Off by default. When on, a model uses the search tool and writes a summary into `output`.

### model

The model used for the search and summary. Offers `gpt-4.1`, `gpt-4.1-mini` (default), `gpt-4.1-nano` and `gpt-5`. The parameter is hidden by default in the editor.

### search_engine

DuckDuckGo (default), Google or Exa. The other engines need keys in Settings under API Keys & Secrets.

| Engine | Keys needed |
| --- | --- |
| DuckDuckGo | None |
| Google | `GOOGLE_API_KEY` and `GOOGLE_API_SEARCH_ID` |
| Exa | `EXA_API_KEY` |

If a key is missing the node shows a warning and the workflow fails validation. Pick another engine to continue.

### output

The search results, or the summary when `summarize` is on.
