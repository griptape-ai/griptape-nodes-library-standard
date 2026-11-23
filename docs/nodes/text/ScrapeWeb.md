---
description: Fetches a web page and returns its main text.
when_to_use: Use it when you have a URL and want the readable text of the page in your workflow.
---

## When to use it

You have the address of a page, such as a vendor's release notes, and want its text. Connect `output` to an [Agent](../agents/Agent.md) or a text node to work with it.

The node asks a model on Griptape Cloud to call a scraper on the URL, then returns the scraped page text as `output`. It needs a Griptape key for that model call. The node does not check for a key before the run, so a missing key shows up as an error when the node runs.

To let an Agent decide for itself which pages to read, use the [Web Scraper](../tools/WebScraper.md) tool.

## Example

1. Add **Scrape Web** and set `prompt` to the URL of a software's release notes page.
1. Connect `output` to an Agent and ask it which changes affect rendering.

## Parameters

### prompt

The URL to scrape. The text is passed to the model as "Scrape the web for information about: ..." and the model picks the page to fetch, so give it a complete URL and nothing else.

### model

The model that calls the scraper. Offers `gpt-4.1`, `gpt-4.1-mini` (default), `gpt-4.1-nano` and `gpt-5`. The parameter is hidden by default in the editor.

### output

The main text content of the page.
