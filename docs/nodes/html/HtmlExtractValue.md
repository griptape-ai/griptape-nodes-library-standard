---
description: Pulls elements, attributes or text out of HTML using an XPath expression.
when_to_use: Use it when you need part of a web page's HTML, such as a heading, link addresses or a block with a given class.
---

## When to use it

You have HTML, for example from a scraped page, and only need part of it. Enter an XPath expression in `path` and the node returns the matches. The result updates as soon as `html` or `path` changes.

## Example

A vendor's release notes page, saved as HTML:

```html
<html><body>
  <h1>Pipeline Release Notes</h1>
  <div class="main">
    <p>Fixed EXR channel ordering.</p>
    <a href="https://example.com/v2.4/notes">Full notes</a>
    <a href="https://example.com/download">Download</a>
  </div>
</body></html>
```

| `path` | What it asks for | Output |
| --- | --- | --- |
| `//h1` | The heading element | `<h1>Pipeline Release Notes</h1>` |
| `//h1/text()` | The heading's words only | `Pipeline Release Notes` |
| `//div[@class="main"]/p/text()` | The text of the paragraph in the main block | `Fixed EXR channel ordering.` |
| `//a/@href` | The address of every link | `https://example.com/v2.4/notes` and `https://example.com/download`, one per line |

The same pattern works on a scraped page: find the heading, a block by its class, or the links, then pass the text on to an [Agent](../agents/Agent.md) to summarize or sort.


## Parameters

### path

An XPath expression.

| Expression | Result |
| --- | --- |
| `//h1` | Every `h1` element |
| `//a/@href` | The link address of every `a` |
| `//div[@class="main"]` | Every `div` with that class |

If `path` is empty, the whole parsed document is returned.

### output

Text. Elements come back as HTML. Attributes and text results come back as their string value. When there are several matches, they are joined with one per line. If nothing matches, or `html` is empty, the output is empty.

> [!NOTE]
> An invalid XPath expression raises an error that names the field. The HTML itself is parsed leniently and is not validated.
