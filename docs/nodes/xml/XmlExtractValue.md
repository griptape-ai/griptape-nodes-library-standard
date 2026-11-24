---
description: Pulls elements, attributes or text out of XML using an XPath expression.
when_to_use: Use it when you need part of an XML document, such as a title, an attribute value or every matching element.
---

## When to use it

You have XML and only need part of it. Enter an XPath expression in `path` and the node returns the matches. The result updates as soon as `xml` or `path` changes.

## Example

A shot list exported from an editorial or tracking tool:

```xml
<shots>
  <shot id="SH010" status="in_progress"><title>Ship arrival</title><frames>96</frames></shot>
  <shot id="SH020" status="approved"><title>Dock fly-by</title><frames>120</frames></shot>
</shots>
```

| `path` | What it asks for | Output |
| --- | --- | --- |
| `//shot/@id` | The `id` of every shot | `SH010` and `SH020`, one per line |
| `//shot/title/text()` | The text of every title | `Ship arrival` and `Dock fly-by`, one per line |
| `//shot/title` | The whole `title` element of every shot | `<title>Ship arrival</title>` and `<title>Dock fly-by</title>`, one per line |
| `//shot[@status='approved']/title/text()` | The title of approved shots only | `Dock fly-by` |
| `//shot[1]/frames/text()` | The frame count of the first shot | `96` |
| `//shot[1]` | The whole first shot | The `shot` element, re-indented, with its children |

End a path with `/text()` for the words and `/@name` for an attribute. Without them you get the element, tags included.


## Parameters

### path

An XPath expression.

| Expression | Result |
| --- | --- |
| `//book/title` | Every `title` inside a `book` |
| `//item/@id` | The `id` attribute of every `item` |
| `//book[1]` | The first `book` |

If `path` is empty, the whole document is returned, re-indented.

### output

Text. Elements come back as indented XML. Attributes and text results come back as their string value. When there are several matches, they are joined with one per line. If nothing matches, or `xml` is empty, the output is empty.

> [!NOTE]
> Invalid XML or an invalid XPath expression raises an error that names the field.
