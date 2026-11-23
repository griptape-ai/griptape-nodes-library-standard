---
description: Replaces the text of elements or the value of attributes in XML, selected with an XPath expression.
when_to_use: Use it when you need to change a value in an XML document, such as a title or an attribute, without retyping the rest.
---

## When to use it

You have XML and want to swap one value. Set `path` to the element or attribute, put the new text in `replacement_value`, and read the changed XML from `output`. The result updates when `xml`, `path` or `replacement_value` changes.

## Parameters

### path

An XPath expression such as `//book/title` or `//item/@id`.

- Every match is replaced, not only the first. Use `//book[1]/title` to change one.
- For an element, its text is replaced and all of its child elements are removed.
- For an attribute, only the attribute value changes.
- If nothing matches, the XML is returned unchanged.
- If `path` is empty, nothing is replaced and the XML is just re-indented.

### replacement_value

Plain text. It is not parsed as XML, so you cannot use it to insert new elements.

### output

The full XML after the replacement, re-indented. If `xml` is empty, the output is empty.
