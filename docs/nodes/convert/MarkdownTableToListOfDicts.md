---
description: Parses a Markdown table into a list of dictionaries, using the header row as the keys.
when_to_use: Use it when text or an LLM reply contains a Markdown table and you want one record per row.
---

## When to use it

Connect Markdown text that holds a table. Each row after the header becomes a dictionary. The output updates as you edit.

## Example

```markdown
| name | age |
|------|-----|
| Alice | 30 |
| Bob | 25 |
```

gives `[{"name": "Alice", "age": "30"}, {"name": "Bob", "age": "25"}]`.

## Parameters

### markdown

The first non-empty line is the header. Lines made only of dashes and colons, like `|---|:--:|`, are skipped. All lines are read as table rows, so the text should hold only the table. A short row is padded with empty text and extra cells are dropped. Use `\|` for a literal pipe inside a cell. All values are text.

### results

A summary with the row count, column names and the first row.
