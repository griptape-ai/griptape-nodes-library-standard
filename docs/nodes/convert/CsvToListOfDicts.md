---
description: Parses CSV text into a list of dictionaries, using the header row as the keys.
when_to_use: Use it when you have a spreadsheet exported as CSV and want one record per row to loop over.
---

## When to use it

Paste CSV text or connect it from a file. Each row after the header becomes a dictionary. Use the result with loops, [SortList](../lists/SortList.md) or [GroupByKey](../dict/GroupByKey.md).

## Example

```text
name,age,city
Alice,30,NYC
Bob,25,LA
```

gives `[{"name": "Alice", "age": "30", "city": "NYC"}, {"name": "Bob", "age": "25", "city": "LA"}]`.

All values are text. Convert numbers downstream, for example with [ToInteger](ToInteger.md). A cell missing from a short row is empty (null).

## Parameters

### csv

The first row is the header. The output updates as you edit.

### delimiter

A single character such as `,`, `;` or a tab. A blank value falls back to a comma.

### results

A summary with the row count, column names and the first row.
