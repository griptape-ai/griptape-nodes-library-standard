---
description: Joins up to four pieces of text into one, with a separator between them. The output updates as soon as an input or option changes, and again when the workflow runs.
when_to_use: Use it when you want to combine text from several nodes into one string, such as a title, body and conclusion, or several shot notes.
---

## When to use it

You have separate pieces of text and want one output. The node joins `input_1` to `input_4` in order, with a separator between them.

The node has exactly four inputs. To merge more, chain a second Merge Texts node, or build a list and use [Join List](../lists/JoinList.md).

## Example

To build a document from three sections:

1. Add a **Merge Texts** node.
1. Connect the title, body and conclusion text nodes to `input_1`, `input_2` and `input_3`.
1. Leave `merge_string` as `\n\n` so the sections are separated by a blank line.
1. `output` contains the three sections in order.

## Parameters

### merge_string

The separator placed between inputs. The default is `\n\n`. It is only added between inputs, never at the start or end.

- Type `\n` for a line break. A real newline typed into the field works too. Other escapes such as `\t` are not converted.
- If you clear it, the inputs are joined with nothing between them.

### whitespace

`keep` (the default) leaves the text as is. `trim` strips whitespace from each input and from the final result.

Empty or unconnected inputs are skipped, so they add no extra separator. Inputs that are not text are converted with `str`, so `2.5` becomes `2.5`.

## Edge cases

- If every input is empty, `output` is an empty string.
- In `keep` mode an input that contains only spaces is kept. In `trim` mode it is trimmed and then skipped.

## Common issues

- **Unexpected formatting.** Check that `merge_string` contains the whitespace you want.
- **Missing content.** Check that every input is connected, and that you are not using more than four.
