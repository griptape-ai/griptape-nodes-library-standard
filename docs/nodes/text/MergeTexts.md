## What is it?

The MergeTexts node is a utility node that combines multiple text strings into a single unified text output. It allows you to consolidate separate pieces of text with a configurable separator between them.

## When would I use it?

Use the MergeTexts node when:

- You need to combine text from multiple sources or nodes
- You want to join separate pieces of text with consistent formatting
- You're building a workflow that generates content from multiple components
- You need to create a comprehensive document from individual sections

## How to use it

### Basic Setup

1. Add a MergeTexts node to your workflow
1. Connect multiple text outputs from other nodes to this node's inputs
1. Optionally configure the separator string
1. Connect the output to nodes that require the combined text

### Parameters

- **input_1 through input_4**: Four text inputs to be combined, in order
- **merge_string**: The separator to place between text segments. The default is `\n\n`, which is converted to two line breaks
- **whitespace**: Choose `trim` to strip whitespace from each input and from the final result, or `keep` (the default) to leave the text as is

### Outputs

- **output**: The combined text result as a single string

## Example

A workflow to create a complete document from separate sections:

1. Add a MergeTexts node to your workflow
1. Connect outputs from three different text nodes (e.g., title, body, conclusion)
1. Leave merge_string as `\n\n` for paragraph separation
1. The output will contain all text segments combined with the specified separator

## Important Notes

- Empty inputs are skipped, so they do not add an extra separator. In `keep` mode an input that contains only spaces is not empty and is kept. In `trim` mode it is trimmed first and then skipped
- The separator is only added between inputs, not at the beginning or end
- Type `\n` in the separator for a line break. If you clear the separator, the inputs are joined with nothing between them
- The output updates as soon as an input or option changes, and again when the workflow runs

## Common Issues

- **Unexpected formatting**: Check that your merge_string contains appropriate whitespace
- **Missing content**: Verify all input connections are properly established. The node has exactly four inputs. To merge more, chain a second Merge Texts node or build a list and use [Join List](../lists/JoinList.md)
