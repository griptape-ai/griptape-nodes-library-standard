## What is it?

The SearchReplaceText node allows you to perform search and replace operations on multiline text content. It supports both simple text replacement and regular expression-based search and replace.

## When would I use it?

Use the SearchReplaceText node when:

- You need to modify multiline text by replacing specific patterns
- You want to perform case-sensitive or case-insensitive text replacement
- You need to use regular expressions for complex pattern matching
- You want to replace either all occurrences or just the first occurrence of a pattern
- You need to work with text that contains multiple lines or paragraphs

## How to use it

### Basic Setup

1. Add a SearchReplaceText node to your workflow
1. Connect or set the input text (can be multiline)
1. Set the search pattern
1. Set the replacement text (can be multiline)
1. Configure additional options as needed
1. Connect the output to nodes that accept text input

### Parameters

- **input_text**: The multiline text to perform search and replace on (string)
- **search_pattern**: The text or pattern to search for (string)
- **replacement_text**: The multiline text to replace the search pattern with (string)
- **case_sensitive**: Whether the search should be case sensitive (boolean, default: true)
- **use_regex**: Whether to treat the search pattern as a regular expression (boolean, default: false)
- **replace_all**: Whether to replace all occurrences or just the first one (boolean, default: true)

### Outputs

- **output**: The multiline text after performing search and replace (string)

## Examples

### Simple Text Replacement

1. Add a SearchReplaceText node to your workflow
1. Set the input text to:
    ```
    Hello World
    Welcome to Griptape
    ```
1. Set the search pattern to: "World"
1. Set the replacement text to: "Griptape"
1. The output will be:
    ```
    Hello Griptape
    Welcome to Griptape
    ```

### Case-Insensitive Replacement

1. Add a SearchReplaceText node to your workflow
1. Set the input text to:
    ```
    Hello WORLD
    Welcome to the WORLD
    ```
1. Set the search pattern to: "world"
1. Set the replacement text to: "Griptape"
1. Set case_sensitive to: false
1. The output will be:
    ```
    Hello Griptape
    Welcome to the Griptape
    ```

### Regular Expression Examples

> [!NOTE]
> **Basic Regex Patterns**
>
> ```
> Input: "Line 1\nLine 2\nLine 3"
> Search Pattern: "Line \d"
> Replacement: "Item"
> Use Regex: true
> Output: "Item\nItem\nItem"
> ```
>
> This pattern matches "Line" followed by any digit (`\d`).

> [!NOTE]
> **Removing Numbers**
>
> ```
> Input: "Product123, Item456, Order789"
> Search Pattern: "\d+"
> Replacement: ""
> Use Regex: true
> Output: "Product, Item, Order"
> ```
>
> This pattern matches one or more digits (`\d+`).

> [!NOTE]
> **Word Boundaries**
>
> ```
> Input: "cat in the hat"
> Search Pattern: "\bcat\b"
> Replacement: "dog"
> Use Regex: true
> Output: "dog in the hat"
> ```
>
> This pattern matches the word "cat" only when it appears as a complete word.

## Regex Reference

> [!NOTE]
> **Common Regex Patterns**
>
> | Pattern | Description                | Example                                 |
> | ------- | -------------------------- | --------------------------------------- |
> | `\n`    | Match a newline            | `Line 1\nLine 2`                        |
> | `\s`    | Match any whitespace       | `Hello\sWorld`                          |
> | `\d`    | Match any digit            | `\d+` matches "123"                     |
> | `[a-z]` | Match any lowercase letter | `[a-z]+` matches "hello"                |
> | `[A-Z]` | Match any uppercase letter | `[A-Z]+` matches "WORLD"                |
> | `.`     | Match any character        | `a.c` matches "abc"                     |
> | `*`     | Match 0 or more            | `a*` matches "", "a", "aa"              |
> | `+`     | Match 1 or more            | `a+` matches "a", "aa"                  |
> | `?`     | Match 0 or 1               | `a?` matches "", "a"                    |
> | `\b`    | Word boundary              | `\bcat\b` matches "cat" but not "catch" |

> [!WARNING]
> **Regex Mode**
>
> When using regex mode, special characters in the search pattern are treated as regex syntax. Make sure to escape special characters if you want to match them literally.

> [!TIP]
> **Plain Text Mode**
>
> When not using regex mode, the search pattern is treated as literal text, and special characters are escaped automatically. This is safer for simple text replacements.

## Important Notes

- When using regular expressions, make sure to properly escape special characters
- Case-insensitive search works for both plain text and regular expressions
- If the search pattern is not found, the original text is returned unchanged
- Invalid regular expressions will result in the original text being returned
- Case-insensitive mode only affects matching. The replacement text is inserted exactly as you typed it
- In regex mode, patterns are applied without the multiline flag, so `^` and `$` match only the start and end of the whole text, and `.` does not match a newline
- The replacement text is passed to Python's `re.sub` in both modes, so a backslash in it is interpreted. `\1` inserts the first captured group, and a bad escape returns the original text
- The node updates `output` as soon as any input or option changes, not only when the workflow runs
- If `input_text` or `search_pattern` is empty, the input text is returned unchanged
- The options `case_sensitive`, `use_regex` and `replace_all` are in the Options group, which is hidden until you expand it
- Newlines are preserved in both input and output text
- When using regex mode, you can use `\n` to match newlines in the search pattern

## Common Issues

- An invalid regular expression raises no error. The node returns the input unchanged, so check the pattern if nothing changes

## Additional Resources

For more comprehensive regex examples and patterns, check out:

- [Python Regular Expression HOWTO](https://docs.python.org/3/howto/regex.html)
- [Regex101](https://regex101.com/) - Interactive regex testing and debugging
- [Regular-Expressions.info](https://www.regular-expressions.info/) - Detailed regex tutorials
