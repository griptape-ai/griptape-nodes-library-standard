---
description: Finds text and replaces it with other text, either as plain text or as a regular expression.
when_to_use: Use it when you need to change text by pattern, such as renaming a shot code across notes or stripping numbers from a list.
---

## When to use it

You have multiline text and want to replace a word or pattern in it. You can match plain text or a regular expression, ignore case, and replace every match or only the first. If nothing matches, the original text is returned unchanged. The output updates as soon as any input or option changes, not only when the workflow runs.

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

## Parameters

### input_text

The text to change. Newlines are preserved in the output. If this or `search_pattern` is empty, the input text is returned unchanged.

### search_pattern

Plain text by default. With `use_regex` on, it is a regular expression and special characters are regex syntax, so escape them to match them literally. Without `use_regex`, special characters are escaped for you, which is safer for simple replacements.

### replacement_text

The replacement is passed to Python's `re.sub` in both modes, so backslashes in it are interpreted: `\n` inserts a newline and `\\` inserts one backslash. With `use_regex` on, `\1` inserts the first captured group. In plain-text mode there are no groups, so `\1` or any other bad escape returns the original text. Case-insensitive mode only affects matching.

### case_sensitive

`case_sensitive`, `use_regex` and `replace_all` are in the Options group, which is hidden until you expand it. Case-insensitive search works for both plain text and regular expressions.

### use_regex

In regex mode, patterns are applied without the multiline flag, so `^` and `$` match only the start and end of the whole text, and `.` does not match a newline. Use `\n` in the pattern to match a newline.

## Common issues

An invalid regular expression raises no error. The node returns the input unchanged, so check the pattern if nothing changes (see #1072).

## Additional Resources

For more comprehensive regex examples and patterns, check out:

- [Python Regular Expression HOWTO](https://docs.python.org/3/howto/regex.html)
- [Regex101](https://regex101.com/) - Interactive regex testing and debugging
- [Regular-Expressions.info](https://www.regular-expressions.info/) - Detailed regex tutorials
