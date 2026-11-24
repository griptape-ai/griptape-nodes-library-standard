---
description: Finds text and replaces it with other text, either as plain text or as a regular expression.
when_to_use: Use it when you need to change text by pattern, such as renaming a shot code across notes or stripping numbers from a list.
---

## When to use it

You have multiline text and want to replace a word or pattern in it. You can match plain text or a regular expression, ignore case, and replace every match or only the first. If nothing matches, the original text is returned unchanged. The output updates as soon as any input or option changes, not only when the workflow runs.

## Examples

### Simple text replacement

```
Input text:       Hello World
                  Welcome to Griptape
search_pattern:   World
replacement_text: Griptape

Output:           Hello Griptape
                  Welcome to Griptape
```

### Case-insensitive replacement

```
Input text:       Hello WORLD
search_pattern:   world
replacement_text: Griptape
case_sensitive:   false

Output:           Hello Griptape
```

### Regular expressions

With `use_regex` on:

| Goal | `search_pattern` | `replacement_text` | Input | Output |
| --- | --- | --- | --- | --- |
| Match "Line" and a digit | `Line \d` | `Item` | `Line 1\nLine 2` | `Item\nItem` |
| Remove numbers | `\d+` | empty | `Product123, Item456` | `Product, Item` |
| Match a whole word only | `\bcat\b` | `dog` | `cat in the hat` | `dog in the hat` |

## Regex reference

| Pattern | Description | Example |
| ------- | ----------- | ------- |
| `\n` | Match a newline (regex mode only) | `Line 1\nLine 2` |
| `\s` | Match any whitespace | `Hello\sWorld` |
| `\d` | Match any digit | `\d+` matches "123" |
| `[a-z]` | Match any lowercase letter | `[a-z]+` matches "hello" |
| `.` | Match any character | `a.c` matches "abc" |
| `*` | Match 0 or more | `a*` matches "", "a", "aa" |
| `+` | Match 1 or more | `a+` matches "a", "aa" |
| `?` | Match 0 or 1 | `a?` matches "", "a" |
| `\b` | Word boundary | `\bcat\b` matches "cat" but not "catch" |

For more patterns, see the [Python Regular Expression HOWTO](https://docs.python.org/3/howto/regex.html), [Regex101](https://regex101.com/) for interactive testing, and [Regular-Expressions.info](https://www.regular-expressions.info/).

## Parameters

### input_text

The text to change. Newlines are preserved. If this or `search_pattern` is empty, the input text is returned unchanged.

### search_pattern

Plain text by default, with special characters escaped for you. With `use_regex` on, it is a regular expression, so escape special characters to match them literally. In plain-text mode, `\n` searches for a backslash followed by an n, not a newline. Use regex mode to match a newline.

> [!WARNING]
> An invalid regular expression raises no error. The node returns the input unchanged, so check the pattern if nothing changes.

### replacement_text

Backslashes are interpreted in both modes: `\n` inserts a newline and `\\` inserts one backslash. With `use_regex` on, `\1` inserts the first captured group. In plain-text mode there are no groups, so `\1` or any other bad escape returns the original text.

### case_sensitive

`case_sensitive`, `use_regex` and `replace_all` are in the Options group, which is hidden until you expand it. Case-sensitive is the default. Case-insensitive works for both plain text and regular expressions.

### replace_all

On by default. Turn it off to replace only the first match.

### use_regex

Patterns are applied without the multiline flag, so `^` and `$` match only the start and end of the whole text, and `.` does not match a newline.

## Edge cases

- A pattern that can match nothing, such as `a*`, inserts the replacement at every empty match. `abc` becomes `--b-c-` with replacement `-`.
