---
description: Picks a random character, word, sentence or paragraph from your text, or generates one if there is no text.
when_to_use: Use it for placeholder or varied text, such as picking a random line from a list of review notes or a random word for a test prompt.
---

## When to use it

You want one random piece of a block of text. With text in `input_text`, the node splits it into characters, words, sentences or paragraphs and picks one. With `input_text` empty, it makes one up.

## Example

1. Add **Random Text** and paste several review notes into `input_text`, each separated by a blank line.
1. Set `selection_type` to `paragraph`.
1. Run. `output` holds one of the notes. The same `seed` picks the same note again.

## Parameters

### input_text

The text to choose from. Leave it empty to generate content instead.

### seed

Default 42. With the same input text, selection type and seed, the choice is the same each run. The value is hidden from editing while `randomize_seed` is on.

### randomize_seed

Off by default. When on, a new seed is chosen on each run.

### selection_type

`character`, `word` (default), `sentence` or `paragraph`.

- Sentences are split on `.`, `!` or `?` followed by whitespace.
- Paragraphs are split on blank lines.
- If the text has nothing to split into, the node generates content instead.

### output

The selected or generated text.

## Generated content

When `input_text` is empty:

- `character` is a random letter, digit or punctuation mark.
- `word` is a random common English word.
- `sentence` and `paragraph` are written by the `gpt-4.1-nano` model through Griptape Cloud, so a Griptape key is needed. The seed is only mentioned in the prompt, so the model may not repeat itself for the same seed. Changing `selection_type`, `seed` or `randomize_seed` starts a run immediately.
- Generated characters and words ignore the seed, so they differ on every run.
