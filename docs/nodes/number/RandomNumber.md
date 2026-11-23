---
description: Generates a random whole number or decimal between a minimum and maximum, with an optional seed.
when_to_use: Use it when you need a random value, and want to repeat the same value later by keeping the seed.
---

## When to use it

Get a random value to vary a prompt setting, pick an index, or add noise. Leave `randomize_seed` off and the same seed gives the same number every run. Turn it on to get a new number each run.

## Parameters

### mode

`integer` gives a whole number. `float` gives a decimal.

### minimum and maximum

The range. Integers can land on either end. For floats the tooltip says the maximum is exclusive.

> [!WARNING]
> A `maximum` of 0 is read as not set and becomes 100. A `minimum` of 0 is fine. If you need 0 as the top of a range, shift the range and subtract afterward.

### randomize_seed

When on, a new seed is picked at the start of every run and the `seed` field is locked. The chosen seed is shown so you can copy it and turn this off to repeat a result.

### seed

The starting point for the random sequence. Default 42. Same seed, mode and range give the same result.

### result

The generated number.
