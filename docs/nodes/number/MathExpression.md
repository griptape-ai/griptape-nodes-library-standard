---
description: Evaluates a math expression using up to 26 variables named a to z.
when_to_use: Use it when you need to combine or transform numbers with a formula instead of chaining several math nodes.
---

## When to use it

Type a formula such as `a * b + 10` or `sin(radians(a))`, set the variables, and read the answer from `result`. The result updates as soon as you change an input.

## Example

With `expression` set to `(a + b) / 2`, `a` set to 10 and `b` set to 20, `result` is `15.0`.

## Parameters

### expression

Supports `+ - * / **`, parentheses and these functions: `abs round min max pow sum rand sqrt sin cos tan asin acos atan sinh cosh tanh exp log ceil floor fabs fmod degrees radians`. Constants are `e`, `tau`, `inf` and `nan`.

> [!WARNING]
> `atan2`, `log10`, `log2` and `pi` are not usable yet. An expression that uses any of them returns 0. This is a known bug (#1080). Use `log(x) / log(10)` for a base-10 logarithm and `3.14159265` for pi until it is fixed.

Multiplication can be left out: `2a`, `a(b+c)` and `ab` are read as `2*a`, `a*(b+c)` and `a*b`. Trig functions use radians.

`rand()` returns a number from 0 to 1, `rand(max)` from 0 to max, and `rand(min, max)` between the two. Because it is evaluated each time, the result can change between runs.

Loops, function definitions and comprehensions are not allowed.

### num_variables

How many variables are available, from 1 to 26. Only the first ones, starting at `a`, are shown and used.

### output_type

`float` or `int`. `int` drops the decimals toward zero, so 2.9 becomes 2 and -2.9 becomes -2.

### precision

Decimal places for float output, from 0 to 15. Values outside the range are clamped.

### Errors

A typo, an unknown name or a division by zero does not fail the node. The result is 0 instead. If a result looks like 0 unexpectedly, check the expression.

> [!WARNING]
> The constant `e` replaces the variable `e`. Using the fifth variable in an expression gives you Euler's number, not your input.
