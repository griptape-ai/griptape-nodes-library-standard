---
title: "Math Operations"
description: Does arithmetic on one or two numbers, such as add, divide, round, average or sine.
when_to_use: Use it when you need a single arithmetic step, such as an average, a rounded value or a remainder, and you would rather pick it from a list than write a formula.
---

## When to use it

Pick an operation from the dropdown, connect numbers to `A` and `B`, and read the answer from `result`. It covers basic arithmetic, powers, modulo, rounding, minimum and maximum, averages, square root, absolute value and sine. `A` and `B` default to 0 and `operation` defaults to add. For a formula with several steps, use [Math Expression](MathExpression.md) instead.

## Example

To average two numbers, such as the frame counts of two shots:

1. Add a **Math** node and select `average [avg(A, B)]` in `operation`.
1. Connect the two numbers to `A` and `B`.
1. The average is in `result`.

## Parameters

### operation

The dropdown entries are written `name [expression]`, for example `add [A + B]`. Binary operations use both `A` and `B`:

- `add [A + B]`
- `subtract [A - B]`
- `multiply [A * B]`
- `divide [A / B]`
- `modulo [A % B]`
- `power [A ^ B]`
- `average [avg(A, B)]`
- `min [min(A, B)]`
- `max [max(A, B)]`

Unary operations use only `A`:

- `sqrt [√A]`
- `round [round(A)]`
- `ceil [⌈A⌉]`
- `floor [⌊A⌋]`
- `abs [|A|]`
- `sin [sin(A)]`

`sin` works in radians. `round` follows Python's `round`, so halves go to the nearest even number (2.5 gives 2, 3.5 gives 4). `modulo` follows Python's `%`, so the result has the sign of `B`. `round`, `ceil` and `floor` return whole numbers. The other operations return floats.

### B

Hidden for unary operations. A value already in `B` is kept and ignored.

### result

The result of the operation. It is recalculated as soon as `operation`, `A` or `B` changes.

## Edge cases

- An empty `A` or `B` counts as 0. Text that is not a number is rejected with a `ValueError` when you set it.
- `divide` and `modulo` by zero return infinity (`inf`) instead of failing.
- `sqrt` of a negative number and `sin` of infinity raise `ValueError`. `power` of 0 to a negative exponent raises `ZeroDivisionError`.
- `power` of a negative number to a fractional exponent returns a complex number, such as `-8 ^ 0.5`.
