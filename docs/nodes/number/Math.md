---
title: "Math Operations"
description: Does arithmetic on one or two numbers, such as add, divide, round, average or sine.
when_to_use: Use it when you need a single arithmetic step, such as an average, a rounded value or a remainder, and you would rather pick it from a list than write a formula.
---

## When to use it

Pick an operation from the dropdown, connect numbers to `A` and `B`, and read the answer from `result`. It covers basic arithmetic, powers, modulo, rounding, minimum and maximum, averages, square root, absolute value and sine. For a formula with several steps, use [Math Expression](MathExpression.md) instead.

## Example

To average two numbers, such as the frame counts of two shots:

1. Add a **Math** node and select `average [avg(A, B)]` in `operation`.
1. Connect the two numbers to `A` and `B`.
1. The average is in `result`.

## Parameters

### operation

Binary operations use both `A` and `B`:

- Add (A + B)
- Subtract (A - B)
- Multiply (A * B)
- Divide (A / B)
- Modulo (A % B)
- Power (A ^ B)
- Average (avg(A, B))
- Min (min(A, B))
- Max (max(A, B))

Unary operations use only `A`:

- Square Root (√A)
- Round (round(A))
- Ceiling (⌈A⌉)
- Floor (⌊A⌋)
- Absolute Value (|A|)
- Sine (sin(A))

### B

Hidden when `operation` is unary, since it is not used.

### result

The result of the operation. Inputs are converted to numbers first. Divide and Modulo by zero return infinity instead of failing.
