---
description: Sends the workflow down a Then or Else path based on a true or false value, and can pick between two data values.
when_to_use: Use it when the next step depends on a condition, or when you want to choose between two inputs.
---

## When to use it

Connect a condition to `evaluate`. If it is true the flow continues from **Then**. Otherwise it continues from **Else**. You can also wire two values into the Data Outputs group and use `output` to get the one that matches.

## Parameters

### evaluate

Accepts a bool, an integer or a string. Zero is false and any other integer is true. For strings, these are false after trimming and ignoring case: `false`, `falsey`, `f`, `no`, `n`, `negative`, `off`, `zero`, `0`, `0.0`, an empty string, `nope`, `nah`, `none`, `null`, `nyet`, `nein` and `disabled`. Every other string is true.

### output_if_true and output_if_false

Optional data inputs in the collapsed Data Outputs group. Only the branch that is chosen is computed. The other branch's upstream nodes are not run.

### output

The value from `output_if_true` or `output_if_false`, depending on `evaluate`. Its type follows what you connect. The first value you connect fixes the type, and image, audio and video values and their URL forms are treated as the same type.
