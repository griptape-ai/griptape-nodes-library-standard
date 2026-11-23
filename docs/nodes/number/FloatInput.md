---
description: Provides a floating-point number that you set on the node.
when_to_use: Use it when you need a configurable decimal value, such as a threshold, a rate or a model temperature.
---

## When to use it

You want a decimal value you can set on the node and connect to other nodes, such as thresholds, rates, or inputs to mathematical operations. Values like 0.5, 3.14159 and -2.75 are all valid.

## Example

To set the temperature of an AI model:

1. Add a **Float Input** node and set `float` to 0.7.
1. Connect its output to the model's temperature parameter.
1. The model uses 0.7 as its temperature.

## Parameters

### float

The default is 0.0. The value can also be changed through the node's properties. The output, also named `float`, carries this value.
