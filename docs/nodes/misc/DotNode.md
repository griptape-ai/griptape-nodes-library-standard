---
description: Passes a value through unchanged. Use it as a visual marker or reroute point.
when_to_use: Use it when you want to tidy up a connection or leave a waypoint on the canvas.
---

## When to use it

Place a Dot in the middle of a long connection to bend it, group several wires, or mark a spot. The value goes in and comes out untouched. It works like [Reroute](../execution/Reroute.md).

## Parameters

### value

The node adapts its type to what is connected. With a source wired in, it takes that output type. With only targets wired, it accepts the types they all share. With nothing connected it accepts anything.

A control connection can also pass through a Dot. Control outputs only drive one connection, so a control wire can't be connected to a Dot that fans out to more than one target.
