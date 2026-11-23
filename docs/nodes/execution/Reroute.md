---
description: Passes a connection through unchanged so you can route wires around a crowded canvas.
when_to_use: Use it to tidy long or crossing connections, or to fan one output out to several nodes from a single point.
---

## When to use it

You have a connection that crosses a lot of the canvas, or one output feeds several nodes far apart. Place a Reroute where the wire should bend, connect the source to it, and connect it to the targets.

The node does not change the value. Its single `passThru` parameter takes the type of whatever is connected, so it works with any data type. It also works on control (flow) connections.

## Example

A **Load Image** node at the top of a comp workflow feeds a blur at the bottom. Connect the image to a Reroute near the blur, then connect the Reroute to the blur's input. You can chain Reroutes to bend the wire more than once.

## Parameters

### passThru

Input and output in one. Its type follows the connections: the incoming source if there is one, otherwise the types the connected targets all accept. With no connections it accepts anything.

A control connection can pass through a Reroute, but only one wire can leave it. A control output cannot be connected into a Reroute that fans out to more than one target.
